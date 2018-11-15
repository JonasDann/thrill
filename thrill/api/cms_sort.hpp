/*******************************************************************************
 * thrill/api/cms_sort.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2015 Alexander Noe <aleexnoe@gmail.com>
 * Copyright (C) 2015 Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (C) 2015-2016 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_CMS_SORT_HEADER
#define THRILL_API_CMS_SORT_HEADER

#include <thrill/api/context.hpp>
#include <thrill/api/dia.hpp>
#include <thrill/api/dop_node.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/common/math.hpp>
#include <thrill/common/porting.hpp>
#include <thrill/common/qsort.hpp>
#include <thrill/core/multisequence_selection.hpp>
#include <thrill/core/multiway_merge.hpp>
#include <thrill/data/file.hpp>
#include <thrill/net/group.hpp>
#include <tlx/math/integer_log2.hpp>

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <functional>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace thrill {
namespace api {

/*!
 * A DIANode which performs a Sort operation. Sort sorts a DIA according to a
 * given compare function
 *
 * \tparam ValueType Type of DIA elements
 *
 * \tparam CompareFunction Type of the compare function
 *
 * \tparam SortAlgorithm Type of the local sort function
 *
 * \tparam Stable Whether or not to use stable sorting mechanisms
 *
 * \ingroup api_layer
 */
template <
    typename ValueType,
    typename CompareFunction,
    typename SortAlgorithm>
class CanonicalMergeSortNode final : public DOpNode<ValueType>
{
    static constexpr bool debug = true;

    //! Set this variable to true to enable generation and output of stats
    static constexpr bool stats_enabled = true;

    using Super = DOpNode<ValueType>;
    using Super::context_;

    //! Timer or FakeTimer
    using Timer = common::StatsTimerBaseStopped<stats_enabled>;
    //! RIAA class for running the timer
    using RunTimer = common::RunTimer<Timer>;

    struct MultiwayMergeTree {
        template <typename ReaderIterator,
                  typename Comparator = std::less<ValueType> >
        auto operator () (
            ReaderIterator seqs_begin, ReaderIterator seqs_end,
            const Comparator& comp = Comparator()) {

            return core::make_multiway_merge_tree<ValueType>(
                seqs_begin, seqs_end, comp);
        }
    };

    static const size_t run_capacity_ = 1000;

public:
    /*!
     * Constructor for a sort node.
     */
    template <typename ParentDIA>
    CanonicalMergeSortNode(const ParentDIA& parent,
             const CompareFunction& compare_function,
             const SortAlgorithm& sort_algorithm = SortAlgorithm())
        : Super(parent.ctx(), "Canonical Merge Sort", { parent.id() }, { parent.node() }),
          compare_function_(compare_function),
          sort_algorithm_(sort_algorithm)
    {
        // Hook PreOp(s)
        auto pre_op_fn = [this](const ValueType& input) {
                             PreOp(input);
                         };

        auto lop_chain = parent.stack().push(pre_op_fn).fold();
        parent.node()->AddChild(this, lop_chain);

        // Count of all workers (and count of target partitions)
        p_ = context_.num_workers();
    }

    void StartPreOp(size_t /* id */) final {
        timer_preop_.Start();
        current_run_.reserve(run_capacity_);
    }

    void PreOp(const ValueType& input) {
        if (current_run_.size() >= run_capacity_) {
            sort_algorithm_(current_run_.begin(), current_run_.end(), compare_function_);

            LOG << "Calculating " << p_ - 1 << " splitters.";
            std::vector<std::array<size_t, 1>> local_ranks(p_ - 1);
            core::MultisequenceSelector<VectorSequenceAdapter, CompareFunction, 1> selector(context_, compare_function_);
            VectorSequenceAdapter runAsArray[1] = {current_run_};
            selector.GetEquallyDistantSplitterRanks(runAsArray, local_ranks, p_ - 1);

            // TODO redistribute and save rest to file
        }
        current_run_.push_back(input);
        local_items_++;
    }

    //! Receive a whole data::File of ValueType, but only if our stack is empty.
    bool OnPreOpFile(const data::File& file, size_t /* parent_index */) final {
        (void) file;
        // TODO Sort whole file and redistribute

        return false;
    }

    void StopPreOp(size_t /* id */) final {
        if (current_run_.size() > 0) {
            sort_algorithm_(current_run_.begin(), current_run_.end(), compare_function_);
        }

        timer_preop_.Stop();
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                "Sort() timer_preop_", timer_preop_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                "Sort() preop local_items_", local_items_);
        }
    }

    DIAMemUse ExecuteMemUse() final {
        return DIAMemUse::Max();
    }

    //! Executes the sort operation.
    void Execute() final {
        MainOp();
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                "Sort() timer_execute", timer_execute_.SecondsDouble());
        }
    }

    DIAMemUse PushDataMemUse() final {
        //TODO What does this do?
        if (files_.size() <= 1) {
            // direct push, no merge necessary
            return 0;
        }
        else {
            // need to perform multiway merging
            return DIAMemUse::Max();
        }
    }

    void PushData(bool consume) final {
        Timer timer_pushdata;
        timer_pushdata.Start();

        size_t local_size = 0;
        if (files_.size() == 0) {
            // nothing to push
        }
        else if (files_.size() == 1) {
            local_size = files_[0].num_items();
            this->PushFile(files_[0], consume);
        }
        else {
            MultiwayMergeTree MakeMultiwayMergeTree;
            (void) MakeMultiwayMergeTree;

            // TODO Merge files
        }

        timer_pushdata.Stop();

        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                "Sort() timer_pushdata", timer_pushdata.SecondsDouble());

            context_.PrintCollectiveMeanStdev("Sort() local_size", local_size);
        }
    }

    void Dispose() final {
        files_.clear();
    }

private:
    size_t p_;

    using VectorSequenceAdapter = core::MultisequenceSelectorVectorSequenceAdapter<ValueType>;

    //! The comparison function which is applied to two elements.
    CompareFunction compare_function_;

    //! Sort function class
    SortAlgorithm sort_algorithm_;

    //! \name PreOp Phase
    //! \{

    //! Current run data
    VectorSequenceAdapter current_run_;
    //! Runs in the first phase of the algorithm
    std::vector<data::File> runs_;
    //! Number of items on this worker
    size_t local_items_ = 0;

    //! \}

    //! \name MainOp and PushData
    //! \{

    //! Local data files
    std::deque<data::File> files_;
    //! Total number of local elements after communication
    size_t local_out_size_ = 0;

    //! \}

    //! \name Statistics
    //! \{

    //! time spent in PreOp (including preceding Node's computation)
    Timer timer_preop_;

    //! time spent in Execute
    Timer timer_execute_;

    //! time spent in sort()
    Timer timer_sort_;

    //! \}

    void MainOp() {
        // TODO redistribute globally over all runs
    }
};

class DefaultSortAlgorithm
{
public:
    template <typename Iterator, typename CompareFunction>
    void operator () (Iterator begin, Iterator end, CompareFunction cmp) const {
        return std::sort(begin, end, cmp);
    }
};

template <typename ValueType, typename Stack>
template <typename CompareFunction>
auto DIA<ValueType, Stack>::CanonicalMergeSort(const CompareFunction& compare_function) const {
    assert(IsValid());

    using CanonicalMergeSortNode = api::CanonicalMergeSortNode<
        ValueType, CompareFunction, DefaultSortAlgorithm>;

    static_assert(
        std::is_convertible<
            ValueType,
            typename FunctionTraits<CompareFunction>::template arg<0> >::value,
        "CompareFunction has the wrong input type");

    static_assert(
        std::is_convertible<
            ValueType,
            typename FunctionTraits<CompareFunction>::template arg<1> >::value,
        "CompareFunction has the wrong input type");

    static_assert(
        std::is_convertible<
            typename FunctionTraits<CompareFunction>::result_type,
            bool>::value,
        "CompareFunction has the wrong output type (should be bool)");

    auto node = tlx::make_counting<CanonicalMergeSortNode>(*this, compare_function);

    return DIA<ValueType>(node);
}

} // namespace api
} // namespace thrill

#endif // !THRILL_API_CMS_SORT_HEADER

/******************************************************************************/
