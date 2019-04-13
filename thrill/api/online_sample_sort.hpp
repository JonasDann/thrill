/*******************************************************************************
 * thrill/api/online_sample_sort.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2019 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_ONLINE_SAMPLE_SORT_HEADER
#define THRILL_API_ONLINE_SAMPLE_SORT_HEADER

#include <thrill/api/dia.hpp>
#include <thrill/api/dop_node.hpp>
#include <thrill/core/online_sampler.hpp>

namespace thrill {
namespace api {

/*!
 * A DIANode which performs a Sort operation. It sorts a DIA according to a
 * given compare function
 *
 * \tparam ValueType Type of DIA elements
 *
 * \tparam Comparator Type of the compare function
 *
 * \tparam SortAlgorithm Type of the local sort function
 *
 * \ingroup api_layer
 */
template <
        typename ValueType,
        typename Comparator,
        typename SortAlgorithm>
class OnlineSampleSortNode final : public DOpNode<ValueType>
{
    // TODO Unit test
    static constexpr bool debug = false;

    //! Set this variable to true to enable generation and output of stats
    static constexpr bool stats_enabled = true;

    using Super = DOpNode<ValueType>;
    using Super::context_;

    //! Timer or FakeTimer
    using Timer = common::StatsTimerBaseStopped<stats_enabled>;

    size_t capacity_;

public:
    /*!
     * Constructor for a sort node.
     */
    template <typename ParentDIA>
    OnlineSampleSortNode(const ParentDIA& parent,
                           const Comparator& comparator,
                           const SortAlgorithm& sort_algorithm = SortAlgorithm())
            : Super(parent.ctx(), "Online Sample Sort", { parent.id() }, { parent.node() }),
              comparator_(comparator), sort_algorithm_(sort_algorithm),
              sampler_(10, 60, parent.ctx(), comparator, sort_algorithm)
    {
        // Hook PreOp(s)
        auto pre_op_fn = [this](const ValueType& input) {
            PreOp(input);
        };

        auto lop_chain = parent.stack().push(pre_op_fn).fold();
        parent.node()->AddChild(this, lop_chain);

        // Count of all workers (and count of target partitions)
        p_ = context_.num_workers();
        capacity_ = (context_.mem_limit() / 2) / sizeof(ValueType);
        LOG << "Capacity: " << capacity_;
    }

    void StartPreOp(size_t /* id */) final {
        timer_total_.Start();
        timer_preop_.Start();
    }

    void PreOp(const ValueType& input) {

        local_items_++;
    }

    //! Receive a whole data::File of ValueType, but only if our stack is empty.
    bool OnPreOpFile(const data::File& file, size_t /* parent_index */) final {
        (void) file;
        // TODO What should this do?

        return false;
    }

    void StopPreOp(size_t /* id */) final {

        timer_preop_.Stop();
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                    "CanonicalMergeSort() preop local_items_", local_items_);
            context_.PrintCollectiveMeanStdev(
                    "CanonicalMergeSort() timer_preop_", timer_preop_.SecondsDouble());
        }
    }

    DIAMemUse ExecuteMemUse() final {
        return DIAMemUse::Max();
    }

    //! Executes the sort operation.
    void Execute() final {
        Timer timer_mainop;
        timer_mainop.Start();
        MainOp();
        timer_mainop.Stop();
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                    "CanonicalMergeSort() timer_mainop", timer_mainop.SecondsDouble());
        }
    }

    DIAMemUse PushDataMemUse() final {
        // Communicates how much memory the DIA needs on push data
        // TODO Make it work.
        return 0;
    }

    void PushData(bool consume) final {

    }

    void Dispose() final {
        // TODO This may need to do something.
    }

private:
    size_t p_;

    //! The comparison function which is applied to two elements.
    Comparator comparator_;

    //! Sort function class
    SortAlgorithm sort_algorithm_;

    //! \name PreOp Phase
    //! \{

    //! Online sampler
    core::OnlineSampler<ValueType, Comparator, SortAlgorithm> sampler_;
    //! Number of items on this worker
    size_t local_items_ = 0;

    //! \}

    //! \name MainOp and PushData
    //! \{



    //! \}

    //! \name Statistics
    //! \{

    //! time spent in PreOp (including preceding node's computation)
    Timer timer_preop_;

    //! time spent in sort()
    Timer timer_sort_;

    //! time spent in partitioning
    Timer timer_partition_;

    //! time spent in communication
    Timer timer_communication_;

    //! total time spent
    Timer timer_total_;

    //! \}

    void MainOp() {

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
auto DIA<ValueType, Stack>::OnlineSampleSort(const CompareFunction& compare_function) const {
    assert(IsValid());

    using OnlineSampleSortNode = api::OnlineSampleSortNode<
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

    auto node = tlx::make_counting<OnlineSampleSortNode>(*this, compare_function);

    return DIA<ValueType>(node);
}

} // namespace api
} // namespace thrill

#endif // !THRILL_API_ONLINE_SAMPLE_SORT_HEADER

/******************************************************************************/