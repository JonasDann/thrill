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
#include <thrill/data/block_reader.hpp>
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

    static const size_t run_capacity_ = 15;

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
            FinishCurrentRun();
        }
        current_run_.push_back(input);
        local_items_++;
    }

    //! Receive a whole data::File of ValueType, but only if our stack is empty.
    bool OnPreOpFile(const data::File& file, size_t /* parent_index */) final {
        (void) file;
        // TODO What should this do?

        return false;
    }

    void StopPreOp(size_t /* id */) final {
        if (current_run_.size() > 0) {
            FinishCurrentRun();
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
        // TODO What does this do? Make it work.
        return 0;
    }

    void PushData(bool consume) final {
        Timer timer_pushdata;
        timer_pushdata.Start();

        // TODO Push.
        (void) consume;

        timer_pushdata.Stop();

        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                "Sort() timer_pushdata", timer_pushdata.SecondsDouble());

            //context_.PrintCollectiveMeanStdev("Sort() local_size", local_size);
        }
    }

    void Dispose() final {
        // TODO This may need to do something.
    }

private:
    size_t p_;

    using VectorSequenceAdapter = core::MultisequenceSelectorVectorSequenceAdapter<ValueType>;
    using MultiVectorSelector = core::MultisequenceSelector<VectorSequenceAdapter, CompareFunction>;

    using FileSequenceAdapter = core::MultisequenceSelectorFileSequenceAdapter<ValueType>;
    using MultiFileSelector = core::MultisequenceSelector<FileSequenceAdapter, CompareFunction>;

    using LocalRanks = std::vector<std::vector<size_t>>;

    //! The comparison function which is applied to two elements.
    CompareFunction compare_function_;

    //! Sort function class
    SortAlgorithm sort_algorithm_;

    MultiVectorSelector vector_selector_ {context_, compare_function_};

    //! \name PreOp Phase
    //! \{

    //! Current run data
    VectorSequenceAdapter current_run_;
    //! Runs in the first phase of the algorithm
    std::vector<data::FilePtr> run_files_;
    //! Number of items on this worker
    size_t local_items_ = 0;

    //! \}

    //! \name MainOp and PushData
    //! \{
    //! Runs that are split up perfectly, just need to be merged
    std::vector<data::FilePtr> final_run_files_;
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

    void ScatterRun(std::vector<ValueType> run_seq,
                    data::StreamData::Writers &data_writers,
                    std::vector<size_t> &offsets) {
        size_t run_size = run_seq.size();
        size_t my_rank = context_.my_rank();
        size_t worker_count = offsets.size();
        size_t worker_rank = (my_rank + 1) % worker_count;
        size_t i = offsets[my_rank] % run_size;
        LOG << "Worker rank " << worker_rank << ".";
        while (worker_rank != my_rank || offsets[worker_rank] > i) {
            if (worker_rank != my_rank && offsets[worker_rank] <= i) {
                data_writers[worker_rank].Close();
                if (worker_rank + 1 >= worker_count) // last worker
                    i %= run_size;
                worker_rank = (worker_rank + 1) % worker_count;
                LOG << "Worker rank " << worker_rank << ".";
            } else {
                auto next = run_seq[i];
                data_writers[worker_rank].template Put<ValueType>(next);
                LOG << next;
                i++;
            }
        }
        data_writers[my_rank].Close();
    }

    void FinishCurrentRun() {
        /* Phase 1 {*/
        // Sort Locally
        LOG << "Sort run locally.";
        timer_sort_.Start();
        sort_algorithm_(current_run_.begin(), current_run_.end(), compare_function_);
        timer_sort_.Stop();

        // Calculate Splitters
        auto splitter_count = p_ - 1;
        LOG << "Calculating " << splitter_count << " splitters.";
        LocalRanks local_ranks(splitter_count, std::vector<size_t>(1));
        std::vector<VectorSequenceAdapter> current_run_vector(1);
        current_run_vector[0] = current_run_;
        // TODO What to do when some PEs do not get the same amount of runs. (Dummy runs so every PE creates same amount of streams)
        vector_selector_.GetEquallyDistantSplitterRanks(current_run_vector, local_ranks, splitter_count);
        LOG << "Local splitters: " << local_ranks;

        // Redistribute Elements
        auto data_stream = context_.template GetNewStream<data::CatStream>(this->id());
        auto data_writers = data_stream->GetWriters();

        // Construct offsets vector
        std::vector<size_t> offsets(splitter_count + 1);
        std::transform(local_ranks.begin(), local_ranks.end(), offsets.begin(), [](std::vector<size_t> element) {
            return element[0];
        });
        offsets[splitter_count] = current_run_.size();

        LOG << "Scatter current run.";
        ScatterRun(current_run_, data_writers, offsets);
        current_run_.clear();

        auto data_readers = data_stream->GetReaders();
        LOG << "Building merge tree.";
        auto multiway_merge_tree = core::make_multiway_merge_tree<ValueType>(
                data_readers.begin(), data_readers.end(), compare_function_);

        LOG << "Merging into run file.";
        run_files_.emplace_back(context_.GetFilePtr(this));
        auto current_run_file_writer = run_files_.back()->GetWriter();
        while (multiway_merge_tree.HasNext()) {
            auto next = multiway_merge_tree.Next();
            current_run_file_writer.template Put<ValueType>(next);
            LOG << next;
        }
        current_run_file_writer.Close();
        LOG << "Finished run has " << run_files_.back()->num_items() << " elements.";

        data_stream.reset();
        /* } Phase 1 */
    }

    void MainOp() {
        /* Phase 2 { */
        LOG << "Phase 2.";
        // Calculate Splitters
        auto splitter_count = p_ - 1;
        auto run_count = run_files_.size();
        LOG << "Calculating " << splitter_count << " splitters.";
        LocalRanks local_ranks(splitter_count, std::vector<size_t>(run_count));
        MultiFileSelector selector(context_, compare_function_);
        std::vector<FileSequenceAdapter> run_file_adapters(run_count);
        for (size_t i = 0; i < run_count; i++) {
            run_file_adapters[i] = FileSequenceAdapter(run_files_[i]);
        }
        selector.GetEquallyDistantSplitterRanks(run_file_adapters, local_ranks, splitter_count);
        LOG << "Local splitters: " << local_ranks;

        // Redistribute Elements
        LOG << "Scatter " << run_count << " run files.";

        for (size_t run_index = 0; run_index < run_count; run_index++) {
            auto data_stream = context_.template GetNewStream<data::CatStream>(this->id());

            // Construct offsets vector
            std::vector<size_t> run_offsets(splitter_count + 2);
            run_offsets[0] = 0;
            std::transform(local_ranks.begin(), local_ranks.end(), run_offsets.begin() + 1, [run_index](std::vector<size_t> element) {
                return element[run_index];
            });
            run_offsets[splitter_count + 1] = run_files_[run_index]->num_items();
            LOG << "Offsets: " << run_offsets;

            data_stream->template Scatter<ValueType>(*run_files_[run_index],
                    run_offsets, true);

            auto final_run_file = context_.GetFilePtr(this);
            final_run_files_.emplace_back(final_run_file);
            data_stream->GetFile(final_run_file, true);

            auto reader = final_run_file->GetKeepReader();
            while (reader.HasNext()) {
                LOG << reader.template Next<ValueType>();
            }

            data_stream.reset();
        }
        /* } Phase 2 */

        /* Phase 3 { */
        LOG << "Phase 3.";
        std::vector<data::File::ConsumeReader> file_readers;
        for (size_t i = 0; i < run_count; i++) {
            LOG << "Run file " << i << " has size " << final_run_files_[i]->num_items();
            file_readers.emplace_back(final_run_files_[i]->GetConsumeReader());
        }

        LOG << "Building merge tree.";
        auto file_merge_tree = core::make_multiway_merge_tree<ValueType>(
                file_readers.begin(), file_readers.end(), compare_function_);

        LOG << "Merging.";
        while (file_merge_tree.HasNext()) {
            auto next = file_merge_tree.Next();
            LOG << next;
        }
        LOG << "Finished merging.";
        /* } Phase 3 */
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
