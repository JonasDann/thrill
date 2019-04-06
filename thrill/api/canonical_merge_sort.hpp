/*******************************************************************************
 * thrill/api/canonical_merge_sort.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2018 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_CANONICAL_MERGE_SORT_HEADER
#define THRILL_API_CANONICAL_MERGE_SORT_HEADER

#include <thrill/api/context.hpp>
#include <thrill/api/dia.hpp>
#include <thrill/api/dop_node.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/common/math.hpp>
#include <thrill/common/porting.hpp>
#include <thrill/common/qsort.hpp>
#include <thrill/core/multisequence_selection.hpp>
#include <thrill/core/multiway_merge.hpp>
#include <thrill/data/sampled_file.hpp>
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
 * A DIANode which performs a Sort operation. It sorts a DIA according to a
 * given compare function
 *
 * \tparam ValueType Type of DIA elements
 *
 * \tparam CompareFunction Type of the compare function
 *
 * \tparam SortAlgorithm Type of the local sort function
 *
 * \ingroup api_layer
 */
template <
    typename ValueType,
    typename CompareFunction,
    typename SortAlgorithm>
class CanonicalMergeSortNode final : public DOpNode<ValueType>
{
    // TODO Unit test
    static constexpr bool debug = true;

    //! Set this variable to true to enable generation and output of stats
    static constexpr bool stats_enabled = true;

    using Super = DOpNode<ValueType>;
    using Super::context_;

    //! Timer or FakeTimer
    using Timer = common::StatsTimerBaseStopped<stats_enabled>;

    size_t run_capacity_;

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
        run_capacity_ = (context_.mem_limit() / 2) / sizeof(ValueType);
        LOG << "Run capacity: " << run_capacity_;
    }

    void StartPreOp(size_t /* id */) final {
        timer_total_.Start();
        timer_preop_.Start();
        current_run_.emplace_back(VectorSequenceAdapter());
        current_run_[0].reserve(run_capacity_);
    }

    void PreOp(const ValueType& input) {
        if (current_run_[0].size() >= run_capacity_) {
            FinishCurrentRun();
        }
        current_run_[0].push_back(input);
        local_items_++;
    }

    //! Receive a whole data::File of ValueType, but only if our stack is empty.
    bool OnPreOpFile(const data::File& file, size_t /* parent_index */) final {
        (void) file;
        // TODO What should this do?

        return false;
    }

    void StopPreOp(size_t /* id */) final {
        if (current_run_[0].size() > 0) {
            FinishCurrentRun();
        }
        std::vector<ValueType>().swap(current_run_[0]); // free vector

        timer_preop_.Stop();
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                    "CanonicalMergeSort() preop local_items_", local_items_);
            context_.PrintCollectiveMeanStdev(
                "CanonicalMergeSort() timer_preop_", timer_preop_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                "CanonicalMergeSort() timer_sort_", timer_sort_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "CanonicalMergeSort() timer_selection_", timer_preop_selection_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "CanonicalMergeSort() timer_scatter_", timer_preop_scatter_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "CanonicalMergeSort() timer_merge_", timer_merge_.SecondsDouble());
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
            context_.PrintCollectiveMeanStdev(
                "CanonicalMergeSort() timer_selection_", timer_global_selection_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                "CanonicalMergeSort() timer_scatter_", timer_global_scatter_.SecondsDouble());
        }
    }

    DIAMemUse PushDataMemUse() final {
        // Communicates how much memory the DIA needs on push data
        // TODO Make it work.
        return 0;
    }

    void PushData(bool consume) final {
        /* Phase 3 { */
        LOG << "Phase 3.";

        Timer timer_pushdata;
        timer_pushdata.Start();

        size_t local_size = 0;
        if (final_run_files_.size() == 0) {
            // nothing to push
        }
        else if (final_run_files_.size() == 1) {
            local_size = final_run_files_[0]->num_items();
            this->PushFile(*(final_run_files_[0]), consume);
        }
        else {
            size_t merge_degree, prefetch;
            std::tie(merge_degree, prefetch) =
                    context_.block_pool().MaxMergeDegreePrefetch(final_run_files_.size());

            std::vector<data::File::Reader> file_readers;
            for (size_t i = 0; i < final_run_files_.size(); i++) {
                LOG << "Run file " << i << " has size " << final_run_files_[i]->num_items();
                file_readers.emplace_back(final_run_files_[i]->GetReader(consume, /* prefetch */ 0));
            }

            StartPrefetch(file_readers, prefetch);

            LOG << "Building merge tree.";
            auto file_merge_tree = core::make_multiway_merge_tree<ValueType>(
                    file_readers.begin(), file_readers.end(), compare_function_);


            LOG << "Merging " << final_run_files_.size() << " files with prefetch " << prefetch << ".";
            timer_merge_.Start();
            while (file_merge_tree.HasNext()) {
                auto next = file_merge_tree.Next();
                this->PushItem(next);
                local_size++;
            }
            timer_merge_.Stop();
            LOG << "Finished merging.";
        }

        timer_pushdata.Stop();
        timer_total_.Stop();

        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                    "CanonicalMergeSort() local_size", local_size);
            context_.PrintCollectiveMeanStdev(
                "CanonicalMergeSort() timer_pushdata", timer_pushdata.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                "CanonicalMergeSort() timer_merge_", timer_merge_.SecondsDouble());
            size_t p = context_.num_workers();
            size_t total_time = context_.net.AllReduce(timer_total_.Milliseconds()) / p;
            double sort = (double) context_.net.AllReduce(timer_sort_.Milliseconds()) / p;
            double preop_scatter = (double) context_.net.AllReduce(timer_preop_scatter_.Milliseconds()) / p;
            double merge = (double) context_.net.AllReduce(timer_merge_.Milliseconds()) / p;
            double preop_selection = (double) context_.net.AllReduce(timer_preop_selection_.Milliseconds()) / p;
            double run_formation = (double) context_.net.AllReduce(timer_preop_.Milliseconds()) / p;
            double global_selection = (double) context_.net.AllReduce(timer_global_selection_.Milliseconds()) / p;
            double global_scatter = (double) context_.net.AllReduce(timer_global_scatter_.Milliseconds()) / p;
            double final_merge = (double) context_.net.AllReduce(timer_pushdata.Milliseconds()) / p;
            double other = total_time - sort - preop_scatter - global_scatter -
                    merge - preop_selection - global_selection;
            size_t result_size = context_.net.AllReduce(local_size);
            if (context_.my_rank() == 0) {
                LOG1 << "RESULT " << "operation=canonical_merge_sort"
                     << " total_time=" << total_time << " sort=" << sort
                     << " merge=" << merge
                     << " communication=" << preop_scatter + global_scatter
                     << " selection=" << preop_selection + global_selection
                     << " other=" << other << " run_formation=" << run_formation
                     << " global_selection=" << global_selection
                     << " global_scatter=" << global_scatter
                     << " final_merge=" << final_merge
                     << " workers=" << p << " result_size=" << result_size;
            }
        }

        /* } Phase 3 */
    }

    void Dispose() final {
        // TODO This may need to do something.
    }

private:
    size_t p_;

    using VectorSequenceAdapter = core::MultisequenceSelectorVectorSequenceAdapter<ValueType>;
    using FileSequenceAdapter = core::MultisequenceSelectorSampledFileSequenceAdapter<ValueType>;

    using LocalRanks = std::vector<std::vector<size_t>>;

    //! The comparison function which is applied to two elements.
    CompareFunction compare_function_;

    //! Sort function class
    SortAlgorithm sort_algorithm_;

    //! \name PreOp Phase
    //! \{

    //! Current run data (in a vector so it does not have to be reallocated when passing to run_multisequence...)
    std::vector<VectorSequenceAdapter> current_run_;
    //! Runs in the first phase of the algorithm
    std::vector<data::SampledFilePtr<ValueType>> run_files_;
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

    //! time spent in sort()
    Timer timer_sort_;

    //! time spent preop in multisequence selection
    Timer timer_preop_selection_;

    //! time spent global in multisequence selection
    Timer timer_global_selection_;

    //! time spent preop in communication
    Timer timer_preop_scatter_;

    //! time spent global in communication
    Timer timer_global_scatter_;

    //! time spent in merging lists
    Timer timer_merge_;

    //! total time spent
    Timer timer_total_;

    //! \}

    void ScatterRun(std::vector<ValueType>& run_seq,
                    data::StreamData::Writers& data_writers,
                    std::vector<size_t>& offsets) {
        size_t run_size = run_seq.size();
        size_t my_rank = context_.my_rank();
        size_t worker_count = offsets.size();
        size_t worker_rank = (my_rank + 1) % worker_count;
        size_t i = offsets[my_rank] % run_size;
        LOG0 << "Worker rank " << worker_rank << ".";
        while (worker_rank != my_rank || offsets[worker_rank] > i) {
            if (worker_rank != my_rank && offsets[worker_rank] <= i) {
                data_writers[worker_rank].Close();
                if (worker_rank + 1 >= worker_count) // last worker
                    i %= run_size;
                worker_rank = (worker_rank + 1) % worker_count;
                LOG0 << "Worker rank " << worker_rank << ".";
            } else {
                auto next = run_seq[i];
                data_writers[worker_rank].template Put<ValueType>(next);
                i++;
            }
        }
        data_writers[my_rank].Close();
    }

    void FinishCurrentRun() {
        /* Phase 1 {*/
        // Sort Locally
        LOG << "Phase 1.";
        LOG << "Sort run locally.";
        timer_sort_.Start();
        sort_algorithm_(current_run_[0].begin(), current_run_[0].end(), compare_function_);
        timer_sort_.Stop();

        // Calculate Splitters
        auto splitter_count = p_ - 1;
        LOG << "Calculating " << splitter_count << " splitters.";
        LocalRanks local_ranks(splitter_count, std::vector<size_t>(1));
        // TODO What to do when some PEs do not get the same amount of runs. (Dummy runs so every PE creates same amount of streams)
        timer_preop_selection_.Start();
        core::run_multisequence_selection<VectorSequenceAdapter, CompareFunction>
                (context_, compare_function_, current_run_, local_ranks,
                        splitter_count);
        timer_preop_selection_.Stop();
        LOG << "Local splitters: " << local_ranks;

        // Redistribute Elements
        auto data_stream = context_.template GetNewStream<data::CatStream>(this->dia_id());
        auto data_writers = data_stream->GetWriters();
        auto data_readers = data_stream->GetReaders();

        // Construct offsets vector
        std::vector<size_t> offsets(splitter_count + 1);
        std::transform(local_ranks.begin(), local_ranks.end(), offsets.begin(), [](std::vector<size_t> element) {
            return element[0];
        });
        offsets[splitter_count] = current_run_[0].size();

        LOG << "Scatter current run.";
        timer_preop_scatter_.Start();
        ScatterRun(current_run_[0], data_writers, offsets);
        timer_preop_scatter_.Stop();
        current_run_[0].clear();

        LOG << "Building merge tree.";
        auto multiway_merge_tree = core::make_multiway_merge_tree<ValueType>(
                data_readers.begin(), data_readers.end(), compare_function_);

        LOG << "Merging into run file.";
        run_files_.emplace_back(context_.template GetSampledFilePtr<ValueType>(this));
        auto current_run_file_writer = run_files_.back()->GetWriter();
        timer_merge_.Start();
        while (multiway_merge_tree.HasNext()) {
            auto next = multiway_merge_tree.Next();
            current_run_file_writer.template Put<ValueType>(next);
        }
        timer_merge_.Stop();
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
        std::vector<FileSequenceAdapter> run_file_adapters(run_count);
        for (size_t i = 0; i < run_count; i++) {
            run_file_adapters[i] = FileSequenceAdapter(run_files_[i]);
        }
        timer_global_selection_.Start();
        core::run_multisequence_selection<FileSequenceAdapter, CompareFunction>
                (context_, compare_function_, run_file_adapters, local_ranks,
                        splitter_count);
        timer_global_selection_.Stop();
        LOG << "Local splitters: " << local_ranks;

        // Redistribute Elements
        LOG << "Scatter " << run_count << " run files.";

        for (size_t run_index = 0; run_index < run_count; run_index++) {
            auto data_stream = context_.template GetNewStream<data::CatStream>(this->dia_id());

            // Construct offsets vector
            std::vector<size_t> run_offsets(splitter_count + 2);
            run_offsets[0] = 0;
            std::transform(local_ranks.begin(), local_ranks.end(), run_offsets.begin() + 1, [run_index](std::vector<size_t> element) {
                return element[run_index];
            });
            run_offsets[splitter_count + 1] = run_files_[run_index]->num_items();
            LOG << "Offsets: " << run_offsets;

            timer_global_scatter_.Start();
            data_stream->template Scatter<ValueType>(*run_files_[run_index],
                    run_offsets, true);
            timer_global_scatter_.Stop();

            auto final_run_file = context_.GetFilePtr(this);
            final_run_files_.emplace_back(final_run_file);
            data_stream->GetFile(final_run_file, true);

            data_stream.reset();
        }
        /* } Phase 2 */
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

#endif // !THRILL_API_CANONICAL_MERGE_SORT_HEADER

/******************************************************************************/
