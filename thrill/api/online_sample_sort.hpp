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
#include <thrill/core/multiway_merge.hpp>
#include <thrill/data/sampled_file.hpp>

#include <tlx/vector_free.hpp>

#include <algorithm>
#include <vector>
#include <utility>

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
    static constexpr bool debug = true;

    //! Set this variable to true to enable generation and output of stats
    static constexpr bool stats_enabled = true;

    using Super = DOpNode<ValueType>;
    using Super::context_;

    //! Timer or FakeTimer
    using Timer = common::StatsTimerBaseStopped<stats_enabled>;

    using SampleIndexPair = std::pair<ValueType, size_t>;
    using LocalRanks = std::vector<std::vector<size_t>>;

    const size_t b_ = 5;
    const size_t k_ = 60000;
    size_t run_capacity_;

public:
    /*!
     * Constructor for a sort node.
     */
    template <typename ParentDIA>
    OnlineSampleSortNode(const ParentDIA& parent,
            const Comparator& comparator,
            const size_t r = 1,
            const SortAlgorithm& sort_algorithm = SortAlgorithm())
        : Super(parent.ctx(), "Online Sample Sort", { parent.id() },
                { parent.node() }),
          comparator_(comparator), sort_algorithm_(sort_algorithm),
          parent_stack_empty_(ParentDIA::stack_empty),
          sampler_(b_, k_, parent.ctx(), this->dia_id(), comparator,
                   sort_algorithm, r)
    {
        // Hook PreOp(s).
        auto pre_op_fn = [this](const ValueType& input) {
            PreOp(input);
        };

        auto lop_chain = parent.stack().push(pre_op_fn).fold();
        parent.node()->AddChild(this, lop_chain);

        // Count of all workers (and count of target partitions).
        p_ = context_.num_workers();
        run_capacity_ = (context_.mem_limit() / 2) / sizeof(ValueType);
        LOG << "Run capacity: " << run_capacity_;
    }

    void StartPreOp(size_t /* id */) final {
        LOG << "Phase: Run Formation";
        timer_total_.Start();
        timer_pre_op_.Start();
        current_run_.reserve(run_capacity_);
        final_splitters_.reserve(p_ - 1);
    }

    void PreOp(const ValueType& input) {
        // TODO Handle different amounts of elements on PEs
        auto has_next_capacity = sampler_.Put(input);
        if (!has_next_capacity) {
            LOG0 << "Collapse sampler buffers.";
            timer_sample_.Start();
            sampler_.Collapse();
            timer_sample_.Stop();
        }
        current_run_.emplace_back(input);
        if (current_run_.size() >= run_capacity_) {
            FinishCurrentRun();
        }
    }

    //! Receive a whole data::File of ValueType, but only if our stack is empty.
    bool OnPreOpFile(const data::File& file, size_t /* parent_index */) final {
        if (!parent_stack_empty_) {
            LOGC(common::g_debug_push_file)
            << "OnlineSampleSort rejected File from parent "
            << "due to non-empty function stack.";
            return false;
        }

        // Accept file.
        auto reader = file.Copy().GetConsumeReader();
        while (reader.HasNext()) {
            PreOp(reader.template Next<ValueType>());
        }

        return true;
    }

    void StopPreOp(size_t /* id */) final {
        LOG << "Finish last run.";
        if (current_run_.size() > 0) {
            FinishCurrentRun(true);
        }
        tlx::vector_free(current_run_);

        timer_pre_op_.Stop();
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() pre op local items", local_items_);
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() pre op timer",
                    timer_pre_op_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() pre op sample timer",
                    timer_sample_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() pre op partition timer",
                    timer_partition_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() pre op sort timer",
                    timer_sort_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() pre op communication timer",
                    timer_pre_op_communication_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() pre op file io timer",
                    timer_pre_op_file_io_.SecondsDouble());
            sampler_.PrintStats();
        }
    }

    DIAMemUse ExecuteMemUse() final {
        return DIAMemUse::Max();
    }

    void Execute() final {
        Timer timer_main_op;
        Timer timer_global_splitter;
        LOG << "Phase: Global Redistribution";
        timer_main_op.Start();

        // Calculate splitters.
        auto splitter_count = final_splitters_.size();
        auto run_count = run_files_.size();
        auto max_run_count = context_.net.AllReduce(run_count,
                [](const size_t& a, const size_t& b){
                    return std::max(a, b);
                });
        LOG << "Add " << max_run_count - run_count << " dummy runs";
        while (run_files_.size() < max_run_count) {
            run_files_.push_back(context_.template GetSampledFilePtr
                    <ValueType>(this));
        }
        run_count = max_run_count;

        LOG << "Calculate " << splitter_count << " splitters for " << run_count
            << " runs each.";
        LocalRanks local_ranks(splitter_count, std::vector<size_t>(run_count));
        timer_global_splitter.Start();
        for (size_t s = 0; s < splitter_count; s++) {
            for (size_t r = 0; r < run_count; r++) {
                local_ranks[s][r] = run_files_[r]->GetFastIndexOf(
                        final_splitters_[s], run_files_[r]->num_items(),
                        comparator_);
            }
        }
        timer_global_splitter.Stop();

        // Redistribute elements.
        LOG << "Scatter " << run_count << " run files.";
        local_items_ = 0;
        for (size_t run_index = 0; run_index < run_count; run_index++) {
            auto data_stream = context_.template GetNewStream<data::CatStream>(
                    this->dia_id());

            // Construct offsets vector.
            std::vector<size_t> run_offsets(splitter_count + 2);
            run_offsets[0] = 0;
            std::transform(local_ranks.begin(), local_ranks.end(),
                    run_offsets.begin() + 1,
                    [run_index](std::vector<size_t> element) {
                return element[run_index];
            });
            run_offsets[splitter_count + 1] = run_files_[run_index]->
                    num_items();
            LOG << "Offsets[" << run_index << "]: " << run_offsets;
            LOG << run_offsets[context_.my_rank() + 1] -
                   run_offsets[context_.my_rank()] << " / "
                << run_files_[run_index]->num_items()
                << " elements will stay local.";

            timer_global_communication_.Start();
            data_stream->template Scatter<ValueType>(*run_files_[run_index],
                                                     run_offsets, true);
            timer_global_communication_.Stop();

            auto final_run_file = context_.GetFilePtr(this);
            final_run_files_.push_back(final_run_file);
            data_stream->GetFile(final_run_file);
            local_items_ += final_run_file->num_items();

            data_stream.reset();
        }

        timer_main_op.Stop();

        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() main op local items", local_items_);
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() main op timer",
                    timer_main_op.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() main op splitter timer",
                    timer_global_splitter.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampleSort() main op communication timer",
                    timer_global_communication_.SecondsDouble());
        }
    }

    //! Communicates how much memory the DIA needs on push data
    DIAMemUse PushDataMemUse() final {
        if (final_run_files_.size() <= 1) {
            // Direct push, no merge necessary.
            return 0;
        }
        else {
            // Need to perform multi way merging.
            return DIAMemUse::Max();
        }
    }

    void PushData(bool consume) final {
        // TODO Partial multi way merge, when there are too many runs
        // TODO Remove dummy runs, if still empty
        Timer timer_push_data;
        timer_push_data.Start();

        if (final_run_files_.size() == 0) {
            // nothing to push
        }
        else if (final_run_files_.size() == 1) {
            this->PushFile(*(final_run_files_[0]), consume);
        }
        else {
            size_t merge_degree, prefetch;
            std::tie(merge_degree, prefetch) =
                    context_.block_pool().MaxMergeDegreePrefetch(final_run_files_.size());

            std::vector<data::File::Reader> file_readers;
            for (size_t i = 0; i < final_run_files_.size(); i++) {
                LOG << "Run file " << i << " has size " << final_run_files_[i]->num_items();
                file_readers.push_back(final_run_files_[i]->GetReader(consume, /* prefetch */ 0));
            }

            StartPrefetch(file_readers, prefetch);

            LOG << "Building merge tree.";
            auto file_merge_tree = core::make_multiway_merge_tree<ValueType>(
                    file_readers.begin(), file_readers.end(), comparator_);


            LOG << "Merging " << final_run_files_.size() << " files with prefetch " << prefetch << ".";
            timer_merge_.Start();
            ValueType first_element;
            if (debug && file_merge_tree.HasNext()) {
                first_element = file_merge_tree.Next();
                this->PushItem(first_element);
            }
            ValueType last_element;
            while (file_merge_tree.HasNext()) {
                auto next = file_merge_tree.Next();
                this->PushItem(next);
                if (debug) {
                    last_element = next;
                }
            }
            timer_merge_.Stop();
            LOG << "Finished merging (first element: " << first_element
                << ", last element: " << last_element << ").";
        }
        timer_push_data.Stop();
        timer_total_.Stop();

        if (stats_enabled) {
            size_t p = context_.num_workers();
            size_t total_time = context_.net.AllReduce(
                    timer_total_.Milliseconds()) / p;
            double sort = (double) context_.net.AllReduce(
                    timer_sort_.Milliseconds()) / p;
            double sample = (double) context_.net.AllReduce(
                    timer_sample_.Milliseconds()) / p;
            double pre_op_communication = (double) context_.net.AllReduce(
                    timer_pre_op_communication_.Milliseconds()) / p;
            double merge = (double) context_.net.AllReduce(
                    timer_merge_.Milliseconds()) / p;
            double partition = (double) context_.net.AllReduce(
                    timer_partition_.Milliseconds()) / p;
            double run_formation = (double) context_.net.AllReduce(
                    timer_pre_op_.Milliseconds()) / p;
            double global_communication = (double) context_.net.AllReduce(
                    timer_global_communication_.Milliseconds()) / p;
            double final_merge = (double) context_.net.AllReduce(
                    timer_push_data.Milliseconds()) / p;
            double other = total_time - sort - pre_op_communication -
                           global_communication - merge - sample - partition;
            if (context_.my_rank() == 0) {
                LOG1 << "RESULT " << "operation=online_sample_sort"
                     << " total_time=" << total_time << " sort=" << sort
                     << " merge=" << merge << " sample=" << sample
                     << " communication=" << pre_op_communication + global_communication
                     << " partition=" << partition
                     << " other=" << other << " run_formation=" << run_formation
                     << " global_communication=" << global_communication
                     << " final_merge=" << final_merge
                     << " workers=" << p; //<< " result_size=" << result_size;
            }
        }
    }

    void Dispose() final {
        run_files_.clear();
        final_splitters_.clear();
        final_run_files_.clear();
    }

private:
    size_t p_;

    //! The comparison function which is applied to two elements.
    Comparator comparator_;

    //! Sort function class
    SortAlgorithm sort_algorithm_;

    //! Whether the parent stack is empty
    const bool parent_stack_empty_;

    //! \name PreOp Phase
    //! \{

    //! Online sampler
    core::OnlineSampler<ValueType, Comparator, SortAlgorithm, false> sampler_;
    //! Current run values that are partitioned, redistributed and sorted when
    //! capacity is reached
    std::vector<ValueType> current_run_;
    //! Runs in the first phase of the algorithm
    std::vector<data::SampledFilePtr<ValueType>> run_files_;
    //! Final splitters used for redistribution step
    std::vector<ValueType> final_splitters_;
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

    //! time spent in PreOp (including preceding node's computation)
    Timer timer_pre_op_;
    
    //! time spent in online sampling
    Timer timer_sample_;

    //! time spent in sort()
    Timer timer_sort_;

    //! time spent in partitioning
    Timer timer_partition_;

    //! time spent in merging
    Timer timer_merge_;

    //! time spent in PreOp in communication
    Timer timer_pre_op_communication_;

    //! time spent in PreOp in file io
    Timer timer_pre_op_file_io_;

    //! time spent in MainOp in communication
    Timer timer_global_communication_;

    //! total time spent
    Timer timer_total_;

    //! \}

    // TODO Refactor to vectors
    class TreeBuilder
    {
    public:
        ValueType* tree_;
        const SampleIndexPair* splitters_;
        size_t splitters_size_;

        TreeBuilder(ValueType* tree,
                    const SampleIndexPair* splitters,
                    size_t splitters_size)
                : tree_(tree),
                  splitters_(splitters),
                  splitters_size_(splitters_size) {
            if (splitters_size != 0)
                recurse(splitters, splitters + splitters_size, 1);
        }

        void recurse(const SampleIndexPair* lo, const SampleIndexPair* hi,
                     unsigned int tree_index) {
            // Pick middle element as splitter.
            const SampleIndexPair* mid = lo + (ssize_t)(hi - lo) / 2;
            assert(mid < splitters_ + splitters_size_);
            tree_[tree_index] = mid->first;

            if (2 * tree_index < splitters_size_)
            {
                const SampleIndexPair* mid_lo = mid, * mid_hi = mid + 1;
                recurse(lo, mid_lo, 2 * tree_index + 0);
                recurse(mid_hi, hi, 2 * tree_index + 1);
            }
        }
    };

    bool EqualSampleGreaterIndex(const SampleIndexPair& a, 
            const SampleIndexPair& b) {
        return !comparator_(a.first, b.first) && a.second >= b.second;
    }

    // TODO Refactor to vectors
    void TransmitItems(
            // Tree of splitters, sizeof |splitter|
            const ValueType* const tree,
            // Number of buckets: k = 2^{log_k}
            size_t k,
            size_t log_k,
            // Number of actual workers to send to
            size_t actual_k,
            const SampleIndexPair* const sorted_splitters,
            data::MixStreamPtr& data_stream) {

        auto data_writers = data_stream->GetWriters();

        // enlarge emitters array to next power of two to have direct access,
        // because we fill the splitter set up with sentinels == last splitter,
        // hence all items land in the last bucket.
        assert(data_writers.size() == actual_k);
        assert(actual_k <= k);

        data_writers.reserve(k);
        while (data_writers.size() < k)
            data_writers.push_back(data::MixStream::Writer());

        std::swap(data_writers[actual_k - 1], data_writers[k - 1]);

        // classify all items (take two at once) and immediately transmit them.

        timer_partition_.Start();
        const size_t step_size = 2;

        size_t i = 0;
        for ( ; i < current_run_.size() / step_size; i += step_size)
        {
            // take two items
            size_t j0 = 1;
            ValueType el0 = current_run_[i];

            size_t j1 = 1;
            ValueType el1 = current_run_[i + 1];

            // run items down the tree
            for (size_t l = 0; l < log_k; l++)
            {
                j0 = 2 * j0 + (comparator_(el0, tree[j0]) ? 0 : 1);
                j1 = 2 * j1 + (comparator_(el1, tree[j1]) ? 0 : 1);
            }

            size_t b0 = j0 - k;
            size_t b1 = j1 - k;

            while (b0 && EqualSampleGreaterIndex(
                    sorted_splitters[b0 - 1], SampleIndexPair(el0, i + 0))) {
                b0--;
            }

            while (b1 && EqualSampleGreaterIndex(
                    sorted_splitters[b1 - 1], SampleIndexPair(el1, i + 1))) {
                b1--;
            }

            assert(data_writers[b0].IsValid());
            assert(data_writers[b1].IsValid());

            timer_partition_.Stop();
            timer_pre_op_communication_.Start();
            data_writers[b0].Put(el0);
            data_writers[b1].Put(el1);
            timer_pre_op_communication_.Stop();
            timer_partition_.Start();
        }

        // last iteration of loop if we have an odd number of items.
        for ( ; i < current_run_.size(); i++)
        {
            size_t j0 = 1;
            ValueType el0 = current_run_[i];

            // run item down the tree
            for (size_t l = 0; l < log_k; l++)
            {
                j0 = 2 * j0 + (comparator_(el0, tree[j0]) ? 0 : 1);
            }

            size_t b0 = j0 - k;

            while (b0 && EqualSampleGreaterIndex(
                    sorted_splitters[b0 - 1], SampleIndexPair(el0, i))) {
                b0--;
            }

            assert(data_writers[b0].IsValid());
            timer_partition_.Stop();
            timer_pre_op_communication_.Start();
            data_writers[b0].Put(el0);
            timer_pre_op_communication_.Stop();
            timer_partition_.Start();
        }
        timer_partition_.Stop();

        // implicitly close writers and flush data
    }

    void SortWriteClearCurrentRun() {
        LOG << "Sort current run of size " << current_run_.size() << ".";
        timer_sort_.Start();
        sort_algorithm_(current_run_.begin(), current_run_.end(), comparator_);
        timer_sort_.Stop();

        // Write elements to file.
        LOG << "Write sorted run to file.";
        run_files_.push_back(context_.template GetSampledFilePtr<ValueType>(
                this));
        timer_pre_op_file_io_.Start();
        auto current_run_file_writer = run_files_.back()->GetWriter();
        for (auto element : current_run_) {
            current_run_file_writer.template Put<ValueType>(element);
        }
        current_run_file_writer.Close();
        timer_pre_op_file_io_.Stop();
        current_run_.clear();
    }

    void FinishCurrentRun(bool is_final = false) {
        LOG << "Finish current run.";

        LOG << "Select " << p_ - 1 << " splitters.";
        std::vector<ValueType> splitters;
        splitters.reserve(p_ - 1);
        std::vector<double> quantiles;
        quantiles.reserve(p_ - 1);
        for (size_t i = 0; i < p_ - 1; i++) {
            quantiles.emplace_back(static_cast<double>(i + 1) * (1.0 / p_));
        }

        timer_sample_.Start();
        sampler_.GetSplitters(quantiles, splitters);
        timer_sample_.Stop();

        // Get the ceiling of log(num_total_workers), as SSSS needs 2^n buckets.
        size_t log_tree_size = tlx::integer_log2_ceil(p_);
        size_t tree_size = size_t(1) << log_tree_size;
        std::vector<ValueType> tree(tree_size + 1);

        std::vector<SampleIndexPair> splitters_with_indices;
        splitters_with_indices.reserve(tree_size);
        for (size_t i = 0; i < p_ - 1; i++) {
            splitters_with_indices.push_back(SampleIndexPair(splitters[i],
                    (i + 1) * current_run_.size() / p_));
        }

        // Add sentinel splitters.
        for (size_t i = p_; i < tree_size; i++) {
            splitters_with_indices.push_back(splitters_with_indices.back());
        }

        // Build tree.
        LOG << "Build tree of size " << tree_size << " with height " 
            << log_tree_size << ".";
        TreeBuilder(tree.data(),
                    splitters_with_indices.data(),
                    splitters_with_indices.size());

        // Partition.
        LOG << "Partition and communicate " << current_run_.size() << " elements.";
        auto data_stream = context_.template GetNewStream<data::MixStream>(
                this->dia_id());
        TransmitItems(tree.data(), tree_size, log_tree_size, p_,
                splitters_with_indices.data(), data_stream);
        current_run_.clear();

        // Receive elements and sort.
        LOG << "Receive elements.";
        auto reader = data_stream->GetReader(true);
        timer_pre_op_communication_.Start();
        while (reader.HasNext()) {
            if (current_run_.size() >= run_capacity_) {
                timer_pre_op_communication_.Stop();
                local_items_ += current_run_.size();
                SortWriteClearCurrentRun();
                timer_pre_op_communication_.Start();
            }
            current_run_.push_back(reader.template Next<ValueType>());
        }
        timer_pre_op_communication_.Stop();
        if (current_run_.size() > 0) {
            local_items_ += current_run_.size();
            SortWriteClearCurrentRun();
        }

        data_stream.reset();

        if (is_final) {
            final_splitters_ = std::move(splitters);
        }
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
auto DIA<ValueType, Stack>::OnlineSampleSort(const size_t r,
        const CompareFunction& compare_function) const {
    assert(IsValid());

    using OnlineSampleSortNode = api::OnlineSampleSortNode<
            ValueType, CompareFunction, DefaultSortAlgorithm>;

    static_assert(
        std::is_convertible<
            ValueType,
            typename FunctionTraits<CompareFunction>::template arg<0>>::value,
        "CompareFunction has the wrong input type");

    static_assert(
        std::is_convertible<
            ValueType,
            typename FunctionTraits<CompareFunction>::template arg<1>>::value,
        "CompareFunction has the wrong input type");

    static_assert(
        std::is_convertible<
            typename FunctionTraits<CompareFunction>::result_type, bool>::value,
        "CompareFunction has the wrong output type (should be bool)");

    auto node = tlx::make_counting<OnlineSampleSortNode>(*this, 
            compare_function, r);

    return DIA<ValueType>(node);
}

} // namespace api
} // namespace thrill

#endif // !THRILL_API_ONLINE_SAMPLE_SORT_HEADER

/******************************************************************************/