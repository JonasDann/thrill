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

    const size_t b_ = 10;
    const size_t k_ = 60; // TODO round k_ up to multiple of p_?
    size_t run_capacity_;

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
              sampler_(b_, k_, parent.ctx(), comparator, sort_algorithm)
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
        LOG << "Run formation.";
        timer_total_.Start();
        timer_preop_.Start();
        current_run_.reserve(run_capacity_);
    }

    void PreOp(const ValueType& input) {
        auto has_next_capacity = sampler_.Put(input);
        if (!has_next_capacity) {
            LOG << "Collapse sampler buffers.";
            sampler_.Collapse([this] (ValueType& value){
                if (current_run_.size() >= run_capacity_) {
                    FinishCurrentRun();
                }
                current_run_.push_back(value);
            });
        }
        local_items_++;
    }

    //! Receive a whole data::File of ValueType, but only if our stack is empty.
    bool OnPreOpFile(const data::File& file, size_t /* parent_index */) final {
        (void) file;
        // TODO What should this do?

        return false;
    }

    void StopPreOp(size_t /* id */) final {
        LOG << "Collapse rest of buffers.";
        bool is_collapsible;
        do {
            // TODO Refactor to emit function
            LOG << "Collapse sampler buffers.";
            is_collapsible = sampler_.Collapse([this] (ValueType& value){
                if (current_run_.size() >= run_capacity_) {
                    FinishCurrentRun();
                }
                current_run_.push_back(value);
            });
        } while(is_collapsible);
        LOG << "Finish last run.";
        if (current_run_.size() > 0) {
            FinishCurrentRun(true);
        }
        std::vector<ValueType>().swap(current_run_); // free vector

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

    void Execute() final {
        Timer timer_main_op;
        timer_main_op.Start();

        // Calculate splitters
        auto splitter_count = final_splitters_.size();
        auto run_count = run_files_.size();
        LocalRanks local_ranks(final_splitters_.size(), std::vector<size_t>(run_count));
        for (size_t s = 0; s < splitter_count; s++) {
            for (size_t r = 0; r < run_count; r++) {
                local_ranks[s][r] = run_files_[r]->GetFastIndexOf(final_splitters_[s], 0, 0, run_files_[r]->num_items(), comparator_);
            }
        }

        // Redistribute Elements
        for (size_t run_index = 0; run_index < run_count; run_index++) {
            auto data_stream = context_.template GetNewStream<data::CatStream>(this->dia_id());

            // Construct offsets vector
            std::vector<size_t> run_offsets(splitter_count + 2);
            run_offsets[0] = 0;
            std::transform(local_ranks.begin(), local_ranks.end(), run_offsets.begin() + 1, [run_index](std::vector<size_t> element) {
                return element[run_index];
            });
            run_offsets[splitter_count + 1] = run_files_[run_index]->num_items();

            data_stream->template Scatter<ValueType>(*run_files_[run_index],
                                                     run_offsets, true);

            auto final_run_file = context_.GetFilePtr(this);
            final_run_files_.emplace_back(final_run_file);
            data_stream->GetFile(final_run_file);

            data_stream.reset();
        }

        timer_main_op.Stop();
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                    "CanonicalMergeSort() timer_mainop", timer_main_op.SecondsDouble());
        }
    }

    DIAMemUse PushDataMemUse() final {
        // Communicates how much memory the DIA needs on push data
        // TODO Make it work.
        return 0;
    }

    void PushData(bool consume) final {
        // TODO push splitters
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
                    file_readers.begin(), file_readers.end(), comparator_);


            LOG << "Merging " << final_run_files_.size() << " files with prefetch " << prefetch << ".";
            ValueType first_element;
            if (debug && file_merge_tree.HasNext()) {
                first_element = file_merge_tree.Next();
                this->PushItem(first_element);
                local_size++;
            }
            ValueType last_element;
            while (file_merge_tree.HasNext()) {
                auto next = file_merge_tree.Next();
                this->PushItem(next);
                if (debug) {
                    last_element = next;
                }
                local_size++;
            }
            LOG << "Finished merging (first element: " << first_element
                << ", last element: " << last_element << ").";
        }
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
    //! Current run values that are partitioned, redistributed and sorted when
    //! capacity is reached
    std::vector<ValueType> current_run_;
    //! Runs in the first phase of the algorithm
    std::vector<data::SampledFilePtr<ValueType>> run_files_;
    //! Final splitters used for correcion step
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
            // Pick middle element as splitter
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

    bool EqualSampleGreaterIndex(const SampleIndexPair& a, const SampleIndexPair& b) {
        return !comparator_(a.first, b.first) && a.second >= b.second;
    }

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
            data_writers.emplace_back(data::MixStream::Writer());

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
            timer_communication_.Start();
            data_writers[b0].Put(el0);
            data_writers[b1].Put(el1);
            timer_communication_.Stop();
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
            timer_communication_.Start();
            data_writers[b0].Put(el0);
            timer_communication_.Stop();
            timer_partition_.Start();
        }
        timer_partition_.Stop();

        // implicitly close writers and flush data
    }

    void FinishCurrentRun(bool is_final = false) {
        LOG << "Finish current run.";

        // Select splitters
        LOG << "Select " << p_ - 1 << " splitters.";
        std::vector<ValueType> samples;
        sampler_.GetSamples(samples);

        std::vector<SampleIndexPair> splitters;
        splitters.reserve(p_);
        for (size_t i = 0; i < k_; i += k_ / p_) {
            splitters.emplace_back(SampleIndexPair(samples[i], i));
            if (is_final) {
                final_splitters_.emplace_back(samples[i]);
            }
        }

        // Get the ceiling of log(num_total_workers), as SSSS needs 2^n buckets.
        size_t log_tree_size = tlx::integer_log2_ceil(p_);
        size_t tree_size = size_t(1) << log_tree_size;
        std::vector<ValueType> tree(tree_size + 1);

        // Add sentinel splitters
        for (size_t i = p_; i < tree_size; i++) {
            splitters.push_back(splitters.back());
        }

        // Build tree
        LOG << "Build tree of size " << tree_size << " with height " << log_tree_size << ".";
        TreeBuilder(tree.data(),
                    splitters.data(),
                    splitters.size());

        // Partition
        auto old_run_size = current_run_.size();
        LOG << "Partition and scatter " << old_run_size << " elements.";
        auto data_stream = context_.template GetNewStream<data::MixStream>(this->dia_id());
        TransmitItems(tree.data(), tree_size, log_tree_size, p_,
                splitters.data(), data_stream);
        current_run_.clear();

        // Receive elements and sort
        LOG << "Receive elements.";
        auto reader = data_stream->GetReader(true);
        while (reader.HasNext()) {
            current_run_.emplace_back(reader.template Next<ValueType>());
            // TODO Sort while receiving when memory is full?
        }
        local_items_ += current_run_.size() - old_run_size;
        LOG << "Sort current run of size " << current_run_.size() << ".";
        timer_sort_.Start();
        sort_algorithm_(current_run_.begin(), current_run_.end(), comparator_);
        timer_sort_.Stop();

        // Write elements to file
        LOG << "Write sorted run to file.";
        run_files_.emplace_back(context_.template GetSampledFilePtr<ValueType>(this));
        auto current_run_file_writer = run_files_.back()->GetWriter();
        for (auto element : current_run_) {
            current_run_file_writer.template Put<ValueType>(element);
        }
        current_run_file_writer.Close();
        current_run_.clear();

        data_stream.reset();
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