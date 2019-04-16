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

    using SampleIndexPair = std::pair<ValueType, size_t>;

    const size_t b_ = 10;
    const size_t k_ = 60;
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
        timer_total_.Start();
        timer_preop_.Start();
        current_run_.reserve(run_capacity_);
    }

    void PreOp(const ValueType& input) {
        auto has_next_capacity = sampler_.Put(input);
        if (!has_next_capacity) {
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
        if (current_run_.size() > 0) {
            FinishCurrentRun();
        }
        std::vector<ValueType>().swap(current_run_[0]); // free vector

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

        // TODO MainOp

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
    //! Current run values that are partitioned, redistributed and sorted when
    //! capacity is reached
    std::vector<ValueType> current_run_;
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
                j0 = 2 * j0 + (compare_function_(el0, tree[j0]) ? 0 : 1);
                j1 = 2 * j1 + (compare_function_(el1, tree[j1]) ? 0 : 1);
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
                j0 = 2 * j0 + (compare_function_(el0, tree[j0]) ? 0 : 1);
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

    void FinishCurrentRun() {
        // Select splitters
        std::vector<ValueType> samples;
        sampler_.GetSamples(samples);

        std::vector<ValueType> splitters;
        splitters.reserve(p_);
        for (size_t i = 0; i < k_; i += k_ / p_) {
            // TODO Replace with better splitter selection?
            splitters.emplace_back(samples[i]);
        }

        // Get the ceiling of log(num_total_workers), as SSSS needs 2^n buckets.
        size_t log_tree_size = tlx::integer_log2_ceil(p_);
        size_t tree_size = size_t(1) << log_tree_size;
        std::vector<ValueType> tree(tree_size + 1);

        // Add sentinel splitters
        for (size_t i = p_; i < tree_size; i++) {
            splitters.push_back(splitters.back());
        }

        TreeBuilder(tree.data(),
                    splitters.data(),
                    splitters.size());

        // Partition
        auto data_stream = context_.template GetNewStream<data::MixStream>(this->dia_id());
        TransmitItems(tree.data(), tree_size, log_tree_size, p_,
                splitters.data(), data_stream);

        // TODO Receive elements
        // TODO Sort
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