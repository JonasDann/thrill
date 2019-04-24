/*******************************************************************************
 * thrill/core/online_sampler.hpp
 *
 * Deterministic stream sampling data structure
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2019 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_CORE_ONLINE_SAMPLER_HEADER
#define THRILL_CORE_ONLINE_SAMPLER_HEADER

#include <tlx/container/loser_tree.hpp>

#include <thrill/api/context.hpp>

namespace thrill {
namespace core {

/*!
 * A data structure that deterministically draws equidistant samples from a
 * stream of elements.
 *
 * \tparam ValueType Type of elements
 *
 * \tparam Comparator Type of the comparator
 *
 * \tparam SortAlgorithm Type of the local sort function
 *
 * \tparam Stable Whether or not to use stable sorting mechanisms
 */
template <
        typename ValueType,
        typename Comparator,
        typename SortAlgorithm,
        bool NonUniformSampling = true,
        bool Stable = false>
class OnlineSampler {
    using LoserTree = tlx::LoserTree<Stable, ValueType, Comparator>;

    static constexpr bool debug = false;

    //! Set this variable to true to enable generation and output of sampling
    //! stats
    static constexpr bool stats_enabled = true;

    //! Timer or FakeTimer
    using Timer = common::StatsTimerBaseStopped<stats_enabled>;

    class Buffer {
    public:
        std::vector<ValueType> elements_;
        size_t weight_;
        bool sorted_;

        explicit Buffer(size_t k) : weight_(1), sorted_(false), k_(k) {
            elements_.reserve(k_);
        }

        Buffer(std::vector<ValueType>&& elements, size_t weight, bool sorted)
            : elements_(elements), weight_(weight), sorted_(sorted),
              k_(elements.size()) {}

        bool Put(ValueType value) {
            assert(HasCapacity());
            elements_.emplace_back(value);
            return HasCapacity();
        }

        bool HasCapacity() {
            return elements_.size() < k_;
        }
    private:
        size_t k_;
    };

public:
    /*!
     * Online sampler constructor. The parameters b and k influence the
     * precision of the samples. In general, higher values of b and k result
     * in more precision.
     *
     * \param b Number of buffers used while sampling
     *
     * \param k Number of elements in each buffer
     */
    OnlineSampler(size_t b, size_t k, Context& context, size_t dia_id,
            const Comparator& comparator, const SortAlgorithm& sort_algorithm,
            size_t r = 1)
            : b_(b), k_(k), r_(r), context_(context), dia_id_(dia_id),
              comparator_(comparator), sort_algorithm_(sort_algorithm),
              partial_buffer_(k), pool_buffer_(k), empty_buffer_count_(b),
              new_level_(0) {
        buffers_ = std::vector<Buffer>(b, Buffer(k));
        level_counters_.emplace_back(0);
        LOG << "New OnlineSampler(" << b_ << ", " << k_ << ")";
    }

    /*!
     * Put an element into the data structure. If the random sampling factor r
     * > 1, the value is put into a sample block where a random value is drawn
     * from, if the sample block size reaches r. Thereafter, the sample block is
     * cleared. If this returns false, Collapse has to be called.
     *
     * \returns True if the data structure still has capacity for more elements
     *  after Put operation
     */
    bool Put(const ValueType& value) {
        assert(empty_buffer_count_ > 0);
        bool has_capacity = true;
        if (r_ <= 1) {
            has_capacity = partial_buffer_.Put(value);
        } else {
            sample_block_.emplace_back(value);
            if (sample_block_.size() >= r_) {
                has_capacity = partial_buffer_.
                        Put(sample_block_[0]); // TODO randomize
                sample_block_.clear();
            }
        }
        if (!has_capacity) {
            has_capacity = New();
        }
        return has_capacity;
    }

    /*!
     * Collapse the lowest level of buffers. If it only contains one buffer,
     * collapse the lowest and the second lowest levels.
     *
     * \returns True if there is more than one buffer remaining
     */
    bool Collapse() {
        timer_total_.Start();
        LOG << "Collapse()";
        size_t level_begin = b_ - empty_buffer_count_;
        size_t level_end = b_ - empty_buffer_count_;
        size_t current_level = new_level_;
        if (level_end < 2) {
            LOG << "Nothing to collapse.";
            return false;
        }
        while (level_end - level_begin < 2) {
            level_begin = b_ - empty_buffer_count_ -
                               level_counters_[current_level];
            empty_buffer_count_ += level_counters_[current_level];
            level_counters_[current_level] = 0;
            current_level++;
        }

        LOG << "Collapse " << level_end - level_begin << " buffers.";
        Collapse(buffers_.begin() + level_begin, buffers_.begin() + level_end,
                 pool_buffer_);
        for (auto i = level_begin; i < level_end; i++) {
            buffers_[i].elements_.clear();
        }
        std::swap(buffers_[level_begin], pool_buffer_);
        empty_buffer_count_--;


        if (current_level >= level_counters_.size()) {
            level_counters_.emplace_back(1);
            if (NonUniformSampling) {
                // TODO Resample partial buffer
                r_ *= 2;
                new_level_++;
            }
        } else {
            level_counters_[current_level]++;
        }
        assert(current_level < level_counters_.size());

        LOG << empty_buffer_count_ << " buffers are now empty.";
        timer_total_.Stop();
        return b_ - empty_buffer_count_ > 1;
    }

    /*!
     * Communicates currently highest weighted samples buffer with all PEs and
     * returns collapsed result. The samples vector does not have to be filled
     * completely, when partially filled buffers were collapsed. This has to be
     * handled accordingly.
     *
     * \param out_samples Vector that will contain the resulting samples
     *
     * \returns Weight of elements
     */
    size_t GetSamples(std::vector<ValueType> &out_samples) {
        // TODO Change to GetQuantiles with new partial buffer merge and pseudo collapse
        // TODO Better parallel policy
        timer_total_.Start();
        LOG << "GetSamples()";
        LOG << "Sort highest weighted buffer, if not sorted.";
        if (!buffers_[0].sorted_) {
            sort_algorithm_(buffers_[0].elements_.begin(),
                            buffers_[0].elements_.end(), comparator_);
            buffers_[0].sorted_ = true;
        }

        LOG << "Send full and partial buffer to PE 0.";
        timer_communication_.Start();
        auto all_weights = context_.net.AllGather(buffers_[0].weight_);
        auto weight_sum = 0;
        for (size_t i = 0; i < all_weights->size(); i++) {
            weight_sum += all_weights->at(i);
        }

        auto sample_stream = context_.template GetNewStream<data::CatStream>(
                dia_id_);
        auto sample_writers = sample_stream->GetWriters();
        auto sample_readers = sample_stream->GetReaders();

        for (auto element : buffers_[0].elements_) {
            sample_writers[0].Put(element);
        }
        sample_writers[0].Close();
        timer_communication_.Stop();

        std::vector<ValueType> samples;
        if (context_.my_rank() == 0) {
            timer_communication_.Start();
            LOG << "Construct " << context_.num_workers() << " buffers.";
            std::vector<Buffer> buffers;
            for (size_t p = 0; p < context_.num_workers(); p++) {
                std::vector<ValueType> elements;
                elements.reserve(k_);
                while (sample_readers[p].HasNext()) {
                    elements.emplace_back(
                            sample_readers[p].template Next<ValueType>());
                }
                buffers.emplace_back(Buffer(std::move(elements),
                                            all_weights->at(p), true));
            }
            timer_communication_.Stop();

            LOG << "Collapse buffers.";
            Buffer target_buffer(k_);
            Collapse(buffers.begin(), buffers.end(), target_buffer);

            LOG << "Send resulting samples.";
            timer_communication_.Start();
            for (size_t p = 1; p < context_.num_workers(); p++) {
                for (auto element : target_buffer.elements_) {
                    sample_writers[p].Put(element);
                }
                sample_writers[p].Close();
            }
            timer_communication_.Stop();

            samples = std::move(target_buffer.elements_);
        } else {
            LOG << "Close unused writers.";
            for (size_t p = 1; p < context_.num_workers(); p++) {
                sample_writers[p].Close();
            }

            LOG << "Receive resulting samples.";
            timer_communication_.Start();
            samples.reserve(k_);
            while(sample_readers[0].HasNext()) {
                samples.emplace_back(
                        sample_readers[0].template Next<ValueType>());
            }
            timer_communication_.Stop();
        }

        sample_stream.reset();

        LOG << "Return buffers.";
        out_samples = std::move(samples);
        timer_total_.Stop();
        return weight_sum;
    }

    void PrintStats() {
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampler total timer",
                    timer_total_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampler sort timer",
                    timer_sort_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampler communication timer",
                    timer_communication_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                    "OnlineSampler merge timer",
                    timer_merge_.SecondsDouble());
        }
    }

private:
    size_t b_;
    size_t k_;
    size_t r_;

    Context& context_;
    size_t dia_id_;
    Comparator comparator_;
    SortAlgorithm sort_algorithm_;

    std::vector<ValueType> sample_block_;
    std::vector<Buffer> buffers_;
    Buffer partial_buffer_;
    Buffer pool_buffer_;
    std::vector<size_t> level_counters_;
    size_t empty_buffer_count_;
    size_t new_level_;

    Timer timer_total_;
    Timer timer_sort_;
    Timer timer_communication_;
    Timer timer_merge_;

    bool New() {
        LOG << "New()";
        assert(!partial_buffer_.HasCapacity());
        auto next_index = std::accumulate(level_counters_.begin(),
                                          level_counters_.end(), (size_t) 0);
        std::swap(buffers_[next_index], partial_buffer_);
        level_counters_[new_level_]++;
        partial_buffer_.weight_ = 1;
        partial_buffer_.sorted_ = false;
        empty_buffer_count_--;
        LOG << empty_buffer_count_ << " buffers are still empty.";
        return empty_buffer_count_ > 0;
    }

    size_t GetTargetRank(size_t j, size_t weight) {
        if (weight % 2 == 0) { // even
            return j * weight + (weight + 2 * (j % 2)) / 2;
        } else { // uneven
            return j * weight + (weight + 1) / 2;
        }
    }

    void Collapse(typename std::vector<Buffer>::iterator buffers_begin,
            typename std::vector<Buffer>::iterator buffers_end,
            Buffer& target_buffer) {
        LOG << "Sort buffers and construct loser tree.";
        auto buffers_size = buffers_end - buffers_begin;
        LoserTree loser_tree(buffers_size, comparator_);
        size_t weight_sum = 0;
        size_t i = 0;
        for (auto it = buffers_begin; it != buffers_end; it++) {
            if (!(*it).sorted_) {
                timer_sort_.Start();
                sort_algorithm_((*it).elements_.begin(),
                                (*it).elements_.end(), comparator_);
                timer_sort_.Stop();
                (*it).sorted_ = true;
            }
            weight_sum += (*it).weight_;
            loser_tree.insert_start(&(*it).elements_[0], i, false);
            LOG << "[" << i << "] " << (*it).elements_.size() << "*"
                 << (*it).weight_;
            i++;
        }
        loser_tree.init();

        target_buffer.weight_ = weight_sum;
        target_buffer.sorted_ = true;

        size_t total_index = 0;
        std::vector<size_t> positions(buffers_size, 0);

        // Advance total_index for amount of empty elements and calculate number
        // of empty elements in target buffer.
        timer_merge_.Start();
        for (auto it = buffers_begin; it != buffers_end; it++) {
            size_t empty_elements = k_ - (*it).elements_.size();
            total_index += empty_elements * (*it).weight_;
        }
        size_t target_buffer_empty_element_count = total_index / weight_sum;
        total_index = (total_index % weight_sum) / 2;
        LOG << "Target buffer has " << target_buffer_empty_element_count
            << " empty elements and total index is " << total_index << ".";

        LOG << "Merge buffers.";
        for (size_t j = 0; j < k_ - target_buffer_empty_element_count; j++) {
            size_t target_rank = GetTargetRank(j, weight_sum);
            ValueType sample = (*buffers_begin).elements_[0]; // Init removes warning
            assert(total_index < target_rank);
            while (total_index < target_rank) {
                auto minimum_index = loser_tree.min_source();
                sample = (*(buffers_begin + minimum_index)).
                        elements_[positions[minimum_index]];
                total_index += (*(buffers_begin + minimum_index)).weight_;
                positions[minimum_index]++;
                auto has_next = positions[minimum_index] <
                        (*(buffers_begin + minimum_index)).elements_.size();
                loser_tree.delete_min_insert(
                        has_next ? &(*(buffers_begin + minimum_index)).
                                elements_[positions[minimum_index]] : nullptr,
                        !has_next);
            }
            target_buffer.Put(sample);
        }
        timer_merge_.Stop();
    }
};

} // namespace core
} // namespace thrill

#endif // !THRILL_CORE_ONLINE_SAMPLER_HEADER

/******************************************************************************/