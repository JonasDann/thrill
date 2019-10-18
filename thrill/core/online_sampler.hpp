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

template <typename ValueType>
class Buffer
{
public:
    std::vector<ValueType> elements_;
    size_t weight_;
    bool sorted_;
    size_t k_;

    explicit Buffer(size_t k) : weight_(1), sorted_(false), k_(k) {
        elements_.reserve(k_);
    }

    Buffer(const Buffer& b) noexcept = delete;
    Buffer& operator = (Buffer& b) noexcept = delete;

    Buffer(Buffer&& b) noexcept = default;
    Buffer& operator = (Buffer&& b) noexcept = default;

    Buffer(std::vector<ValueType>&& elements, size_t weight, bool sorted,
           size_t k)
        : elements_(elements), weight_(weight), sorted_(sorted),
          k_(k) { }

    bool Put(ValueType value) {
        assert(HasCapacity());
        elements_.push_back(value);
        return HasCapacity();
    }

    bool HasCapacity() {
        return elements_.size() < k_;
    }
};

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
 * \tparam NonUniformSampling Whether non uniform sampling technique should be
 *  applied
 *
 * \tparam Stable Whether or not to use stable sorting mechanisms
 */
template <
    typename ValueType,
    typename Comparator,
    typename SortAlgorithm,
    bool NonUniformSampling = true,
    bool Stable = false>
class OnlineSampler
{
    using LoserTree = tlx::LoserTree<Stable, ValueType, Comparator>;

    static constexpr bool debug = false;

    //! Set this variable to true to enable generation and output of sampling
    //! stats
    static constexpr bool stats_enabled = true;

    //! Timer or FakeTimer
    using Timer = common::StatsTimerBaseStopped<stats_enabled>;

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
        rng_ = std::mt19937(rd_());
        buffers_.reserve(b_);
        for (size_t i = 0; i < b_; i++) {
            Buffer<ValueType> buffer(k);
            buffers_.push_back(std::move(buffer));
        }
        level_counters_.push_back(0);
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
        }
        else {
            sample_block_.push_back(value);
            if (sample_block_.size() >= r_) {
                std::uniform_int_distribution<size_t> uni(0, r_);
                has_capacity = partial_buffer_.
                               Put(sample_block_[uni(rng_)]);
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
            level_counters_.push_back(1);
            if (NonUniformSampling) {
                r_ *= 2;
                new_level_++;
                ResampleBuffer(partial_buffer_, 1 / 2);
            }
        }
        else {
            level_counters_[current_level]++;
        }
        assert(current_level < level_counters_.size());

        LOG << empty_buffer_count_ << " buffers are now empty.";
        timer_total_.Stop();
        return b_ - empty_buffer_count_ > 1;
    }

    /*!
     * Collapses current buffers and communicates the result with PE 0 which
     * selects splitters based on the quantiles given and communicates result
     * with all PEs. This function does not alter the state of the data
     * structure.
     *
     * \param quantiles Vector that contains the quantiles
     * \param out_splitters Vector that will contain the resulting splitters
     */
    void GetSplitters(const std::vector<double>& quantiles,
                      std::vector<ValueType>& out_splitters) {
        timer_total_.Start();
        LOG << "GetSplitters()";
        LOG << "Collapse full buffers, but do not change state.";
        Buffer<ValueType> local_target_buffer(k_);
        Collapse(buffers_.begin(),
                 buffers_.begin() + (b_ - empty_buffer_count_),
                 local_target_buffer);

        LOG << "Send full and partial buffer to PE 0.";
        timer_communication_.Start();
        auto stream = context_.template GetNewStream<data::CatStream>(
            dia_id_);
        auto writers = stream->GetWriters();
        auto readers = stream->GetReaders();

        writers[0].Put(local_target_buffer);
        writers[0].Put(partial_buffer_);
        writers[0].Close();
        timer_communication_.Stop();

        if (context_.my_rank() == 0) {
            timer_communication_.Start();
            LOG << "Receive " << context_.num_workers() << " buffers.";
            std::vector<Buffer<ValueType> > full_buffers;
            std::vector<Buffer<ValueType> > partial_buffers;
            for (size_t p = 0; p < context_.num_workers(); p++) {
                full_buffers.push_back(std::move(
                                           readers[p].template Next<Buffer<ValueType> >()));
                partial_buffers.push_back(std::move(
                                              readers[p].template Next<Buffer<ValueType> >()));
                sort_algorithm_(partial_buffers.back().elements_.begin(),
                                partial_buffers.back().elements_.end(), comparator_);
                partial_buffers.back().sorted_ = true;
            }
            timer_communication_.Stop();

            Buffer<ValueType> partial_buffer = std::move(partial_buffers[0]);
            for (size_t p = 1; p < context_.num_workers(); p++) {
                if (partial_buffers[p].elements_.size() > 0) {
                    if (partial_buffer.weight_ < partial_buffers[p].weight_) {
                        ResampleBuffer(partial_buffer,
                                       partial_buffer.weight_ /
                                       partial_buffers[p].weight_);
                    }
                    else {
                        ResampleBuffer(partial_buffers[p],
                                       partial_buffers[p].weight_ /
                                       partial_buffer.weight_);
                    }
                    bool has_capacity;
                    for (size_t i = 0; i < partial_buffers[p].elements_.size();
                         i++) {
                        has_capacity = partial_buffer.Put(
                            partial_buffers[p].elements_[i]);
                        partial_buffer.sorted_ = false;
                        if (!has_capacity) {
                            sort_algorithm_(partial_buffer.elements_.begin(),
                                            partial_buffer.elements_.end(),
                                            comparator_);
                            partial_buffer.sorted_ = true;
                            full_buffers.push_back(
                                std::move(partial_buffer));
                            partial_buffer = std::move(partial_buffers[p]);
                            partial_buffer.elements_.erase(
                                partial_buffer.elements_.begin(),
                                partial_buffer.elements_.begin() + i);
                        }
                    }
                }
            }
            if (!partial_buffer.sorted_) {
                sort_algorithm_(partial_buffer.elements_.begin(),
                                partial_buffer.elements_.end(), comparator_);
            }

            LOG << "Collapse buffers.";
            Buffer<ValueType> target_buffer(k_);
            Collapse(full_buffers.begin(), full_buffers.end(), target_buffer);

            size_t sequence_length = k_ * target_buffer.weight_ +
                                     partial_buffer.elements_.size() *
                                     partial_buffer.weight_;
            std::vector<Buffer<ValueType> > remaining_buffers;
            remaining_buffers.push_back(std::move(target_buffer));
            if (partial_buffer.elements_.size() > 0) {
                remaining_buffers.push_back(std::move(partial_buffer));
            }
            std::vector<size_t> positions(remaining_buffers.size(), 0);
            size_t total_position = 0;
            for (auto quantile : quantiles) {
                auto target_rank = quantile * sequence_length;
                assert(target_rank >= total_position);

                ValueType splitter;
                while (total_position < target_rank) {
                    size_t minimum_index = 0;
                    if (remaining_buffers.size() > 1 &&
                        remaining_buffers[1].elements_[positions[1]] <
                        remaining_buffers[0].elements_[positions[0]]) {
                        minimum_index = 1;
                    }

                    if (remaining_buffers.size() < 2) {
                        size_t steps = std::max((target_rank - total_position) /
                                                remaining_buffers[0].weight_, 1.0);
                        total_position += remaining_buffers[0].weight_ * steps;
                        positions[0] += steps;
                        splitter = remaining_buffers[0].
                                   elements_[positions[0] - 1];
                    }
                    else {
                        splitter = remaining_buffers[minimum_index].
                                   elements_[positions[minimum_index]];
                        total_position += remaining_buffers[minimum_index].
                                          weight_;
                        positions[minimum_index]++;
                        if (positions[minimum_index] >=
                            remaining_buffers[minimum_index].elements_.
                            size()) {
                            remaining_buffers.erase(
                                remaining_buffers.begin() + minimum_index);
                        }
                    }
                }
                out_splitters.push_back(splitter);
            }

            LOG << "Send resulting quantiles.";
            timer_communication_.Start();
            for (size_t p = 1; p < context_.num_workers(); p++) {
                for (auto splitter : out_splitters) {
                    writers[p].Put(splitter);
                }
                writers[p].Close();
            }
            timer_communication_.Stop();
        }
        else {
            LOG << "Close unused writers.";
            for (size_t p = 1; p < context_.num_workers(); p++) {
                writers[p].Close();
            }

            LOG << "Receive resulting quantiles.";
            timer_communication_.Start();
            while (readers[0].HasNext()) {
                out_splitters.push_back(readers[0].template Next<ValueType>());
            }
            timer_communication_.Stop();
        }

        stream.reset();
        timer_total_.Stop();
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

    // alternating target rank bias, as described in original paper
    size_t even_bias_ = 0;

    Context& context_;
    size_t dia_id_;
    Comparator comparator_;
    SortAlgorithm sort_algorithm_;

    std::random_device rd_;
    std::mt19937 rng_;

    std::vector<ValueType> sample_block_;
    std::vector<Buffer<ValueType> > buffers_;
    Buffer<ValueType> partial_buffer_;
    Buffer<ValueType> pool_buffer_;
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
                                          level_counters_.end(), (size_t)0);
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
            return j * weight + weight / 2 + even_bias_;
        }
        else {                 // uneven
            return j * weight + (weight + 1) / 2;
        }
    }

    void Collapse(
        typename std::vector<Buffer<ValueType> >::iterator buffers_begin,
        typename std::vector<Buffer<ValueType> >::iterator buffers_end,
        Buffer<ValueType>& target_buffer) {
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

        timer_merge_.Start();
        LOG << "Merge buffers.";
        for (size_t j = 0; j < k_; j++) {
            size_t target_rank = GetTargetRank(j, weight_sum);
            ValueType sample =
                (*buffers_begin).elements_[0];     // Init removes warning
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

        if (weight_sum % 2 == 0) {
            if (even_bias_) {
                even_bias_ = 0;
            }
            else {
                even_bias_ = 1;
            }
        }
    }

    void ResampleBuffer(Buffer<ValueType>& buffer, double factor) {
        assert((factor > 0) && (factor <= 1));
        assert(buffer.sorted_);

        std::vector<ValueType> new_elements;
        new_elements.reserve(k_);
        auto new_size = buffer.elements_.size() * factor;
        double step_size = 1 / factor;
        for (size_t i = 0; i < new_size; i++) {
            auto index =
                static_cast<size_t>(i * step_size + (step_size / 2));
            new_elements.push_back(buffer.elements_[index]);
        }
        buffer.elements_ = std::move(new_elements);
    }
};

} // namespace core

namespace data {

template <typename Archive, typename T>
struct Serialization<Archive, core::Buffer<T> > {
    static void Serialize(const core::Buffer<T>& b, Archive& ar) {
        Serialization<Archive, std::vector<T> >::Serialize(b.elements_, ar);
        Serialization<Archive, size_t>::Serialize(b.weight_, ar);
        Serialization<Archive, bool>::Serialize(b.sorted_, ar);
        Serialization<Archive, size_t>::Serialize(b.k_, ar);
    }

    static core::Buffer<T> Deserialize(Archive& ar) {
        std::vector<T> elements =
            Serialization<Archive, std::vector<T> >::Deserialize(ar);
        size_t weight = Serialization<Archive, size_t>::Deserialize(ar);
        bool sorted = Serialization<Archive, bool>::Deserialize(ar);
        size_t k = Serialization<Archive, size_t>::Deserialize(ar);
        return core::Buffer<T>(std::move(elements), weight, sorted, k);
    }
    static constexpr bool   is_fixed_size = false;
    static constexpr size_t fixed_size = 0;
};

} // namespace data
} // namespace thrill

#endif // !THRILL_CORE_ONLINE_SAMPLER_HEADER

/******************************************************************************/
