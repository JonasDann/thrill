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
        bool Stable = false>
class OnlineSampler {
    using LooserTree = tlx::LoserTree<Stable, ValueType, Comparator>;

    static constexpr bool debug = false;

    //! Set this variable to true to enable generation and output of sampling
    //! stats
    static constexpr bool stats_enabled = true;
    // TODO Implement stats

    class Buffer {
    public:
        std::vector<ValueType> elements_;
        size_t weight_;
        bool sorted_;

        explicit Buffer(size_t k) : weight_(0), sorted_(false), k_(k) {}

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
    OnlineSampler(size_t b, size_t k, Context& context,
            const Comparator& comparator, const SortAlgorithm& sort_algorithm)
            : b_(b), k_(k), context_(context), comparator_(comparator),
              sort_algorithm_(sort_algorithm), current_buffer_(b),
              empty_buffers_(b), minimum_level_(0) {
        buffers_ = std::vector<Buffer>(b, Buffer(k));
        level_counters_.emplace_back(0);
    }

    /*!
     * Put an element into the data structure. If this returns false, Collapse
     * has to be called.
     *
     * \returns True if the data structure still has capacity for more elements
     *  after Put operation
     */
    bool Put(const ValueType& value) {
        if (current_buffer_ >= b_ ||
                !buffers_[current_buffer_].HasCapacity()) {
            New();
        }
        return buffers_[current_buffer_].Put(value) || empty_buffers_ > 0;
    }

    /*!
     * Put an element into the data structure.
     *
     * \tparam Emitter Type of emitter function that is called with elements
     *  that are no longer in the buffers after collapse operation
     *
     * \param emit Emitter function void emit(ValueType element)
     *
     * \returns True if there is more than one buffer remaining
     */
    template <typename Emitter>
    bool Collapse(const Emitter& emit) {
        std::vector<Buffer> level;
        auto level_begin = b_ - empty_buffers_ -
                level_counters_[minimum_level_];
        for (size_t i = level_begin; i < b_ - empty_buffers_; i++) {
            level.emplace_back(std::move(buffers_[i]));
        }

        Collapse(level, buffers_[level_begin], emit);

        empty_buffers_ += level.size() - 1;
        level_counters_[minimum_level_] = 0;
        current_buffer_ = b_;

        minimum_level_++;
        if (minimum_level_ >= level_counters_.size()) {
            level_counters_.emplace_back(1);
        } else {
            level_counters_[minimum_level_]++;
        }
        assert(minimum_level_ < level_counters_.size());

        return b_ - empty_buffers_ > 1;
    }

    /*!
     * Communicates currently highest weighted samples buffer with all PEs and
     * returns collapsed result.
     *
     * \param out_samples Vector that will contain the resulting samples
     *
     * \returns Weight of elements
     */
    size_t GetSamples(std::vector<ValueType> &out_samples) {
        // TODO Pseudo concat to use all knowledge in the buffers?
        if (!buffers_[0].sorted_) {
            sort_algorithm_(buffers_[0].elements_.begin(),
                            buffers_[0].elements_.end(), comparator_);
            buffers_[0].sorted_ = true;
        }

        auto all_weights = context_.net.AllGather(buffers_[0].weight_);
        auto all_elements = context_.net.AllGather(buffers_[0].elements_);
        std::vector<Buffer> buffers;
        for (size_t p = 0; p < context_.num_workers(); p++) {
            buffers.emplace_back(Buffer(std::move(all_elements->at(p)),
                    all_weights->at(p), true));
        }

        Buffer target_buffer(k_);
        Collapse(buffers, target_buffer, [] (ValueType){});

        out_samples = std::move(target_buffer.elements_);
        return target_buffer.weight_;
    }

private:
    size_t b_;
    size_t k_;

    Context& context_;
    Comparator comparator_;
    SortAlgorithm sort_algorithm_;

    std::vector<Buffer> buffers_;
    std::vector<size_t> level_counters_;
    size_t current_buffer_;
    size_t empty_buffers_;
    size_t minimum_level_;

    void New() {
        current_buffer_ = std::accumulate(level_counters_.begin(),
                                          level_counters_.end(), (size_t) 0);
        if (empty_buffers_ > 1) {
            minimum_level_ = 0;
        }
        level_counters_[minimum_level_]++;
        buffers_[current_buffer_].elements_.reserve(k_);
        buffers_[current_buffer_].weight_ = 1;
        buffers_[current_buffer_].sorted_ = false;
        empty_buffers_--;
    }

    size_t GetTargetRank(size_t j, size_t weight) {
        if (weight % 2 == 0) { // even
            return j * weight + (weight + 2 * (j % 2)) / 2;
        } else { // uneven
            return j * weight + (weight + 1) / 2;
        }
    }

    template <typename Emitter>
    void Collapse(std::vector<Buffer>& buffers, Buffer& target_buffer,
            const Emitter& emit) {
        LooserTree looser_tree(buffers.size(), comparator_);
        size_t weight_sum = 0;
        for (size_t i = 0; i < buffers.size(); i++) {
            if (!buffers[i].sorted_) {
                sort_algorithm_(buffers[i].elements_.begin(),
                                buffers[i].elements_.end(), comparator_);
                buffers[i].sorted_ = true;
            }
            weight_sum += buffers[i].weight_;
            looser_tree.insert_start(&buffers[i].elements_[0], i, false);
        }
        looser_tree.init();

        target_buffer.weight_ = weight_sum;
        target_buffer.sorted_ = true;

        size_t total_index = 0;
        std::vector<size_t> positions(buffers.size(), 0);

        // Advance total_index for amount of empty elements and calculate number
        // of empty elements in target buffer.
        for (size_t i = 0; i < buffers.size(); i++) {
            size_t empty_elements = k_ - buffers[i].elements_.size();
            total_index += empty_elements * buffers[i].weight_;
        }
        size_t target_buffer_empty = total_index / weight_sum;
        total_index = (total_index / 2) % weight_sum;

        for (size_t j = 0; j < k_ - target_buffer_empty; j++) {
            size_t target_rank = GetTargetRank(j, weight_sum);
            ValueType sample = buffers[0].elements_[0]; // Init removes warning
            bool first = true;
            assert(total_index < target_rank);
            while (total_index < target_rank) {
                auto minimum_index = looser_tree.min_source();
                if (first) {
                    first = false;
                } else {
                    emit(sample);
                }
                sample = buffers[minimum_index].
                        elements_[positions[minimum_index]];
                total_index += buffers[minimum_index].weight_;
                positions[minimum_index]++;
                auto has_next = positions[minimum_index] <
                                buffers[minimum_index].elements_.size();
                looser_tree.delete_min_insert(
                        has_next ? &buffers[minimum_index].
                                elements_[positions[minimum_index]] : nullptr,
                        !has_next);
            }
            target_buffer.Put(sample);
        }
    }
};

} // namespace core
} // namespace thrill

#endif // !THRILL_CORE_ONLINE_SAMPLER_HEADER

/******************************************************************************/