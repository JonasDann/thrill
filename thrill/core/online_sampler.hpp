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

namespace thrill {
namespace core {

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

    class Buffer {
    public:
        size_t weight_;
        std::vector<ValueType> elements_;
        bool sorted_;

        explicit Buffer(size_t k) : weight_(0), sorted_(false), k_(k) {
            elements_.reserve(k);
        }

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
    OnlineSampler(size_t b, size_t k, Context& context,
            const Comparator& comparator, const SortAlgorithm& sort_algorithm)
            : b_(b), k_(k), context_(context), comparator_(comparator),
              sort_algorithm_(sort_algorithm), current_buffer_(-1),
              empty_buffers_(b), minimum_level_(0) {
        buffers_ = std::vector<Buffer>(b, Buffer(k));
        level_counters_.emplace_back(0);
    }

    bool Put(ValueType value) {
        if (current_buffer_ == -1 || !buffers_[current_buffer_].HasCapacity()) {
            New();
        }
        return buffers_[current_buffer_].Put(value) || empty_buffers_ > 0;
    }

    template <typename Emitter>
    bool Collapse(const Emitter& emit) {
        LooserTree looser_tree(level_counters_[minimum_level_], comparator_);
        std::vector<Buffer> level;
        auto level_begin = b_ - empty_buffers_ - level_counters_[minimum_level_];
        auto element_count = 0;
        for (size_t i = level_begin; i < b_ - empty_buffers_; i++) {
            level.emplace_back(std::move(buffers_[i]));
            element_count += level.back().elements_.size();
        }
        size_t weight_sum = 0;
        for (int i = 0; i < level.size(); i++) {
            if (!level[i].sorted_) {
                sort_algorithm_(level[i].elements_.begin(),
                        level[i].elements_.end(), comparator_);
                level[i].sorted_ = true;
            }
            weight_sum += level[i].weight_;
            looser_tree.insert_start(level[i].elements_[0], i, false);
        }
        looser_tree.init();

        auto target_buffer_index = level_begin;
        buffers_[target_buffer_index].weight_ = weight_sum;
        buffers_[target_buffer_index].sorted_ = true;

        size_t total_index = 0;
        std::vector<size_t> positions(level.size(), 0);
        for (int j = 0; j < k_; j++) {
            size_t target_rank = GetTargetRank(j, weight_sum);
            ValueType sample;
            bool first = true;
            while (total_index < target_rank) {
                auto minimum_index = looser_tree.min_source();
                if (first) {
                    first = false;
                } else {
                    emit(sample);
                }
                sample = level[minimum_index].elements_[positions[minimum_index]];
                total_index += level[minimum_index].weight_;
                positions[minimum_index]++;
                looser_tree.delete_min_insert(
                        level[minimum_index].elements_[positions[minimum_index]],
                        false);
            }
            buffers_[target_buffer_index].Put(sample);
        }

        empty_buffers_ += level.size() - 1;
        level_counters_[minimum_level_] = 0;
        current_buffer_ = -1;

        minimum_level_++;
        if (minimum_level_ >= level_counters_.size()) {
            level_counters_.emplace_back(1);
        } else {
            level_counters_[minimum_level_]++;
        }
        assert(minimum_level_ < level_counters_.size());

        return b_ - empty_buffers_ > 1;
    }

    // TODO pseudo concat to use all knowledge in the buffers?
    size_t GetSamples(std::vector<ValueType> &out_samples) {
        out_samples = buffers_[0].elements_;
        return buffers_[0].weight_;
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
};

} // namespace core
} // namespace thrill

#endif // !THRILL_CORE_ONLINE_SAMPLER_HEADER

/******************************************************************************/