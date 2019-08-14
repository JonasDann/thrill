/*******************************************************************************
 * thrill/data/sampled_file.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2019 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_DATA_SAMPLED_FILE_HEADER
#define THRILL_DATA_SAMPLED_FILE_HEADER

#include <thrill/data/file.hpp>
#include <thrill/data/fwd.hpp>
#include <tlx/die.hpp>

namespace thrill {
namespace data {

//! \addtogroup data_layer
//! \{

/*!
 * A SampledFile is a File with in memory samples of every Block. This is useful
 * whenever a search like GetIndexOf is performed often on a file that is on
 * disk. A notable example is Canonical Merge Sort and in particular the
 * Multi Sequence Selection used there.
 *
 * With the in memory samples, only one block has to be accessed from disk for
 * every search.
 */
template <typename ItemType>
class SampledFile : public File
{
    static constexpr bool debug = false;
    static constexpr bool self_verify = false;

public:
    //! Constructor from BlockPool
    SampledFile(BlockPool& block_pool, size_t local_worker_id, size_t dia_id) :
        File(block_pool, local_worker_id, dia_id) {}

    //! non-copyable: delete copy-constructor
    SampledFile(const SampledFile&) = delete;
    //! non-copyable: delete assignment operator
    SampledFile& operator = (const SampledFile&) = delete;
    //! move-constructor: default
    SampledFile(SampledFile&&) noexcept = default;
    //! move-assignment operator: default
    SampledFile& operator = (SampledFile&&) noexcept = default;

    //! Append a block to this file, the block must contain given number of
    //! items after the offset first. Also saves the first element of the block
    //! as a sample.
    void AppendBlock(const Block& b) override {
        auto sample_index = num_items();
        File::AppendBlock(b);
        block_samples_.emplace_back(GetItemAt<ItemType>(sample_index));
        LOG << "AppendBlock (" << sample_index << ", " << block_samples_.back() << ")";
    }

    //! Append a block to this file, the block must contain given number of
    //! items after the offset first. Also saves the first element of the block
    //! as a sample.
    void AppendBlock(Block&& b) override {
        auto sample_index = num_items();
        File::AppendBlock(std::move(b));
        block_samples_.emplace_back(GetItemAt<ItemType>(sample_index));
        LOG << "AppendBlockMove (" << sample_index << ", " << block_samples_.back() << ")";
    }

    //! Append a block to this file, the block must contain given number of
    //! items after the offset first. Also saves the first element of the block
    //! as a sample.
    void AppendBlock(const Block& b, bool /* is_last_block */) override {
        return AppendBlock(b);
    }

    //! Append a block to this file, the block must contain given number of
    //! items after the offset first. Also saves the first element of the block
    //! as a sample.
    void AppendBlock(Block&& b, bool /* is_last_block */) override {
        return AppendBlock(std::move(b));
    }

    //! Appends the PinnedBlock
    void AppendPinnedBlock(PinnedBlock&& b, bool is_last_block) override {
        auto sample_index = num_items();
        AppendBlock(std::move(b).MoveToBlock(), is_last_block);
        block_samples_.emplace_back(GetItemAt<ItemType>(sample_index));
        LOG << "AppendPinnedBlock (" << sample_index << ", " << block_samples_.back() << ")";
    }

    /*!
     * Get index of the given item, or the next greater item, in this file. The
     * file has to be ordered according to the given compare function. The tie
     * value can be used to make a decision in case of many successive equal
     * elements.  The tie is compared with the local rank of the element.
     *
     * The search is sped up by the Block samples.
     */
    template <typename Comparator = std::less<ItemType>>
    size_t GetFastIndexOf(const ItemType& item, size_t tie, size_t left,
            size_t right, const Comparator& less = Comparator()) const {
        LOG << "item: " << item;
        // TODO replace with custom binary search with tie breaker
        size_t block_index = std::lower_bound(block_samples_.begin(),
                block_samples_.end(), item, less) - block_samples_.begin();
        size_t result = 0;
        if (block_index > 0) {
            block_index--;

            size_t block_left = 0;
            if (block_index > 0) {
                block_left = num_items_sum_[block_index - 1];
            }
            // TODO Tie
            /*if (item == block_samples_[block_index] && block_index > 0) {
                block_left = num_items_sum_[block_index - 1];
            }*/
            size_t block_right = num_items_sum_[block_index];
            LOG << "block_index: " << block_index << " / " << block_samples_.size();
            LOG << "range: [" << block_left << ", " << block_right << ")";
            result = GetIndexOf<ItemType, Comparator>(item, tie, block_left, block_right, less);
        }
        if (self_verify) {
            auto real_index = GetIndexOf<ItemType, Comparator>(item, tie, left, right, less);
            LOG << "result: " << result << " / " << real_index;
            die_unless(result == real_index);
        }
        return result;
    }

    std::deque<size_t> num_items_sum() {
        return num_items_sum_;
    }

    std::deque<ItemType> block_samples() {
        return block_samples_;
    }
private:
    std::deque<ItemType> block_samples_;
};

//! \}

} // namespace data
} // namespace thrill

#endif // !THRILL_DATA_SAMPLED_FILE_HEADER

/******************************************************************************/