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

namespace thrill {
namespace data {

//! \addtogroup data_layer
//! \{

/*!
 * A SampledFile is a File with in memory samples of every Block. This is useful
 * whenever a search like GetIndexOf is performed often on a file that is on
 * disk. A notable example is Canonical Merge Sort and in particular the
 * Multisequence Selection used there.
 *
 * With the in memory samples, only one block has to be accessed from disk for
 * every search.
 */
template <typename ItemType>
class SampledFile : public File
{
public:
    //! Append a block to this file, the block must contain given number of
    //! items after the offset first. Also saves the first element of the block
    //! as a sample.
    void AppendBlock(const Block& b) override {
        auto sample_index = num_items();
        File::AppendBlock(b);
        block_samples_.emplace_back(GetItemAt<ItemType>(sample_index));
    }

    //! Append a block to this file, the block must contain given number of
    //! items after the offset first. Also saves the first element of the block
    //! as a sample.
    void AppendBlock(Block&& b) override {
        auto sample_index = num_items();
        File::AppendBlock(std::move(b));
        block_samples_.emplace_back(GetItemAt<ItemType>(sample_index));
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

    //! Get BlockWriter.
    Writer GetWriter(size_t block_size) override {
        return Writer(
                FileBlockSink(tlx::CountingPtrNoDelete<SampledFile>(this)),
                block_size);
    }

    //! Get BlockWriter with default block size.
    Writer GetWriter() override {
        return GetWriter(default_block_size);
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
    size_t GetFastIndexOf(const ItemType& item, size_t tie,
            const Comparator& func = Comparator()) const {
        // TODO Use binary search here
        if (item < block_samples_[0]) {
            return 0;
        }
        size_t block_index = 0;
        while (block_index < block_samples_.size() &&
               item < block_samples_[block_index + 1]) {
            block_index++;
        }
        if (block_index + 1 == block_samples_.size()) {
            return num_items_sum_[block_index];
        }
        auto left = num_items_sum_[block_index];
        auto right = num_items_sum_[block_index + 1] - 1;
        return GetIndexOf<ItemType, Comparator>(item, tie, left, right, func);
    }
private:
    std::deque<ItemType> block_samples_;
};

template <typename ItemType>
using SampledFilePtr = tlx::CountingPtr<SampledFile<ItemType>>;

//! \}

} // namespace data
} // namespace thrill

#endif // !THRILL_DATA_SAMPLED_FILE_HEADER

/******************************************************************************/