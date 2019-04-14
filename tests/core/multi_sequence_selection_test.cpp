/*******************************************************************************
 * tests/core/multi_sequence_selection_test.cpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2019 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>

#include <thrill/core/multi_sequence_selection.hpp>

#include <thrill/common/logger.hpp>

using namespace thrill; // NOLINT

using LocalRanks = std::vector<std::vector<size_t>>;
using VectorSequenceAdapter = core::MultiSequenceSelectorVectorSequenceAdapter<int>;
using Comparator = std::less<int>;

TEST(MultiSequenceSelection, MultiVectorSequenceSelection) {
    const auto run_count = 3;
    const auto elements_per_run = 10000;
    const auto splitter_count = 7;

    auto test_lambda = [](Context& context) {
        auto comparator = Comparator();
        std::vector<int> global_sequence;

        LocalRanks local_ranks(splitter_count, std::vector<size_t>(run_count));
        std::vector<VectorSequenceAdapter> sequences(run_count);
        for (size_t r = 0; r < run_count; r++) {
            for (size_t i = 0; i < elements_per_run; i++) {
                sequences[r].emplace_back(0);
                global_sequence.emplace_back(0);
            }
            std::sort(sequences[r].begin(), sequences[r].end(), comparator);
        }
        std::sort(global_sequence.begin(), global_sequence.end(), comparator);

        core::run_multi_sequence_selection<VectorSequenceAdapter, Comparator>
                (context, comparator, sequences, local_ranks, splitter_count);

        size_t splitter_distance = run_count * elements_per_run /
                (splitter_count + 1);
        for (size_t s = 0; s < splitter_count; s++) {
            size_t sum_ranks = 0;
            for (size_t r = 0; r < run_count; r++) {
                sum_ranks += local_ranks[s][r];
            }
            ASSERT_EQ((s + 1) * splitter_distance, sum_ranks);
        }
    };

    api::RunLocalTests(test_lambda);
}

/******************************************************************************/