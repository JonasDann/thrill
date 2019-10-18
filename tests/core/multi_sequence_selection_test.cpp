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

using LocalRanks = std::vector<std::vector<size_t> >;
using VectorSequenceAdapter = core::MultiSequenceSelectorVectorSequenceAdapter<int>;
using Comparator = std::less<int>;

TEST(MultiSequenceSelection, MultiVectorSequenceSelection) {
    const auto run_count = 3;
    const auto elements_per_run_goal = 10000;
    const auto splitter_count = 7;

    auto test_lambda = [](Context& context) {
                           const auto p = context.num_workers();
                           const auto elements_per_run = elements_per_run_goal -
                                                         (elements_per_run_goal % p);

                           auto comparator = Comparator();
                           std::uniform_int_distribution<int> uni;
                           std::random_device rd;
                           std::mt19937 rng(rd());

                           LocalRanks local_ranks(splitter_count, std::vector<size_t>(run_count));
                           std::vector<VectorSequenceAdapter> sequences(run_count);
                           for (size_t r = 0; r < run_count; r++) {
                               for (size_t i = 0; i < elements_per_run / p; i++) {
                                   auto element = uni(rng);
                                   sequences[r].emplace_back(element);
                               }
                               std::sort(sequences[r].begin(), sequences[r].end(), comparator);
                           }

                           core::run_multi_sequence_selection<VectorSequenceAdapter, Comparator>
                               (context, comparator, sequences, local_ranks, splitter_count);

                           size_t global_splitter_distance = run_count * elements_per_run /
                                                             (splitter_count + 1);
                           size_t correction;
                           for (size_t s = 0; s < splitter_count; s++) {
                               if (s < (run_count * elements_per_run) % (splitter_count + 1)) {
                                   correction = 1;
                               }
                               else {
                                   correction = 0;
                               }
                               size_t sum_ranks = 0;
                               for (size_t r = 0; r < run_count; r++) {
                                   sum_ranks += local_ranks[s][r];
                               }
                               sum_ranks = context.net.AllReduce(sum_ranks);
                               ASSERT_EQ((s + 1) * global_splitter_distance + correction,
                                         sum_ranks);
                           }
                       };

    api::RunLocalTests(test_lambda);
}

/******************************************************************************/
