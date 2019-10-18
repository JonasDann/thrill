/*******************************************************************************
 * tests/core/online_sampler_test.cpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2019 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>

#include <thrill/core/online_sampler.hpp>

#include <thrill/common/logger.hpp>

using namespace thrill; // NOLINT

using Comparator = std::less<>;

class DefaultSortAlgorithm
{
public:
    template <typename Iterator, typename CompareFunction>
    void operator () (Iterator begin, Iterator end, CompareFunction cmp) const {
        return std::sort(begin, end, cmp);
    }
};

template <typename ValueType>
double calculate_error(std::vector<ValueType>& sequence,
                       std::vector<double>& quantiles, std::vector<ValueType>& samples) {
    size_t N = sequence.size();
    size_t k = samples.size();

    float error = 0;
    for (size_t i = 0; i < quantiles.size(); i++) {
        int target_rank = quantiles[i] * N;
        int actual_rank = 0;
        while (sequence[actual_rank] < samples[i]) {
            actual_rank++;
        }
        error += abs(target_rank - actual_rank);
    }
    return error / k / N;
}

TEST(OnlineSampler, IntUniFullSortedBufferSampling) {
    auto test_lambda = [](Context& context) {
                           auto comparator = Comparator();
                           DefaultSortAlgorithm sort_algorithm;
                           std::uniform_int_distribution<int> uni;
                           std::random_device rd;
                           std::mt19937 rng(rd());

                           auto p = context.num_workers();
                           size_t b = 3;
                           size_t k = 2778;
                           size_t N_pow = 5;
                           size_t N_p = ((size_t)(pow(10, N_pow) / (p * b * k)) + 1) * b * k;
                           core::OnlineSampler<int, Comparator, DefaultSortAlgorithm, false>
                           sampler(b, k, context, 0, comparator, sort_algorithm);

                           // generate input sequence
                           std::vector<int> sequence;
                           sequence.reserve(N_p);
                           for (size_t i = 0; i < N_p; i++) {
                               sequence.emplace_back(uni(rng));
                           }
                           std::sort(sequence.begin(), sequence.end(), comparator);

                           // stream data into buffers
                           for (size_t i = 0; i < N_p; i++) {
                               auto has_capacity = sampler.Put(sequence[i]);
                               if (!has_capacity) {
                                   sampler.Collapse();
                               }
                           }

                           std::vector<int> global_sequence;
                           global_sequence = context.net.AllReduce(sequence,
                                                                   common::VectorConcat<int>());

                           std::sort(global_sequence.begin(), global_sequence.end(), comparator);

                           std::vector<int> splitters;
                           splitters.reserve(k);
                           std::vector<double> quantiles;
                           quantiles.reserve(k);
                           for (size_t i = 1; i < k; i++) {
                               quantiles.emplace_back(i / k);
                           }
                           sampler.GetSplitters(quantiles, splitters);

                           auto error = calculate_error(global_sequence, quantiles, splitters);

                           ASSERT_GT(0.001, error);
                       };

    api::RunLocalTests(test_lambda);
}

/******************************************************************************/
