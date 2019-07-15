/*******************************************************************************
 * benchmarks/core/sampling_benchmark.cpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2019 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <thrill/common/logger.hpp>
#include <tlx/cmdline_parser.hpp>

#include <thrill/core/online_sampler.hpp>
#include <thrill/common/reservoir_sampling.hpp>

using namespace thrill; // NOLINT

using Type = long;

using Comparator = std::less<>;

class DefaultSortAlgorithm
{
public:
    template <typename Iterator, typename CompareFunction>
    void operator () (Iterator begin, Iterator end, CompareFunction cmp) const {
        return std::sort(begin, end, cmp);
    }
};

double calculate_error(std::vector<Type>& sequence, std::vector<Type>& samples) {
    size_t n = sequence.size();
    size_t k = samples.size();

    float error = 0;
    int actual_rank = 0;
    for (size_t j = 0; j < k; j++) {
        int target_rank = (n / k) * (j + 1);
        while (sequence[actual_rank] < samples[j]) {
            actual_rank++;
        }
        error += abs(target_rank - actual_rank);
    }
    return error / k / n;
}

int main(int argc, char *argv[]) {

    tlx::CmdlineParser clp;

    int iterations;
    clp.add_param_int("i", iterations, "Iterations.");

    int n;
    clp.add_param_int("n", n, "Amount of elements.");

    std::string generator_type;
    clp.add_param_string("generator", generator_type,
            "Type of generator (uni, expo, sort, ones, almost, dup).");

    if (!clp.process(argc, argv)) {
        return -1;
    }

    clp.print_result();

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> distribution;

    auto generator = [&rng, &distribution](int i){return i;};

    // TODO Generators

    api::Run(
            [&iterations, &n, &generator, &rng](api::Context &ctx) {
                for (int i = 0; i < iterations; i++) {
                    std::vector<std::vector<Type>> rs_samples_list;
                    std::vector<std::vector<Type>> os_samples_list;

                    std::vector<Type> rs_samples;
                    common::ReservoirSamplingGrow<Type, std::mt19937> rs(
                            rs_samples, rng);

                    core::OnlineSampler
                            <Type, Comparator, DefaultSortAlgorithm, false>
                            os(5, 600, ctx, 0, Comparator(),
                                    DefaultSortAlgorithm());

                    LOG1 << "start generating";
                    std::vector<Type> sequence;
                    sequence.reserve(n);
                    for (int j = 0; j < n; j++) {
                        sequence.emplace_back(generator(j));

                        rs.add(sequence.back());
                        auto has_capacity = os.Put(sequence.back());
                        if (!has_capacity) {
                            os.Collapse();
                        }

                        if (/*ctx.my_rank() == 0 &&*/ j % static_cast<int>(n / 10.0) == 0 || j == n - 1) {
                            LOG1 << "draw samples";
                            std::vector<Type> os_samples;
                            os.GetSamples(os_samples);
                            os_samples_list.push_back(os_samples);
                            rs_samples_list.push_back(rs.samples());
                            std::sort(rs_samples_list.back().begin(),
                                    rs_samples_list.back().end());
                        }
                    }

                    if (ctx.my_rank() == 0) {
                        LOG1 << "iteration " << i;
                        std::sort(sequence.begin(), sequence.end());
                        for (size_t x = 1; x < rs_samples_list.size(); x++) {
                            auto rs_error = calculate_error(sequence,
                                                            rs_samples_list[x]);
                            auto os_error = calculate_error(sequence,
                                                            os_samples_list[x]);
                            LOG1 << rs_error << "\t" << os_error;
                        }
                    }
                }
            });
}

/******************************************************************************/
