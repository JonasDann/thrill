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

#include <fstream>

using namespace thrill; // NOLINT

bool ERROR = false;
bool ST_DEV = true;
bool HISTOGRAM = false;

using Type = int;
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
        int target_rank = (n / (k + 1)) * (j + 1);
        while (sequence[actual_rank] < samples[j]) {
            actual_rank++;
        }
        error += abs(target_rank - actual_rank);
    }
    return error / k / n;
}

size_t get_rank(std::vector<Type>& sequence, Type element) {
    size_t rank = 0;
    while (sequence[rank] < element) {
        rank++;
    }
    return rank;
}

Type generator(size_t i, size_t n, std::string generator_type,
        std::default_random_engine rng) {
    std::uniform_int_distribution<int> uni;
    std::exponential_distribution<double> expo(2.0);
    if (generator_type == "uni") {
        return uni(rng);
    } else if (generator_type == "expo") {
        return expo(rng);
    } else if (generator_type == "sort") {
        return i;
    } else if (generator_type == "ones") {
        return 1;
    } else if (generator_type == "almost") {
        return 1; // TODO
    } else if (generator_type == "window") {
        size_t window_size = 10000;
        return i + (uni(rng) % window_size);
    } else if (generator_type == "dup") {
        return static_cast<Type>(std::fmod((pow(i, 2) + static_cast<double>(n) / 2), n));
    }
    return -1;
}

int main(int argc, char *argv[]) {

    tlx::CmdlineParser clp;

    int iterations;
    clp.add_param_int("i", iterations, "Iterations.");

    int n;
    clp.add_param_int("n", n, "Amount of elements.");

    std::string generator_type;
    clp.add_param_string("generator", generator_type,
            "Type of generator (uni, expo, sort, ones, almost, window, dup).");

    if (!clp.process(argc, argv)) {
        return -1;
    }

    clp.print_result();

    std::random_device rd;
    std::default_random_engine rng(rd());

    api::Run(
            [&iterations, &n, &generator_type, &rng](api::Context &ctx) {
                size_t num_splitters = 5;
                std::vector<int> rs_histogram(n);
                std::vector<int> os_histogram(n);

                std::vector<std::vector<size_t>> all_rs_splitters;
                std::vector<std::vector<size_t>> all_os_splitters;

                for (int i = 0; i < iterations; i++) {
                    std::vector<std::vector<Type>> rs_splitters_list;
                    std::vector<std::vector<Type>> os_splitters_list;

                    std::vector<Type> rs_samp;
                    common::ReservoirSampling<Type, std::default_random_engine> rs(
                            1000, rs_samp, rng);

                    core::OnlineSampler
                            <Type, Comparator, DefaultSortAlgorithm, false>
                            os(10, 100, ctx, 0, Comparator(),
                                    DefaultSortAlgorithm());

                    LOG0 << "start generating";
                    std::vector<Type> sequence;
                    sequence.reserve(n);
                    for (int j = 0; j < n; j++) {
                        sequence.emplace_back(generator(j, n, generator_type, rng));

                        rs.add(sequence.back());
                        auto has_capacity = os.Put(sequence.back());
                        if (!has_capacity) {
                            os.Collapse();
                        }

                        if (/*ctx.my_rank() == 0 &&*/ j % static_cast<int>(n / 10.0) == n / 10.0 - 1) {
                            if (ctx.my_rank() == 0) {
                                LOG0 << "draw samples";
                            }
                            std::vector<double> quantiles;
                            quantiles.reserve(num_splitters);
                            for (size_t s = 0; s < num_splitters; s++) {
                                quantiles.emplace_back(static_cast<double>(s + 1) / (num_splitters + 1));
                            }
                            std::vector<Type> os_splitters;
                            os.GetSplitters(quantiles, os_splitters);
                            os_splitters_list.push_back(os_splitters);
                            std::vector<Type> rs_samples;
                            rs_samples = rs.samples();
                            std::sort(rs_samples.begin(), rs_samples.end());
                            //TODO Communication
                            std::vector<Type> rs_splitters;
                            for (size_t s = 0; s < num_splitters; s++) {
                                rs_splitters.push_back(rs_samples[
                                        static_cast<double>(rs_samples.size()) *
                                        quantiles[s]]);
                            }
                            rs_splitters_list.push_back(rs_splitters);
                        }
                    }

                    if (ctx.my_rank() == 0) {
                        LOG1 << "iteration " << i;
                        std::sort(sequence.begin(), sequence.end());
                        for (size_t x = 0; x < rs_splitters_list.size(); x++) {
                            if (ERROR) {
                                auto rs_error = calculate_error(sequence,
                                                                rs_splitters_list[x]);
                                auto os_error = calculate_error(sequence,
                                                                os_splitters_list[x]);
                                LOG1 << rs_error << "\t" << os_error;
                            }

                            if (ST_DEV) {
                                std::vector<size_t> rs_splitter_ranks;
                                std::vector<size_t> os_splitter_ranks;
                                for (size_t s = 0; s < num_splitters; s++) {
                                    rs_splitter_ranks.push_back(
                                            get_rank(sequence,
                                                     rs_splitters_list[x][s]));
                                    os_splitter_ranks.push_back(
                                            get_rank(sequence,
                                                     os_splitters_list[x][s]));
                                }
                                all_rs_splitters.push_back(rs_splitter_ranks);
                                all_os_splitters.push_back(os_splitter_ranks);
                            }
                        }

                        if (HISTOGRAM) {
                            for (size_t s = 0; s < num_splitters; s++) {
                                rs_histogram[get_rank(sequence,
                                                      rs_splitters_list.back()[s])]++;
                                os_histogram[get_rank(sequence,
                                                      os_splitters_list.back()[s])]++;
                            }
                        }
                    }
                }
                if (ctx.my_rank() == 0) {
                    if (ST_DEV) {
                        double avg_rs_st_dev = 0;
                        double avg_os_st_dev = 0;
                        for (size_t s = 0; s < num_splitters; s++) {
                            double rs_avg = 0;
                            double os_avg = 0;
                            for (size_t x = 0;
                                 x < all_rs_splitters.size(); x++) {
                                rs_avg += all_rs_splitters[x][s];
                                os_avg += all_os_splitters[x][s];
                            }
                            rs_avg /= all_rs_splitters.size();
                            os_avg /= all_os_splitters.size();
                            double rs_st_dev = 0;
                            double os_st_dev = 0;
                            for (size_t x = 0;
                                 x < all_rs_splitters.size(); x++) {
                                rs_st_dev += pow(
                                        all_rs_splitters[x][s] - rs_avg, 2);
                                os_st_dev += pow(
                                        all_os_splitters[x][s] - os_avg, 2);
                            }
                            avg_rs_st_dev += sqrt(
                                    rs_st_dev / all_rs_splitters.size());
                            avg_os_st_dev += sqrt(
                                    os_st_dev / all_os_splitters.size());
                        }
                        LOG1 << "RS standard deviation: "
                             << avg_rs_st_dev / num_splitters;
                        LOG1 << "OS standard deviation: "
                             << avg_os_st_dev / num_splitters;
                    }

                    if (HISTOGRAM) {
                        std::ofstream rs_file;
                        rs_file.open("rs_file");
                        LOG1 << "Write RS histogram";
                        for (size_t i = 0; i < rs_histogram.size(); i++) {
                            rs_file << i << "\t" << rs_histogram[i] << "\n";
                        }
                        rs_file.close();
                        std::ofstream os_file;
                        os_file.open("os_file");
                        LOG1 << "Write OS histogram";
                        for (size_t i = 0; i < os_histogram.size(); i++) {
                            os_file << i << "\t" << os_histogram[i] << "\n";
                        }
                        os_file.close();
                    }
                }
            });
}

/******************************************************************************/
