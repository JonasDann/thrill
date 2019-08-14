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

using Type = int;
using Comparator = std::less<>;
using RandomGenerator = std::mt19937_64;

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

size_t calculate_wrong_processor(const size_t n, const size_t P,
        const size_t start, const size_t end, const size_t num_splitters,
        std::vector<Type>& splitters, std::vector<Type>& sequence_all,
        std::vector<std::vector<Type>>& sequence_gather) {
    size_t wrong_processor = 0;

    for (auto y = start; y < end; y++) {
        auto element = sequence_gather[y % P][
                y / P];
        size_t target_processor =
                (std::lower_bound(
                        sequence_all.begin(),
                        sequence_all.end(),
                        element) -
                 sequence_all.begin()) /
                (n / (num_splitters + 1));
        size_t rs_processor = std::lower_bound(
                splitters.begin(),
                splitters.end(),
                element) - splitters.begin();
        if (target_processor != rs_processor) {
            wrong_processor++;
        }
    }

    return wrong_processor;
}

void get_splitters(api::Context& ctx, const size_t P, const size_t rank,
        const std::vector<double>& quantiles, const std::vector<Type>& samples,
        std::vector<Type>& out_splitters) {
    auto stream = ctx.template GetNewStream<data::CatStream>(0);
    auto writers = stream->GetWriters();
    auto readers = stream->GetReaders();

    writers[0].Put(samples);
    writers[0].Close();

    if (rank == 0) {
        std::vector<Type> rs_samples_all;
        for (auto& reader : readers) {
            auto s = reader.template Next<std::vector<Type>>();
            rs_samples_all.insert(rs_samples_all.end(),
                                  s.begin(),
                                  s.end());
        }
        std::sort(rs_samples_all.begin(), rs_samples_all.end());

        for (double quantile : quantiles) {
            out_splitters.push_back(rs_samples_all[
                    static_cast<size_t >(quantile * rs_samples_all.size())]);
        }
    }

    for (size_t p = 1; p < P; p++) {
        writers[p].Close();
    }
    stream.reset();
}

void gather_sequence(api::Context& ctx, const size_t rank, const size_t P,
        const std::vector<Type>& sequence, std::vector<Type>& out_sequence_all,
        std::vector<std::vector<Type>>& out_sequence_gather) {
    auto stream = ctx.template GetNewStream<data::CatStream>(0);
    auto writers = stream->GetWriters();
    auto readers = stream->GetReaders();

    writers[0].Put(sequence);
    writers[0].Close();

    if (rank == 0) {
        for (auto &reader : readers) {
            auto s = reader.template Next<std::vector<Type>>();
            out_sequence_all.insert(out_sequence_all.end(),
                                s.begin(),
                                s.end());
            out_sequence_gather.push_back(s);
        }
    }

    for (size_t p = 1; p < P; p++) {
        writers[p].Close();
    }
    stream.reset();
}

Type generator(size_t i, size_t n, const std::string& generator_type,
        RandomGenerator rng) {
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

void iteration(const size_t i, const size_t rank, const size_t n,
        const size_t P, const size_t num_measurements, api::Context& ctx,
        RandomGenerator& rng, const std::string& generator_type, const size_t b,
        const size_t k, const std::vector<double>& quantiles,
        std::vector<std::vector<Type>>& out_rs_splitters,
        std::vector<std::vector<Type>>& out_os_splitters,
        std::vector<Type>& out_sequence_all,
        std::vector<std::vector<Type>>& out_sequence_gather) {
    const size_t nP = n / P;

    if (rank == 0) {
        LOG1 << "ITERATION " << i;
    }

    // Initialize
    std::vector<Type> samples;
    common::ReservoirSampling<Type, RandomGenerator> rs(1000 / P, samples, rng);

    core::OnlineSampler<Type, Comparator, DefaultSortAlgorithm, false>
            os(b, k, ctx, 0, Comparator(), DefaultSortAlgorithm());

    std::vector<Type> sequence;
    sequence.reserve(nP);
    for (size_t j = 0; j < nP; j++) {
        // Generate and insert element
        auto global_idx = j * P + rank;
        sequence.emplace_back(generator(global_idx, n,
                                        generator_type, rng));

        rs.add(sequence.back());
        auto has_capacity = os.Put(sequence.back());
        if (!has_capacity) {
            os.Collapse();
        }

        // Draw splitters
        if ((num_measurements > 1 &&
             j % (nP / num_measurements) == nP / num_measurements - 1) ||
             j == nP - 1) {
            std::vector<Type> os_splitters;
            os.GetSplitters(quantiles, os_splitters);
            if (rank == 0) {
                out_os_splitters.push_back(os_splitters);
            }

            std::vector<Type> rs_splitters;
            const auto& rs_samples = rs.samples();
            get_splitters(ctx, P, rank, quantiles,
                          rs_samples, rs_splitters);
            out_rs_splitters.push_back(rs_splitters);
        }
    }

    // Gather sequence
    gather_sequence(ctx, rank, P, sequence, out_sequence_all,
            out_sequence_gather);

    std::sort(out_sequence_all.begin(), out_sequence_all.end());
}

int main(int argc, char *argv[]) {

    tlx::CmdlineParser clp;

    std::string benchmark;
    clp.add_param_string("benchmark", benchmark,
                         "Type of benchmark (histogram, error, parameter).");

    int iterations;
    clp.add_param_int("i", iterations, "Iterations.");

    size_t n;
    clp.add_param_size_t("n", n, "Amount of elements.");

    std::string generator_type;
    clp.add_param_string("generator", generator_type,
            "Type of generator (uni, expo, sort, ones, almost, window, dup).");

    if (!clp.process(argc, argv)) {
        return -1;
    }

    clp.print_result();

    std::random_device rd;
    RandomGenerator rng(rd());

    api::Run(
            [&iterations, &n, &generator_type, &rng, &benchmark]
            (api::Context &ctx) {
                const size_t P = ctx.num_workers();
                const size_t rank = ctx.my_rank();

                const size_t num_splitters = 9; // TODO console parameter

                std::vector<double> quantiles(num_splitters);
                for (size_t s = 0; s < num_splitters; s++) {
                    quantiles[s] = static_cast<double>(s + 1) / (num_splitters + 1);
                }

                if (benchmark == "error") {
                    n -= n % P;

                    const size_t num_measurements = 10;
                    const size_t b = 10 / P;
                    const size_t k = 100;

                    std::string generators[] = {"uni", "sort", "dup", "window"};

                    std::vector<std::vector<double>> errors(8);

                    for (size_t g = 0; g < 4; g++) {
                        std::vector<std::vector<Type>> rs_splitters;
                        std::vector<std::vector<Type>> os_splitters;
                        std::vector<Type> sequence_all;
                        std::vector<std::vector<Type>> sequence_gather;

                        iteration(0, rank, n, P, num_measurements, ctx, rng,
                                  generators[g], b, k, quantiles, rs_splitters,
                                  os_splitters, sequence_all, sequence_gather);

                        if (rank == 0) {
                            // Calculate mean error of splitters
                            for (size_t m = 0; m < num_measurements; m++) {
                                errors[g * 2].push_back(calculate_error(
                                        sequence_all, rs_splitters[m]));
                                errors[g * 2 + 1].push_back(calculate_error(
                                        sequence_all, os_splitters[m]));
                            }

                            // Calculate number of elements on wrong host
                            // TODO plot
                            LOG1 << "";
                            LOG1 << "WRONG PROCESSOR";
                            double rs_wrong_processor = 0;
                            double os_wrong_processor = 0;
                            for (size_t x = 0; x < num_measurements; x++) {
                                size_t start = x * n / num_measurements;
                                size_t end = (x + 1) * n / num_measurements;

                                rs_wrong_processor += static_cast<double>(
                                        calculate_wrong_processor(n, P,
                                                                  start, end, num_splitters,
                                                                  rs_splitters[x], sequence_all,
                                                                  sequence_gather));

                                os_wrong_processor += static_cast<double>(
                                        calculate_wrong_processor(n, P,
                                                                  start, end, num_splitters,
                                                                  os_splitters[x], sequence_all,
                                                                  sequence_gather));

                                LOG1 << rs_wrong_processor / n << "\t"
                                     << os_wrong_processor / n;
                            }
                        }
                    }

                    if (rank == 0) {
                        std::ofstream errors_file;
                        errors_file.open("sampling_convergence.dat");
                        for (size_t m = 0; m < num_measurements; m++) {
                            errors_file << (m + 1) * n / num_measurements << "\t";
                            for (size_t d = 0; d < errors.size(); d++) {
                                errors_file << errors[d][m];
                                if (d == errors.size() - 1) {
                                    errors_file << "\n";
                                } else {
                                    errors_file << "\t";
                                }
                            }
                        }
                        LOG1 << "Sampling convergence written to sampling_convergence.dat";
                        errors_file.close();
                    }
                } else if (benchmark == "histogram") {
                    n -= n % P;

                    const size_t b = 8 / P;
                    const size_t k = 119;

                    std::vector<int> rs_histogram(n);
                    std::vector<int> os_histogram(n);

                    std::vector<std::vector<size_t>> rs_splitter_ranks_by_iter;
                    std::vector<std::vector<size_t>> os_splitter_ranks_by_iter;

                    for (int i = 0; i < iterations; i++) {
                        std::vector<std::vector<Type>> rs_splitters;
                        std::vector<std::vector<Type>> os_splitters;
                        std::vector<Type> sequence_all;
                        std::vector<std::vector<Type>> sequence_gather;

                        iteration(i, rank, n, P, 1, ctx, rng,
                                  generator_type, b, k, quantiles, rs_splitters,
                                  os_splitters, sequence_all, sequence_gather);

                        if (rank == 0) {
                            std::vector<size_t> rs_splitter_ranks;
                            std::vector<size_t> os_splitter_ranks;
                            for (size_t s = 0; s < num_splitters; s++) {
                                auto rs_rank = std::lower_bound(
                                        sequence_all.begin(),
                                        sequence_all.end(),
                                        rs_splitters[0][s]) -
                                               sequence_all.begin();
                                rs_splitter_ranks.push_back(rs_rank);
                                auto os_rank = std::lower_bound(
                                        sequence_all.begin(),
                                        sequence_all.end(),
                                        os_splitters[0][s]) -
                                               sequence_all.begin();
                                os_splitter_ranks.push_back(os_rank);
                            }
                            rs_splitter_ranks_by_iter.push_back(
                                    rs_splitter_ranks);
                            os_splitter_ranks_by_iter.push_back(
                                    os_splitter_ranks);

                            for (size_t s = 0; s < num_splitters; s++) {
                                auto rs_rank =
                                        std::lower_bound(sequence_all.begin(),
                                                         sequence_all.end(),
                                                         rs_splitters.back()[s]) -
                                        sequence_all.begin();
                                rs_histogram[rs_rank]++;
                                auto os_rank =
                                        std::lower_bound(sequence_all.begin(),
                                                         sequence_all.end(),
                                                         os_splitters.back()[s]) -
                                        sequence_all.begin();
                                os_histogram[os_rank]++;
                            }
                        }
                    }

                    if (rank == 0) {
                        double avg_rs_st_dev = 0;
                        double avg_os_st_dev = 0;
                        double avg_rs_avg_err = 0;
                        double avg_os_avg_err = 0;
                        for (size_t s = 0; s < num_splitters; s++) {
                            double rs_avg = 0;
                            double os_avg = 0;
                            for (size_t x = 0;
                                 x < rs_splitter_ranks_by_iter.size(); x++) {
                                rs_avg += rs_splitter_ranks_by_iter[x][s];
                                os_avg += os_splitter_ranks_by_iter[x][s];
                            }
                            rs_avg /= rs_splitter_ranks_by_iter.size();
                            os_avg /= os_splitter_ranks_by_iter.size();
                            double rs_st_dev = 0;
                            double os_st_dev = 0;
                            for (size_t x = 0;
                                 x < rs_splitter_ranks_by_iter.size(); x++) {
                                rs_st_dev += pow(
                                        rs_splitter_ranks_by_iter[x][s] - rs_avg, 2);
                                os_st_dev += pow(
                                        os_splitter_ranks_by_iter[x][s] - os_avg, 2);
                            }
                            avg_rs_st_dev += sqrt(
                                    rs_st_dev / rs_splitter_ranks_by_iter.size());
                            avg_os_st_dev += sqrt(
                                    os_st_dev / os_splitter_ranks_by_iter.size());
                            auto target_rank = (n / (num_splitters + 1)) * (s + 1);
                            avg_rs_avg_err += rs_avg - target_rank;
                            avg_os_avg_err += os_avg - target_rank;
                        }
                        LOG1 << "RS standard deviation: "
                             << avg_rs_st_dev / num_splitters << "(mean mean error: "
                             << avg_rs_avg_err / num_splitters << ")";
                        LOG1 << "OS standard deviation: "
                             << avg_os_st_dev / num_splitters << "(mean mean error: "
                             << avg_os_avg_err / num_splitters << ")";

                        std::ofstream rs_file;
                        rs_file.open("reservoir_sampling_histogram.dat");
                        for (size_t i = 0; i < rs_histogram.size(); i++) {
                            rs_file << i << "\t" << rs_histogram[i] << "\n";
                        }
                        LOG1 << "RS histogram written to reservoir_sampling_histogram.dat";
                        rs_file.close();
                        std::ofstream os_file;
                        os_file.open("online_sampling_histogram.dat");
                        for (size_t i = 0; i < os_histogram.size(); i++) {
                            os_file << i << "\t" << os_histogram[i] << "\n";
                        }
                        LOG1 << "OS histogram written to online_sampling_histogram.dat";
                        os_file.close();
                    }
                } else if (benchmark == "parameter") {
                    const size_t b_start = 2;
                    const size_t b_end = 50;
                    const size_t b_step = 5;

                    const size_t k_start = 20;
                    const size_t k_end = 1000;
                    const size_t k_step = 50;

                    const auto combinations = static_cast<size_t>(ceil(
                            static_cast<double>(b_end - b_start) / b_step) *
                            ceil(static_cast<double>(k_end - k_start) / k_step));
                    if (rank == 0) {
                        LOG1 << combinations << " combinations";
                    }

                    size_t i = 0;
                    std::ofstream result_file;
                    if (rank == 0) {
                        result_file.open("online_sampling_parameter.dat");
                    }
                    for (size_t b = b_start; b < b_end; b += b_step) {
                        for (size_t k = k_start; k < k_end; k += k_step) {
                            std::vector<std::vector<Type>> rs_splitters;
                            std::vector<std::vector<Type>> os_splitters;
                            std::vector<Type> sequence_all;
                            std::vector<std::vector<Type>> sequence_gather;

                            iteration(i++, rank, b * k * P * 1000, P, 1, ctx, rng,
                                      generator_type, b, k, quantiles, rs_splitters,
                                      os_splitters, sequence_all, sequence_gather);

                            if (rank == 0) {
                                auto os_error = calculate_error(sequence_all,
                                                                os_splitters[0]);

                                result_file << k << "\t" << b * P << "\t"
                                        << os_error << "\n";
                            }
                        }
                    }
                    LOG1 << "Result written to online_sampling_parameter.dat";
                    result_file.close();
                }
            });
}

/******************************************************************************/
