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

void get_splitters(api::Context& ctx, const size_t P, const size_t rank,
        const size_t num_splitters, const std::vector<double>& quantiles,
        const std::vector<Type>& samples, std::vector<Type>& out_splitters) {
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

        for (size_t s = 0; s < num_splitters; s++) {
            out_splitters.push_back(rs_samples_all[
                    static_cast<size_t >(quantiles[s] * rs_samples_all.size())]);
        }
    }

    for (size_t p = 1; p < P; p++) {
        writers[p].Close();
    }
    stream.reset();
}

void gather_sequence(api::Context& ctx, const size_t rank, const size_t P, const std::vector<Type>& sequence, std::vector<Type>& out_sequence_all, std::vector<std::vector<Type>>& out_sequence_gather) {
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

int main(int argc, char *argv[]) {

    tlx::CmdlineParser clp;

    std::string benchmark;
    clp.add_param_string("benchmark", benchmark,
                         "Type of benchmark (histogram, error).");

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

    const bool HISTOGRAM = benchmark == "histogram";
    const bool ERROR = benchmark == "error";
    if (ERROR) {
        iterations = 1;
    }

    api::Run(
            [&iterations, &n, &generator_type, &rng, &ERROR, &HISTOGRAM](api::Context &ctx) {
                const size_t P = ctx.num_workers();
                const size_t rank = ctx.my_rank();
                const size_t nP = n / P;
                n = nP * P;

                const size_t num_splitters = 9;
                const double num_measurements = 10.0;

                std::vector<double> quantiles(num_splitters);
                for (size_t s = 0; s < num_splitters; s++) {
                    quantiles[s] = static_cast<double>(s + 1) / (num_splitters + 1);
                }

                std::vector<int> rs_histogram(n);
                std::vector<int> os_histogram(n);

                std::vector<std::vector<size_t>> rs_splitter_ranks_by_iter;
                std::vector<std::vector<size_t>> os_splitter_ranks_by_iter;

                for (int i = 0; i < iterations; i++) {
                    if (rank == 0) {
                        LOG1 << "ITERATION " << i;
                    }

                    std::vector<std::vector<Type>> rs_splitters_by_meas;
                    std::vector<std::vector<Type>> os_splitters_by_meas;

                    // Initialize
                    std::vector<Type> samples;
                    common::ReservoirSampling<Type, RandomGenerator> rs(
                            1000 / P, samples, rng);

                    core::OnlineSampler
                            <Type, Comparator, DefaultSortAlgorithm, false>
                            os(8 / P, 119, ctx, 0, Comparator(),
                                    DefaultSortAlgorithm());

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
                        if ((ERROR &&
                                j % static_cast<int>(nP / num_measurements) ==
                                nP / num_measurements - 1) ||
                                j == nP - 1) {
                            std::vector<Type> os_splitters;
                            os.GetSplitters(quantiles, os_splitters);
                            if (rank == 0) {
                                os_splitters_by_meas.push_back(os_splitters);
                            }

                            std::vector<Type> rs_splitters;
                            const auto& rs_samples = rs.samples();
                            get_splitters(ctx, P, rank, num_splitters, quantiles,
                                    rs_samples, rs_splitters);
                            rs_splitters_by_meas.push_back(rs_splitters);
                        }
                    }

                    // Gather sequence
                    std::vector<Type> sequence_all;
                    sequence_all.reserve(n);
                    std::vector<std::vector<Type>> sequence_gather;
                    gather_sequence(ctx, rank, P, sequence, sequence_all,
                            sequence_gather);

                    std::sort(sequence_all.begin(), sequence_all.end());

                    if (rank == 0) {
                        LOG1 << "ITERATION " << i;

                        // Calculate mean error of splitters
                        if (ERROR) {
                            LOG1 << "";
                            LOG1 << "ERROR";

                            for (size_t m = 0; m < num_measurements; m++) {
                                auto rs_error = calculate_error(
                                        sequence_all,
                                        rs_splitters_by_meas[m]);
                                auto os_error = calculate_error(
                                        sequence_all,
                                        os_splitters_by_meas[m]);
                                LOG1 << rs_error << "\t" << os_error;
                            }
                        }

                        if (HISTOGRAM) {
                            std::vector<size_t> rs_splitter_ranks;
                            std::vector<size_t> os_splitter_ranks;
                            for (size_t s = 0; s < num_splitters; s++) {
                                auto rs_rank = std::lower_bound(
                                        sequence_all.begin(),
                                        sequence_all.end(),
                                        rs_splitters_by_meas[0][s]) -
                                               sequence_all.begin();
                                rs_splitter_ranks.push_back(rs_rank);
                                auto os_rank = std::lower_bound(
                                        sequence_all.begin(),
                                        sequence_all.end(),
                                        os_splitters_by_meas[0][s]) -
                                               sequence_all.begin();
                                os_splitter_ranks.push_back(os_rank);
                            }
                            rs_splitter_ranks_by_iter.push_back(
                                    rs_splitter_ranks);
                            os_splitter_ranks_by_iter.push_back(
                                    os_splitter_ranks);
                        }

                        if (ERROR) {
                            LOG1 << "";
                            LOG1 << "WRONG PROCESSOR";
                            double rs_wrong_processor = 0;
                            double os_wrong_processor = 0;
                            for (size_t x = 0; x < num_measurements; x++) {
                                for (auto y = static_cast<size_t>(x * n / num_measurements);
                                     y < (x + 1) * n / num_measurements; y++) {
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
                                            rs_splitters_by_meas[x].begin(),
                                            rs_splitters_by_meas[x].end(),
                                            element) -
                                                          rs_splitters_by_meas[x].begin();
                                    size_t os_processor = std::lower_bound(
                                            os_splitters_by_meas[x].begin(),
                                            os_splitters_by_meas[x].end(),
                                            element) -
                                                          os_splitters_by_meas[x].begin();
                                    if (target_processor != rs_processor) {
                                        rs_wrong_processor++;
                                    }
                                    if (target_processor != os_processor) {
                                        os_wrong_processor++;
                                    }
                                }

                                LOG1 << rs_wrong_processor / n << "\t"
                                     << os_wrong_processor / n;
                            }
                            LOG1 << "";
                            LOG1 << "";
                        }

                        if (HISTOGRAM) {
                            for (size_t s = 0; s < num_splitters; s++) {
                                auto rs_rank = std::lower_bound(sequence_all.begin(),
                                                                sequence_all.end(),
                                                                rs_splitters_by_meas.back()[s]) -
                                               sequence_all.begin();
                                rs_histogram[rs_rank]++;
                                auto os_rank = std::lower_bound(sequence_all.begin(),
                                                                sequence_all.end(),
                                                                os_splitters_by_meas.back()[s]) -
                                               sequence_all.begin();
                                os_histogram[os_rank]++;
                            }
                        }
                    }
                }
                if (rank == 0) {
                    if (HISTOGRAM) {
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
                        LOG1 << "Write RS histogram";
                        for (size_t i = 0; i < rs_histogram.size(); i++) {
                            rs_file << i << "\t" << rs_histogram[i] << "\n";
                        }
                        rs_file.close();
                        std::ofstream os_file;
                        os_file.open("online_sampling_histogram.dat");
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
