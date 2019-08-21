/*******************************************************************************
 * benchmarks/api/online_sample_sort.cpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2019 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <thrill/api/dia.hpp>
#include <thrill/api/generate.hpp>
#include <thrill/api/size.hpp>
#include <thrill/api/all_gather.hpp>
#include <thrill/api/online_sample_sort.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/common/stats_timer.hpp>
#include <tlx/cmdline_parser.hpp>

#include <limits>
#include <string>
#include <utility>

using namespace thrill; // NOLINT

// Same element structure as in original CMS paper
struct Record {
    uint64_t key;
    uint64_t value;

    bool operator < (const Record& b) const {
        return key < b.key;
    }

    friend std ::ostream& operator << (std::ostream& os, const Record& r) {
        return os << r.key << r.value;
    }
} TLX_ATTRIBUTE_PACKED;

constexpr bool self_verify = false;

using RandomGenerator = std::mt19937_64;

uint64_t generator(size_t i, size_t n, const std::string& generator_type,
               RandomGenerator rng) {
    std::uniform_int_distribution<int> uni;
    std::exponential_distribution<double> expo(2.0);
    if (generator_type == "uni") {
        return uni(rng);
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
        return static_cast<uint64_t>(std::fmod((pow(i, 2) + static_cast<double>(n) / 2), n));
    }
    return -1;
}

int main(int argc, char* argv[]) {

    tlx::CmdlineParser clp;

    int iterations;
    clp.add_param_int("i", iterations, "Iterations");

    uint64_t size;

    clp.add_param_bytes("size", size,
                        "Amount of data transfered between peers (example: 1 GiB).");

    std::string generator_type;
    clp.add_param_string("generator", generator_type,
                         "Type of generator (uni, sort, ones, almost, window, dup).");

    if (!clp.process(argc, argv)) {
        return -1;
    }

    clp.print_result();

    api::Run(
            [&iterations, &size, &generator_type](api::Context& ctx) {
                for (int i = 0; i < iterations; i++) {
                    std::random_device rd;
                    RandomGenerator rng(rd());

                    size_t element_count = size / sizeof(Record);
                    common::StatsTimerStart timer;
                    auto sorted = api::Generate(
                            ctx, element_count,
                            [&ctx, &element_count, &generator_type, &rng](size_t i) -> Record {
                                auto global_idx = i * ctx.num_workers() + ctx.my_rank();
                                uint64_t key = generator(global_idx, element_count, generator_type, rng);
                                Record r{key, key};
                                return r;
                            })
                            .OnlineSampleSort();
                    if (self_verify) {
                        auto result = sorted.AllGather();
                        die_unless(result.size() == element_count);
                        for (size_t j = 1; j < result.size(); j++) {
                            die_unless(result[i - 1] < result[i]);
                        }
                    } else {
                        die_unless(sorted.Size() == element_count);
                    }
                    timer.Stop();
                    if (!ctx.my_rank()) {
                        LOG1 << "ITERATION " << i << " RESULT" << " time=" << timer;
                    }
                }
            });
}

/******************************************************************************/
