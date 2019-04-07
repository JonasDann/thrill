/*******************************************************************************
 * benchmarks/api/cms_sort.cpp
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
#include <thrill/api/canonical_merge_sort.hpp>
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

int main(int argc, char* argv[]) {

    tlx::CmdlineParser clp;

    int iterations;
    clp.add_param_int("i", iterations, "Iterations");

    uint64_t size;

    clp.add_param_bytes("size", size,
                        "Amount of data transfered between peers (example: 1 GiB).");

    if (!clp.process(argc, argv)) {
        return -1;
    }

    clp.print_result();

    api::Run(
            [&iterations, &size](api::Context& ctx) {
                for (int i = 0; i < iterations; i++) {
                    std::default_random_engine generator(std::random_device {} ());
                    std::uniform_int_distribution<uint64_t> distribution(0, std::numeric_limits<uint64_t>::max());

                    common::StatsTimerStart timer;
                    api::Generate(
                            ctx, size / sizeof(Record),
                            [&distribution, &generator](size_t) -> Record {
                                uint64_t key = distribution(generator);
                                Record r{key, key};
                                return r;
                            })
                            .CanonicalMergeSort().Size();
                    timer.Stop();
                    if (!ctx.my_rank()) {
                        LOG1 << "ITERATION " << i << " RESULT" << " time=" << timer;
                    }
                }
            });
}

/******************************************************************************/
