/*******************************************************************************
 * examples/canonical_merge_sort/cms_sort.cpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2018 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <thrill/api/generate.hpp>
#include <thrill/api/read_binary.hpp>
#include <thrill/api/size.hpp>
#include <thrill/api/cm_sort.hpp>
#include <thrill/api/write_binary.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/common/string.hpp>
#include <tlx/cmdline_parser.hpp>

#include <tlx/string/hexdump.hpp>
#include <tlx/string/parse_si_iec_units.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace thrill;               // NOLINT

// TODO Build minimal example for cms sort

struct Record {
    uint8_t key[10];
    uint8_t value[90];

    bool operator < (const Record& b) const {
        return std::lexicographical_compare(key, key + 10, b.key, b.key + 10);
    }
    friend std ::ostream& operator << (std::ostream& os, const Record& c) {
        return os << tlx::hexdump(c.key, 10);
    }
} TLX_ATTRIBUTE_PACKED;

static_assert(sizeof(Record) == 100, "struct Record packing incorrect.");

class GenerateRecord
{
public:
    Record operator () (size_t index) {
        (void)index;
        Record r;

        for (size_t i = 0; i < 10; ++i) {
            r.key[i] = static_cast<uint8_t>(rng_());
        }

        return r;
    }

private:
    std::default_random_engine rng_ { std::random_device { } () };
};

int main(int argc, char* argv[]) {

    tlx::CmdlineParser clp;

    bool generate = false;
    clp.add_bool('g', "generate", generate,
                 "generate binary record on-the-fly for testing."
                 " size: first input pattern, default: false");

    bool generate_only = false;
    clp.add_bool('G', "generate-only", generate_only,
                 "write unsorted generated binary records to output.");

    std::string output;
    clp.add_string('o', "output", output,
                   "output file pattern");

    std::vector<std::string> input;
    clp.add_param_stringlist("input", input,
                             "input file pattern(s)");

    if (!clp.process(argc, argv)) {
        return -1;
    }

    clp.print_result();

    return api::Run(
        [&](api::Context& ctx) {
            ctx.enable_consume();

            common::StatsTimerStart timer;

            if (generate_only || generate) {
                die_unequal(input.size(), 1u);
                // parse first argument like "100mib" size
                uint64_t size;
                die_unless(tlx::parse_si_iec_units(input[0].c_str(), &size));

                if (generate_only) {
                    Generate(ctx, size / sizeof(Record), GenerateRecord())
                    .WriteBinary(output);
                } else {
                    auto r =
                        Generate(ctx, size / sizeof(Record), GenerateRecord())
                        .CanonicalMergeSort();

                    if (output.size())
                        r.WriteBinary(output);
                    else
                        r.Size();
                }
            }
            else {
                auto r = ReadBinary<Record>(ctx, input).CanonicalMergeSort();

                if (output.size())
                    r.WriteBinary(output);
                else
                    r.Size();
            }

            ctx.net.Barrier();
            if (ctx.my_rank() == 0) {
                auto traffic = ctx.net_manager().Traffic();
                LOG1 << "RESULT"
                     << " example=canonical_merge_sort"
                     << " time=" << timer
                     << " traffic=" << traffic.total()
                     << " hosts=" << ctx.num_hosts();
            }
        });
}

/******************************************************************************/
