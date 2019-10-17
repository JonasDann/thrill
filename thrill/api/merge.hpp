/*******************************************************************************
 * thrill/api/merge.hpp
 *
 * DIANode for a merge operation. Performs the actual merge operation
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2015-2016 Timo Bingmann <tb@panthema.net>
 * Copyright (C) 2015 Emanuel Jöbstl <emanuel.joebstl@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_MERGE_HEADER
#define THRILL_API_MERGE_HEADER

#include <thrill/api/dia.hpp>
#include <thrill/api/dop_node.hpp>
#include <thrill/common/functional.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/common/stats_counter.hpp>
#include <thrill/common/stats_timer.hpp>
#include <thrill/common/string.hpp>
#include <thrill/core/multiway_merge.hpp>
#include <thrill/core/multi_sequence_selection.hpp>
#include <thrill/data/dyn_block_reader.hpp>
#include <thrill/data/file.hpp>

#include <tlx/math/abs_diff.hpp>
#include <tlx/meta/call_foreach_with_index.hpp>
#include <tlx/meta/vexpand.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace thrill {
namespace api {

/*!
 * Implementation of Thrill's merge. This merge implementation balances all data
 * before merging, so each worker has the same amount of data when merge
 * finishes.
 *
 * The algorithm performs a distributed multi-sequence selection by picking
 * random pivots (from the largest remaining interval) for each DIA. The pivots
 * are selected via a global AllReduce. There is one pivot per DIA.
 *
 * Then the pivots are searched for in the interval [left,left + width) in each
 * local File's partition, where these are initialized with left = 0 and width =
 * File.size(). This delivers the local_rank of each pivot. From the local_ranks
 * the corresponding global_ranks of each pivot is calculated via a AllReduce.
 *
 * The global_ranks are then compared to the target_ranks (which are n/p *
 * rank). If global_ranks is smaller, the interval [left,left + width) is
 * reduced to [left,idx), where idx is the rank of the pivot in the local
 * File. If global_ranks is larger, the interval is reduced to [idx,left+width).
 *
 * left  -> width
 * V            V      V           V         V                   V
 * +------------+      +-----------+         +-------------------+ DIA 0
 *    ^
 *    local_ranks,  global_ranks = sum over all local_ranks
 *
 * \tparam ValueType The type of the first and second input DIA
 * \tparam Comparator The comparator defining input and output order.
 * \tparam ParentDIA0 The type of the first input DIA
 * \tparam ParentDIAs The types of the other input DIAs
 *
 * \ingroup api_layer
 */
template <typename ValueType, typename Comparator, size_t kNumInputs>
class MergeNode : public DOpNode<ValueType>
{
    static constexpr bool debug = false;
    static constexpr bool self_verify = debug && common::g_debug_mode;

    //! Set this variable to true to enable generation and output of merge stats
    static constexpr bool stats_enabled = false;

    using Super = DOpNode<ValueType>;
    using Super::context_;

    static_assert(kNumInputs >= 2, "Merge requires at least two inputs.");

public:
    template <typename ParentDIA0, typename... ParentDIAs>
    MergeNode(const Comparator& comparator,
              const ParentDIA0& parent0, const ParentDIAs& ... parents)
        : Super(parent0.ctx(), "Merge",
                { parent0.id(), parents.id() ... },
                { parent0.node(), parents.node() ... }),
          comparator_(comparator),
          // this weirdness is due to a MSVC2015 parser bug
          parent_stack_empty_(
              std::array<bool, kNumInputs>{
                  { ParentDIA0::stack_empty, (ParentDIAs::stack_empty)... }
              }) {
        // allocate files.
        for (size_t i = 0; i < kNumInputs; ++i)
            files_[i] = context_.GetFilePtr(this);

        for (size_t i = 0; i < kNumInputs; ++i)
            writers_[i] = files_[i]->GetWriter();

        tlx::call_foreach_with_index(
            RegisterParent(this), parent0, parents...);
    }

    //! Register Parent PreOp Hooks, instantiated and called for each Merge
    //! parent
    class RegisterParent
    {
    public:
        explicit RegisterParent(MergeNode* merge_node)
            : merge_node_(merge_node) { }

        template <typename Index, typename Parent>
        void operator () (const Index&, Parent& parent) {

            // construct lambda with only the writer in the closure
            data::File::Writer* writer = &merge_node_->writers_[Index::index];
            auto pre_op_fn = [writer](const ValueType& input) -> void {
                                 writer->Put(input);
                             };

            // close the function stacks with our pre ops and register it at
            // parent nodes for output
            auto lop_chain = parent.stack().push(pre_op_fn).fold();

            parent.node()->AddChild(merge_node_, lop_chain, Index::index);
        }

    private:
        MergeNode* merge_node_;
    };

    //! Receive a whole data::File of ValueType, but only if our stack is empty.
    bool OnPreOpFile(const data::File& file, size_t parent_index) final {
        assert(parent_index < kNumInputs);
        if (!parent_stack_empty_[parent_index]) return false;

        // accept file
        assert(files_[parent_index]->num_items() == 0);
        *files_[parent_index] = file.Copy();
        return true;
    }

    void StopPreOp(size_t parent_index) final {
        writers_[parent_index].Close();
    }

    void Execute() final {
        MainOp();
    }

    void PushData(bool consume) final {
        size_t result_count = 0;
        static constexpr bool debug = false;

        stats_.merge_timer_.Start();

        // get inbound readers from all Channels
        std::vector<data::CatStream::CatReader> readers;
        readers.reserve(kNumInputs);

        for (size_t i = 0; i < kNumInputs; i++)
            readers.emplace_back(streams_[i]->GetCatReader(consume));

        auto puller = core::make_multiway_merge_tree<ValueType>(
            readers.begin(), readers.end(), comparator_);

        while (puller.HasNext())
            this->PushItem(puller.Next());

        stats_.merge_timer_.Stop();

        sLOG << "Merge: result_count" << result_count;

        stats_.result_size_ = result_count;
        stats_.Print(context_);
    }

    void Dispose() final { }

private:
    //! Merge comparator
    Comparator comparator_;

    using FileSequenceAdapter = core::MultisequenceSelectorFileSequenceAdapter<ValueType>;

    using LocalRanks = std::vector<std::vector<size_t>>;

    //! Whether the parent stack is empty
    const std::array<bool, kNumInputs> parent_stack_empty_;

    //! Files for intermediate storage
    data::FilePtr files_[kNumInputs];

    //! Writers to intermediate files
    data::File::Writer writers_[kNumInputs];

    //! Array of inbound CatStreams
    data::CatStreamPtr streams_[kNumInputs];

    using StatsTimer = common::StatsTimerBaseStopped<stats_enabled>;

    /*!
     * Stats holds timers for measuring merge performance, that supports
     * accumulating the output and printing it to the standard out stream.
     */
    class Stats
    {
    public:
        //! A Timer accumulating all time spent while actually merging.
        StatsTimer merge_timer_;
        //! A Timer accumulating all time spent calling the scatter method of
        //! the data subsystem.
        StatsTimer scatter_timer_;
        //! The count of all elements processed on this host.
        size_t result_size_ = 0;

        void PrintToSQLPlotTool(
            const std::string& label, size_t p, size_t value) {

            LOG1 << "RESULT " << "operation=" << label << " time=" << value
                 << " workers=" << p << " result_size_=" << result_size_;
        }

        void Print(Context& ctx) {
            if (stats_enabled) {
                size_t p = ctx.num_workers();
                size_t merge =
                    ctx.net.AllReduce(merge_timer_.Milliseconds()) / p;
                size_t scatter =
                    ctx.net.AllReduce(scatter_timer_.Milliseconds()) / p;

                result_size_ = ctx.net.AllReduce(result_size_);

                if (ctx.my_rank() == 0) {
                    PrintToSQLPlotTool("merge", p, merge);
                    PrintToSQLPlotTool("scatter", p, scatter);
                }
            }
        }
    };

    //! Instance of merge statistics
    Stats stats_;

    /*!
     * Receives elements from other workers and re-balance them, so each worker
     * has the same amount after merging.
     */
    void MainOp() {
        // Count of all workers (and count of target partitions)
        size_t p = context_.num_workers();
        LOG << "splitting to " << p << " workers";

        LocalRanks local_ranks(p - 1, std::vector<size_t>(kNumInputs));

        std::vector<FileSequenceAdapter> sequences(kNumInputs);
        for (size_t i = 0; i < kNumInputs; i++)
            sequences[i] = FileSequenceAdapter(files_[i]);

        core::run_multi_sequence_selection<FileSequenceAdapter, Comparator>
                (context_, comparator_, sequences, local_ranks, p - 1);

        LOG << "Creating channels";

        // Initialize channels for distributing data.
        for (size_t j = 0; j < kNumInputs; j++)
            streams_[j] = context_.GetNewCatStream(this);

        stats_.scatter_timer_.Start();

        LOG << "Scattering.";

        // For each file, initialize an array of offsets according to the
        // splitters we found. Then call Scatter to distribute the data.

        std::vector<size_t> tx_items(p);
        for (size_t j = 0; j < kNumInputs; j++) {

            std::vector<size_t> offsets(p + 1, 0);

            for (size_t r = 0; r < p - 1; r++)
                offsets[r + 1] = local_ranks[r][j];

            offsets[p] = files_[j]->num_items();

            LOG << "Scatter from file " << j << " to other workers: "
                << offsets;

            for (size_t r = 0; r < p; ++r) {
                tx_items[r] += offsets[r + 1] - offsets[r];
            }

            streams_[j]->template ScatterConsume<ValueType>(
                *files_[j], offsets);
        }

        LOG << "tx_items: " << tx_items;

        // calculate total items on each worker after Scatter
        tx_items = context_.net.AllReduce(
            tx_items, common::ComponentSum<std::vector<size_t> >());
        if (context_.my_rank() == 0)
            LOG1 << "Merge(): total_items: " << tx_items;

        stats_.scatter_timer_.Stop();
    }
};

/*!
 * Merge is a DOp, which merges any number of sorted DIAs to a single sorted
 * DIA.  All input DIAs must be sorted conforming to the given comparator.  The
 * type of the output DIA will be the type of this DIA.
 *
 * The merge operation balances all input data, so that each worker will have an
 * equal number of elements when the merge completes.
 *
 * \tparam Comparator Comparator to specify the order of input and output.
 *
 * \param comparator Comparator to specify the order of input and output.
 *
 * \param first_dia first DIA
 * \param dias DIAs, which is merged with this DIA.
 *
 * \ingroup dia_dops
 */
template <typename Comparator, typename FirstDIA, typename... DIAs>
auto Merge(const Comparator& comparator,
           const FirstDIA& first_dia, const DIAs& ... dias) {

    tlx::vexpand((first_dia.AssertValid(), 0), (dias.AssertValid(), 0) ...);

    using ValueType = typename FirstDIA::ValueType;

    using CompareResult =
        typename common::FunctionTraits<Comparator>::result_type;

    using MergeNode = api::MergeNode<
        ValueType, Comparator, 1 + sizeof ... (DIAs)>;

    // Assert comparator types.
    static_assert(
        std::is_convertible<
            ValueType,
            typename common::FunctionTraits<Comparator>::template arg<0>
            >::value,
        "Comparator has the wrong input type in argument 0");

    static_assert(
        std::is_convertible<
            ValueType,
            typename common::FunctionTraits<Comparator>::template arg<1>
            >::value,
        "Comparator has the wrong input type in argument 1");

    // Assert meaningful return type of comperator.
    static_assert(
        std::is_convertible<
            CompareResult,
            bool
            >::value,
        "Comparator must return bool");

    auto merge_node =
        tlx::make_counting<MergeNode>(comparator, first_dia, dias...);

    return DIA<ValueType>(merge_node);
}

template <typename ValueType, typename Stack>
template <typename Comparator, typename SecondDIA>
auto DIA<ValueType, Stack>::Merge(
    const SecondDIA& second_dia, const Comparator& comparator) const {
    return api::Merge(comparator, *this, second_dia);
}

} // namespace api

//! imported from api namespace
using api::Merge;

} // namespace thrill

#endif // !THRILL_API_MERGE_HEADER

/******************************************************************************/
