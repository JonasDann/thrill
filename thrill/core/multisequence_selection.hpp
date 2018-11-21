/*******************************************************************************
 * thrill/core/multisequence_selection.hpp
 *
 * Distributed, external multisequence selection algorithm
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2018 Timo Bingmann <tb@panthema.net>
 * Copyright (C) 2018 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_MULTISEQUENCE_SELECTION_HEADER
#define THRILL_API_MULTISEQUENCE_SELECTION_HEADER

#include <thrill/api/context.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/common/stats_counter.hpp>

#include <tlx/math/abs_diff.hpp>

#include <vector>

namespace thrill {
namespace core {

template <typename ValueType_>
class MultisequenceSelectorFileSequenceAdapter
{
public:
    typedef ValueType_ ValueType;

    MultisequenceSelectorFileSequenceAdapter() = default;

    explicit MultisequenceSelectorFileSequenceAdapter(data::FilePtr& file)
        : file_(file)
    {}

    size_t size()
    {
        return file_->num_items();
    }

    ValueType operator [](size_t index) {
        return file_->template GetItemAt<ValueType>(index);
    }

    template<typename Comparator>
    size_t GetIndexOf(const ValueType& item, size_t tie, size_t left, size_t right, const Comparator& comparator)
    {
        return file_->GetIndexOf(item, tie, left, right, comparator);
    }

private:
    data::FilePtr file_;
};

template <typename ValueType_>
class MultisequenceSelectorVectorSequenceAdapter : public std::vector<ValueType_>
{
public:
    typedef ValueType_ ValueType;

    template<typename Comparator>
    size_t GetIndexOf(const ValueType& item, size_t tie, size_t left, size_t right, const Comparator& less)
    {
        static constexpr bool debug = false;

        static_assert(
                std::is_convertible<
                        bool, typename common::FunctionTraits<Comparator>::result_type
                >::value,
                "Comperator must return boolean.");

        LOG << "MultisequenceSelectorVectorSequenceAdapter::GetIndexOf() looking for item " << item << " tie " << tie
            << " in range [" << left << "," << right << ") ="
            << " size " << right - left;

        assert(left <= right);
        assert(left <= std::vector<ValueType>::size());
        assert(right <= std::vector<ValueType>::size());

        // Use a binary search to find the item.
        while (left < right) {
            size_t mid = (right + left) >> 1;
            LOG << "left: " << left << "right: " << right << "mid: " << mid;
            ValueType currentItem = std::vector<ValueType>::operator [](mid);
            LOG << "Item at mid: " << currentItem;
            if (less(item, currentItem) ||
                (!less(item, currentItem) && !less(currentItem, item) &&
                  tie <= mid)) {
                right = mid;
            }
            else {
                left = mid + 1;
            }
        }

        return left;
    }
};

template <typename SequenceAdapterType, typename Comparator, size_t kNumInputs>
class MultisequenceSelector
{
    using ValueType = typename SequenceAdapterType::ValueType;

    static constexpr bool debug = true;
    static constexpr bool self_verify = debug && common::g_debug_mode;

    //! Set this variable to true to enable generation and output of merge stats
    static constexpr bool stats_enabled = true;

    using ArrayNumInputsSizeT = std::array<size_t, kNumInputs>;

public:
    MultisequenceSelector(Context& context, const Comparator& comparator)
        : context_(context), comparator_(comparator)
    {}

    void GetEquallyDistantSplitterRanks(SequenceAdapterType sequences[kNumInputs],
                          std::vector<ArrayNumInputsSizeT>& out_local_ranks, size_t numSplitters)
    {
        // *** Setup Environment for merging ***

        // Count of all local elements.
        size_t local_size = 0;

        for (size_t i = 0; i < kNumInputs; i++) {
            local_size += sequences[i].size();
        }

        // TODO Make this work again
        // test that the data we got is sorted!
        /*if (self_verify) {
            for (size_t i = 0; i < kNumInputs; i++) {
                auto reader = files[i]->GetKeepReader();
                if (!reader.HasNext()) continue;

                ValueType prev = reader.template Next<ValueType>();
                while (reader.HasNext()) {
                    ValueType next = reader.template Next<ValueType>();
                    if (comparator_(next, prev)) {
                        die("Merge input was not sorted!");
                    }
                    prev = std::move(next);
                }
            }
        }*/

        // Count of all global elements.
        stats_.comm_timer_.Start();
        size_t global_size = context_.net.AllReduce(local_size);
        stats_.comm_timer_.Stop();

        LOG << "local size: " << local_size;
        LOG << "global size: " << global_size;

        // Calculate and remember the ranks we search for.  In our case, we
        // search for ranks that split the data into equal parts.
        std::vector<size_t> target_ranks(numSplitters);

        for (size_t r = 0; r < numSplitters; r++) {
            target_ranks[r] = (global_size / (numSplitters + 1)) * (r + 1);
            // Modify all ranks 0..(globalSize % p), in case global_size is not
            // divisible by p.
            if (r < global_size % (numSplitters + 1))
                target_ranks[r] += 1;
        }

        if (debug) {
            LOG << "target_ranks: " << target_ranks;

            stats_.comm_timer_.Start();
            assert(context_.net.Broadcast(target_ranks) == target_ranks);
            stats_.comm_timer_.Stop();
        }

        // buffer for the global ranks of selected pivots
        std::vector<size_t> global_ranks(numSplitters);

        // Search range bounds.
        std::vector<ArrayNumInputsSizeT> left(numSplitters), width(numSplitters);

        // Auxillary array.
        std::vector<Pivot> pivots(numSplitters);

        // Initialize all lefts with 0 and all widths with size of their
        // respective file.
        for (size_t r = 0; r < numSplitters; r++) {
            for (size_t q = 0; q < kNumInputs; q++) {
                width[r][q] = sequences[q].size();
            }
        }

        bool finished = false;
        stats_.balancing_timer_.Start();

        // Iterate until we find a pivot which is within the prescribed balance
        // tolerance
        while (!finished) {

            LOG << "iteration: " << stats_.iterations_;
            LOG0 << "left: " << left;
            LOG0 << "width: " << width;

            if (debug) {
                for (size_t q = 0; q < kNumInputs; q++) {
                    std::ostringstream oss;
                    for (size_t i = 0; i < numSplitters; ++i) {
                        if (i != 0) oss << " # ";
                        oss << '[' << left[i][q] << ',' << left[i][q] + width[i][q] << ')';
                    }
                    LOG1 << "left/right[" << q << "]: " << oss.str();
                }
            }

            // Find pivots.
            stats_.pivot_selection_timer_.Start();
            SelectPivots(sequences, left, width, pivots);
            stats_.pivot_selection_timer_.Stop();

            LOG << "final pivots: " << VToStr(pivots);

            // Get global ranks and shrink ranges.
            stats_.search_step_timer_.Start();
            GetGlobalRanks(sequences, pivots, global_ranks, out_local_ranks, left, width);

            LOG << "global_ranks: " << global_ranks;
            LOG << "local_ranks: " << out_local_ranks;

            SearchStep(global_ranks, out_local_ranks, target_ranks, left, width);

            if (debug) {
                for (size_t q = 0; q < kNumInputs; q++) {
                    std::ostringstream oss;
                    for (size_t i = 0; i < numSplitters; ++i) {
                        if (i != 0) oss << " # ";
                        oss << '[' << left[i][q] << ',' << left[i][q] + width[i][q] << ')';
                    }
                    LOG1 << "left/right[" << q << "]: " << oss.str();
                }
            }

            // We check for accuracy of kNumInputs + 1
            finished = true;
            for (size_t i = 0; i < numSplitters; i++) {
                size_t a = global_ranks[i], b = target_ranks[i];
                if (tlx::abs_diff(a, b) > kNumInputs + 5) {
                    finished = false;
                    break;
                }
            }

            stats_.search_step_timer_.Stop();
            stats_.iterations_++;
        }
        stats_.balancing_timer_.Stop();

        LOG << "Finished after " << stats_.iterations_ << " iterations";
    }

private:
    Context& context_;

    Comparator comparator_;

    struct Pivot {
        ValueType value;
        size_t    tie_idx;
        size_t    segment_len;
    };

    //! Logging helper to print vectors of vectors of pivots.
    static std::string VToStr(const std::vector<Pivot>& data) {
        std::stringstream oss;
        for (const Pivot& elem : data) {
            oss << "(" << elem.value
                << ", itie: " << elem.tie_idx
                << ", len: " << elem.segment_len << ") ";
        }
        return oss.str();
    }

    //! Reduce functor that returns the pivot originating from the biggest
    //! range.  That removes some nasty corner cases, like selecting the same
    //! pivot over and over again from a tiny range.
    class ReducePivots
    {
    public:
        Pivot operator () (const Pivot& a, const Pivot& b) const {
            return a.segment_len > b.segment_len ? a : b;
        }
    };

    using StatsTimer = common::StatsTimerBaseStopped<stats_enabled>;

    /*!
     * Stats holds timers for measuring multisequence selection performance, that supports
     * accumulating the output and printing it to the standard out stream.
     */
    class Stats
    {
    public:
        //! A Timer accumulating all time spent in file operations.
        StatsTimer file_op_timer_;
        //! A Timer accumulating all time spent while re-balancing the data.
        StatsTimer balancing_timer_;
        //! A Timer accumulating all time spent for selecting the global pivot
        //! elements.
        StatsTimer pivot_selection_timer_;
        //! A Timer accumulating all time spent in global search steps.
        StatsTimer search_step_timer_;
        //! A Timer accumulating all time spent communicating.
        StatsTimer comm_timer_;
        //! The count of search iterations needed for balancing.
        size_t iterations_ = 0;

        void PrintToSQLPlotTool(
                const std::string& label, size_t p, size_t value) {

            LOG1 << "RESULT " << "operation=" << label << " time=" << value
                 << " workers=" << p;
        }

        void Print(Context& ctx) {
            if (stats_enabled) {
                size_t p = ctx.num_workers();
                size_t balance =
                        ctx.net.AllReduce(balancing_timer_.Milliseconds()) / p;
                size_t pivot_selection =
                        ctx.net.AllReduce(pivot_selection_timer_.Milliseconds()) / p;
                size_t search_step =
                        ctx.net.AllReduce(search_step_timer_.Milliseconds()) / p;
                size_t file_op =
                        ctx.net.AllReduce(file_op_timer_.Milliseconds()) / p;
                size_t comm =
                        ctx.net.AllReduce(comm_timer_.Milliseconds()) / p;

                if (ctx.my_rank() == 0) {
                    PrintToSQLPlotTool("balance", p, balance);
                    PrintToSQLPlotTool("pivot_selection", p, pivot_selection);
                    PrintToSQLPlotTool("search_step", p, search_step);
                    PrintToSQLPlotTool("file_op", p, file_op);
                    PrintToSQLPlotTool("communication", p, comm);
                    PrintToSQLPlotTool("iterations", p, iterations_);
                }
            }
        }
    };

    //! Instance of merge statistics
    Stats stats_;

    /*!
     * Selects random global pivots for all splitter searches based on all
     * worker's search ranges.
     *
     * \param left The left bounds of all search ranges for all files.  The
     * first index identifies the splitter, the second index identifies the
     * file.
     *
     * \param width The width of all search ranges for all files.  The first
     * index identifies the splitter, the second index identifies the file.
     *
     * \param out_pivots The output pivots.
     */
    void SelectPivots(
            SequenceAdapterType sequences[kNumInputs],
            const std::vector<ArrayNumInputsSizeT>& left,
            const std::vector<ArrayNumInputsSizeT>& width,
            std::vector<Pivot>& out_pivots) {

        // Select a random pivot for the largest range we have for each
        // splitter.
        for (size_t s = 0; s < width.size(); s++) {
            size_t mp = 0;

            // Search for the largest range.
            for (size_t p = 1; p < width[s].size(); p++) {
                if (width[s][p] > width[s][mp]) {
                    mp = p;
                }
            }

            // We can leave pivot_elem uninitialized.  If it is not initialized
            // below, then an other worker's pivot will be taken for this range,
            // since our range is zero.
            ValueType pivot_elem = ValueType();
            size_t pivot_idx = left[s][mp];

            if (width[s][mp] > 0) {
                pivot_idx = left[s][mp] + (context_.rng_() % width[s][mp]);
                assert(pivot_idx < sequences[mp].size());
                stats_.file_op_timer_.Start();
                pivot_elem = sequences[mp][pivot_idx];
                stats_.file_op_timer_.Stop();
            }

            out_pivots[s] = Pivot {
                    pivot_elem,
                    pivot_idx,
                    width[s][mp]
            };
        }

        LOG << "local pivots: " << VToStr(out_pivots);

        // Reduce vectors of pivots globally to select the pivots from the
        // largest ranges.
        stats_.comm_timer_.Start();
        out_pivots = context_.net.AllReduce(
                out_pivots, common::ComponentSum<std::vector<Pivot>, ReducePivots>());
        stats_.comm_timer_.Stop();
    }

    /*!
     * Calculates the global ranks of the given pivots.
     * Additionally returns the local ranks so we can use them in the next step.
     */
    void GetGlobalRanks(
            SequenceAdapterType sequences[kNumInputs],
            const std::vector<Pivot>& pivots,
            std::vector<size_t>& global_ranks,
            std::vector<ArrayNumInputsSizeT>& out_local_ranks,
            const std::vector<ArrayNumInputsSizeT>& left,
            const std::vector<ArrayNumInputsSizeT>& width) {

        // Simply get the rank of each pivot in each file. Sum the ranks up
        // locally.
        for (size_t s = 0; s < pivots.size(); s++) {
            size_t rank = 0;
            for (size_t i = 0; i < kNumInputs; i++) {
                stats_.file_op_timer_.Start();

                size_t idx = sequences[i].template GetIndexOf<Comparator>(
                        pivots[s].value, pivots[s].tie_idx,
                        left[s][i], left[s][i] + width[s][i],
                        comparator_);

                stats_.file_op_timer_.Stop();

                rank += idx;
                out_local_ranks[s][i] = idx;
            }
            global_ranks[s] = rank;
        }

        stats_.comm_timer_.Start();
        // Sum up ranks globally.
        global_ranks = context_.net.AllReduce(
                global_ranks, common::ComponentSum<std::vector<size_t> >());
        stats_.comm_timer_.Stop();
    }

    /*!
     * Shrinks the search ranges according to the global ranks of the pivots.
     *
     * \param global_ranks The global ranks of all pivots.
     *
     * \param local_ranks The local ranks of each pivot in each file.
     *
     * \param target_ranks The desired ranks of the splitters we are looking
     * for.
     *
     * \param left The left bounds of all search ranges for all files.  The
     * first index identifies the splitter, the second index identifies the
     * file.  This parameter will be modified.
     *
     * \param width The width of all search ranges for all files.  The first
     * index identifies the splitter, the second index identifies the file.
     * This parameter will be modified.
     */
    void SearchStep(
            const std::vector<size_t>& global_ranks,
            const std::vector<ArrayNumInputsSizeT>& local_ranks,
            const std::vector<size_t>& target_ranks,
            std::vector<ArrayNumInputsSizeT>& left,
            std::vector<ArrayNumInputsSizeT>& width) {

        for (size_t s = 0; s < width.size(); s++) {
            for (size_t p = 0; p < width[s].size(); p++) {

                if (width[s][p] == 0)
                    continue;

                size_t local_rank = local_ranks[s][p];
                size_t old_width = width[s][p];
                assert(left[s][p] <= local_rank);

                if (global_ranks[s] < target_ranks[s]) {
                    width[s][p] -= local_rank - left[s][p];
                    left[s][p] = local_rank;
                }
                else if (global_ranks[s] >= target_ranks[s]) {
                    width[s][p] = local_rank - left[s][p];
                }

                if (debug) {
                    die_unless(width[s][p] <= old_width);
                }
            }
        }
    }
};

} // namespace core
} // namespace thrill

#endif // !THRILL_API_MULTISEQUENCE_SELECTION_HEADER

/******************************************************************************/