/*******************************************************************************
 * thrill/core/multi_sequence_selection.hpp
 *
 * Distributed, external multi sequence selection algorithm
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2018 Timo Bingmann <tb@panthema.net>
 * Copyright (C) 2018 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_MULTI_SEQUENCE_SELECTION_HEADER
#define THRILL_API_MULTI_SEQUENCE_SELECTION_HEADER

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
    size_t GetIndexOf(const ValueType& item, size_t tie,
                      size_t left, size_t right, const Comparator& comparator)
    {
        return file_->GetIndexOf(item, tie, left, right, comparator);
    }

private:
    data::FilePtr file_;
};

template <typename ValueType_>
class MultiSequenceSelectorSampledFileSequenceAdapter
{
public:
    typedef ValueType_ ValueType;

    MultiSequenceSelectorSampledFileSequenceAdapter() = default;

    explicit MultiSequenceSelectorSampledFileSequenceAdapter(
            data::SampledFilePtr<ValueType>& file)
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
    size_t GetIndexOf(const ValueType& item, size_t tie,
                      size_t left, size_t right, const Comparator& comparator)
    {
        (void) left; (void) right;
        return file_->GetFastIndexOf(item, tie, comparator);
    }

    size_t ItemsStartIn(size_t i) {
        return file_->ItemsStartIn(i);
    }

    std::deque<ValueType_> block_samples() {
        return file_->block_samples();
    }

    std::deque<size_t> num_items_sum() {
        return file_->num_items_sum();
    }

    size_t num_blocks() {
        return file_->num_blocks();
    }

private:
    data::SampledFilePtr<ValueType> file_;
};

template <typename ValueType_>
class MultiSequenceSelectorVectorSequenceAdapter : public std::vector<ValueType_>
{
public:
    typedef ValueType_ ValueType;

    template<typename Comparator>
    size_t GetIndexOf(const ValueType& item, size_t tie, size_t left,
            size_t right, const Comparator& less)
    {
        static constexpr bool debug = false;

        static_assert(
                std::is_convertible<
                        bool, typename common::FunctionTraits<Comparator>::result_type
                >::value,
                "Comparator must return boolean.");

        LOG << "MultiSequenceSelectorVectorSequenceAdapter::GetIndexOf()"
            << " looking for item " << tlx::wrap_unp(item) << " tie " << tie
            << " in range [" << left << "," << right << ") ="
            << " size " << right - left;

        assert(left <= right);
        assert(left <= std::vector<ValueType>::size());
        assert(right <= std::vector<ValueType>::size());

        // Use a binary search to find the item.
        while (left < right) {
            size_t mid = (right + left) >> static_cast<size_t>(1);
            LOG << "left: " << left << "right: " << right << "mid: " << mid;
            ValueType currentItem = std::vector<ValueType>::operator [](mid);
            LOG << "Item at mid: " << tlx::wrap_unp(currentItem);
            if (less(item, currentItem) ||
                (!less(item, currentItem) && !less(currentItem, item) &&
                  tie <= mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }
};

template <typename SequenceAdapterType, typename Comparator>
class MultiSequenceSelector
{
    using ValueType = typename SequenceAdapterType::ValueType;
    using SequenceAdapters = typename std::vector<SequenceAdapterType>;

    static constexpr bool debug = false;
    static constexpr bool self_verify = debug && common::g_debug_mode;

    //! Set this variable to true to enable generation and output of selection
    //! stats
    static constexpr bool stats_enabled = true;

public:
    MultiSequenceSelector(Context& context, const Comparator& comparator)
        : context_(context), comparator_(comparator)
    {}

    virtual void GetEquallyDistantSplitterRanks(SequenceAdapters& sequences,
                          std::vector<std::vector<size_t>>& out_local_ranks,
                          size_t splitter_count)
    {
        auto seq_count = sequences.size();

        std::vector<size_t> target_ranks(splitter_count);
        GetTargetRanks(sequences, splitter_count, target_ranks);

        // Search range bounds.
        std::vector<std::vector<size_t>>
                left(splitter_count, std::vector<size_t>(seq_count)),
                width(splitter_count, std::vector<size_t>(seq_count));

        // Initialize all lefts with 0 and all widths with size of their
        // respective file.
        for (size_t r = 0; r < splitter_count; r++) {
            for (size_t q = 0; q < sequences.size(); q++) {
                width[r][q] = sequences[q].size();
            }
        }

        GetSplitterRanks(sequences, out_local_ranks, target_ranks, left, width);
    }

protected:
    Context& context_;

    Comparator comparator_;

    struct Pivot {
        ValueType value;
        size_t    tie_idx;
        size_t    segment_len;
        size_t    worker_rank;
        size_t    sequence_idx;
    };

    //! Logging helper to print vectors of vectors of pivots.
    static std::string VToStr(const std::vector<Pivot>& data) {
        std::stringstream oss;
        for (const Pivot& elem : data) {
            oss << "(" << tlx::wrap_unp(elem.value)
                << ", tie_idx: " << elem.tie_idx
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

    bool EqualPivot(const Pivot& a, const Pivot& b) {
        auto equal = !comparator_(a.value, b.value) &&
                     !comparator_(b.value, a.value);
        return equal && a.tie_idx == b.tie_idx;
    }

    using StatsTimer = common::StatsTimerBaseStopped<stats_enabled>;

    /*!
     * Stats holds timers for measuring multi sequence selection performance,
     * that supports accumulating the output and printing it to the standard
     * out stream.
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

    //! Instance of selection statistics
    Stats stats_;

    void GetSplitterRanks(SequenceAdapters& sequences,
            std::vector<std::vector<size_t>>& out_local_ranks,
            std::vector<size_t>& target_ranks,
            std::vector<std::vector<size_t>>& left,
            std::vector<std::vector<size_t>>& width)
    {
        auto seq_count = sequences.size();
        auto splitter_count = target_ranks.size();

        // buffer for the global ranks of selected pivots
        std::vector<size_t> global_ranks(splitter_count);

        // Auxiliary array.
        std::vector<Pivot> pivots(splitter_count);

        bool finished = false;
        stats_.balancing_timer_.Start();

        if (debug) {
            for (size_t s = 0; s < seq_count; s++) {
                std::ostringstream oss;
                for (size_t r = 0; r < splitter_count; ++r) {
                    if (r != 0) oss << " # ";
                    oss << '[' << left[r][s] << ',' << left[r][s] + width[r][s] << ')';
                }
                LOG1 << "initial left/right[" << s << "]: " << oss.str();
            }
        }

        // Iterate until we find a pivot which is within the prescribed balance
        // tolerance
        while (!finished) {

            LOG << "iteration: " << stats_.iterations_;
            LOG0 << "left: " << left;
            LOG0 << "width: " << width;

            // Find pivots.
            stats_.pivot_selection_timer_.Start();
            SelectPivots(sequences, left, width, pivots);
            stats_.pivot_selection_timer_.Stop();

            LOG << "final pivots: " << VToStr(pivots);

            size_t pivots_len_sum = 0;
            for (auto pivot : pivots) {
                pivots_len_sum += pivot.segment_len;
            }

            // Get global ranks and shrink ranges.
            stats_.search_step_timer_.Start();
            GetGlobalRanks(sequences, pivots, global_ranks, out_local_ranks, left, width);

            LOG << "global_ranks: " << global_ranks;
            LOG << "local_ranks: " << out_local_ranks;

            SearchStep(pivots, global_ranks, out_local_ranks, target_ranks, left, width);

            if (debug) {
                for (size_t s = 0; s < seq_count; s++) {
                    std::ostringstream oss;
                    for (size_t r = 0; r < splitter_count; ++r) {
                        if (r != 0) oss << " # ";
                        oss << '[' << left[r][s] << ',' << left[r][s] + width[r][s] << ')';
                    }
                    LOG1 << "left/right[" << s << "]: " << oss.str();
                }
            }

            finished = true;
            for (size_t r = 0; r < splitter_count; r++) {
                size_t a = global_ranks[r], b = target_ranks[r];
                if (tlx::abs_diff(a, b) > 0) {
                    finished = false;
                    break;
                }
            }
            if (pivots_len_sum == 0) {
                finished = true;
            }

            stats_.search_step_timer_.Stop();
            stats_.iterations_++;
        }
        stats_.balancing_timer_.Stop();

        LOG << "Finished after " << stats_.iterations_ << " iterations";
        stats_.Print(context_);
    }

    void GetTargetRanks(SequenceAdapters& sequences,
            const size_t splitter_count,
            std::vector<size_t>& out_target_ranks) {
        auto seq_count = sequences.size();

        // Count of all local elements.
        size_t local_size = 0;

        for (size_t s = 0; s < sequences.size(); s++) {
            local_size += sequences[s].size();
        }

        // Test that the data we got is sorted.
        if (self_verify) {
            for (size_t s = 0; s < seq_count; s++) {
                for (size_t n = 1; n < sequences[s].size(); n++) {
                    //if (comparator_(sequences[s][n], sequences[s][n - 1])) {
                    //    die("Input was not sorted!");
                    //}
                }
            }
        }

        // Count of all global elements.
        stats_.comm_timer_.Start();
        size_t global_size = context_.net.AllReduce(local_size);
        stats_.comm_timer_.Stop();

        LOG << "local size: " << local_size;
        LOG << "global size: " << global_size;

        // Calculate and remember the ranks we search for.  In our case, we
        // search for ranks that split the data into equal parts.
        for (size_t r = 0; r < splitter_count; r++) {
            out_target_ranks[r] = (global_size / (splitter_count + 1)) * (r + 1);
            // Modify all ranks 0..(globalSize % p), in case global_size is not
            // divisible by p.
            if (r < global_size % (splitter_count + 1))
                out_target_ranks[r] += 1;
        }

        if (debug) {
            LOG << "target_ranks: " << out_target_ranks;

            stats_.comm_timer_.Start();
            assert(context_.net.Broadcast(out_target_ranks) == out_target_ranks);
            stats_.comm_timer_.Stop();
        }
    }

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
            SequenceAdapters& sequences,
            const std::vector<std::vector<size_t>>& left,
            const std::vector<std::vector<size_t>>& width,
            std::vector<Pivot>& out_pivots) {

        // Select a random pivot for the largest range we have for each
        // splitter.
        for (size_t r = 0; r < width.size(); r++) {
            size_t ms = 0;

            // Search for the largest range.
            for (size_t s = 1; s < width[r].size(); s++) {
                if (width[r][s] > width[r][ms]) {
                    ms = s;
                }
            }

            // We can leave pivot_elem uninitialized.  If it is not initialized
            // below, then an other worker's pivot will be taken for this range,
            // since our range is zero.
            ValueType pivot_elem = ValueType();
            size_t pivot_idx = left[r][ms];

            if (width[r][ms] > 0) {
                pivot_idx = left[r][ms] + (width[r][ms] / 2);
                assert(pivot_idx < sequences[ms].size());
                stats_.file_op_timer_.Start();
                pivot_elem = sequences[ms][pivot_idx];
                stats_.file_op_timer_.Stop();
            }

            out_pivots[r] = Pivot {
                    pivot_elem,
                    pivot_idx,
                    width[r][ms],
                    context_.my_rank(),
                    ms
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
            SequenceAdapters& sequences,
            std::vector<Pivot>& pivots,
            std::vector<size_t>& global_ranks,
            std::vector<std::vector<size_t>>& out_local_ranks,
            std::vector<std::vector<size_t>>& left,
            std::vector<std::vector<size_t>>& width) {

        // Simply get the rank of each pivot in each file. Sum the ranks up
        // locally.
        std::vector<Pivot*> last_pivot = std::vector<Pivot*>(sequences.size(), nullptr);
        std::vector<size_t> last_idx = std::vector<size_t>(sequences.size(), 0);
        for (size_t r = 0; r < pivots.size(); r++) {
            size_t rank = 0;
            for (size_t s = 0; s < sequences.size(); s++) {
                stats_.file_op_timer_.Start();

                size_t idx = left[r][s];
                if (width[r][s] > 0) {
                    if (last_pivot[s] && EqualPivot(*last_pivot[s], pivots[r])) {
                        idx = last_idx[s];
                    } else {
                        idx = sequences[s].template GetIndexOf<Comparator>(
                                pivots[r].value, pivots[r].tie_idx,
                                left[r][s], left[r][s] + width[r][s],
                                comparator_);
                        last_pivot[s] = &pivots[r];
                        last_idx[s] = idx;
                    }

                    if (idx > left[r][s] + width[r][s]) {
                        width[r][s] = idx - left[r][s];
                        if (left[r][s] + width[r][s] < sequences[s].size()) {
                            width[r][s]++;
                        }
                    }
                    if (idx < left[r][s]) {
                        width[r][s] += left[r][s] - idx;
                        left[r][s] = idx;
                    }
                }

                stats_.file_op_timer_.Stop();

                rank += idx;
                out_local_ranks[r][s] = idx;
            }
            global_ranks[r] = rank;
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
            const std::vector<Pivot>& pivots,
            const std::vector<size_t>& global_ranks,
            const std::vector<std::vector<size_t>>& local_ranks,
            const std::vector<size_t>& target_ranks,
            std::vector<std::vector<size_t>>& left,
            std::vector<std::vector<size_t>>& width) {

        for (size_t r = 0; r < width.size(); r++) {
            for (size_t s = 0; s < width[r].size(); s++) {
                size_t local_rank = local_ranks[r][s];
                size_t old_width = width[r][s];

                if (width[r][s] == 0)
                    continue;

                if (target_ranks[r] > global_ranks[r]) {
                    // +1 binary search only on worker that pivot is from and
                    // only on sequence that pivot is from
                    if (pivots[r].worker_rank == context_.my_rank() &&
                        pivots[r].sequence_idx == s) {
                        LOG << "[" << stats_.iterations_
                            << "] increment (split: " << r << " seq: " << s
                            << " pivot: " << tlx::wrap_unp(pivots[r].value)
                            << ").";
                        local_rank++;
                    }
                    width[r][s] -= local_rank - left[r][s];
                    left[r][s] = local_rank;
                }
                else if (target_ranks[r] < global_ranks[r]) {
                    width[r][s] = local_rank - left[r][s];
                } else {
                    left[r][s] = local_rank;
                    width[r][s] = 0;
                }

                if (debug) {
                    die_unless(width[r][s] <= old_width);
                }
            }
        }
    }
};

template <typename ValueType, typename Comparator, typename SortAlgorithm>
class MultiSequenceSelectorSampled : public MultiSequenceSelector<MultiSequenceSelectorSampledFileSequenceAdapter<ValueType>, Comparator> {
    using Sequences = typename std::vector<MultiSequenceSelectorSampledFileSequenceAdapter<ValueType>>;
    using Samples = typename std::vector<std::deque<ValueType>>;
    using Pivot = typename MultiSequenceSelector<MultiSequenceSelectorSampledFileSequenceAdapter<ValueType>, Comparator>::Pivot;
    using ReducePivots = typename MultiSequenceSelector<MultiSequenceSelectorSampledFileSequenceAdapter<ValueType>, Comparator>::ReducePivots;

    static constexpr bool debug = false;

public:
    MultiSequenceSelectorSampled(Context& context, size_t dia_id,
                                 const Comparator& comparator,
                                 const SortAlgorithm& sort_algorithm)
         : MultiSequenceSelector<MultiSequenceSelectorSampledFileSequenceAdapter<ValueType>, Comparator>(context, comparator),
                 dia_id_(dia_id), sort_algorithm_(sort_algorithm) {

    }

    void GetEquallyDistantSplitterRanks(Sequences &sequences,
                                        std::vector<std::vector<size_t>> &out_local_ranks,
                                        size_t splitter_count) {
        auto seq_count = sequences.size();

        std::vector<size_t> target_ranks(splitter_count);
        this->GetTargetRanks(sequences, splitter_count, target_ranks);

        // Search range bounds.
        std::vector<std::vector<size_t>>
                left(splitter_count, std::vector<size_t>(seq_count)),
                width(splitter_count, std::vector<size_t>(seq_count));

        // Initialize all lefts with 0 and widths with 1 if sequence.size() > 0.
        for (size_t r = 0; r < splitter_count; r++) {
            for (size_t q = 0; q < sequences.size(); q++) {
                if (sequences[q].size() > 0) {
                    width[r][q] = 1;
                }
            }
        }

        auto stream = this->context_.template GetNewStream<data::CatStream>(
                this->dia_id_);
        auto writers = stream->GetWriters();
        auto readers = stream->GetReaders();

        std::vector<BlockSample> samples;
        for (size_t s = 0; s < seq_count; s++) {
            auto num_blocks = sequences[s].num_blocks();
            auto block_samples = sequences[s].block_samples();

            std::vector<BlockSample> sequence_samples;
            sequence_samples.reserve(num_blocks);

            for (size_t i = 0; i < num_blocks; i++) {
                samples.push_back(
                        BlockSample {
                            block_samples[i],
                            i,
                            sequences[s].ItemsStartIn(i),
                            this->context_.my_rank(),
                            s
                        });
            }
        }

        auto P = this->context_.num_workers();

        if (this->context_.my_rank() != 0) {
            writers[0].Put(samples);
            writers[0].Close();
        }

        if (this->context_.my_rank() == 0) {
            for (size_t p = 1; p < P; p++) {
                auto foreign_samples = std::move(
                        readers[p].template Next<std::vector<BlockSample>>());

                samples.insert(samples.end(), foreign_samples.begin(),
                        foreign_samples.end());
            }

            // TODO replace with merges
            sort_algorithm_(samples.begin(), samples.end(),
                    [this](BlockSample a, BlockSample b)
                    {
                        return this->comparator_(a.value, b.value);
                    });

            for (size_t i = 1; i < samples.size(); i++) {
                samples[i + 1].size += samples[i].size;
            }
            for (size_t i = samples.size() - 1; i > 0; i--) {
                samples[i].size = samples[i - 1].size;
            }
            samples[0].size = 0;

            for (size_t r = 0; r < splitter_count; r++) {
                // TODO init next lower_bound with this iterator
                size_t block_index = std::lower_bound(samples.begin(),
                                                      samples.end(),
                                                      target_ranks[r],
                                                      [](BlockSample b, size_t t){
                    return b.size < t;
                }) - samples.begin();

                if (block_index > 0) {
                    size_t b = 0;
                    std::vector<std::vector<bool>> workers_sent(P,
                            std::vector<bool>(seq_count, false));

                    while (b < P * seq_count && block_index > 0) {
                        block_index--;

                        auto block_sample = samples[block_index];
                        if (!workers_sent[block_sample.worker_rank]
                                [block_sample.sequence_idx]) {
                            writers[block_sample.worker_rank].Put(
                                    ReplyBlockSample{
                                            block_sample.idx,
                                            block_sample.sequence_idx,
                                            r
                                    });
                            workers_sent[block_sample.worker_rank]
                                [block_sample.sequence_idx] = true;
                            b++;
                        }
                    }
                }
            }

            writers[0].Close();
        }

        for (size_t p = 1; p < P; p++) {
            writers[p].Close();
        }

        while(readers[0].HasNext()) {
            auto sample = readers[0].template Next<ReplyBlockSample>();

            width[sample.splitter_idx][sample.sequence_idx] =
                    sequences[sample.sequence_idx].ItemsStartIn(sample.idx);
            left[sample.splitter_idx][sample.sequence_idx] =
                    sequences[sample.sequence_idx].num_items_sum()[sample.idx] -
                    width[sample.splitter_idx][sample.sequence_idx];
        }

        stream.reset();

        this->GetSplitterRanks(sequences, out_local_ranks, target_ranks, left,
                width);
    }

private:
    size_t dia_id_;

    SortAlgorithm sort_algorithm_;

    struct BlockSample {
        ValueType value;
        size_t    idx;
        size_t    size;
        size_t    worker_rank;
        size_t    sequence_idx;
    };

    struct ReplyBlockSample {
        size_t idx;
        size_t sequence_idx;
        size_t splitter_idx;
    };
};

template <typename SequenceAdapterType, typename Comparator>
void run_multi_sequence_selection(Context &context,
                                  const Comparator &comparator,
                                  std::vector<SequenceAdapterType> &sequences,
                                  std::vector<std::vector<size_t>> &out_local_ranks,
                                  size_t splitter_count) {
    MultiSequenceSelector<SequenceAdapterType, Comparator> selector(context,
                                                                    comparator);
    return selector.GetEquallyDistantSplitterRanks(sequences, out_local_ranks,
                                                   splitter_count);

}

template <typename ValueType, typename Comparator,
        typename SortAlgorithm>
void run_sampled_multi_sequence_selection(Context &context, size_t dia_id,
                                  const Comparator &comparator,
                                  const SortAlgorithm &sort_algorithm,
                                  std::vector<MultiSequenceSelectorSampledFileSequenceAdapter<ValueType>> &sequences,
                                  std::vector<std::vector<size_t>> &out_local_ranks,
                                  size_t splitter_count) {
    MultiSequenceSelectorSampled<ValueType, Comparator, SortAlgorithm>
            selector(context, dia_id, comparator, sort_algorithm);
    return selector.GetEquallyDistantSplitterRanks(sequences, out_local_ranks,
            splitter_count);

}

} // namespace core
} // namespace thrill

#endif // !THRILL_API_MULTI_SEQUENCE_SELECTION_HEADER

/******************************************************************************/
