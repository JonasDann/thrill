/*******************************************************************************
 * thrill/api/cms_sort.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2015 Alexander Noe <aleexnoe@gmail.com>
 * Copyright (C) 2015 Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (C) 2015-2016 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_CMS_SORT_HEADER
#define THRILL_API_CMS_SORT_HEADER

#include <thrill/api/context.hpp>
#include <thrill/api/dia.hpp>
#include <thrill/api/dop_node.hpp>
#include <thrill/common/logger.hpp>
#include <thrill/common/math.hpp>
#include <thrill/common/porting.hpp>
#include <thrill/common/qsort.hpp>
#include <thrill/core/multisequence_selection.hpp>
#include <thrill/core/multiway_merge.hpp>
#include <thrill/data/file.hpp>
#include <thrill/net/group.hpp>
#include <tlx/math/integer_log2.hpp>

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <functional>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace thrill {
namespace api {

/*!
 * A DIANode which performs a Sort operation. Sort sorts a DIA according to a
 * given compare function
 *
 * \tparam ValueType Type of DIA elements
 *
 * \tparam CompareFunction Type of the compare function
 *
 * \tparam SortAlgorithm Type of the local sort function
 *
 * \tparam Stable Whether or not to use stable sorting mechanisms
 *
 * \ingroup api_layer
 */
template <
    typename ValueType,
    typename CompareFunction,
    typename SortAlgorithm>
class CanonicalMergeSortNode final : public DOpNode<ValueType>
{
    static constexpr bool debug = true;

    //! Set this variable to true to enable generation and output of stats
    static constexpr bool stats_enabled = true;

    using Super = DOpNode<ValueType>;
    using Super::context_;

    //! Timer or FakeTimer
    using Timer = common::StatsTimerBaseStopped<stats_enabled>;
    //! RIAA class for running the timer
    using RunTimer = common::RunTimer<Timer>;

    struct MultiwayMergeTree {
        template <typename ReaderIterator,
                  typename Comparator = std::less<ValueType> >
        auto operator () (
            ReaderIterator seqs_begin, ReaderIterator seqs_end,
            const Comparator& comp = Comparator()) {

            return core::make_multiway_merge_tree<ValueType>(
                seqs_begin, seqs_end, comp);
        }
    };

    static const size_t run_capacity_ = 15;

public:
    /*!
     * Constructor for a sort node.
     */
    template <typename ParentDIA>
    CanonicalMergeSortNode(const ParentDIA& parent,
             const CompareFunction& compare_function,
             const SortAlgorithm& sort_algorithm = SortAlgorithm())
        : Super(parent.ctx(), "Canonical Merge Sort", { parent.id() }, { parent.node() }),
          compare_function_(compare_function),
          sort_algorithm_(sort_algorithm)
    {
        // Hook PreOp(s)
        auto pre_op_fn = [this](const ValueType& input) {
                             PreOp(input);
                         };

        auto lop_chain = parent.stack().push(pre_op_fn).fold();
        parent.node()->AddChild(this, lop_chain);

        // Count of all workers (and count of target partitions)
        p_ = context_.num_workers();
    }

    void StartPreOp(size_t /* id */) final {
        timer_preop_.Start();
        current_run_.reserve(run_capacity_);
    }

    void PreOp(const ValueType& input) {
        if (current_run_.size() >= run_capacity_) {
            FinishCurrentRun();
        }
        current_run_.push_back(input);
        local_items_++;
    }

    //! Receive a whole data::File of ValueType, but only if our stack is empty.
    bool OnPreOpFile(const data::File& file, size_t /* parent_index */) final {
        (void) file;
        // TODO What should this do?

        return false;
    }

    void StopPreOp(size_t /* id */) final {
        if (current_run_.size() > 0) {
            FinishCurrentRun();
        }

        timer_preop_.Stop();
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                "Sort() timer_preop_", timer_preop_.SecondsDouble());
            context_.PrintCollectiveMeanStdev(
                "Sort() preop local_items_", local_items_);
        }
    }

    DIAMemUse ExecuteMemUse() final {
        return DIAMemUse::Max();
    }

    //! Executes the sort operation.
    void Execute() final {
        MainOp();
        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                "Sort() timer_execute", timer_execute_.SecondsDouble());
        }
    }

    DIAMemUse PushDataMemUse() final {
        // TODO What does this do? Make it work.
        return 0;
    }

    void PushData(bool consume) final {
        Timer timer_pushdata;
        timer_pushdata.Start();

        // TODO Push.
        (void) consume;

        timer_pushdata.Stop();

        if (stats_enabled) {
            context_.PrintCollectiveMeanStdev(
                "Sort() timer_pushdata", timer_pushdata.SecondsDouble());

            //context_.PrintCollectiveMeanStdev("Sort() local_size", local_size);
        }
    }

    void Dispose() final {
        // TODO This may need to do something.
    }

private:
    size_t p_;

    using VectorSequenceAdapter = core::MultisequenceSelectorVectorSequenceAdapter<ValueType>;
    using MultiVectorSelector = core::MultisequenceSelector<VectorSequenceAdapter, CompareFunction>;

    using FileSequenceAdapter = core::MultisequenceSelectorFileSequenceAdapter<ValueType>;
    using MultiFileSelector = core::MultisequenceSelector<FileSequenceAdapter, CompareFunction>;

    using LocalRanks = std::vector<std::vector<size_t>>;

    //! The comparison function which is applied to two elements.
    CompareFunction compare_function_;

    //! Sort function class
    SortAlgorithm sort_algorithm_;

    MultiVectorSelector vector_selector_ {context_, compare_function_};

    //! \name PreOp Phase
    //! \{

    //! Current run data
    VectorSequenceAdapter current_run_;
    //! Runs in the first phase of the algorithm
    std::vector<data::FilePtr> run_files_;
    //! Number of items on this worker
    size_t local_items_ = 0;

    //! \}

    //! \name MainOp and PushData
    //! \{



    //! \}

    //! \name Statistics
    //! \{

    //! time spent in PreOp (including preceding Node's computation)
    Timer timer_preop_;

    //! time spent in Execute
    Timer timer_execute_;

    //! time spent in sort()
    Timer timer_sort_;

    //! \}

    class VectorConsumeReaderAdapter {
    public:
        using ValueTypeVector = std::vector<ValueType>;
        using ValueTypeVectorIterator = typename std::vector<ValueType>::iterator;

        VectorConsumeReaderAdapter(ValueTypeVector& vector)
        : vector_(vector),
          current_(vector.begin())
        {}

        template <typename T>
        ValueType Next() {
            assert(HasNext());
            return *(++current_);
        }

        bool HasNext() {
            return std::next(current_) != vector_.end();
        }

        void Skip(size_t items, size_t bytes) {
            (void) bytes;
            current_ += items;
        }

        bool typecode_verify() {
            return false;
        }
    private:
        ValueTypeVector& vector_;
        ValueTypeVectorIterator current_;
    };

    template <typename SequenceReaderType, bool skip_local>
    void TransmitElements(std::vector<SequenceReaderType>& seq_readers, data::StreamData::Writers& data_writers, LocalRanks& local_ranks) {
        size_t my_rank = context_.my_rank();
        size_t seq_count = seq_readers.size();
        size_t splitter_count = local_ranks.size();
        for (size_t seq_index = 0; seq_index < seq_count; seq_index++) {
            LOG << "Transmitting element of sequence " << seq_index << ".";
            auto seq_reader = &seq_readers[seq_index];
            size_t worker_rank = 0;
            size_t i = 0;
            LOG << "Worker rank " << worker_rank << ".";
            while (seq_reader->HasNext()) {
                while (worker_rank < splitter_count && local_ranks[worker_rank][seq_index] <= i) {
                    worker_rank++;
                    LOG << "Worker rank " << worker_rank << ".";
                }

                if (skip_local && worker_rank == my_rank) {
                    // if last worker
                    if (my_rank == splitter_count) {
                        LOG << "Go to next sequence.";
                        break;
                    } else {
                        const size_t items = local_ranks[worker_rank][seq_index] - i;
                        LOG << "Skip " << items <<" elements.";
                        const size_t bytes_per_item =
                                (seq_reader->typecode_verify() ? sizeof(size_t) : 0)
                                + data::Serialization<data::File::ConsumeReader, ValueType>::fixed_size;
                        seq_reader->Skip(items, items * bytes_per_item);
                        i += items;
                    }
                } else {
                    auto next = seq_reader->template Next<ValueType>();
                    data_writers[worker_rank].template Put<ValueType>(next);
                    i++;
                }
            }
        }
    }

    void FinishCurrentRun() {
        /* Phase 1 {*/
        // Sort Locally
        LOG << "Sort run locally.";
        timer_sort_.Start();
        sort_algorithm_(current_run_.begin(), current_run_.end(), compare_function_);
        timer_sort_.Stop();

        // Calculate Splitters
        auto splitter_count = p_ - 1;
        LOG << "Calculating " << splitter_count << " splitters.";
        LocalRanks local_ranks(splitter_count, std::vector<size_t>(1));
        std::vector<VectorSequenceAdapter> current_run_vector(1);
        current_run_vector[0] = current_run_;
        vector_selector_.GetEquallyDistantSplitterRanks(current_run_vector, local_ranks, splitter_count);
        LOG << "Local splitters: " << local_ranks;

        // Redistribute Elements
        auto data_stream = context_.template GetNewStream<data::CatStream>(this->id());
        auto data_writers = data_stream->GetWriters();
        std::vector<VectorConsumeReaderAdapter> current_run_reader(1, VectorConsumeReaderAdapter(current_run_));

        LOG << "Transmitting elements.";
        TransmitElements<VectorConsumeReaderAdapter, false>(current_run_reader, data_writers, local_ranks);
        current_run_.clear();
        for (size_t worker_rank = 0; worker_rank < p_; worker_rank++) {
            data_writers[worker_rank].Close();
        }

        auto data_readers = data_stream->GetReaders();
        LOG << "Building merge tree.";
        auto multiway_merge_tree = core::make_multiway_merge_tree<ValueType>(
                data_readers.begin(), data_readers.end(), compare_function_);

        LOG << "Merging into run file.";
        run_files_.emplace_back(context_.GetFilePtr(this));
        auto current_run_file_writer = run_files_.back()->GetWriter();
        while (multiway_merge_tree.HasNext()) {
            auto next = multiway_merge_tree.Next();
            current_run_file_writer.template Put<ValueType>(next);
            LOG << next;
        }
        current_run_file_writer.Close();
        LOG << "Finished run has " << run_files_.back()->num_items() << " elements.";

        data_stream.reset();
        /* } Phase 1 */
    }

    void MainOp() {
        /* Phase 2 { */
        LOG << "Phase 2.";
        // Calculate Splitters
        auto splitter_count = p_ - 1;
        auto run_count = run_files_.size();
        LOG << "Calculating " << splitter_count << " splitters.";
        LocalRanks local_ranks(splitter_count, std::vector<size_t>(run_count));
        MultiFileSelector selector(context_, compare_function_);
        std::vector<FileSequenceAdapter> run_file_adapters(run_count);
        for (size_t i = 0; i < run_count; i++) {
            run_file_adapters[i] = FileSequenceAdapter(run_files_[i]);
        }
        selector.GetEquallyDistantSplitterRanks(run_file_adapters, local_ranks, splitter_count);
        LOG << "Local splitters: " << local_ranks;

        // Redistribute Elements
        auto data_stream = context_.template GetNewStream<data::CatStream>(this->id());
        auto data_writers = data_stream->GetWriters();
        std::vector<data::File::ConsumeReader> run_readers;
        for (size_t i = 0; i < run_count; i++) {
            run_readers.emplace_back(run_files_[i]->GetConsumeReader());
        }

        LOG << "Transmitting elements.";
        TransmitElements<data::File::ConsumeReader, true>(run_readers, data_writers, local_ranks);
        for (size_t worker_rank = 0; worker_rank < p_; worker_rank++) {
            data_writers[worker_rank].Close();
        }

        auto data_readers = data_stream->GetReaders();
        /* } Phase 2 */

        /* Phase 3 { */
        LOG << "Phase 3.";
        // TODO Phase 3: Merge everything.
        /* } Phase 3 */
    }
};

class DefaultSortAlgorithm
{
public:
    template <typename Iterator, typename CompareFunction>
    void operator () (Iterator begin, Iterator end, CompareFunction cmp) const {
        return std::sort(begin, end, cmp);
    }
};

template <typename ValueType, typename Stack>
template <typename CompareFunction>
auto DIA<ValueType, Stack>::CanonicalMergeSort(const CompareFunction& compare_function) const {
    assert(IsValid());

    using CanonicalMergeSortNode = api::CanonicalMergeSortNode<
        ValueType, CompareFunction, DefaultSortAlgorithm>;

    static_assert(
        std::is_convertible<
            ValueType,
            typename FunctionTraits<CompareFunction>::template arg<0> >::value,
        "CompareFunction has the wrong input type");

    static_assert(
        std::is_convertible<
            ValueType,
            typename FunctionTraits<CompareFunction>::template arg<1> >::value,
        "CompareFunction has the wrong input type");

    static_assert(
        std::is_convertible<
            typename FunctionTraits<CompareFunction>::result_type,
            bool>::value,
        "CompareFunction has the wrong output type (should be bool)");

    auto node = tlx::make_counting<CanonicalMergeSortNode>(*this, compare_function);

    return DIA<ValueType>(node);
}

} // namespace api
} // namespace thrill

#endif // !THRILL_API_CMS_SORT_HEADER

/******************************************************************************/
