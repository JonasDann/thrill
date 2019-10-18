/*******************************************************************************
 * thrill/api/sort.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2019 Timo Bingmann <tb@panthema.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_API_SORT_HEADER
#define THRILL_API_SORT_HEADER

#include <thrill/api/sort_canonical_merge_sort.hpp>
#include <thrill/api/sort_online_sample_sort.hpp>
#include <thrill/api/sort_sample_sort.hpp>

namespace thrill {
namespace api {

/******************************************************************************/
// DIA::Sort()

enum class SortingAlgorithm {
    Default,
    SampleSort,
    CanonicalMergeSort,
    OnlineSampleSort
};

class DefaultSortAlgorithm
{
public:
    template <typename Iterator, typename CompareFunction>
    void operator () (Iterator begin, Iterator end, CompareFunction cmp) const {
        return std::sort(begin, end, cmp);
    }
};

class DefaultSortConfig
{
public:
    //! distributed sorting algorithm
    static constexpr SortingAlgorithm algorithm_ = SortingAlgorithm::Default;

    //! base sequential sorting algorithm
    DefaultSortAlgorithm base_sort_;
};

template <typename ValueType, typename Stack>
template <typename CompareFunction, typename SortConfig>
auto DIA<ValueType, Stack>::Sort(const CompareFunction& compare_function,
                                 const SortConfig& sort_config) const {
    assert(IsValid());

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

    switch (sort_config.algorithm_)
    {
    default:
    case SortingAlgorithm::Default:
    case SortingAlgorithm::CanonicalMergeSort:
        using CanonicalMergeSortNode = api::CanonicalMergeSortNode<
            ValueType, CompareFunction, DefaultSortAlgorithm2>;

        return DIA<ValueType>(
            tlx::make_counting<CanonicalMergeSortNode>(
                *this, compare_function));

    case SortingAlgorithm::SampleSort:
        using SampleSortNode = api::SampleSortNode<
            ValueType, CompareFunction, SortConfig>;

        return DIA<ValueType>(
            tlx::make_counting<SampleSortNode>(
                *this, compare_function, sort_config));
    }
}

/******************************************************************************/
// DIA::StableSort()

class DefaultStableSortAlgorithm
{
public:
    template <typename Iterator, typename CompareFunction>
    void operator () (Iterator begin, Iterator end, CompareFunction cmp) const {
        return std::stable_sort(begin, end, cmp);
    }
};

class DefaultStableSortConfig
{
public:
    //! base sequential sorting algorithm
    DefaultStableSortAlgorithm base_sort_;
};

template <typename ValueType, typename Stack>
template <typename CompareFunction, typename SortConfig>
auto DIA<ValueType, Stack>::SortStable(const CompareFunction& compare_function,
                                       const SortConfig& sort_config) const {
    assert(IsValid());

    using SortStableNode = api::SampleSortNode<
        ValueType, CompareFunction, SortConfig, /* Stable */ true>;

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

    auto node = tlx::make_counting<SortStableNode>(
        *this, compare_function, sort_config);

    return DIA<ValueType>(node);
}

/******************************************************************************/

} // namespace api
} // namespace thrill

#endif // !THRILL_API_SORT_HEADER

/******************************************************************************/
