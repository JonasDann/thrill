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

class DefaultSortAlgorithm
{
public:
    template <typename Iterator, typename CompareFunction>
    void operator () (Iterator begin, Iterator end, CompareFunction cmp) const {
        return std::sort(begin, end, cmp);
    }
};

template <typename ValueType, typename Stack>
template <typename CompareFunction, typename SortAlgorithm>
auto DIA<ValueType, Stack>::Sort(const CompareFunction& compare_function,
                                 const SortAlgorithm& sort_algorithm) const {
    assert(IsValid());

    using SortNode = api::SampleSortNode<
        ValueType, CompareFunction, SortAlgorithm>;

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

    auto node = tlx::make_counting<SortNode>(
        *this, compare_function, sort_algorithm);

    return DIA<ValueType>(node);
}

template <typename ValueType, typename Stack>
template <typename CompareFunction>
auto DIA<ValueType, Stack>::Sort(const CompareFunction& compare_function) const {
    return Sort(compare_function, DefaultSortAlgorithm());
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

template <typename ValueType, typename Stack>
template <typename CompareFunction, typename SortAlgorithm>
auto DIA<ValueType, Stack>::SortStable(
    const CompareFunction& compare_function,
    const SortAlgorithm& sort_algorithm) const {

    assert(IsValid());

    using SortStableNode = api::SampleSortNode<
        ValueType, CompareFunction, SortAlgorithm, /* Stable */ true>;

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
        *this, compare_function, sort_algorithm);

    return DIA<ValueType>(node);
}


template <typename ValueType, typename Stack>
template <typename CompareFunction>
auto DIA<ValueType, Stack>::SortStable(
    const CompareFunction& compare_function) const {
    return SortStable(compare_function, DefaultStableSortAlgorithm());
}

/******************************************************************************/

} // namespace api
} // namespace thrill

#endif // !THRILL_API_SORT_HEADER

/******************************************************************************/
