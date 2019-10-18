/*******************************************************************************
 * thrill/data/fwd.hpp
 *
 * Part of Project Thrill - http://project-thrill.org
 *
 * Copyright (C) 2019 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once
#ifndef THRILL_DATA_FWD_HEADER
#define THRILL_DATA_FWD_HEADER

#include <tlx/counting_ptr.hpp>

namespace thrill {
namespace data {

template <typename ItemType>
class SampledFile;
template <typename ItemType>
using SampledFilePtr = tlx::CountingPtr<SampledFile<ItemType> >;

} // namespace data
} // namespace thrill

#endif // !THRILL_DATA_FWD_HEADER

/******************************************************************************/
