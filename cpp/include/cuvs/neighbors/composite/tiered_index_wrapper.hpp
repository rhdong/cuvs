/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 */

#pragma once

#include "iindex.hpp"
#include <raft/util/raft_expectations.hpp>

namespace cuvs::neighbors::composite {

template <typename T, typename IdxT>
class TieredIndexWrapper : public IIndex<T, IdxT> {
 public:
  void search(const raft::resources&,
              const cuvs::neighbors::search_params&,
              raft::device_matrix_view<const T, int64_t, raft::row_major>,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major>,
              raft::device_matrix_view<float, int64_t, raft::row_major>,
              const cuvs::neighbors::filtering::base_filter&) const override
  {
    RAFT_FAIL("TieredIndexWrapper not implemented");
  }
};

}  // namespace cuvs::neighbors::composite

