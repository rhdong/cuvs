/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "iindex.hpp"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::composite {

/**
 * @brief Wrapper for a CAGRA index providing the `IIndex` interface.
 */
template <typename T, typename IdxT>
class CagraIndexWrapper : public IIndex<T, IdxT> {
 public:
  explicit CagraIndexWrapper(cuvs::neighbors::cagra::index<T, IdxT>&& idx)
    : index_(std::move(idx))
  {
  }

  const cuvs::neighbors::cagra::index<T, IdxT>& get() const { return index_; }

  void search(const raft::resources& handle,
              const cuvs::neighbors::search_params& params,
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,
              const cuvs::neighbors::filtering::base_filter& filter =
                cuvs::neighbors::filtering::none_sample_filter{}) const override
  {
    auto const& cagra_params = static_cast<cuvs::neighbors::cagra::search_params const&>(params);
    cuvs::neighbors::cagra::search(handle, cagra_params, index_, queries, neighbors, distances, filter);
  }

 private:
  cuvs::neighbors::cagra::index<T, IdxT> index_;
};

}  // namespace cuvs::neighbors::composite

