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

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::composite {

/**
 * @brief Polymorphic interface for cuVS nearest neighbor indices.
 *
 * This interface allows using different index implementations through
 * a common `search()` method.
 */
template <typename T, typename IdxT>
struct IIndex {
  using value_type  = T;
  using index_type  = IdxT;

  virtual ~IIndex() = default;

  /**
   * @brief Perform kNN search on the index.
   */
  virtual void search(const raft::resources& handle,
                      const cuvs::neighbors::search_params& params,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                      raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
                      raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                      const cuvs::neighbors::filtering::base_filter& filter =
                        cuvs::neighbors::filtering::none_sample_filter{}) const = 0;
};

}  // namespace cuvs::neighbors::composite

