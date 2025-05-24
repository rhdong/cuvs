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
#include <memory>
#include <vector>

namespace cuvs::neighbors::composite {

/**
 * @brief A wrapper that aggregates multiple `IIndex` implementations.
 *
 * The current implementation simply dispatches search to the first child index
 * and ignores the others. This can be extended to merge results from all
 * sub-indices.
 */
template <typename T, typename IdxT>
class CompositeIndexWrapper : public IIndex<T, IdxT> {
 public:
  explicit CompositeIndexWrapper(std::vector<std::shared_ptr<IIndex<T, IdxT>>> children)
    : children_(std::move(children))
  {
  }

  void search(const raft::resources& handle,
              const cuvs::neighbors::search_params& params,
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,
              const cuvs::neighbors::filtering::base_filter& filter =
                cuvs::neighbors::filtering::none_sample_filter{}) const override
  {
    if (children_.empty()) return;
    // TODO: merge results from all child indices
    children_.front()->search(handle, params, queries, neighbors, distances, filter);
  }

 private:
  std::vector<std::shared_ptr<IIndex<T, IdxT>>> children_;
};

}  // namespace cuvs::neighbors::composite

