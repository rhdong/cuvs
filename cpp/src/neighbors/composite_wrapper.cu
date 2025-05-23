/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/composite/index_iface.hpp>
#include <cuvs/selection/select_k.hpp>
#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>

namespace cuvs::neighbors::composite {

template <typename T, typename IdxT>
void CompositeIndexWrapper<T, IdxT>::search(
  const raft::resources& handle,
  const cuvs::neighbors::search_params& params,
  raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
  raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const cuvs::neighbors::filtering::base_filter& filter) const
{
  if (children_.empty()) { return; }
  if (children_.size() == 1) {
    children_.front()->search(handle, params, queries, neighbors, distances, filter);
    return;
  }

  auto n_query = queries.extent(0);
  auto k       = neighbors.extent(1);
  auto total_k = k * static_cast<int64_t>(children_.size());

  auto tmp_dists = raft::make_device_matrix<float, int64_t>(handle, n_query, total_k);
  auto tmp_inds  = raft::make_device_matrix<IdxT, int64_t>(handle, n_query, total_k);

  int64_t offset = 0;
  for (auto const& child : children_) {
    auto sub_neigh = raft::device_matrix_view<IdxT, int64_t, raft::row_major>(
      tmp_inds.data_handle() + offset, raft::make_matrix_extent<int64_t>(n_query, k));
    auto sub_dist = raft::device_matrix_view<float, int64_t, raft::row_major>(
      tmp_dists.data_handle() + offset, raft::make_matrix_extent<int64_t>(n_query, k));
    child->search(handle, params, queries, sub_neigh, sub_dist, filter);
    offset += k * n_query;
  }

  cuvs::selection::select_k(handle,
                            tmp_dists.view(),
                            tmp_inds.view(),
                            distances,
                            neighbors,
                            true,
                            true);
}

// explicit instantiation
template class CompositeIndexWrapper<float, uint32_t>;

}  // namespace cuvs::neighbors::composite

