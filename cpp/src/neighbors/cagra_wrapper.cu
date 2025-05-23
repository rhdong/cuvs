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

namespace cuvs::neighbors::composite {

template <typename T, typename IdxT>
void CagraIndexWrapper<T, IdxT>::search(
  const raft::resources& handle,
  const cuvs::neighbors::search_params& params,
  raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
  raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  const cuvs::neighbors::filtering::base_filter& filter) const
{
  auto const& cagra_params = dynamic_cast<const cuvs::neighbors::cagra::search_params&>(params);
  cuvs::neighbors::cagra::search(handle, cagra_params, *index_, queries, neighbors, distances, filter);
}

// explicit instantiations
template class CagraIndexWrapper<float, uint32_t>;
template class CagraIndexWrapper<half, uint32_t>;
template class CagraIndexWrapper<int8_t, uint32_t>;
template class CagraIndexWrapper<uint8_t, uint32_t>;

}  // namespace cuvs::neighbors::composite

