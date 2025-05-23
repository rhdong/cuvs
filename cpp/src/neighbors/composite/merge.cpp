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

#include <cuvs/neighbors/composite/merge.hpp>

namespace cuvs::neighbors::composite {

template <typename T, typename IdxT>
std::shared_ptr<IIndex<T, IdxT>> merge(
  const raft::resources& handle,
  const cuvs::neighbors::cagra::merge_params& params,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices)
{
  if (params.strategy == cuvs::neighbors::cagra::MergeStrategy::MERGE_STRATEGY_LOGICAL) {
    auto comp = cuvs::neighbors::cagra::make_composite_index<T, IdxT>(params, indices);
    std::vector<std::shared_ptr<IIndex<T, IdxT>>> wrappers;
    wrappers.reserve(comp.sub_indices.size());
    for (auto* idx : comp.sub_indices) {
      wrappers.push_back(std::make_shared<CagraIndexWrapper<T, IdxT>>(idx));
    }
    return std::make_shared<CompositeIndexWrapper<T, IdxT>>(std::move(wrappers));
  } else {
    auto out = cuvs::neighbors::cagra::merge<T, IdxT>(handle, params, indices);
    auto* idx = new decltype(out)(std::move(out));
    return std::make_shared<CagraIndexWrapper<T, IdxT>>(idx);
  }
}

// explicit instantiation
#define INSTANTIATE(T, IdxT) \
  template std::shared_ptr<IIndex<T, IdxT>> merge<T, IdxT>( \
    const raft::resources&, const cuvs::neighbors::cagra::merge_params&, \
    std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>&);

INSTANTIATE(float, uint32_t);
INSTANTIATE(half, uint32_t);
INSTANTIATE(int8_t, uint32_t);
INSTANTIATE(uint8_t, uint32_t);

#undef INSTANTIATE

}  // namespace cuvs::neighbors::composite

