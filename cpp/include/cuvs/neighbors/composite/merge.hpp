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

#include "cagra_index_wrapper.hpp"
#include "composite_index_wrapper.hpp"
#include <cuvs/neighbors/cagra.hpp>
#include <memory>
#include <vector>

namespace cuvs::neighbors::composite {

/**
 * @brief Create a unified index from multiple CAGRA indices.
 *
 * Depending on the merge strategy, either physically merges the indices or
 * creates a logical composite. The returned object implements `IIndex`.
 */
template <typename T, typename IdxT>
std::shared_ptr<IIndex<T, IdxT>> merge(
  const raft::resources& handle,
  const cuvs::neighbors::cagra::merge_params& params,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices)
{
  using namespace cuvs::neighbors;
  if (params.strategy == cagra::MergeStrategy::MERGE_STRATEGY_PHYSICAL) {
    auto merged = cagra::merge<T, IdxT>(handle, params, indices);
    return std::make_shared<CagraIndexWrapper<T, IdxT>>(std::move(merged));
  } else {
    auto comp = cagra::make_composite_index<T, IdxT>(params, indices);
    std::vector<std::shared_ptr<IIndex<T, IdxT>>> children;
    children.reserve(comp.sub_indices.size());
    for (auto* idx : comp.sub_indices) {
      children.push_back(std::make_shared<CagraIndexWrapper<T, IdxT>>(*idx));
    }
    return std::make_shared<CompositeIndexWrapper<T, IdxT>>(std::move(children));
  }
}

}  // namespace cuvs::neighbors::composite

