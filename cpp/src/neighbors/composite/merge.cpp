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

#include <cuvs/neighbors/composite/merge.hpp>

namespace cuvs::neighbors::composite {

#define CUVS_INSTANTIATE_MERGE(T, IdxT)                                       \
  template std::shared_ptr<IIndex<T, IdxT>> merge<T, IdxT>(                   \
    const raft::resources&,                                                   \
    const cuvs::neighbors::cagra::merge_params&,                              \
    std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>&);

CUVS_INSTANTIATE_MERGE(float, uint32_t);
CUVS_INSTANTIATE_MERGE(half, uint32_t);
CUVS_INSTANTIATE_MERGE(uint8_t, uint32_t);
CUVS_INSTANTIATE_MERGE(int8_t, uint32_t);

#undef CUVS_INSTANTIATE_MERGE

}  // namespace cuvs::neighbors::composite

