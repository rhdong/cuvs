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

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <memory>
#include <vector>

namespace cuvs::neighbors::composite {

/**
 * @brief Polymorphic index interface used by composite indexing.
 */
template <typename T, typename IdxT>
struct IIndex {
  using value_type  = T;
  using index_type  = IdxT;
  using dataset_idx = int64_t;

  virtual ~IIndex() = default;

  /**
   * @brief Search the index.
   */
  virtual void search(const raft::resources& handle,
                      const cuvs::neighbors::search_params& params,
                      raft::device_matrix_view<const T, dataset_idx, raft::row_major> queries,
                      raft::device_matrix_view<IdxT, dataset_idx, raft::row_major> neighbors,
                      raft::device_matrix_view<float, dataset_idx, raft::row_major> distances,
                      const cuvs::neighbors::filtering::base_filter& filter =
                        cuvs::neighbors::filtering::none_sample_filter{}) const = 0;
};

/**
 * @brief Wrapper for a CAGRA index implementing IIndex.
 */
template <typename T, typename IdxT>
class CagraIndexWrapper : public IIndex<T, IdxT> {
 public:
  using index_type  = IdxT;
  using value_type  = T;

  explicit CagraIndexWrapper(cuvs::neighbors::cagra::index<T, IdxT>* idx) : index_(idx) {}

  void search(const raft::resources& handle,
              const cuvs::neighbors::search_params& params,
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,
              const cuvs::neighbors::filtering::base_filter& filter =
                cuvs::neighbors::filtering::none_sample_filter{}) const override;

 private:
  cuvs::neighbors::cagra::index<T, IdxT>* index_;
};

/**
 * @brief Composite index made of other IIndex implementations.
 */
template <typename T, typename IdxT>
class CompositeIndexWrapper : public IIndex<T, IdxT> {
 public:
  using index_ptr = std::shared_ptr<IIndex<T, IdxT>>;

  explicit CompositeIndexWrapper(std::vector<index_ptr> children)
    : children_(std::move(children))
  {
  }

  void search(const raft::resources& handle,
              const cuvs::neighbors::search_params& params,
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,
              const cuvs::neighbors::filtering::base_filter& filter =
                cuvs::neighbors::filtering::none_sample_filter{}) const override;

 private:
  std::vector<index_ptr> children_;
};

}  // namespace cuvs::neighbors::composite

