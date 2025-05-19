/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_cagra_merge : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  struct search_param : public search_param_base {};

  struct build_param {
    cuvs::neighbors::cagra::index_params cagra_params;
    uint32_t split_size = 0;
    uint32_t n_splits   = 1;
    cuvs::neighbors::cagra::MergeStrategy strategy =
      cuvs::neighbors::cagra::MergeStrategy::MERGE_STRATEGY_PHYSICAL;
  };

  cuvs_cagra_merge(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim), params_(param)
  {
    params_.cagra_params.metric = parse_metric_type(metric);
  }

  void build(const T* dataset, size_t nrow) final
  {
    if (params_.split_size == 0 && params_.n_splits > 0) {
      params_.split_size = raft::ceildiv<uint32_t>(static_cast<uint32_t>(nrow), params_.n_splits);
    }
    if (params_.n_splits == 0 && params_.split_size > 0) {
      params_.n_splits = raft::ceildiv<uint32_t>(static_cast<uint32_t>(nrow), params_.split_size);
    }
    if (params_.n_splits == 0) { params_.n_splits = 1; }
    if (params_.split_size == 0) { params_.split_size = static_cast<uint32_t>(nrow); }

    indices_.clear();
    index_ptrs_.clear();

    bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;

    for (uint32_t i = 0; i < params_.n_splits; ++i) {
      IdxT start = static_cast<IdxT>(i) * params_.split_size;
      if (start >= static_cast<IdxT>(nrow)) break;
      IdxT rows      = std::min<IdxT>(params_.split_size, static_cast<IdxT>(nrow) - start);
      auto extents   = raft::make_extents<IdxT>(rows, this->dim_);
      auto view_host = raft::make_mdspan<const T, IdxT, raft::row_major, true, false>(
        dataset + static_cast<size_t>(start) * this->dim_, extents);
      auto view_dev = raft::make_mdspan<const T, IdxT, raft::row_major, false, true>(
        dataset + static_cast<size_t>(start) * this->dim_, extents);
      indices_.emplace_back(
        std::move(dataset_is_on_host
                    ? cuvs::neighbors::cagra::build(handle_, params_.cagra_params, view_host)
                    : cuvs::neighbors::cagra::build(handle_, params_.cagra_params, view_dev)));
      index_ptrs_.push_back(&indices_.back());
    }

    cuvs::neighbors::cagra::merge_params m_params{params_.cagra_params};
    m_params.strategy = params_.strategy;
    merged_index_     = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(
      std::move(cuvs::neighbors::cagra::merge(handle_, m_params, index_ptrs_)));
  }

  void set_search_param(const search_param_base&, const void*) override {}

  void search(const T*, int, int, algo_base::index_type*, float*) const override
  {
    throw std::runtime_error("search not supported for merge benchmark");
  }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property{};
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  void save(const std::string& file) const override
  {
    cuvs::neighbors::cagra::serialize(handle_, *merged_index_, file);
  }

  void load(const std::string& file) override
  {
    merged_index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(
      cuvs::neighbors::cagra::deserialize<T, IdxT>(handle_, file));
  }

  std::unique_ptr<algo<T>> copy() override
  {
    return std::make_unique<cuvs_cagra_merge<T, IdxT>>(*this);
  }

 private:
  configured_raft_resources handle_{};
  build_param params_{};
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>> indices_{};
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> index_ptrs_{};
  std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>> merged_index_{};
};

}  // namespace cuvs::bench
