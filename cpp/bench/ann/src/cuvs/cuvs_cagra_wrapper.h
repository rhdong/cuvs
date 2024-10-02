/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../../../../src/neighbors/detail/cagra/utils.hpp"
#include "../common/ann_types.hpp"
#include "../common/cuda_huge_page_resource.hpp"
#include "../common/cuda_pinned_resource.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
// #include <raft/neighbors/dataset.hpp>
// #include <raft/neighbors/detail/cagra/cagra_build.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuvs::bench {

enum class AllocatorType { kHostPinned, kHostHugePage, kDevice };
enum class CagraBuildAlgo { kAuto, kIvfPq, kNnDescent };

template <typename T, typename IdxT>
class cuvs_cagra : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  struct search_param : public search_param_base {
    cuvs::neighbors::cagra::search_params p;
    float refine_ratio;
    AllocatorType graph_mem   = AllocatorType::kDevice;
    AllocatorType dataset_mem = AllocatorType::kDevice;
    [[nodiscard]] auto needs_dataset() const -> bool override { return true; }
  };

  struct build_param {
    cuvs::neighbors::cagra::index_params cagra_params;
    CagraBuildAlgo algo;
    std::optional<cuvs::neighbors::nn_descent::index_params> nn_descent_params = std::nullopt;
    std::optional<float> ivf_pq_refine_rate                                    = std::nullopt;
    std::optional<cuvs::neighbors::ivf_pq::index_params> ivf_pq_build_params   = std::nullopt;
    std::optional<cuvs::neighbors::ivf_pq::search_params> ivf_pq_search_params = std::nullopt;
  };

  int64_t create_sparse_bitset(int64_t total, float sparsity, std::vector<uint32_t>& bitset) const
  {
    int64_t num_ones = static_cast<int64_t>((total * 1.0f) * (1.0f - sparsity));
    int64_t res      = num_ones;

    for (auto& item : bitset) {
      item = static_cast<uint32_t>(0);
    }

    if(sparsity == 0.0) {
      for (auto& item : bitset) {
        item = static_cast<uint32_t>(0xffffffff);
      }
      return total;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dis(0, total - 1);

    while (num_ones > 0) {
      int64_t index = dis(gen);

      uint32_t& element    = bitset[index / (8 * sizeof(uint32_t))];
      int64_t bit_position = index % (8 * sizeof(uint32_t));

      if (((element >> bit_position) & 1) == 0) {
        element |= (static_cast<uint32_t>(1) << bit_position);
        num_ones--;
      }
    }
    return res;
  }

  cuvs_cagra(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim),
      index_params_(param),
      dimension_(dim),

      dataset_(std::make_shared<raft::device_matrix<T, int64_t, raft::row_major>>(
        std::move(raft::make_device_matrix<T, int64_t>(handle_, 0, 0)))),
      graph_(std::make_shared<raft::device_matrix<IdxT, int64_t, raft::row_major>>(
        std::move(raft::make_device_matrix<IdxT, int64_t>(handle_, 0, 0)))),
      input_dataset_v_(
        std::make_shared<raft::device_matrix_view<const T, int64_t, raft::row_major>>(
          nullptr, 0, 0)),
      bitset_filter_(
        std::make_shared<raft::device_vector<std::uint32_t, int64_t>>(
          std::move(raft::make_device_vector<std::uint32_t, int64_t>(handle_, 0))))

  {
    index_params_.cagra_params.metric         = parse_metric_type(metric);
    index_params_.ivf_pq_build_params->metric = parse_metric_type(metric);
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param) override;

  void set_search_dataset(const T* dataset, size_t nrow) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  void search_base(const T* queries,
                   int batch_size,
                   int k,
                   algo_base::index_type* neighbors,
                   float* distances) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  void save_to_hnswlib(const std::string& file) const;
  std::unique_ptr<algo<T>> copy() override;

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  raft::mr::cuda_pinned_resource mr_pinned_;
  raft::mr::cuda_huge_page_resource mr_huge_page_;
  AllocatorType graph_mem_{AllocatorType::kDevice};
  AllocatorType dataset_mem_{AllocatorType::kDevice};
  float refine_ratio_;
  build_param index_params_;
  bool need_dataset_update_{true};
  cuvs::neighbors::cagra::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>> index_;
  int dimension_;
  std::shared_ptr<raft::device_matrix<IdxT, int64_t, raft::row_major>> graph_;
  std::shared_ptr<raft::device_matrix<T, int64_t, raft::row_major>> dataset_;
  std::shared_ptr<raft::device_matrix_view<const T, int64_t, raft::row_major>> input_dataset_v_;

  std::shared_ptr<raft::device_vector<std::uint32_t, int64_t>> bitset_filter_;
  double sparsity_ = 0.9;

  inline rmm::device_async_resource_ref get_mr(AllocatorType mem_type)
  {
    switch (mem_type) {
      case (AllocatorType::kHostPinned): return &mr_pinned_;
      case (AllocatorType::kHostHugePage): return &mr_huge_page_;
      default: return rmm::mr::get_current_device_resource();
    }
  }
};

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_extents = raft::make_extents<IdxT>(nrow, dimension_);

  auto& params = index_params_.cagra_params;

  if (index_params_.algo == CagraBuildAlgo::kIvfPq) {
    auto pq_params =
      cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(dataset_extents, params.metric);
    if (index_params_.ivf_pq_build_params) {
      pq_params.build_params = *index_params_.ivf_pq_build_params;
    }
    if (index_params_.ivf_pq_search_params) {
      pq_params.search_params = *index_params_.ivf_pq_search_params;
    }
    if (index_params_.ivf_pq_refine_rate) {
      pq_params.refinement_rate = *index_params_.ivf_pq_refine_rate;
    }
    params.graph_build_params = pq_params;
  } else if (index_params_.algo == CagraBuildAlgo::kNnDescent) {
    auto nn_params = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
      params.intermediate_graph_degree);
    if (index_params_.nn_descent_params) { nn_params = *index_params_.nn_descent_params; }
    params.graph_build_params = nn_params;
  }
  auto dataset_view_host =
    raft::make_mdspan<const T, IdxT, raft::row_major, true, false>(dataset, dataset_extents);
  auto dataset_view_device =
    raft::make_mdspan<const T, IdxT, raft::row_major, false, true>(dataset, dataset_extents);
  bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;

  index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(std::move(
    dataset_is_on_host ? cuvs::neighbors::cagra::build(handle_, params, dataset_view_host)
                       : cuvs::neighbors::cagra::build(handle_, params, dataset_view_device)));
}

inline auto allocator_to_string(AllocatorType mem_type) -> std::string
{
  if (mem_type == AllocatorType::kDevice) {
    return "device";
  } else if (mem_type == AllocatorType::kHostPinned) {
    return "host_pinned";
  } else if (mem_type == AllocatorType::kHostHugePage) {
    return "host_huge_page";
  }
  return "<invalid allocator type>";
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_search_param(const search_param_base& param)
{
  auto sp        = dynamic_cast<const search_param&>(param);
  search_params_ = sp.p;
  refine_ratio_  = sp.refine_ratio;
  if (sp.graph_mem != graph_mem_) {
    // Move graph to correct memory space
    graph_mem_ = sp.graph_mem;
    RAFT_LOG_DEBUG("moving graph to new memory space: %s", allocator_to_string(graph_mem_).c_str());
    // We create a new graph and copy to it from existing graph
    auto mr        = get_mr(graph_mem_);
    auto new_graph = raft::make_device_mdarray<IdxT, int64_t>(
      handle_, mr, raft::make_extents<int64_t>(index_->graph().extent(0), index_->graph_degree()));

    raft::copy(new_graph.data_handle(),
               index_->graph().data_handle(),
               index_->graph().size(),
               raft::resource::get_cuda_stream(handle_));

    index_->update_graph(handle_, make_const_mdspan(new_graph.view()));
    // update_graph() only stores a view in the index. We need to keep the graph object alive.
    *graph_ = std::move(new_graph);
  }

  if (sp.dataset_mem != dataset_mem_ || need_dataset_update_) {
    dataset_mem_ = sp.dataset_mem;

    // First free up existing memory
    *dataset_ = raft::make_device_matrix<T, int64_t>(handle_, 0, 0);
    index_->update_dataset(handle_, make_const_mdspan(dataset_->view()));

    // Allocate space using the correct memory resource.
    RAFT_LOG_DEBUG("moving dataset to new memory space: %s",
                   allocator_to_string(dataset_mem_).c_str());

    // Brute force doesn't support padding dataset.
    bool without_brute_force = false;
    if(without_brute_force) {
      auto mr = get_mr(dataset_mem_);
      cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);

      auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
        dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
      index_->update_dataset(handle_, dataset_view);
    } else {
      *dataset_ = raft::make_device_matrix<T, int64_t>(
                      handle_,
                      input_dataset_v_->extent(0),
                      input_dataset_v_->extent(1));
      auto stream = raft::resource::get_cuda_stream(handle_);
      auto data_size =  input_dataset_v_->extent(0) * input_dataset_v_->extent(1);
      raft::copy(dataset_->data_handle(), input_dataset_v_->data_handle(), data_size, stream);
      index_->update_dataset(handle_, make_const_mdspan(dataset_->view()));
    }

    need_dataset_update_ = false;
  }

  { // create bitset filter in advance.
    auto stream_ = raft::resource::get_cuda_stream(handle_);
    size_t filter_n_elements = size_t((input_dataset_v_->extent(0) + 31) / 32);
    *bitset_filter_ = raft::make_device_vector<std::uint32_t, int64_t>(
                        handle_,
                        filter_n_elements);

    std::vector<std::uint32_t> bitset_cpu(filter_n_elements);

    create_sparse_bitset(input_dataset_v_->extent(0), sparsity_, bitset_cpu);
    raft::copy(bitset_filter_->data_handle(), bitset_cpu.data(), filter_n_elements, stream_);
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  using ds_idx_type = decltype(index_->data().n_rows());
  bool is_vpq =
    dynamic_cast<const cuvs::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
    dynamic_cast<const cuvs::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
  // It can happen that we are re-using a previous algo object which already has
  // the dataset set. Check if we need update.
  if (static_cast<size_t>(input_dataset_v_->extent(0)) != nrow ||
      input_dataset_v_->data_handle() != dataset) {
    *input_dataset_v_ = raft::make_device_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
    need_dataset_update_ = !is_vpq;  // ignore update if this is a VPQ dataset.
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::save(const std::string& file) const
{
  cuvs::neighbors::cagra::serialize(handle_, file, *index_);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::save_to_hnswlib(const std::string& file) const
{
  cuvs::neighbors::cagra::serialize_to_hnswlib(handle_, file, *index_);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::load(const std::string& file)
{
  index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(handle_);
  cuvs::neighbors::cagra::deserialize(handle_, file, index_.get());
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_cagra<T, IdxT>::copy()
{
  return std::make_unique<cuvs_cagra<T, IdxT>>(*this);  // use copy constructor
}

template <typename IdxT>
auto cagra_calc_recall(const std::vector<IdxT>& expected_idx,
                       const std::vector<IdxT>& actual_idx,
                       size_t rows,
                       size_t cols)
{
  size_t match_count = 0;
  size_t total_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k = i * cols + k;  // row major assumption!
      auto act_idx = actual_idx[idx_k];
      for (size_t j = 0; j < cols; ++j) {
        size_t idx   = i * cols + j;  // row major assumption!
        auto exp_idx = expected_idx[idx];
        if (act_idx == exp_idx) {
          match_count++;
          break;
        }
      }
    }
  }
  return std::make_tuple(
    static_cast<double>(match_count) / static_cast<double>(total_count),
    match_count,
    total_count);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search_base(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  IdxT* neighbors_idx_t;
  std::optional<rmm::device_uvector<IdxT>> neighbors_storage{std::nullopt};
  if constexpr (sizeof(IdxT) == sizeof(algo_base::index_type)) {
    neighbors_idx_t = reinterpret_cast<IdxT*>(neighbors);
  } else {
    neighbors_storage.emplace(batch_size * k, raft::resource::get_cuda_stream(handle_));
    neighbors_idx_t = neighbors_storage->data();
  }

  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, dimension_);
  auto neighbors_view =
    raft::make_device_matrix_view<IdxT, int64_t>(neighbors_idx_t, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);


  auto neighbors_brute_force =
    raft::make_device_matrix<IdxT, int64_t>(handle_, batch_size, k);
  auto neighbors_brute_force_view = neighbors_brute_force.view();

  constexpr bool need_check_recall = true;

  if constexpr ((std::is_same_v<T, float> || std::is_same_v<T, half>) && need_check_recall) {
    auto stream_ = raft::resource::get_cuda_stream(handle_);
    cuvs::core::bitset<std::uint32_t, int64_t> bitset_filter(handle_, index_->data().n_rows(), false);

    std::vector<uint32_t> bitset_cpu(bitset_filter.n_elements());

    create_sparse_bitset(bitset_filter.size(), sparsity_, bitset_cpu);
    raft::copy(bitset_filter.data(), bitset_cpu.data(), bitset_filter.n_elements(), stream_);

    auto filter =
      std::make_optional(cuvs::neighbors::filtering::bitset_filter(bitset_filter.view()));
    cuvs::neighbors::cagra::search(
      handle_, search_params_, *index_, queries_view, neighbors_view, distances_view, filter, 1.000);

    if(need_check_recall) {
      assert(((index_->data().n_rows() * 1.0) * (1.0 - sparsity_) < k));
      std::vector<IdxT> actual_idx_cpu(neighbors_view.size());
      std::vector<IdxT> expected_idx_cpu(neighbors_view.size());

      raft::copy(actual_idx_cpu.data(), neighbors_view.data_handle(), neighbors_view.size(), stream_);

      cuvs::neighbors::cagra::search(
        handle_, search_params_, *index_, queries_view, neighbors_brute_force_view, distances_view, filter, 0.0000);

      raft::copy(expected_idx_cpu.data(), neighbors_brute_force_view.data_handle(), neighbors_brute_force_view.size(), stream_);
      raft::resource::sync_stream(handle_, stream_);

      auto [actual_recall, match_count, total_count] =
        cagra_calc_recall(expected_idx_cpu, actual_idx_cpu, batch_size, k);
      std::cout << "actual recall rate: " << actual_recall
                << ", match_count: " << match_count
                << ", total_count: " << total_count
                << std::endl;
    }
  } else {
    cuvs::neighbors::cagra::search(
      handle_, search_params_, *index_, queries_view, neighbors_view, distances_view);
  }

  if constexpr (sizeof(IdxT) != sizeof(algo_base::index_type)) {
    raft::linalg::unaryOp(neighbors,
                          neighbors_idx_t,
                          batch_size * k,
                          raft::cast_op<algo_base::index_type>(),
                          raft::resource::get_cuda_stream(handle_));
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);
  const raft::resources& res    = handle_;

  if (disable_refinement) {
    search_base(queries, batch_size, k, neighbors, distances);
  } else {
    auto queries_v = raft::make_device_matrix_view<const T, algo_base::index_type>(
      queries, batch_size, dimension_);
    auto candidate_ixs =
      raft::make_device_matrix<algo_base::index_type, algo_base::index_type>(res, batch_size, k0);
    auto candidate_dists =
      raft::make_device_matrix<float, algo_base::index_type>(res, batch_size, k0);
    search_base(
      queries, batch_size, k0, candidate_ixs.data_handle(), candidate_dists.data_handle());
    refine_helper(
      res, *input_dataset_v_, queries_v, candidate_ixs, k, neighbors, distances, index_->metric());
  }
}
}  // namespace cuvs::bench
