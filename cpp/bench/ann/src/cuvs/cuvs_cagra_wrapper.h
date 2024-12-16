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
#include <cuvs/neighbors/dynamic_batching.hpp>
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

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>

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
    /* Dynamic batching */
    bool dynamic_batching = false;
    int64_t dynamic_batching_k;
    int64_t dynamic_batching_max_batch_size     = 4;
    double dynamic_batching_dispatch_timeout_ms = 0.01;
    size_t dynamic_batching_n_queues            = 8;
    bool dynamic_batching_conservative_dispatch = false;
  };

  struct build_param {
    cuvs::neighbors::cagra::index_params cagra_params;
    CagraBuildAlgo algo;
    std::optional<cuvs::neighbors::nn_descent::index_params> nn_descent_params = std::nullopt;
    std::optional<float> ivf_pq_refine_rate                                    = std::nullopt;
    std::optional<cuvs::neighbors::ivf_pq::index_params> ivf_pq_build_params   = std::nullopt;
    std::optional<cuvs::neighbors::ivf_pq::search_params> ivf_pq_search_params = std::nullopt;

    void prepare_build_params(const raft::extent_2d<IdxT>& dataset_extents)
    {
      if (algo == CagraBuildAlgo::kIvfPq) {
        auto pq_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
          dataset_extents, cagra_params.metric);
        if (ivf_pq_build_params) { pq_params.build_params = *ivf_pq_build_params; }
        if (ivf_pq_search_params) { pq_params.search_params = *ivf_pq_search_params; }
        if (ivf_pq_refine_rate) { pq_params.refinement_rate = *ivf_pq_refine_rate; }
        cagra_params.graph_build_params = pq_params;
      } else if (algo == CagraBuildAlgo::kNnDescent) {
        auto nn_params = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
          cagra_params.intermediate_graph_degree);
        if (nn_descent_params) { nn_params = *nn_descent_params; }
        cagra_params.graph_build_params = nn_params;
      }
    }
  };

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
          nullptr, 0, 0))

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
                   float* distances,
                   IdxT* neighbors_idx_t) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  [[nodiscard]] auto uses_stream() const noexcept -> bool override
  {
    // If the algorithm uses persistent kernel, the CPU has to synchronize by the end of computing
    // the result. Hence it guarantees the benchmark CUDA stream is empty by the end of the
    // execution. Hence we inform the benchmark to not waste the time on recording & synchronizing
    // the event.
    return !search_params_.persistent;
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

  auto get_index() const -> const cuvs::neighbors::cagra::index<T, IdxT>* { return index_.get(); }

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
  std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>> index_B_;
  int dimension_;
  std::shared_ptr<raft::device_matrix<IdxT, int64_t, raft::row_major>> graph_;
  std::shared_ptr<raft::device_matrix<T, int64_t, raft::row_major>> dataset_;
  std::shared_ptr<raft::device_matrix_view<const T, int64_t, raft::row_major>> input_dataset_v_;

  std::shared_ptr<cuvs::neighbors::dynamic_batching::index<T, IdxT>> dynamic_batcher_;
  cuvs::neighbors::dynamic_batching::search_params dynamic_batcher_sp_{};
  int64_t dynamic_batching_max_batch_size_;
  size_t dynamic_batching_n_queues_;
  bool dynamic_batching_conservative_dispatch_;

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
  constexpr float extend_ratio = 0.5f;
  auto stream                  = raft::resource::get_cuda_stream(handle_);

  auto A_extents = raft::make_extents<IdxT>(IdxT(nrow / 2), dimension_);
  auto B_extents = raft::make_extents<IdxT>(nrow - IdxT(nrow / 2), dimension_);

  auto addtl_A_extents =
    raft::make_extents<int64_t>(IdxT(A_extents.extent(0) * extend_ratio), dimension_);
  auto addtl_B_extents =
    raft::make_extents<int64_t>(IdxT(B_extents.extent(0) * extend_ratio), dimension_);

  auto A_offset = IdxT(0);
  auto B_offset = A_extents.extent(0) * dimension_;

  auto addtl_A_offset = (A_extents.extent(0) - addtl_A_extents.extent(0)) * dimension_;
  auto addtl_B_offset = (nrow - addtl_B_extents.extent(0)) * dimension_;

  index_params_.prepare_build_params(A_extents);
  auto& params = index_params_.cagra_params;

  // prep A
  auto A_view_h =
    raft::make_mdspan<const T, IdxT, raft::row_major, true, false>(dataset + A_offset, A_extents);
  auto A_view_d =
    raft::make_mdspan<const T, IdxT, raft::row_major, false, true>(dataset + A_offset, A_extents);

  bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;

  std::cout << "dataset_is_on_host: " << dataset_is_on_host << std::endl;

  index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(
    std::move(dataset_is_on_host ? cuvs::neighbors::cagra::build(handle_, params, A_view_h)
                                 : cuvs::neighbors::cagra::build(handle_, params, A_view_d)));

  // prep B
  index_params_.prepare_build_params(B_extents);

  auto B_view_h =
    raft::make_mdspan<const T, IdxT, raft::row_major, true, false>(dataset + B_offset, B_extents);
  auto B_view_d =
    raft::make_mdspan<const T, IdxT, raft::row_major, false, true>(dataset + B_offset, B_extents);

  index_B_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(
    std::move(dataset_is_on_host ? cuvs::neighbors::cagra::build(handle_, params, B_view_h)
                                 : cuvs::neighbors::cagra::build(handle_, params, B_view_d)));

  cuvs::neighbors::cagra::extend_params extend_params;
  extend_params.max_chunk_size = 1024;

  // insert 20% of B to A
  {
    // prep additional matrix
    auto target_index = index_.get();

    auto addtl_extents = addtl_B_extents;
    auto addtl_offset  = addtl_B_offset;

    auto addtl_matrix = raft::make_device_matrix<T, int64_t>(
      handle_, addtl_extents.extent(0), addtl_extents.extent(1));

    raft::copy(addtl_matrix.data_handle(),
               static_cast<const T*>(dataset) + addtl_offset,
               addtl_matrix.size(),
               stream);

    std::cout << "A target_index->size before extend: " << index_.get()->size() << std::endl;
    if constexpr (std::is_same_v<float, T> && extend_ratio > 0.0f) {
      std::cout << "extend start A + 20% of B" << std::endl;
      cuvs::neighbors::cagra::extend(
        handle_, extend_params, raft::make_const_mdspan(addtl_matrix.view()), *target_index);

      raft::resource::sync_stream(handle_);
      std::cout << "extend end A + 20% of B" << std::endl;
    }
    std::cout << "A target_index->size: " << index_.get()->size() << std::endl;
  }

  // insert 20% of A to B
  {
    // prep additional matrix
    auto target_index = index_B_.get();

    auto addtl_extents = addtl_A_extents;
    auto addtl_offset  = addtl_A_offset;

    auto addtl_matrix = raft::make_device_matrix<T, int64_t>(
      handle_, addtl_extents.extent(0), addtl_extents.extent(1));

    raft::copy(addtl_matrix.data_handle(),
               static_cast<const T*>(dataset) + addtl_offset,
               addtl_matrix.size(),
               stream);

    if constexpr (std::is_same_v<float, T> && extend_ratio > 0.0f) {
      std::cout << "extend start B + 20% of A" << std::endl;
      cuvs::neighbors::cagra::extend(
        handle_, extend_params, raft::make_const_mdspan(addtl_matrix.view()), *target_index);

      raft::resource::sync_stream(handle_);
      std::cout << "extend end B + 20% of A" << std::endl;
    }
    std::cout << "B target_index->size: " << index_B_.get()->size() << std::endl;
  }

  raft::resource::sync_stream(handle_);
  size_t A_graph_size = index_.get()->size() * index_.get()->graph_degree();
  size_t B_graph_size = index_B_.get()->size() * index_B_.get()->graph_degree();
  std::cout << "A_graph_size: " << A_graph_size << std::endl;
  std::cout << "B_graph_size: " << B_graph_size << std::endl;

  std::cout << std::endl;
  std::cout << "index_.get()->graph_degree(): " << index_.get()->graph_degree() << std::endl;

  std::vector<IdxT> A_graph_h(A_graph_size);
  std::vector<IdxT> B_graph_h(B_graph_size);

  {
    raft::copy(A_graph_h.data(), index_.get()->graph().data_handle(), A_graph_size, stream);
    raft::resource::sync_stream(handle_);

    raft::copy(B_graph_h.data(), index_B_.get()->graph().data_handle(), B_graph_size, stream);
    raft::resource::sync_stream(handle_);

    size_t counter             = 0;
    auto addtl_B_adjust_offset = nrow - addtl_B_extents.extent(0) - A_extents.extent(0);

    std::cout << "nrow: " << nrow << std::endl;
    std::cout << "addtl_B_adjust_offset: " << addtl_B_adjust_offset << std::endl;

    for (IdxT& n : A_graph_h) {
      if (n >= A_extents.extent(0)) { n += addtl_B_adjust_offset; }
    }
    std::cout << "A counter: " << counter << std::endl;

    counter                    = 0;
    auto addtl_A_adjust_offset = 0;

    for (IdxT& n : B_graph_h) {
      if (n < B_extents.extent(0)) { counter++; }

      addtl_A_adjust_offset =
        (n < B_extents.extent(0))
          ? A_extents.extent(0)
          : (A_extents.extent(0) - addtl_A_extents.extent(0) - B_extents.extent(0));
      n += addtl_A_adjust_offset;
    }
    std::cout << "B counter: " << counter << std::endl;

    std::cout << "addtl_A_adjust_offset1: " << A_extents.extent(0) << std::endl;
    std::cout << "addtl_A_adjust_offset2: "
              << A_extents.extent(0) - addtl_A_extents.extent(0) - B_extents.extent(0) << std::endl;
  }

  // zip merge
  size_t merged_graph_size = nrow * index_.get()->graph_degree();
  std::vector<IdxT> merged_graph_h(merged_graph_size);
  {
    raft::copy(merged_graph_h.data(),
               A_graph_h.data(),
               (A_extents.extent(0) - addtl_A_extents.extent(0)) * index_.get()->graph_degree(),
               stream);

    std::unordered_set<IdxT> temp_set(index_.get()->graph_degree());

    for (int64_t r = 0; r < addtl_A_extents.extent(0); r++) {
      IdxT* A = A_graph_h.data() + (A_extents.extent(0) - addtl_A_extents.extent(0) + r) *
                                     index_.get()->graph_degree();
      IdxT* B = B_graph_h.data() + (B_extents.extent(0) + r) * index_.get()->graph_degree();

      IdxT* merged = merged_graph_h.data() + (A_extents.extent(0) - addtl_A_extents.extent(0) + r) *
                                               index_.get()->graph_degree();

      size_t counter = 0;
      temp_set.clear();
      for (size_t n = 0; n < index_.get()->graph_degree(); n++) {
        IdxT candi_A = A[n];
        IdxT candi_B = B[n];
        if (temp_set.find(candi_A) == temp_set.end()) {
          merged[counter] = candi_A;
          counter++;
          temp_set.insert(candi_A);
          if (counter >= index_.get()->graph_degree()) break;
        }
        if (temp_set.find(candi_B) == temp_set.end()) {
          merged[counter] = candi_B;
          counter++;
          temp_set.insert(candi_B);
          if (counter >= index_.get()->graph_degree()) break;
        }
      }
    }

    raft::copy(merged_graph_h.data() + A_extents.extent(0) * index_.get()->graph_degree(),
               B_graph_h.data(),
               (B_extents.extent(0) - addtl_B_extents.extent(0)) * index_.get()->graph_degree(),
               stream);

    for (int64_t r = 0; r < addtl_B_extents.extent(0); r++) {
      IdxT* B = B_graph_h.data() + (B_extents.extent(0) - addtl_B_extents.extent(0) + r) *
                                     index_.get()->graph_degree();
      IdxT* A = A_graph_h.data() + (A_extents.extent(0) + r) * index_.get()->graph_degree();

      IdxT* merged = merged_graph_h.data() +
                     (nrow - addtl_B_extents.extent(0) + r) * index_.get()->graph_degree();

      size_t counter = 0;
      temp_set.clear();
      for (size_t n = 0; n < index_.get()->graph_degree(); n++) {
        IdxT candi_A = A[n];
        IdxT candi_B = B[n];
        if (temp_set.find(candi_B) == temp_set.end()) {
          merged[counter] = candi_B;
          counter++;
          temp_set.insert(candi_B);
          if (counter >= index_.get()->graph_degree()) break;
        }
        if (temp_set.find(candi_A) == temp_set.end()) {
          merged[counter] = candi_A;
          counter++;
          temp_set.insert(candi_A);
          if (counter >= index_.get()->graph_degree()) break;
        }
      }
    }
  }

  // update dataset and graph
  if constexpr (std::is_same_v<float, T>) {
    const auto degree    = index_.get()->graph_degree();
    auto dataset_extents = raft::make_extents<int64_t>(nrow, dimension_);

    auto dataset_view_host =
      raft::make_mdspan<const T, int64_t, raft::row_major, false, true>(dataset, dataset_extents);
    index_.get()->update_dataset(handle_, dataset_view_host);

    // update graph
    auto updated_graph = raft::make_host_matrix<IdxT, std::int64_t>(nrow, degree);
    assert(updated_graph.size() == merged_graph_h.size());
    raft::copy(updated_graph.data_handle(),
               merged_graph_h.data(),
               merged_graph_h.size(),
               raft::resource::get_cuda_stream(handle_));

    index_.get()->update_graph(handle_, raft::make_const_mdspan(updated_graph.view()));
    std::cout << "lase dataset().size(): " << index_.get()->dataset().size() << std::endl;
    std::cout << "lase graph()->size() " << index_.get()->size() << std::endl;
  }
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
  auto sp = dynamic_cast<const search_param&>(param);
  bool needs_dynamic_batcher_update =
    (dynamic_batching_max_batch_size_ != sp.dynamic_batching_max_batch_size) ||
    (dynamic_batching_n_queues_ != sp.dynamic_batching_n_queues) ||
    (dynamic_batching_conservative_dispatch_ != sp.dynamic_batching_conservative_dispatch);
  dynamic_batching_max_batch_size_        = sp.dynamic_batching_max_batch_size;
  dynamic_batching_n_queues_              = sp.dynamic_batching_n_queues;
  dynamic_batching_conservative_dispatch_ = sp.dynamic_batching_conservative_dispatch;
  search_params_                          = sp.p;
  refine_ratio_                           = sp.refine_ratio;
  if (sp.graph_mem != graph_mem_) {
    // Move graph to correct memory space
    graph_mem_ = sp.graph_mem;
    RAFT_LOG_DEBUG("moving graph to new memory space: %s", allocator_to_string(graph_mem_).c_str());
    // We create a new graph and copy to it from existing graph
    auto mr = get_mr(graph_mem_);
    *graph_ = raft::make_device_mdarray<IdxT, int64_t>(
      handle_, mr, raft::make_extents<int64_t>(index_->graph().extent(0), index_->graph_degree()));

    raft::copy(graph_->data_handle(),
               index_->graph().data_handle(),
               index_->graph().size(),
               raft::resource::get_cuda_stream(handle_));

    // NB: update_graph() only stores a view in the index. We need to keep the graph object alive.
    index_->update_graph(handle_, make_const_mdspan(graph_->view()));
    needs_dynamic_batcher_update = true;
  }

  if (sp.dataset_mem != dataset_mem_ || need_dataset_update_) {
    dataset_mem_ = sp.dataset_mem;

    // First free up existing memory
    *dataset_ = raft::make_device_matrix<T, int64_t>(handle_, 0, 0);
    index_->update_dataset(handle_, make_const_mdspan(dataset_->view()));

    // Allocate space using the correct memory resource.
    RAFT_LOG_DEBUG("moving dataset to new memory space: %s",
                   allocator_to_string(dataset_mem_).c_str());

    auto mr = get_mr(dataset_mem_);
    cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);

    auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
    index_->update_dataset(handle_, dataset_view);

    need_dataset_update_         = false;
    needs_dynamic_batcher_update = true;
  }

  // dynamic batching
  if (sp.dynamic_batching) {
    if (!dynamic_batcher_ || needs_dynamic_batcher_update) {
      dynamic_batcher_ = std::make_shared<cuvs::neighbors::dynamic_batching::index<T, IdxT>>(
        handle_,
        cuvs::neighbors::dynamic_batching::index_params{{},
                                                        sp.dynamic_batching_k,
                                                        sp.dynamic_batching_max_batch_size,
                                                        sp.dynamic_batching_n_queues,
                                                        sp.dynamic_batching_conservative_dispatch},
        *index_,
        search_params_);
    }
    dynamic_batcher_sp_.dispatch_timeout_ms = sp.dynamic_batching_dispatch_timeout_ms;
  } else {
    if (dynamic_batcher_) { dynamic_batcher_.reset(); }
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
  using ds_idx_type = decltype(index_->data().n_rows());
  bool is_vpq =
    dynamic_cast<const cuvs::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
    dynamic_cast<const cuvs::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
  cuvs::neighbors::cagra::serialize(handle_, file, *index_, is_vpq);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::save_to_hnswlib(const std::string& file) const
{
  if constexpr (!std::is_same_v<T, half>) {
    cuvs::neighbors::cagra::serialize_to_hnswlib(handle_, file, *index_);
  } else {
    RAFT_FAIL("Cannot save fp16 index to hnswlib format");
  }
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
  return std::make_unique<cuvs_cagra<T, IdxT>>(std::cref(*this));  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search_base(const T* queries,
                                      int batch_size,
                                      int k,
                                      algo_base::index_type* neighbors,
                                      float* distances,
                                      IdxT* neighbors_idx_t) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  if constexpr (sizeof(IdxT) == sizeof(algo_base::index_type)) {
    neighbors_idx_t = reinterpret_cast<IdxT*>(neighbors);
  }

  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, dimension_);
  auto neighbors_view =
    raft::make_device_matrix_view<IdxT, int64_t>(neighbors_idx_t, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);

  if (dynamic_batcher_) {
    cuvs::neighbors::dynamic_batching::search(handle_,
                                              dynamic_batcher_sp_,
                                              *dynamic_batcher_,
                                              queries_view,
                                              neighbors_view,
                                              distances_view);
  } else {
    cuvs::neighbors::cagra::search(
      handle_, search_params_, *index_, queries_view, neighbors_view, distances_view);
  }

  if constexpr (sizeof(IdxT) != sizeof(algo_base::index_type)) {
    if (raft::get_device_for_address(neighbors) < 0 &&
        raft::get_device_for_address(neighbors_idx_t) < 0) {
      // Both pointers on the host, let's use host-side mapping
      if (uses_stream()) {
        // Need to wait for GPU to finish filling source
        raft::resource::sync_stream(handle_);
      }
      for (int i = 0; i < batch_size * k; i++) {
        neighbors[i] = algo_base::index_type(neighbors_idx_t[i]);
      }
    } else {
      raft::linalg::unaryOp(neighbors,
                            neighbors_idx_t,
                            batch_size * k,
                            raft::cast_op<algo_base::index_type>(),
                            raft::resource::get_cuda_stream(handle_));
    }
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);
  constexpr bool kNeedsIoMapping = sizeof(IdxT) != sizeof(algo_base::index_type);

  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);
  const raft::resources& res    = handle_;
  auto mem_type =
    raft::get_device_for_address(neighbors) >= 0 ? MemoryType::kDevice : MemoryType::kHostPinned;

  // If dynamic batching is used and there's no sync between benchmark laps, multiple sequential
  // requests can group together. The data is copied asynchronously, and if the same intermediate
  // buffer is used for multiple requests, they can override each other's data. Hence, we need to
  // allocate as much space as required by the maximum number of sequential requests.
  auto max_dyn_grouping = dynamic_batcher_ ? raft::div_rounding_up_safe<int64_t>(
                                               dynamic_batching_max_batch_size_, batch_size) *
                                               dynamic_batching_n_queues_
                                           : 1;
  auto tmp_buf_size = ((disable_refinement ? 0 : (sizeof(float) + sizeof(algo_base::index_type))) +
                       (kNeedsIoMapping ? sizeof(IdxT) : 0)) *
                      batch_size * k0;
  auto& tmp_buf = get_tmp_buffer_from_global_pool(tmp_buf_size * max_dyn_grouping);
  thread_local static int64_t group_id = 0;
  auto* candidates_ptr                 = reinterpret_cast<algo_base::index_type*>(
    reinterpret_cast<uint8_t*>(tmp_buf.data(mem_type)) + tmp_buf_size * group_id);
  group_id = (group_id + 1) % max_dyn_grouping;
  auto* candidate_dists_ptr =
    reinterpret_cast<float*>(candidates_ptr + (disable_refinement ? 0 : batch_size * k0));
  auto* neighbors_idx_t =
    reinterpret_cast<IdxT*>(candidate_dists_ptr + (disable_refinement ? 0 : batch_size * k0));

  if (disable_refinement) {
    search_base(queries, batch_size, k, neighbors, distances, neighbors_idx_t);
  } else {
    search_base(queries, batch_size, k0, candidates_ptr, candidate_dists_ptr, neighbors_idx_t);

    if (mem_type == MemoryType::kHostPinned && uses_stream()) {
      // If the algorithm uses a stream to synchronize (non-persistent kernel), but the data is in
      // the pinned host memory, we need to synchronize before the refinement operation to wait for
      // the data being available for the host.
      raft::resource::sync_stream(res);
    }

    auto candidate_ixs =
      raft::make_device_matrix_view<const algo_base::index_type, algo_base::index_type>(
        candidates_ptr, batch_size, k0);
    auto queries_v = raft::make_device_matrix_view<const T, algo_base::index_type>(
      queries, batch_size, dimension_);
    refine_helper(
      res, *input_dataset_v_, queries_v, candidate_ixs, k, neighbors, distances, index_->metric());
  }
}
}  // namespace cuvs::bench