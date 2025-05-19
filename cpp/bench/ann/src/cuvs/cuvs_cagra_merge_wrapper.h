#pragma once

#include "cuvs_ann_bench_utils.h"
#include "../common/ann_types.hpp"
#include <cuvs/neighbors/cagra.hpp>
#include <memory>
#include <vector>

namespace cuvs::bench {

template <typename T, typename IdxT>
class cuvs_cagra_merge : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  struct build_param {
    cuvs::neighbors::cagra::index_params index_params{};
    uint32_t splits{2};
    cuvs::neighbors::cagra::MergeStrategy strategy{
      cuvs::neighbors::cagra::MergeStrategy::MERGE_STRATEGY_PHYSICAL};
  };
  struct search_param : public search_param_base {
    cuvs::neighbors::cagra::search_params params{};
  };

  cuvs_cagra_merge(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim), build_param_(param)
  {
    build_param_.index_params.metric = parse_metric_type(metric);
  }

  void build(const T* dataset, size_t nrow) final;
  void set_search_param(const search_param_base& param, const void* filter) override;
  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    return {MemoryType::kHost, MemoryType::kHost};
  }

  void save(const std::string& file) const override;
  void load(const std::string& file) override;
  std::unique_ptr<algo<T>> copy() override
  {
    return std::make_unique<cuvs_cagra_merge<T, IdxT>>(*this);
  }

 private:
  configured_raft_resources handle_{};
  build_param build_param_{};
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>> sub_indices_{};
  std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>> index_{};
  std::shared_ptr<cuvs::neighbors::cagra::composite_index<T, IdxT>> composite_{};
  cuvs::neighbors::cagra::search_params search_params_{};
};

// Implementation

template <typename T, typename IdxT>
void cuvs_cagra_merge<T, IdxT>::build(const T* dataset, size_t nrow)
{
  sub_indices_.clear();
  sub_indices_.reserve(build_param_.splits);
  size_t base = 0;
  size_t chunk = nrow / build_param_.splits;
  for (uint32_t i = 0; i < build_param_.splits; ++i) {
    size_t size = (i == build_param_.splits - 1) ? (nrow - base) : chunk;
    auto view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      dataset + base * this->dim_, size, this->dim_);
    sub_indices_.push_back(
      cuvs::neighbors::cagra::build(handle_, build_param_.index_params, view));
    base += size;
  }
  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> ptrs;
  ptrs.reserve(sub_indices_.size());
  for (auto& idx : sub_indices_) ptrs.push_back(&idx);
  cuvs::neighbors::cagra::merge_params mp{build_param_.index_params};
  mp.strategy = build_param_.strategy;
  if (build_param_.strategy == cuvs::neighbors::cagra::MergeStrategy::MERGE_STRATEGY_PHYSICAL) {
    index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(cuvs::neighbors::cagra::merge(handle_, mp, ptrs));
    composite_.reset();
  } else {
    composite_ = std::make_shared<cuvs::neighbors::cagra::composite_index<T, IdxT>>(cuvs::neighbors::cagra::make_composite_index(mp, ptrs));
    index_.reset();
  }
}

template <typename T, typename IdxT>
void cuvs_cagra_merge<T, IdxT>::set_search_param(const search_param_base& param, const void*)
{
  auto& sp = dynamic_cast<const search_param&>(param);
  search_params_ = sp.params;
}

template <typename T, typename IdxT>
void cuvs_cagra_merge<T, IdxT>::search(const T* queries,
                                       int batch_size,
                                       int k,
                                       algo_base::index_type* neighbors,
                                       float* distances) const
{
  auto qv = raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, this->dim_);
  auto nv = raft::make_device_matrix_view<IdxT, int64_t>((IdxT*)neighbors, batch_size, k);
  auto dv = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);
  if (index_) {
    cuvs::neighbors::cagra::search(handle_, search_params_, *index_, qv, nv, dv);
  } else if (composite_) {
    cuvs::neighbors::cagra::search(handle_, search_params_, *composite_, qv, nv, dv);
  }
}

template <typename T, typename IdxT>
void cuvs_cagra_merge<T, IdxT>::save(const std::string& file) const
{
  if (index_) { cuvs::neighbors::cagra::serialize(handle_, *index_, file); }
}

template <typename T, typename IdxT>
void cuvs_cagra_merge<T, IdxT>::load(const std::string& file)
{
  index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(handle_);
  cuvs::neighbors::cagra::deserialize(handle_, file, index_.get());
  composite_.reset();
}

}  // namespace cuvs::bench

