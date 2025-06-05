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

#include <cuvs/neighbors/index_base.hpp>
#include <cuvs/neighbors/index_wrappers.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <memory>

// Forward declarations to avoid circular dependencies
namespace cuvs::neighbors::cagra {
template <typename T, typename IdxT>
struct index;
struct merge_params;
}  // namespace cuvs::neighbors::cagra

namespace cuvs::neighbors::cagra {

/**
 * @brief Wrapper for CAGRA index implementing IndexWrapper.
 *
 * This class wraps a CAGRA index and provides compatibility with the IndexBase interface.
 * It serves as a bridge to help the CAGRA index implementation transition from its
 * original design to the new object-oriented polymorphic design based on IndexBase.
 *
 * The wrapper enables:
 * - CAGRA index to work seamlessly with the new polymorphic IndexBase interface
 * - Gradual migration from the original CAGRA API to the unified index architecture
 * - Compatibility with composite index patterns and other polymorphic usage scenarios
 * - Preservation of existing CAGRA functionality while adopting the new design patterns
 *
 * This allows existing CAGRA users to benefit from the new architecture without
 * requiring immediate changes to their existing code, while new users can adopt
 * the unified interface from the start.
 */
template <typename T, typename IdxT, typename OutputIdxT>
class IndexWrapper : public cuvs::neighbors::IndexWrapper<T, IdxT, OutputIdxT> {
 public:
  using base_type         = cuvs::neighbors::IndexWrapper<T, IdxT, OutputIdxT>;
  using value_type        = typename base_type::value_type;
  using index_type        = typename base_type::index_type;
  using out_index_type    = typename base_type::out_index_type;
  using matrix_index_type = typename base_type::matrix_index_type;

  explicit IndexWrapper(cuvs::neighbors::cagra::index<T, IdxT>* idx);
  IndexWrapper(cuvs::neighbors::cagra::index<T, IdxT>* idx, bool owns_index);

  ~IndexWrapper()
  {
    if (owns_index_ && index_) { delete index_; }
  }

  void search(
    const raft::resources& handle,
    const cuvs::neighbors::search_params& params,
    raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> queries,
    raft::device_matrix_view<out_index_type, matrix_index_type, raft::row_major> neighbors,
    raft::device_matrix_view<float, matrix_index_type, raft::row_major> distances,
    const cuvs::neighbors::filtering::base_filter& filter) const override;

  index_type size() const noexcept override;

  cuvs::distance::DistanceType metric() const noexcept override;

  /**
   * @brief Build the index from the dataset (device memory).
   *
   * This method builds a CAGRA index from device dataset using the specified parameters.
   * The generic index_params will be cast to cagra::index_params for CAGRA-specific building.
   *
   * @param[in] handle CUDA resources for executing operations
   * @param[in] params Build parameters (must be cagra::index_params)
   * @param[in] dataset Device matrix of vectors to index [n_rows, dim]
   */
  void build(const raft::resources& handle,
             const cuvs::neighbors::index_params& params,
             raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> dataset)
    override;

  /**
   * @brief Build the index from the dataset (host memory).
   *
   * This method builds a CAGRA index from host dataset using the specified parameters.
   * The generic index_params will be cast to cagra::index_params for CAGRA-specific building.
   *
   * @param[in] handle CUDA resources for executing operations
   * @param[in] params Build parameters (must be cagra::index_params)
   * @param[in] dataset Host matrix of vectors to index [n_rows, dim]
   */
  void build(
    const raft::resources& handle,
    const cuvs::neighbors::index_params& params,
    raft::host_matrix_view<const value_type, matrix_index_type, raft::row_major> dataset) override;

  /**
   * @brief Merge this index with other indices.
   *
   * This interface provides polymorphic merge capability for index types that support merging.
   * The merge strategy and parameters are determined by the specific merge_params implementation.
   * Not all index types need to support merging, so this has a default implementation that
   * throws an error.
   *
   * @param[in] handle RAFT resources for executing operations
   * @param[in] params Merge parameters containing strategy and algorithm-specific settings
   * @param[in] other_indices Vector of other indices to merge with this one
   * @return Shared pointer to merged index
   */
  std::shared_ptr<cuvs::neighbors::IndexBase<T, IdxT, OutputIdxT>> merge(
    const raft::resources& handle,
    const cuvs::neighbors::merge_params& params,
    const std::vector<std::shared_ptr<cuvs::neighbors::IndexBase<T, IdxT, OutputIdxT>>>&
      other_indices) const override;

 private:
  cuvs::neighbors::cagra::index<T, IdxT>* index_;
  bool owns_index_ = false;
};

/**
 * @brief Factory function for creating a wrapped CAGRA index.
 *
 * This function creates a shared pointer to an IndexWrapper that wraps a CAGRA index,
 * enabling it to work with the polymorphic IndexBase interface and composite operations.
 *
 * @tparam T Data type
 * @tparam IdxT Index type
 * @tparam OutputIdxT Output index type
 * @param index Pointer to the CAGRA index
 * @return Shared pointer to the wrapped index
 *
 * @par Example usage:
 * @code{.cpp}
 * // Create multiple CAGRA indices
 * auto cagra_index1 = cuvs::neighbors::cagra::build(res, params, dataset1);
 * auto cagra_index2 = cuvs::neighbors::cagra::build(res, params, dataset2);
 *
 * // Wrap them for polymorphic usage
 * auto wrapped_index1 = cuvs::neighbors::cagra::make_index_wrapper(&cagra_index1);
 * auto wrapped_index2 = cuvs::neighbors::cagra::make_index_wrapper(&cagra_index2);
 *
 * // Merge indices using the composite merge function
 * std::vector<std::shared_ptr<cuvs::neighbors::IndexWrapper<float, uint32_t>>> indices;
 * indices.push_back(wrapped_index1);
 * indices.push_back(wrapped_index2);
 *
 * cuvs::neighbors::cagra::merge_params merge_params;
 * auto merged_index = cuvs::neighbors::composite::merge(res, merge_params, indices);
 * @endcode
 */
template <typename T, typename IdxT, typename OutputIdxT = IdxT>
inline auto make_index_wrapper(cuvs::neighbors::cagra::index<T, IdxT>* index)
  -> std::shared_ptr<cuvs::neighbors::IndexWrapper<T, IdxT, OutputIdxT>>
{
  return std::make_shared<IndexWrapper<T, IdxT, OutputIdxT>>(index);
}

}  // namespace cuvs::neighbors::cagra
