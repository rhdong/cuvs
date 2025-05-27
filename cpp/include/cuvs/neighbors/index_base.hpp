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

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <memory>
#include <vector>

namespace cuvs::neighbors {

/**
 * @brief Polymorphic index interface used by indexing algorithms.
 *
 * This interface provides a common abstraction layer for different ANN (Approximate Nearest
 * Neighbor) index implementations, enabling polymorphic usage across the library. It defines the
 * core operations that all index types must support, such as search, size querying, and metric type
 * access.
 *
 * Supported index types include:
 * - CAGRA (Clustering-based Approximate Graph Accelerator)
 * - IVF-PQ (Inverted File with Product Quantization)
 * - IVF-Flat (Inverted File with Flat Storage)
 * - HNSW (Hierarchical Navigable Small World)
 *
 * The interface is designed to:
 * - Enable polymorphic usage of different index types through a common interface
 * - Facilitate easy extension for new index implementations
 * - Support composite index patterns for distributed search scenarios
 * - Provide compatibility between new and existing implementations
 *
 * @tparam T Data element type (e.g., float, int8, uint8)
 * @tparam IdxT Index type for vector indices (must be able to represent dataset.extent(0))
 * @tparam OutputIdxT Output index type, defaults to IdxT
 */
template <typename T, typename IdxT, typename OutputIdxT = IdxT>
struct IndexBase {
  using value_type        = T;
  using index_type        = IdxT;
  using out_index_type    = OutputIdxT;
  using matrix_index_type = int64_t;

  virtual ~IndexBase() = default;

  /**
   * @brief Build interface for constructing index from dataset.
   *
   * This interface is designed to be extensible for future enhancements. Current and planned
   * features include:
   *
   * Current:
   * - Basic index construction from dataset
   * - Support for different distance metrics
   * - Configurable build parameters
   *
   * @param[in] handle CUDA resources for executing operations
   * @param[in] params Build parameters specific to the index implementation
   * @param[in] dataset Matrix of vectors to index [n_samples, dim]
   *
   * @note This interface is currently commented out and will be implemented in future versions.
   *       The documentation is kept as a design reference for the planned implementation.
   */
  /* Future implementation:
  virtual void build(
    const raft::resources& handle,
    const cuvs::neighbors::build_params& params,
    raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> dataset) = 0;
  */

  /**
   * @brief Perform approximate nearest neighbor search.
   *
   * Searches the index for the k-nearest neighbors of each query point.
   * The number of neighbors to find is determined by the neighbors matrix extent.
   *
   * @param[in] handle CUDA resources for executing operations
   * @param[in] params Search parameters specific to the index implementation
   * @param[in] queries Matrix of query vectors to search for [n_queries, dim]
   * @param[out] neighbors Matrix to store neighbor indices [n_queries, k]
   * @param[out] distances Matrix to store distances to neighbors [n_queries, k]
   * @param[in] filter Optional filter to exclude certain vectors from search results
   */
  virtual void search(
    const raft::resources& handle,
    const cuvs::neighbors::search_params& params,
    raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> queries,
    raft::device_matrix_view<out_index_type, matrix_index_type, raft::row_major> neighbors,
    raft::device_matrix_view<float, matrix_index_type, raft::row_major> distances,
    const cuvs::neighbors::filtering::base_filter& filter =
      cuvs::neighbors::filtering::none_sample_filter{}) const = 0;

  /**
   * @brief Get the number of vectors in the index.
   *
   * @return Number of indexed vectors
   */
  virtual index_type size() const noexcept = 0;

  /**
   * @brief Get the distance metric used by the index.
   *
   * @return Distance metric type (e.g., L2, InnerProduct)
   */
  virtual cuvs::distance::DistanceType metric() const noexcept = 0;
};

}  // namespace cuvs::neighbors
