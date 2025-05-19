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
#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_param_parser.h"
#include "cuvs_cagra_merge_wrapper.h"

namespace cuvs::bench {

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  [[maybe_unused]] cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if (algo_name == "cuvs_cagra_merge") {
    typename cuvs::bench::cuvs_cagra_merge<T, uint32_t>::build_param param;
    parse_build_param<T, uint32_t>(conf, param);
    a = std::make_unique<cuvs::bench::cuvs_cagra_merge<T, uint32_t>>(metric, dim, param);
  }

  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  (void)conf;
  if (algo_name == "cuvs_cagra_merge") {
    return std::make_unique<typename cuvs::bench::cuvs_cagra_merge<T, uint32_t>::search_param>();
  }

  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

}  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(half);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
