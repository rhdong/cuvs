/*
 * Benchmark for cuvs::neighbors::cagra::merge API
 */
#include "cuvs_cagra_merge_wrapper.h"
#include "cuvs_cagra_wrapper.h"
#include "cuvs_ann_bench_param_parser.h"
#include "../common/ann_types.hpp"

namespace cuvs::bench {

template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename cuvs::bench::cuvs_cagra_merge<T, IdxT>::build_param& param)
{
  typename cuvs::bench::cuvs_cagra<T, IdxT>::build_param tmp;
  cuvs::bench::parse_build_param<T, IdxT>(conf, tmp);
  param.index_params = tmp.cagra_params;
  if (conf.contains("splits")) { param.splits = conf.at("splits"); }
  if (conf.contains("merge_strategy")) {
    std::string s = conf.at("merge_strategy");
    if (s == "PHYSICAL" || s == "physical") {
      param.strategy = cuvs::neighbors::cagra::MergeStrategy::MERGE_STRATEGY_PHYSICAL;
    } else if (s == "LOGICAL" || s == "logical") {
      param.strategy = cuvs::neighbors::cagra::MergeStrategy::MERGE_STRATEGY_LOGICAL;
    }
  }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuvs_cagra_merge<T, IdxT>::search_param& param)
{
  cuvs::bench::parse_search_param<T, IdxT>(conf, param);
}

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> ||
                std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
    if (algo_name == "cuvs_cagra_merge") {
      typename cuvs::bench::cuvs_cagra_merge<T, uint32_t>::build_param param;
      cuvs::bench::parse_build_param<T, uint32_t>(conf, param);
      a = std::make_unique<cuvs::bench::cuvs_cagra_merge<T, uint32_t>>(metric, dim, param);
    }
  }
  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }
  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if (algo_name == "cuvs_cagra_merge") {
    auto param = std::make_unique<typename cuvs::bench::cuvs_cagra_merge<T, uint32_t>::search_param>();
    cuvs::bench::parse_search_param<T, uint32_t>(conf, *param);
    return param;
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

