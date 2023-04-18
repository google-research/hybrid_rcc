#ifndef THIRD_PARTY_HYBRID_RCC_SRC_PY_INTERFACE_H_
#define THIRD_PARTY_HYBRID_RCC_SRC_PY_INTERFACE_H_

#include <cstdint>
#include <sstream>

#include "Eigen/Core"
#include "algorithm/helper.h"
#include "pybind11/detail/common.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace rcc::interface {
using VecType = Eigen::ArrayXd;

enum class SamplingAlgorithm { PFR, SIS };

class SamplingOutput {
 public:
  SamplingOutput(VecType z, int n, int i, int seed, VecType k, VecType M) {
    box_dimensions_ = M;
    sample_opt_ = z;
    signal_ = k;
    sample_index_ = n;
    total_number_samples_ = i;
    seed_ = seed;
  }
  SamplingOutput(VecType z, int n, int i, int seed) {
    sample_opt_ = z;
    sample_index_ = n;
    total_number_samples_ = i;
    seed_ = seed;
  }
  std::string ToString() const {
    std::stringstream ss;
    ss << "SamplingOutput("
       << "box_dimensions="
       << box_dimensions_.transpose().format(eigen_format()) << ", "
       << "sample_opt=" << sample_opt_.transpose().format(eigen_format())
       << ", "
       << "signal=" << signal_.transpose().format(eigen_format()) << ", "
       << "sample_index=" << sample_index_ << ", "
       << "total_number_samples=" << total_number_samples_ << ", "
       << "seed=" << seed_ << ")";
    return ss.str();
  }
  VecType box_dimensions_, sample_opt_, signal_;
  int sample_index_, total_number_samples_, seed_;
};

SamplingOutput sample_gaussian_hybrid(VecType q_mean, VecType q_std,
                                      VecType p_mean, VecType p_std,
                                      SamplingAlgorithm sampling_algorithm,
                                      double eps, uint64_t seed, uint32_t N_max,
                                      bool verbose);

SamplingOutput sample_gaussian(VecType q_mean, VecType q_std, VecType p_mean,
                               VecType p_std,
                               SamplingAlgorithm sampling_algorithm,
                               uint64_t seed, uint32_t N_max, bool verbose);

VecType decode_gaussian_hybrid(SamplingOutput h, VecType p_mean, VecType p_std);
}  // namespace rcc::interface

void AddModules(pybind11::module &m);
#endif  // THIRD_PARTY_HYBRID_RCC_SRC_PY_INTERFACE_H_
