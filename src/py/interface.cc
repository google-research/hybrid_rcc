#include "py/interface.h"

#include "algorithm/reverse_channel.h"
#include "stats/distributions/multivariate/continuous/gaussian.h"
#include "include/pcg_random.hpp"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"

namespace rcc::interface {
using stats::multivariates::IndependentGaussian;

SamplingOutput sample_gaussian_hybrid(VecType q_mean, VecType q_std,
                                      VecType p_mean, VecType p_std,
                                      SamplingAlgorithm sampling_algorithm,
                                      double eps, uint64_t seed, uint32_t N_max,
                                      bool verbose) {
  IndependentGaussian p(p_mean, p_std);
  IndependentGaussian q(q_mean, q_std);
  pcg32 rs(seed);
  auto [z, n, k, i, M] = rcc::algorithm::sample_gaussian_hybrid(
      &q, &p, sampling_algorithm == SamplingAlgorithm::PFR, eps, rs, N_max,
      verbose);
  return SamplingOutput(z, n, i, seed, k, M);
}

SamplingOutput sample_gaussian(VecType q_mean, VecType q_std, VecType p_mean,
                               VecType p_std,
                               SamplingAlgorithm sampling_algorithm,
                               uint64_t seed, uint32_t N_max, bool verbose) {
  IndependentGaussian p(p_mean, p_std);
  IndependentGaussian q(q_mean, q_std);
  pcg32 rs(seed);
  auto [z, n, i] = rcc::algorithm::sample_gaussian(
      &q, &p, sampling_algorithm == SamplingAlgorithm::PFR, rs, N_max, verbose);
  return SamplingOutput(z, n, i, seed);
}

VecType decode_gaussian_hybrid(SamplingOutput h, VecType p_mean,
                               VecType p_std) {
  IndependentGaussian p(p_mean, p_std);
  pcg32 rs(h.seed_);
  return rcc::algorithm::decode_hybrid(h.sample_index_, h.signal_,
                                       h.box_dimensions_, p, p_mean.size(), rs);
}
}  // namespace rcc::interface

namespace py = ::pybind11;
void AddModules(pybind11::module& m) {
  py::class_<rcc::interface::SamplingOutput>(m, "SamplingOutput")
      // Class properties.
      .def_readonly("sample_opt", &rcc::interface::SamplingOutput::sample_opt_)
      .def_readonly("seed", &rcc::interface::SamplingOutput::seed_)
      .def_readonly("sample_index",
                    &rcc::interface::SamplingOutput::sample_index_)
      .def_readonly("signal", &rcc::interface::SamplingOutput::signal_)
      .def_readonly("box_dimensions",
                    &rcc::interface::SamplingOutput::box_dimensions_)
      .def_readonly("total_number_samples",
                    &rcc::interface::SamplingOutput::total_number_samples_)
      // Constructors.
      .def(py::init([](rcc::interface::VecType optimal_sampe, int sample_index,
                       int total_number_sambles, int seed) {
        return rcc::interface::SamplingOutput(optimal_sampe, sample_index,
                                              total_number_sambles, seed);
      }))
      .def(py::init([](rcc::interface::VecType optimal_sampe, int sample_index,
                       int total_number_sambles, int seed,
                       rcc::interface::VecType signal,
                       rcc::interface::VecType box_dimensions) {
        return rcc::interface::SamplingOutput(optimal_sampe, sample_index,
                                              total_number_sambles, seed,
                                              signal, box_dimensions);
      }))
      .def(py::init<rcc::interface::SamplingOutput>())
      // String representation.
      .def("__str__", &rcc::interface::SamplingOutput::ToString);
  py::enum_<rcc::interface::SamplingAlgorithm>(m, "SamplingAlgorithm")
      .value("SIS", rcc::interface::SamplingAlgorithm::SIS)
      .value("PFR", rcc::interface::SamplingAlgorithm::PFR);
  m.def("decode_gaussian_hybrid", &rcc::interface::decode_gaussian_hybrid);
  m.def("sample_gaussian_hybrid", &rcc::interface::sample_gaussian_hybrid);
  m.def("sample_gaussian", &rcc::interface::sample_gaussian);
}