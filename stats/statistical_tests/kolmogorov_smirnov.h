/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_HYBRID_RCC_STATS_STATISTICAL_TESTS_KOLMOGOROV_SMIRNOV_H_
#define THIRD_PARTY_HYBRID_RCC_STATS_STATISTICAL_TESTS_KOLMOGOROV_SMIRNOV_H_

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>

#include "stats/distributions/probability_distribution.h"
#include "stats/distributions/univariate/univariate.h"

namespace py = pybind11;

namespace stats {
const double KS_95 =
    1.36;  // 95th percential of kolmogorov smirnov distribution.

const double KS_EPS = 1e-6;

double kolmogorov_smirnov_statistic(
    Eigen::ArrayXd rvs, ProbabilityDistribution<ContinuousSingleVariable>& p);

double kolmogorov_smirnov_statistic(const Eigen::ArrayXd rvs,
                                    const Eigen::ArrayXd cdf);

double kolmogorov_smirnov_statistic(const Eigen::ArrayXXd& rvs,
                                    const Eigen::ArrayXXd& cdf);
}  // namespace stats

PYBIND11_MODULE(hybrid_rcc, m) {
  py::module stats = m.def_submodule("stats", "statistics submodule.");
  py::module tests =
      stats.def_submodule("tests", "statistical tests submodule.");
  tests.def("kolmogorov_smirnov_statistic",
            py::overload_cast<const Eigen::ArrayXXd&, const Eigen::ArrayXXd&>(
                &stats::kolmogorov_smirnov_statistic),
            "Computes the Kolmogorov-Smirnov statistic");
  tests.def("kolmogorov_smirnov_statistic",
            py::overload_cast<const Eigen::ArrayXd, const Eigen::ArrayXd>(
                &stats::kolmogorov_smirnov_statistic),
            "Computes the Kolmogorov-Smirnov statistic");
}

#endif  // THIRD_PARTY_HYBRID_RCC_STATS_STATISTICAL_TESTS_KOLMOGOROV_SMIRNOV_H_
