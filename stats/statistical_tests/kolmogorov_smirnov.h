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

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "stats/distributions/multivariate/multivariate.h"
#include "stats/distributions/probability_distribution.h"
#include "stats/distributions/univariate/univariate.h"
#include <Eigen/Core>

namespace stats {
const double KS_95 =
    1.36;  // 95th percential of kolmogorov smirnov distribution.

const double KS_EPS = 1e-6;

inline double kolmogorov_smirnov_statistic(
    Eigen::ArrayXd rvs, ProbabilityDistribution<ContinuousSingleVariable>& p) {
  int N = rvs.size();
  std::sort(rvs.begin(), rvs.end());
  Eigen::ArrayXd expectedCDF = p.cdf(rvs);
  Eigen::ArrayXd gotCDF(N);
  std::iota(gotCDF.begin(), gotCDF.end(), 1);
  gotCDF /= N;
  return (expectedCDF - gotCDF).abs().maxCoeff();
}

inline double kolmogorov_smirnov_statistic(const Eigen::ArrayXd rvs,
                                           const Eigen::ArrayXd cdf) {
  int N = rvs.size();
  std::vector<uint32_t> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&rvs](const int a, const int b) { return rvs[a] < rvs[b]; });
  double maxDiff = 0.0;
  for (int i = 0; i < N; i++)
    maxDiff = std::max(maxDiff, std::abs(cdf[idx[i]] - (i + 1) / (double)N));
  return maxDiff;
}

inline double kolmogorov_smirnov_statistic(const Eigen::ArrayXXd& rvs,
                                           const Eigen::ArrayXXd& cdf) {
  int N = rvs.rows();
  double maxDiff = 0.0;
  for (int i = 0; i < N; i++) {
    int leq = 0;
    for (int j = 0; j < N; j++) {
      leq += (rvs.row(j) <= rvs.row(i) + KS_EPS).all();
    }
    double empirical_cdf = leq / static_cast<double>(N);
    maxDiff = std::max(maxDiff, std::abs(cdf.row(i).prod() - empirical_cdf));
  }
  return maxDiff;
}
}  // namespace stats

#endif  // THIRD_PARTY_HYBRID_RCC_STATS_STATISTICAL_TESTS_KOLMOGOROV_SMIRNOV_H_
