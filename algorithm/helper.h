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

#ifndef THIRD_PARTY_HYBRID_RCC_ALGORITHM_HELPER_H_
#define THIRD_PARTY_HYBRID_RCC_ALGORITHM_HELPER_H_

#include <math.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>

#include "stats/distributions/multivariate/multivariate.h"
#include "stats/distributions/probability_distribution.h"

#ifdef __STDCPP_MATH_SPEC_FUNCS__
using std::riemann_zeta;
#else
namespace {
// this function is never actually used since we use std::riemann_zeta.
inline double riemann_zeta(double s) {
  const double eps = 1e-9;
  if (s > 1) {
    double ret = 0, term;
    int i = 1;
    do {
      term = std::pow(i, -s);
      ret += term;
      i++;
    } while (term > eps);
    return ret;
  }
  if (s >= 0) {
    double scaler = std::pow(2, 1 - s) - 1;
    double ret = 0, term;
    int i = 1;
    int sgn = -1;
    do {
      term = std::pow(i, -s) / scaler;
      ret += sgn * term;
      i += 1;
      sgn = -sgn;
    } while (term > eps);
    return ret;
  }
  return std::pow(2, s) * std::pow(M_PI, s - 1) * std::sin(M_PI * s / 2) *
         std::tgamma(1 - s) * riemann_zeta(1 - s);
}
}  // namespace
#endif


namespace rcc {
namespace internal {
inline double estimate_w(
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p,
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &q,
    std::unique_ptr<stats::RandomNumberGenerator> & rng, int N) {
  double w_min = std::numeric_limits<double>::infinity();
  for (int i = 0; i < N; i++) {
    auto u = p.rvs(rng);
    w_min = std::min(w_min, p.pdf(u).prod() / q.pdf(u).prod());
  }
  return w_min;
}

inline std::tuple<double, double, double> codingCostHyprid(
    const stats::ProbabilityDistribution<stats::ContinuousMultiVariable> *&q,
    const stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p,
    const Eigen::ArrayXd &M, int n) {
  static double log2 = std::log(2), e = std::exp(1);
  double marg_diff_entropy = p.entropy().sum() / log2;
  double cond_diff_entropy = q->entropy().sum() / log2;
  double mi = marg_diff_entropy - cond_diff_entropy;

  double log2M = M.log2().sum();
  double coding_cost_bound = mi + std::log2(mi - log2M + 1) + 4;

  double exponent = 1.0 + 1.0 / (1.0 + std::log2(e) / e + mi - log2M);
  double log2Pn = -exponent * std::log2(n + 1);
  log2Pn = log2Pn - std::log2(riemann_zeta(exponent));

  double coding_cost_zipf = -log2Pn + log2M;
  return std::tuple<double, double, double>(coding_cost_zipf, mi,
                                            coding_cost_bound);
}

inline std::tuple<double, double> codingCostBounds(
    const stats::ProbabilityDistribution<stats::ContinuousMultiVariable> *q,
    const stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p,
    const Eigen::ArrayXd &M) {
  static double log2 = std::log(2);
  double marg_diff_entropy = p.entropy().sum() / log2;
  double cond_diff_entropy = q->entropy().sum() / log2;
  double mi = marg_diff_entropy - cond_diff_entropy;

  double log2M = M.log2().sum();
  double coding_cost_bound = mi + std::log2(mi - log2M + 1) + 4;

  return std::tuple<double, double>(mi, coding_cost_bound);
}

inline double codingCost(
    const stats::ProbabilityDistribution<stats::ContinuousMultiVariable> *q,
    const stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p,
    double mi, double log2M, int n) {
  static double e = std::exp(1);
  double exponent = 1.0 + 1.0 / (1.0 + std::log2(e) / e + mi - log2M);
  double log2Pn = -exponent * std::log2(n + 1);
  log2Pn = log2Pn - riemann_zeta(exponent);
  return -log2Pn + log2M;
}

inline Eigen::ArrayXd minimum_weight(
    const stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &q,
    const stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p) {
  Eigen::ArrayXd x_min =
      (p.var() * q.mean() - q.var() * p.mean()) / (p.var() - q.var());
  Eigen::ArrayXd logW = (p.pdf(x_min).log() - q.pdf(x_min).log());
  logW = logW.unaryExpr([](double x) { return isnan(x) ? 0 : x; });
  return logW.exp();
}

inline Eigen::ArrayXd linspace(double s, double e, int m) {
  Eigen::ArrayXd X(m);
  double d = (e - s) / std::max(1, m - 1);
  for (int i = 0; i < m; i++) X[i] = s + d * i;
  return X;
}
}  // namespace internal

Eigen::IOFormat eigen_format();
}  // namespace rcc
#endif  // THIRD_PARTY_HYBRID_RCC_ALGORITHM_HELPER_H_
