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

#ifndef THIRD_PARTY_HYBRID_RCC_ALGORITHM_REVERSE_CHANNEL_H_
#define THIRD_PARTY_HYBRID_RCC_ALGORITHM_REVERSE_CHANNEL_H_

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <random>
#include <tuple>
#include <utility>

#include "algorithm/helper.h"
#include "stats/distributions/multivariate/continuous/gaussian.h"
#include "stats/distributions/multivariate/continuous/truncated_gaussian.h"
#include "stats/distributions/multivariate/continuous/uniform.h"
#include "stats/distributions/multivariate/multivariate.h"
#include "stats/distributions/probability_distribution.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/pcg_random/include/pcg_random.hpp"

namespace rcc {
namespace algorithm {
template <typename STD_URBG>
std::tuple<Eigen::ArrayXd, int, Eigen::ArrayXd, int> sample_hybrid_pfr(
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &q,
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p,
    const Eigen::ArrayXd &M, uint32_t limit, double w_min, STD_URBG urbg,
    bool verbose = false) {
  std::exponential_distribution<> exponential(1);
  int dim = q.mean().size();
  Eigen::ArrayXd logM = M.log();

  stats::multivariates::IndependentUniform U(dim);
  auto rng = U.make_rng(urbg);

  auto [q_a, q_b] = q.support();
  Eigen::ArrayXd q_support_a = p.cdf(q_a) * M;
  Eigen::ArrayXd q_support_b = p.cdf(q_b) * M;
  Eigen::ArrayXd c = (q_support_a + q_support_b) / 2.0;
  c = c.max(0.5).min(M - 0.5);
  if (verbose) {
    std::cerr << q_a.transpose() << "\t" << q_b.transpose() << "\t"
              << c.transpose() << std::endl;
  }

  // apply reverse channel coding
  double t = 0;
  double s = std::numeric_limits<double>::infinity();
  int n = 0;  // index of last accepted proposal
  int i = 0;  // index of current proposal
  double exp_s = std::numeric_limits<double>::infinity();
  typename stats::ContinuousMultiVariable::pointProbabilityType y, k;
  double prodM = M.prod();

  while (i < limit && exp_s > t * w_min * prodM) {
    // generate candidate using universal quantization
    Eigen::ArrayXd u = U.rvs(rng);
    Eigen::ArrayXd k_ = (c - u + 0.5).floor();
    Eigen::ArrayXd y_ = k_ + u;

    // evaluate candidate
    t += exponential(urbg);
    Eigen::ArrayXd phi = p.ppf(y_ / M);
    Eigen::ArrayXd q_phi_logpdf = q.logpdf(phi) + (-p.logpdf(phi) - logM);
    double s_ = std::log(t) - q_phi_logpdf.sum();
    if (verbose) {
      std::cerr << "u " << u.transpose() << " k_ " << k_.transpose() << " y_ "
                << y_.transpose() << std::endl;
      std::cerr << "t = " << t << " s_ = " << s_
                << ": phi = " << phi.transpose() << ", p.pdf "
                << p.pdf(phi).transpose() << ", q.pdf "
                << q.pdf(phi).transpose() << ", M = " << M.transpose()
                << std::endl;
    }
    // accept/reject candidate
    if (i == 0 || s_ < s) {
      n = i;
      s = s_;
      k = k_;
      y = y_;
      exp_s = std::exp(s);
    }
    i++;
  }
  // transform sample back
  auto z = p.ppf(y / M);
  return std::tuple<Eigen::ArrayXd, int, Eigen::ArrayXd, int>(z, n, k, i);
}

template <typename STD_URBG>
std::tuple<Eigen::ArrayXd, int, Eigen::ArrayXd, int> sample_hybrid_sis(
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &q,
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p,
    const Eigen::ArrayXd &M, uint32_t N_max, double w_min, STD_URBG urbg,
    bool verbose = false) {
  std::exponential_distribution<> exponential(1);
  int dim = q.mean().size();
  Eigen::ArrayXd logM = M.log();

  stats::multivariates::IndependentUniform U(dim);
  auto rng = U.make_rng(urbg);

  auto [q_a, q_b] = q.support();
  Eigen::ArrayXd q_support_a = p.cdf(q_a) * M;
  Eigen::ArrayXd q_support_b = p.cdf(q_b) * M;
  Eigen::ArrayXd c = (q_support_a + q_support_b) / 2.0;
  c = c.max(0.5).min(M - 0.5);
  if (verbose) {
    std::cerr << q_a.transpose() << "\t" << q_b.transpose() << "\t"
              << c.transpose() << std::endl;
  }

  // apply reverse channel coding
  double t = 0;
  double s = std::numeric_limits<double>::infinity();
  int n = 0;  // index of last accepted proposal
  int i = 0;  // index of current proposal
  double exp_s = std::numeric_limits<double>::infinity();
  typename stats::ContinuousMultiVariable::pointProbabilityType y, k;
  double prodM = M.prod();

  while (i < N_max && exp_s > t * w_min * prodM) {
    // generate candidate using universal quantization
    Eigen::ArrayXd u = U.rvs(rng);
    Eigen::ArrayXd k_ = (c - u + 0.5).floor();
    Eigen::ArrayXd y_ = k_ + u;

    // evaluate candidate
    double w = N_max / static_cast<double>(N_max - n);
    t += w * exponential(urbg);
    Eigen::ArrayXd phi = p.ppf(y_ / M);
    Eigen::ArrayXd q_phi_logpdf = q.logpdf(phi) + (-p.logpdf(phi) - logM);
    double s_ = std::log(t) - q_phi_logpdf.sum();
    if (verbose) {
      std::cerr << i << "/" << N_max << ": u " << u.transpose() << " k_ "
                << k_.transpose() << " y_ " << y_.transpose() << std::endl;
      std::cerr << "t = " << t << " s_ = " << s_
                << ": phi = " << phi.transpose() << ", p.pdf "
                << p.pdf(phi).transpose() << ", q.pdf "
                << q.pdf(phi).transpose() << ", M = " << M.transpose()
                << std::endl;
    }
    // accept/reject candidate
    if (i == 0 || s_ < s) {
      n = i;
      s = s_;
      k = k_;
      y = y_;
      exp_s = std::exp(s);
    }
    i++;
  }
  // transform sample back
  auto z = p.ppf(y / M);
  return std::tuple<Eigen::ArrayXd, int, Eigen::ArrayXd, int>(z, n, k, i);
}

template <typename AdvanceURBG>
inline Eigen::ArrayXd decode_hybrid(
    int n, const Eigen::ArrayXd &k, const Eigen::ArrayXd &M,
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p, int dim,
    AdvanceURBG urbg) {
  urbg.advance(n * (dim + 1LL));
  stats::multivariates::IndependentUniform U(dim);
  auto rng = U.make_rng(urbg);
  Eigen::ArrayXd u = U.rvs(rng);
  return p.ppf((k + u) / M);
}

template <typename STD_URBG>
std::tuple<Eigen::ArrayXd, int, int> sample_sis(
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &q,
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p,
    double w_min, int N_max, STD_URBG &urbg, bool verbose = false) {
  std::exponential_distribution<> exponential(1);
  int dim = q.mean().size();
  stats::multivariates::IndependentUniform U(dim);
  auto rng = U.make_rng(urbg);

  double t = 0;
  int n = 0;
  double s_star = std::numeric_limits<double>::infinity();
  int n_star = 1;
  Eigen::ArrayXd z_star;

  do {
    Eigen::ArrayXd u = U.rvs(rng);
    Eigen::ArrayXd z = p.ppf(u);
    if (verbose) {
      std::cerr << n << "/" << N_max << ": " << u.transpose() << "\t"
                << z.transpose() << std::endl;
    }

    double w = N_max / static_cast<double>(N_max - n);
    t += w * exponential(urbg);
    double s = std::log(t) + p.logpdf(z).sum() - q.logpdf(z).sum();
    if (isnan(s))
      s = std::numeric_limits<double>::infinity();
    else
      s = std::exp(s);

    if (n == 0 || s < s_star) {
      s_star = s;
      n_star = n;
      z_star = z;
    }
    n++;
  } while (s_star > t * w_min && n < N_max);
  return std::tuple<Eigen::ArrayXd, int, int>(z_star, n_star, n);
}

template <typename STD_URBG>
std::tuple<Eigen::ArrayXd, int, int> sample_pfr(
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &q,
    stats::ProbabilityDistribution<stats::ContinuousMultiVariable> &p,
    double w_min, uint32_t N_max, STD_URBG &urbg, bool verbose = false) {
  std::exponential_distribution<> exponential(1);
  int dim = q.mean().size();
  stats::multivariates::IndependentUniform U(dim);
  auto rng = U.make_rng(urbg);

  double t = 0;
  double s = std::numeric_limits<double>::infinity();
  int n = 0;
  int i = 0;
  Eigen::ArrayXd z;

  while (i < N_max && s > t * w_min) {
    Eigen::ArrayXd u = U.rvs(rng);
    Eigen::ArrayXd z_ = p.ppf(u);

    if (verbose) {
      std::cerr << i << ": " << u.transpose() << "\t" << z_.transpose()
                << std::endl;
    }
    t += exponential(urbg);
    double s_ = std::log(t) + p.logpdf(z_).sum() - q.logpdf(z_).sum();
    if (isnan(s_))
      s_ = std::numeric_limits<double>::infinity();
    else
      s_ = std::exp(s_);

    if (i == 0 || s_ < s) {
      n = i;
      s = s_;
      z = z_;
    }
    i++;
  }
  return std::tuple<Eigen::ArrayXd, int, int>(z, n, i);
}

template <typename STD_URBG>
std::tuple<Eigen::ArrayXd, int, Eigen::ArrayXd, int, Eigen::ArrayXd>
sample_gaussian_hybrid(stats::multivariates::IndependentGaussian *q,
                       stats::multivariates::IndependentGaussian *p, bool pfr,
                       double eps = 1e-4, STD_URBG rs = pcg32(0),
                       uint32_t N_max = 0, bool verbose = false) {
  int dim = q->mean().size();
  Eigen::ArrayXd D(dim);
  D = eps;
  D = 1 - (1 - D).pow(1.0 / dim);
  stats::multivariates::IndependentGaussian standardNormal(dim);

  auto a = standardNormal.ppf(D / 2.0);
  auto b = standardNormal.ppf(1 - D / 2.0);

  auto mu = q->mean();
  auto std = q->std();
  stats::multivariates::IndependentTruncatedGaussian q_tr(mu, std, a * std + mu,
                                                          b * std + mu);
  std::tie(a, b) = q_tr.support();

  Eigen::ArrayXd c = p->mean() - q_tr.mean() + a;
  Eigen::ArrayXd d = p->mean() - q_tr.mean() + b;
  c = p->cdf(c);
  d = p->cdf(d);
  Eigen::ArrayXd M = (1.0 / (d - c)).floor();

  double w_min = (internal::minimum_weight(*q, *p) * (1 - D)).prod();

  Eigen::ArrayXd z, k;
  int n, i;
  if (pfr)
    std::tie(z, n, k, i) =
        sample_hybrid_pfr(q_tr, *p, M, N_max, w_min, rs, verbose);
  else
    std::tie(z, n, k, i) =
        sample_hybrid_sis(q_tr, *p, M, N_max, w_min, rs, verbose);
  return std::tuple<Eigen::ArrayXd, int, Eigen::ArrayXd, int, Eigen::ArrayXd>(
      z, n, k, i, M);
}

template <typename STD_URBG>
std::tuple<Eigen::ArrayXd, int, int> sample_gaussian(
    stats::multivariates::IndependentGaussian *q,
    stats::multivariates::IndependentGaussian *p, bool pfr,
    STD_URBG rs = pcg32(0), uint32_t N_max = 0, bool verbose = false) {
  double w_min = internal::minimum_weight(*q, *p).prod();
  if (pfr)
    return sample_pfr(*q, *p, w_min, N_max, rs, verbose);
  else
    return sample_sis(*q, *p, w_min, N_max, rs, verbose);
}
}  // namespace algorithm
}  // namespace rcc
#endif  // THIRD_PARTY_HYBRID_RCC_ALGORITHM_REVERSE_CHANNEL_H_
