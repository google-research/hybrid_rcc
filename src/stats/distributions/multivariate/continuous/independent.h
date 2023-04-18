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

#ifndef THIRD_PARTY_HYBRID_RCC_SRC_STATS_DISTRIBUTIONS_MULTIVARIATE_CONTINUOUS_INDEPENDENT_H_
#define THIRD_PARTY_HYBRID_RCC_SRC_STATS_DISTRIBUTIONS_MULTIVARIATE_CONTINUOUS_INDEPENDENT_H_

#include <memory>
#include <random>

#include "stats/distributions/multivariate/multivariate.h"
#include "stats/distributions/probability_distribution.h"
#include "stats/random_number_generator/stl_urbg.h"

namespace stats::multivariates {
template <typename Distribution>
class IndependentDistributions
    : public ProbabilityDistribution<ContinuousMultiVariable> {
 protected:
  std::vector<Distribution> univariates_;
  int dim_;
  Eigen::ArrayXd lower_corner_, upper_corner_, mu_, std_;

 public:
  template <typename STD_URBG>
  std::unique_ptr<RandomNumberGenerator> make_rng(STD_URBG& urbg) {
    std::uniform_real_distribution<> d(0, 1);
    return std::make_unique<URBG<STD_URBG, std::uniform_real_distribution<>>>(
        urbg, d);
  }
  Eigen::ArrayXd rvs(std::unique_ptr<RandomNumberGenerator>& rng) override {
    return ppf(rng->sample(0, dim_));
  }
  Eigen::ArrayXXd rvs(std::unique_ptr<RandomNumberGenerator>& rng,
                      int n) override {
    Eigen::ArrayXXd X(n, dim_);
    for (int i = 0; i < n; i++) {
      X.row(i) = rvs(rng);
    }
    return X;
  }
  Eigen::ArrayXd pdf(const Eigen::ArrayXd& X) const override {
    Eigen::ArrayXd p(dim_);
    std::transform(
        univariates_.begin(), univariates_.end(), X.begin(), p.begin(),
        [](const Distribution& G, const double x) { return G.pdf(x); });
    return p;
  }
  Eigen::ArrayXXd pdf(const Eigen::ArrayXXd& X) const override {
    auto cX = X.colwise();
    Eigen::ArrayXXd p(X.rows(), dim_);
    auto cp = p.colwise();
    std::transform(univariates_.begin(), univariates_.end(), cX.begin(),
                   cp.begin(),
                   [](const Distribution& G, const Eigen::ArrayXd& x) {
                     return G.pdf(x);
                   });
    return p;
  }
  Eigen::ArrayXd logpdf(const Eigen::ArrayXd& X) const override {
    Eigen::ArrayXd p(dim_);
    std::transform(
        univariates_.begin(), univariates_.end(), X.begin(), p.begin(),
        [](const Distribution& G, const double x) { return G.logpdf(x); });
    return p;
  }
  Eigen::ArrayXXd logpdf(const Eigen::ArrayXXd& X) const override {
    auto cX = X.colwise();
    Eigen::ArrayXXd logp(X.rows(), dim_);
    auto clogp = logp.colwise();
    std::transform(univariates_.begin(), univariates_.end(), cX.begin(),
                   clogp.begin(),
                   [](const Distribution& G, const Eigen::ArrayXd& x) {
                     return G.logpdf(x);
                   });
    return logp;
  }
  Eigen::ArrayXd cdf(const Eigen::ArrayXd& X) const override {
    Eigen::ArrayXd p(dim_);
    std::transform(
        univariates_.begin(), univariates_.end(), X.begin(), p.begin(),
        [](const Distribution& G, const double x) { return G.cdf(x); });
    return p;
  }
  Eigen::ArrayXXd cdf(const Eigen::ArrayXXd& X) const override {
    auto cX = X.colwise();
    Eigen::ArrayXXd p(X.rows(), dim_);
    auto cp = p.colwise();
    std::transform(univariates_.begin(), univariates_.end(), cX.begin(),
                   cp.begin(),
                   [](const Distribution& G, const Eigen::ArrayXd& x) {
                     return G.cdf(x);
                   });
    return p;
  }
  std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> support() const override {
    return std::tuple<Eigen::ArrayXd, Eigen::ArrayXd>(lower_corner_,
                                                      upper_corner_);
  }

  Eigen::ArrayXd ppf(Eigen::ArrayXd P) const override {
    Eigen::ArrayXd X(dim_);
    std::transform(
        univariates_.begin(), univariates_.end(), P.begin(), X.begin(),
        [](const Distribution& G, const double p) { return G.ppf(p); });
    return X;
  }

  Eigen::ArrayXd mean() const override { return mu_; }
  Eigen::ArrayXd std() const override { return std_; }
  Eigen::ArrayXd var() const override { return std_ * std_; }
  Eigen::ArrayXd entropy() const override {
    Eigen::ArrayXd H(dim_);
    std::transform(univariates_.begin(), univariates_.end(), H.begin(),
                   [](const Distribution& N) { return N.entropy(); });
    return H;
  }
};
}  // namespace stats::multivariates
#endif  // THIRD_PARTY_HYBRID_RCC_SRC_STATS_DISTRIBUTIONS_MULTIVARIATE_CONTINUOUS_INDEPENDENT_H_
