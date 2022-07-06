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

#ifndef THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_UNIVARIATE_CONTINUOUS_UNIFORM_H_
#define THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_UNIVARIATE_CONTINUOUS_UNIFORM_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "hybrid_rcc/stats/distributions/probability_distribution.h"
#include "hybrid_rcc/stats/distributions/univariate/univariate.h"
#include "hybrid_rcc/stats/random_number_generator/stl_urbg.h"
#include "eigen3/Eigen/Core"

namespace stats::univariates {
class Uniform : public ProbabilityDistribution<ContinuousSingleVariable> {
 private:
  std::uniform_real_distribution<> uniform_rng_;
  double lower_end_, upper_end_;

 public:
  Uniform(double s, double e) {
    lower_end_ = s, upper_end_ = e;
    uniform_rng_ = std::uniform_real_distribution<>(s, e);
  }

  double rvs(std::unique_ptr<RandomNumberGenerator>& rng) override {
    return rng->sample(0);
  }

  Eigen::ArrayXd rvs(std::unique_ptr<RandomNumberGenerator>& rng,
                     int n) override {
    return rng->sample(0, n);
  }

  double pdf(const double& x) const override {
    return (lower_end_ <= x && x <= upper_end_) / (upper_end_ - lower_end_);
  }
  Eigen::ArrayXd pdf(const Eigen::ArrayXd& X) const override {
    return (lower_end_ <= X && X <= upper_end_).cast<double>() /
           (upper_end_ - lower_end_);
  }
  double logpdf(const double& x) const override {
    if (lower_end_ <= x && x <= upper_end_)
      return -std::log(upper_end_ - lower_end_);
    return -std::numeric_limits<double>::infinity();
  }
  Eigen::ArrayXd logpdf(const Eigen::ArrayXd& X) const override {
    auto in_support = (lower_end_ <= X && X <= upper_end_).cast<double>() /
                      (upper_end_ - lower_end_);
    return in_support +
           (1 - in_support) * -std::numeric_limits<double>::infinity();
  }
  double cdf(const double& x) const override {
    return std::min(std::max(x - lower_end_, 0.0) / (upper_end_ - lower_end_),
                    1.0);
  }
  Eigen::ArrayXd cdf(const Eigen::ArrayXd& X) const override {
    return ((X - lower_end_) / (upper_end_ - lower_end_)).min(1).max(0);
  }
  std::tuple<double, double> support() const override {
    return std::tuple<double, double>(lower_end_, upper_end_);
  }

  template <typename RNG>
  std::unique_ptr<RandomNumberGenerator> make_rng(RNG rng) {
    return std::make_unique<URBG<RNG, std::uniform_real_distribution<>>>(
        rng, uniform_rng_);
  }

  double ppf(double p) const override {
    return lower_end_ + (upper_end_ - lower_end_) * p;
  }
  double mean() const override { return (lower_end_ + upper_end_) / 2.0; }
  double std() const override { return std::sqrt(var()); }
  double var() const override {
    return (upper_end_ - lower_end_) * (upper_end_ - lower_end_) / 12.0;
  }
  double entropy() const override { return std::log(upper_end_ - lower_end_); }
};
}  // namespace stats::univariates
#endif  // THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_UNIVARIATE_CONTINUOUS_UNIFORM_H_
