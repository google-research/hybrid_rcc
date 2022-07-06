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

#ifndef THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_UNIVARIATE_CONTINUOUS_GAUSSIAN_H_
#define THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_UNIVARIATE_CONTINUOUS_GAUSSIAN_H_

#include <cmath>
#include <random>
#include <tuple>

#include "hybrid_rcc/stats/distributions/probability_distribution.h"
#include "hybrid_rcc/stats/distributions/univariate/univariate.h"
#include "hybrid_rcc/stats/random_number_generator/stl_urbg.h"
#include "eigen3/Eigen/Core"

namespace stats::univariates {
class Gaussian : public ProbabilityDistribution<ContinuousSingleVariable> {
 private:
  double mu_, std_;
  const double sqrt2_ = sqrt(2);
  const double sqrt2pi_ = sqrt(2 * M_PI);
  std::normal_distribution<> normal_rng_;

 public:
  Gaussian(double, double);

  ContinuousSingleVariable::instanceType rvs(
      std::unique_ptr<RandomNumberGenerator>&) override;
  ContinuousSingleVariable::listType rvs(std::unique_ptr<RandomNumberGenerator>&,
                                         int) override;
  ContinuousSingleVariable::pointProbabilityType pdf(
      const ContinuousSingleVariable::instanceType&) const override;
  Eigen::ArrayXd pdf(const ContinuousSingleVariable::listType&) const override;
  ContinuousSingleVariable::pointProbabilityType logpdf(
      const ContinuousSingleVariable::instanceType&) const override;
  Eigen::ArrayXd logpdf(
      const ContinuousSingleVariable::listType&) const override;
  ContinuousSingleVariable::pointProbabilityType cdf(
      const ContinuousSingleVariable::instanceType&) const override;
  ContinuousSingleVariable::listProbabilityType cdf(
      const ContinuousSingleVariable::listType&) const override;
  std::tuple<ContinuousSingleVariable::instanceType,
             ContinuousSingleVariable::instanceType>
  support() const override;

  template <typename RNG>
  std::unique_ptr<RandomNumberGenerator> make_rng(RNG rng) {
    return std::make_unique<URBG<RNG, std::normal_distribution<>>>(rng,
                                                                   normal_rng_);
  }

  double ppf(double) const override;

  double mean() const override { return mu_; }
  double std() const override { return std_; }
  double var() const override { return std_ * std_; }
  double entropy() const override {
    return 0.5 * std::log(2 * M_PI * std_ * std_) + 0.5;
  }
};
}  // namespace stats::univariates

#endif  // THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_UNIVARIATE_CONTINUOUS_GAUSSIAN_H_
