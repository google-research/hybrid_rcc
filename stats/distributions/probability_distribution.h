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

#ifndef THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_PROBABILITY_DISTRIBUTION_H_
#define THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_PROBABILITY_DISTRIBUTION_H_
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "hybrid_rcc/stats/random_number_generator/random_number_generator.h"

namespace stats {

// http://web.mit.edu/urban_or_book/www/book/chapter7/7.1.3.html
template <typename DistributionType>
class ProbabilityDistribution {
 public:
  virtual typename DistributionType::instanceType rvs(
      std::unique_ptr<RandomNumberGenerator>&) = 0;
  virtual typename DistributionType::listType rvs(
      std::unique_ptr<RandomNumberGenerator>&, int) = 0;
  virtual typename DistributionType::pointProbabilityType pdf(
      const typename DistributionType::instanceType&) const = 0;
  virtual typename DistributionType::listProbabilityType pdf(
      const typename DistributionType::listType&) const = 0;
  virtual typename DistributionType::pointProbabilityType cdf(
      const typename DistributionType::instanceType&) const = 0;
  virtual typename DistributionType::listProbabilityType cdf(
      const typename DistributionType::listType&) const = 0;
  virtual typename DistributionType::pointProbabilityType logpdf(
      const typename DistributionType::instanceType&) const = 0;
  virtual typename DistributionType::listProbabilityType logpdf(
      const typename DistributionType::listType&) const = 0;
  virtual typename DistributionType::instanceType ppf(
      typename DistributionType::pointProbabilityType) const = 0;
  virtual typename DistributionType::supportType support() const = 0;
  virtual typename DistributionType::instanceType mean() const = 0;
  virtual typename DistributionType::instanceType std() const = 0;
  virtual typename DistributionType::instanceType var() const = 0;
  virtual typename DistributionType::pointProbabilityType entropy() const = 0;
  virtual ~ProbabilityDistribution() = default;
};
}  // namespace stats

#endif  // THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_PROBABILITY_DISTRIBUTION_H_
