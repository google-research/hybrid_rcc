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

#ifndef THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_UNIVARIATE_UNIVARIATE_H_
#define THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_UNIVARIATE_UNIVARIATE_H_

#include <cstdint>
#include <tuple>
#include <vector>

#include <Eigen/Core>

namespace stats {
class ContinuousSingleVariable{
 public:
  using instanceType = double;
  using listType = Eigen::Array<instanceType, Eigen::Dynamic, 1>;
  using supportType = std::tuple<double, double>;
  using pointProbabilityType = double;
  using listProbabilityType = Eigen::ArrayXd;
};
class DiscreteSingleVariable{
 public:
  using instanceType = int64_t;
  using listType = Eigen::Array<instanceType, Eigen::Dynamic, 1>;
  using supportType = std::vector<instanceType>;
  using pointProbabilityType = double;
  using listProbabilityType = Eigen::ArrayXd;
};
}  // namespace stats

#endif  // THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_UNIVARIATE_UNIVARIATE_H_
