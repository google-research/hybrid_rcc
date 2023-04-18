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

#ifndef THIRD_PARTY_HYBRID_RCC_SRC_STATS_DISTRIBUTIONS_MULTIVARIATE_MULTIVARIATE_H_
#define THIRD_PARTY_HYBRID_RCC_SRC_STATS_DISTRIBUTIONS_MULTIVARIATE_MULTIVARIATE_H_

#include <cstdint>
#include <tuple>
#include <vector>

#include "Eigen/Core"

namespace stats {
class ContinuousMultiVariable{
 public:
  using instanceType = Eigen::ArrayXd;
  using listType = Eigen::ArrayXXd;
  using supportType = std::tuple<instanceType, instanceType>;
  using pointProbabilityType = Eigen::ArrayXd;
  using listProbabilityType = Eigen::ArrayXXd;
};

class DiscreteMultiVariable{
 public:
  using instanceType = Eigen::Array<int64_t, Eigen::Dynamic, 1>;
  using listType = Eigen::Array<int64_t, Eigen::Dynamic, Eigen::Dynamic>;
  using supportType = std::vector<std::vector<int64_t>>;
  using pointProbabilityType = Eigen::ArrayXd;
  using listProbabilityType = Eigen::ArrayXXd;
};
}  // namespace stats

#endif  // THIRD_PARTY_HYBRID_RCC_SRC_STATS_DISTRIBUTIONS_MULTIVARIATE_MULTIVARIATE_H_
