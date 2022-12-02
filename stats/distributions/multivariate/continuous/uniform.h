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

#ifndef THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_MULTIVARIATE_CONTINUOUS_UNIFORM_H_
#define THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_MULTIVARIATE_CONTINUOUS_UNIFORM_H_

#include <algorithm>
#include <iostream>
#include <random>
#include <utility>

#include "stats/distributions/multivariate/continuous/independent.h"
#include "stats/distributions/univariate/continuous/uniform.h"
#include "third_party/eigen3/Eigen/Core"

namespace stats::multivariates {
class IndependentUniform
    : public IndependentDistributions<univariates::Uniform> {
 public:
  explicit IndependentUniform(const Eigen::ArrayXd& _start,
                              const Eigen::ArrayXd& _end) {
    dim_ = _start.size();
    lower_corner_ = _start;
    upper_corner_ = _end;
    univariates_.reserve(dim_);
    mu_ = Eigen::ArrayXd(dim_);
    std_ = Eigen::ArrayXd(dim_);
    for (int d = 0; d < dim_; d++) {
      univariates_.emplace_back(_start[d], _end[d]);
      mu_[d] = univariates_[d].mean();
      std_[d] = univariates_[d].std();
    }
  }
  explicit IndependentUniform(int dim) {
    dim_ = dim;
    lower_corner_ = Eigen::ArrayXd(dim);
    lower_corner_ = 0;
    upper_corner_ = Eigen::ArrayXd(dim);
    upper_corner_ = 1;
    mu_ = Eigen::ArrayXd(dim_);
    std_ = Eigen::ArrayXd(dim_);
    for (int d = 0; d < dim_; d++) {
      univariates_.emplace_back(lower_corner_[d], upper_corner_[d]);
      mu_[d] = univariates_[d].mean();
      std_[d] = univariates_[d].std();
    }
  }
};
}  // namespace stats::multivariates

#endif  // THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_MULTIVARIATE_CONTINUOUS_UNIFORM_H_
