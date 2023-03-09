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

#ifndef THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_MULTIVARIATE_CONTINUOUS_GAUSSIAN_H_
#define THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_MULTIVARIATE_CONTINUOUS_GAUSSIAN_H_

#include <algorithm>
#include <cassert>
#include <exception>
#include <limits>
#include <random>

#include "stats/distributions/multivariate/continuous/independent.h"
#include "stats/distributions/univariate/continuous/gaussian.h"
#include <Eigen/Core>

namespace stats::multivariates {
class IndependentGaussian
    : public IndependentDistributions<univariates::Gaussian> {
 public:
  IndependentGaussian(const Eigen::ArrayXd& mu, const Eigen::ArrayXd& std) {
    assert(mu.size() == std.size());
    dim_ = mu.size();
    lower_corner_.resize(dim_);
    upper_corner_.resize(dim_);
    mu_ = mu;
    std_ = std;
    for (int i = 0; i < dim_; i++) {
      univariates_.emplace_back(mu[i], std[i]);
      lower_corner_[i] = -std::numeric_limits<double>::infinity();
      upper_corner_[i] = std::numeric_limits<double>::infinity();
    }
  }
  explicit IndependentGaussian(int dim) {
    dim_ = dim;
    lower_corner_.resize(dim_);
    upper_corner_.resize(dim_);
    mu_ = Eigen::ArrayXd(dim);
    std_ = Eigen::ArrayXd(dim);
    mu_ = 0;
    std_ = 1;
    for (int i = 0; i < dim_; i++) {
      univariates_.emplace_back(mu_[i], std_[i]);
      lower_corner_[i] = -std::numeric_limits<double>::infinity();
      upper_corner_[i] = std::numeric_limits<double>::infinity();
    }
  }
};
}  // namespace stats::multivariates
#endif  // THIRD_PARTY_HYBRID_RCC_STATS_DISTRIBUTIONS_MULTIVARIATE_CONTINUOUS_GAUSSIAN_H_
