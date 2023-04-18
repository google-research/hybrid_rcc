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

#ifndef THIRD_PARTY_HYBRID_RCC_SRC_STATS_RANDOM_NUMBER_GENERATOR_STL_URBG_H_
#define THIRD_PARTY_HYBRID_RCC_SRC_STATS_RANDOM_NUMBER_GENERATOR_STL_URBG_H_

#include <any>
#include <climits>
#include <cstdint>
#include <functional>
#include <iostream>
#include <typeinfo>
#include <vector>

#include "stats/random_number_generator/random_number_generator.h"
#include "Eigen/Core"

namespace stats {
template <typename RNG, typename Distribution>
class URBG : public RandomNumberGenerator {
 private:
  RNG state_;
  std::vector<Distribution> distributions_;

 public:
  URBG() = delete;
  URBG(RNG& rng, Distribution& ds) {
    state_ = rng;
    distributions_.push_back(ds);
  }

  URBG(RNG& rng, std::vector<Distribution>& ds) {
    state_ = rng;
    distributions_ = std::vector<Distribution>(ds);
  }

  uint64_t step() override { return state_(); }

  double sample(uint32_t distribution_id) override {
    return distributions_[distribution_id](state_);
  }
  Eigen::ArrayXd sample(uint32_t distribution_id, uint32_t n) override {
    auto& d = distributions_[distribution_id];
    return (Eigen::ArrayXd)Eigen::ArrayXd(n).unaryExpr(
        [&d, this](const double x) { return d(state_); });
  }

  void advance(uint64_t n) override {}
};
}  // namespace stats

#endif  // THIRD_PARTY_HYBRID_RCC_SRC_STATS_RANDOM_NUMBER_GENERATOR_STL_URBG_H_
