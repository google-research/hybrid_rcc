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

#ifndef THIRD_PARTY_HYBRID_RCC_STATS_RANDOM_NUMBER_GENERATOR_RANDOM_NUMBER_GENERATOR_H_
#define THIRD_PARTY_HYBRID_RCC_STATS_RANDOM_NUMBER_GENERATOR_RANDOM_NUMBER_GENERATOR_H_
#include <cstdint>
#include <functional>
#include <tuple>
#include "eigen3/Eigen/Core"

namespace stats {
class RandomNumberGenerator {
 public:
  virtual uint64_t step() = 0;
  virtual void advance(uint64_t) = 0;
  virtual double sample(uint32_t distribution_id) = 0;
  virtual Eigen::ArrayXd sample(uint32_t distribution_id, uint32_t n) = 0;
  virtual ~RandomNumberGenerator() = default;
};
}  // namespace stats

#endif  // THIRD_PARTY_HYBRID_RCC_STATS_RANDOM_NUMBER_GENERATOR_RANDOM_NUMBER_GENERATOR_H_
