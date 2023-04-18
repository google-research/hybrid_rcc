// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "stats/distributions/univariate/continuous/gaussian.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

#include "unsupported/Eigen/SpecialFunctions"

namespace stats::univariates {
Gaussian::Gaussian(double mu, double std) {
  mu_ = mu;
  std_ = std;
  normal_rng_ = std::normal_distribution<>{mu_, std_};
}

double Gaussian::rvs(std::unique_ptr<RandomNumberGenerator>& rng) {
  return rng->sample(0);
}

Eigen::ArrayXd Gaussian::rvs(std::unique_ptr<RandomNumberGenerator>& rng,
                             int n) {
  return rng->sample(0, n);
}
double Gaussian::pdf(const double& x) const {
  return std::exp(-0.5 * pow((x - mu_) / std_, 2)) / (std_ * sqrt2pi_);
}
double Gaussian::logpdf(const double& x) const {
  return (-0.5 * pow((x - mu_) / std_, 2)) - std::log(std_ * sqrt2pi_);
}
double Gaussian::cdf(const double& x) const {
  return 0.5 * (1 + std::erf((x - mu_) / (std_ * sqrt2_)));
}

Eigen::ArrayXd Gaussian::pdf(const Eigen::ArrayXd& X) const {
  return (-0.5 * ((X - mu_) / std_).pow(2)).exp() / (std_ * sqrt2pi_);
}
Eigen::ArrayXd Gaussian::logpdf(const Eigen::ArrayXd& X) const {
  return (-0.5 * ((X - mu_) / std_).pow(2)) - std::log(std_ * sqrt2pi_);
}

Eigen::ArrayXd Gaussian::cdf(const Eigen::ArrayXd& X) const {
  return (((X - mu_) / (std_ * sqrt2_)).erf() + 1) * 0.5;
}

std::tuple<double, double> Gaussian::support() const {
  return std::tuple<double, double>(-std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::infinity());
}

double Gaussian::ppf(double p) const {
  double size = 1;
  while (cdf(-size) > p || cdf(size) < p) size *= 2;
  double s = -size, e = size;
  while (e - s > 1e-12) {
    double m = s + (e - s) / 2;
    if (cdf(m) >= p)
      e = m;
    else
      s = m;
  }
  return e;
}
};  // namespace stats::univariates
