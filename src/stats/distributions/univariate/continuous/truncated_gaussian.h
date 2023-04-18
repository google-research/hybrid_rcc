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

#ifndef THIRD_PARTY_HYBRID_RCC_SRC_STATS_DISTRIBUTIONS_UNIVARIATE_CONTINUOUS_TRUNCATED_GAUSSIAN_H_
#define THIRD_PARTY_HYBRID_RCC_SRC_STATS_DISTRIBUTIONS_UNIVARIATE_CONTINUOUS_TRUNCATED_GAUSSIAN_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <utility>

#include "stats/distributions/univariate/continuous/gaussian.h"
#include "stats/random_number_generator/random_number_generator.h"

namespace stats::univariates {
class TruncatedGaussian : public Gaussian {
 private:
  double z_, cdf_a_, a_, b_, f1_, f2_;

 public:
  TruncatedGaussian(double mu, double std, double A, double B)
      : Gaussian(mu, std) {
    a_ = A, b_ = B;
    cdf_a_ = Gaussian::cdf(A);
    z_ = Gaussian::cdf(B) - cdf_a_;
    double alpha = (a_ - mu) / std;
    double beta = (b_ - mu) / std;
    Gaussian phi(0, 1);
    f1_ = (alpha * phi.pdf(alpha) - beta * phi.pdf(beta)) / z_;
    f2_ = (phi.pdf(alpha) - phi.pdf(beta)) / z_;
  }

  double rvs(std::unique_ptr<RandomNumberGenerator>&) override;
  Eigen::ArrayXd rvs(std::unique_ptr<RandomNumberGenerator>&, int) override;
  double pdf(const double& x) const override {
    return Gaussian::pdf(x) / z_ * (a_ <= x && x <= b_);
  }
  Eigen::ArrayXd pdf(const Eigen::ArrayXd& X) const override {
    return Gaussian::pdf(X) * (a_ <= X && X <= b_).cast<double>() / z_;
  }
  double logpdf(const double& x) const override {
    if (a_ <= x && x <= b_)
      return Gaussian::logpdf(x) - std::log(z_);
    else
      return -std::numeric_limits<double>::infinity();
  }
  Eigen::ArrayXd logpdf(const Eigen::ArrayXd& X) const override {
    auto in_support = (a_ <= X && X <= b_).cast<double>();
    return (Gaussian::logpdf(X) - std::log(z_)) * in_support +
           (1 - in_support) * -std::numeric_limits<double>::infinity();
  }
  double cdf(const double& x) const override {
    return std::min(1.0, std::max(0.0, (Gaussian::cdf(x) - cdf_a_) / z_));
  }
  Eigen::ArrayXd cdf(const Eigen::ArrayXd& X) const override {
    return ((Gaussian::cdf(X) - cdf_a_) / z_).min(1).max(0);
  }
  std::tuple<double, double> support() const override {
    return std::tuple<double, double>(a_, b_);
  }

  double ppf(double p) const override {
    double s = a_, e = b_;
    while (e - s > 1e-12) {
      double m = s + (e - s) / 2;
      if (cdf(m) >= p)
        e = m;
      else
        s = m;
    }
    return e;
  }
  double mean() const override {
    double mu = Gaussian::mean();
    double sigma = Gaussian::std();
    return mu + f2_ * sigma;
  }
  double var() const override {
    double sigma = Gaussian::std();
    return sigma * sigma * (1 + f1_ - f2_ * f2_);
  }
  double std() const override { return std::sqrt(var()); }
  double entropy() const override {
    return Gaussian::entropy() + std::log(z_) + f1_ / 2;
  }
  template <typename RNG>
  std::unique_ptr<RandomNumberGenerator> make_rng(RNG rng) {
    std::uniform_real_distribution<> d(0, 1);
    return std::make_unique<URBG<RNG, std::uniform_real_distribution<>>>(rng,
                                                                         d);
  }
};

inline double TruncatedGaussian::rvs(
    std::unique_ptr<RandomNumberGenerator>& rng) {
  return ppf(rng->sample(0));
}

inline Eigen::ArrayXd TruncatedGaussian::rvs(
    std::unique_ptr<RandomNumberGenerator>& rng, int n) {
  return (Eigen::ArrayXd)Eigen::ArrayXd(n)
      .unaryExpr(
          [this, &rng](const double) { return this->rvs(rng); })
      .cast<double>();
}
}  // namespace stats::univariates
#endif  // THIRD_PARTY_HYBRID_RCC_SRC_STATS_DISTRIBUTIONS_UNIVARIATE_CONTINUOUS_TRUNCATED_GAUSSIAN_H_
