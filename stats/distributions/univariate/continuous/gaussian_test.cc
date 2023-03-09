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

#include <math.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "stats/statistical_tests/kolmogorov_smirnov.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include <pcg_random.hpp>
namespace stats {
namespace univariates {
namespace {

class GaussianTest : public ::testing::Test {
 protected:
  GaussianTest() {}

  ~GaussianTest() override {}

  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(GaussianTest, PDF) {
  std::vector<std::array<double, 4>> tests{
      {0.25330232072667835, -0.8899518314300903, 0.2392063282606547,
       1.8280591781660975e-05},
      {-0.5810752474581407, -2.4375936299865706, 0.10387697227725928,
       1.6732643003980243e-69},
      {0.4193234152171857, -0.08492383825829136, 0.09377033816258806,
       2.236434213557814e-06},
      {-0.5605259164456273, -0.8831995526207206, 1.7721099591823173,
       0.2214215767599463},
      {1.2604356503220835, 0.18378043968932584, 1.223080873193922,
       0.2214055722715615},
      {0.043054772373413354, -1.129415569361523, 1.0385368574760807,
       0.20310527704292855},
      {-2.6145772124036895, -0.02837250181973601, 0.5369439838648985,
       6.8139481452575486e-06},
      {-0.9784632799002942, 1.6164352441018535, 0.26489216162458945,
       2.1867612112791966e-21},
      {0.4747244271156488, -0.1846919394052537, 0.970396595539162,
       0.32635466736690516},
      {-0.26044389557640363, 0.445020244888626, 1.213978100876403,
       0.2775677955734867}};
  for (auto test : tests) {
    auto [x, mu, std, want] = test;
    double got = stats::univariates::Gaussian(mu, std).pdf(x);
    EXPECT_THAT(got, testing::DoubleNear(want, 1e-12));
  }
}
TEST_F(GaussianTest, VECPDF) {
  std::vector<std::array<double, 4>> tests{
      {0.25330232072667835, -0.8899518314300903, 0.2392063282606547,
       1.8280591781660975e-05},
      {-0.5810752474581407, -2.4375936299865706, 0.10387697227725928,
       1.6732643003980243e-69},
      {0.4193234152171857, -0.08492383825829136, 0.09377033816258806,
       2.236434213557814e-06},
      {-0.5605259164456273, -0.8831995526207206, 1.7721099591823173,
       0.2214215767599463},
      {1.2604356503220835, 0.18378043968932584, 1.223080873193922,
       0.2214055722715615},
      {0.043054772373413354, -1.129415569361523, 1.0385368574760807,
       0.20310527704292855},
      {-2.6145772124036895, -0.02837250181973601, 0.5369439838648985,
       6.8139481452575486e-06},
      {-0.9784632799002942, 1.6164352441018535, 0.26489216162458945,
       2.1867612112791966e-21},
      {0.4747244271156488, -0.1846919394052537, 0.970396595539162,
       0.32635466736690516},
      {-0.26044389557640363, 0.445020244888626, 1.213978100876403,
       0.2775677955734867}};
  for (auto test : tests) {
    auto [x, mu, std, want] = test;
    Eigen::ArrayXd X(1);
    X << x;
    double got = stats::univariates::Gaussian(mu, std).pdf(X)[0];
    EXPECT_THAT(got, testing::DoubleNear(want, 1e-12));
  }
}
TEST_F(GaussianTest, LOGPDF) {
  std::vector<std::array<double, 4>> tests{
      {0.25330232072667835, -0.8899518314300903, 0.2392063282606547,
       1.8280591781660975e-05},
      {-0.5810752474581407, -2.4375936299865706, 0.10387697227725928,
       1.6732643003980243e-69},
      {0.4193234152171857, -0.08492383825829136, 0.09377033816258806,
       2.236434213557814e-06},
      {-0.5605259164456273, -0.8831995526207206, 1.7721099591823173,
       0.2214215767599463},
      {1.2604356503220835, 0.18378043968932584, 1.223080873193922,
       0.2214055722715615},
      {0.043054772373413354, -1.129415569361523, 1.0385368574760807,
       0.20310527704292855},
      {-2.6145772124036895, -0.02837250181973601, 0.5369439838648985,
       6.8139481452575486e-06},
      {-0.9784632799002942, 1.6164352441018535, 0.26489216162458945,
       2.1867612112791966e-21},
      {0.4747244271156488, -0.1846919394052537, 0.970396595539162,
       0.32635466736690516},
      {-0.26044389557640363, 0.445020244888626, 1.213978100876403,
       0.2775677955734867}};
  for (auto test : tests) {
    auto [x, mu, std, e_want] = test;
    double got = stats::univariates::Gaussian(mu, std).logpdf(x);
    double want = std::log(e_want);
    EXPECT_THAT(got, testing::DoubleNear(want, 1e-12));
  }
}

TEST_F(GaussianTest, CDF) {
  std::vector<std::array<double, 4>> tests{
      {-0.2917607230862372, -0.7717786702555447, 0.13399498290489412,
       0.9998297463899888},
      {-0.504073560971343, 0.6359214833795479, 1.3274607073542632,
       0.1952312973122169},
      {0.5555944702196062, -0.6655975036071224, 0.006077103994893407, 1.0},
      {-0.3221429304545142, 0.5588749904248047, 1.3571179093506494,
       0.2581100385236984},
      {-1.3323641904417425, -0.26630594768706173, 0.2410107941935971,
       4.860679349830667e-06},
      {0.7159486000197106, -1.6979186574779386, 0.4617640793636631,
       0.9999999140868068},
      {-0.024311521918037114, -1.2682424367996266, 0.812300511456048,
       0.9371607515138738},
      {1.4756117966760995, 0.6982737327190504, 0.06618531360815383, 1.0},
      {0.26339660566258305, 0.5395354519581823, 0.6025464181721987,
       0.3233733363832933},
      {-3.202209021489422, 1.0438114894546096, 0.3410217343610789,
       6.915629786514816e-36}};
  for (auto test : tests) {
    auto [x, mu, std, want] = test;
    double got = stats::univariates::Gaussian(mu, std).cdf(x);
    EXPECT_THAT(got, testing::DoubleNear(want, 1e-12));
  }
}
TEST_F(GaussianTest, VECCDF) {
  std::vector<std::array<double, 4>> tests{
      {-0.2917607230862372, -0.7717786702555447, 0.13399498290489412,
       0.9998297463899888},
      {-0.504073560971343, 0.6359214833795479, 1.3274607073542632,
       0.1952312973122169},
      {0.5555944702196062, -0.6655975036071224, 0.006077103994893407, 1.0},
      {-0.3221429304545142, 0.5588749904248047, 1.3571179093506494,
       0.2581100385236984},
      {-1.3323641904417425, -0.26630594768706173, 0.2410107941935971,
       4.860679349830667e-06},
      {0.7159486000197106, -1.6979186574779386, 0.4617640793636631,
       0.9999999140868068},
      {-0.024311521918037114, -1.2682424367996266, 0.812300511456048,
       0.9371607515138738},
      {1.4756117966760995, 0.6982737327190504, 0.06618531360815383, 1.0},
      {0.26339660566258305, 0.5395354519581823, 0.6025464181721987,
       0.3233733363832933},
      {-3.202209021489422, 1.0438114894546096, 0.3410217343610789,
       6.915629786514816e-36}};
  for (auto test : tests) {
    auto [x, mu, std, want] = test;
    Eigen::ArrayXd X(1);
    X << x;
    double got = stats::univariates::Gaussian(mu, std).cdf(X)[0];
    EXPECT_THAT(got, testing::DoubleNear(want, 1e-12));
  }
}

// Kolmogorov–Smirnov_test.
TEST_F(GaussianTest, RVS) {
  std::vector<std::array<double, 2>> tests{
      {-0.7717786702555447, 0.13399498290489412},
      {0.6359214833795479, 1.3274607073542632},
      {-0.6655975036071224, 0.006077103994893407},
      {0.5588749904248047, 1.3571179093506494},
      {-0.26630594768706173, 0.2410107941935971},
      {-1.6979186574779386, 0.4617640793636631},
      {-1.2682424367996266, 0.812300511456048},
      {0.6982737327190504, 0.06618531360815383},
      {0.5395354519581823, 0.6025464181721987},
      {1.0438114894546096, 0.3410217343610789}};
  for (auto [mu, std] : tests) {
    auto P = stats::univariates::Gaussian(mu, std);
    pcg32 rng(static_cast<uint64_t>(std::abs(mu * std)));
    auto RNG = P.make_rng(rng);
    for (int N = 10; N <= 10000; N *= 10) {
      auto random_values = P.rvs(RNG, N);
      double maxDifference = kolmogorov_smirnov_statistic(random_values, P);
      EXPECT_LE(maxDifference, KS_95 / sqrt(N));
    }
  }
}

}  // namespace
}  // namespace univariates
}  // namespace stats
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
