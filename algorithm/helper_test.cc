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

#include "algorithm/helper.h"

#include <algorithm>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace rcc::internal {
namespace {
class HelperTest : public ::testing::Test {
 private:
 protected:
  HelperTest() {}

  ~HelperTest() override {}

  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(HelperTest, linspace) {
  std::vector<std::tuple<double, double, int>> tests{
      {0, 1, 10},
      {10, -1, 20},
      {0, 2, 1},
  };
  for (auto [s, e, n] : tests) {
    auto X = linspace(s, e, n);
    EXPECT_EQ(X.size(), n);
    double maxDiff = 0;
    for (int i = 0; i < n; i++) {
      double want = s + (e - s) * i / std::max(n - 1, 1);
      maxDiff = std::max(maxDiff, std::abs(want - X[i]));
    }
    EXPECT_LT(maxDiff, 2e-15);
  }
}

TEST_F(HelperTest, MINIMUMWEIGHT){
  
}
}  // namespace
}  // namespace rcc::internal

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
