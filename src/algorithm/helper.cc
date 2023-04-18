#include "algorithm/helper.h"

namespace rcc {
Eigen::IOFormat eigen_format() {
  return Eigen::IOFormat(3, 0, ", ", "\n", "", "", "[", "]");
}
}  // namespace rcc
