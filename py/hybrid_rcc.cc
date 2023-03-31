#include <pybind11/pybind11.h>

#include "py/interface.h"
#include "pybind11/detail/common.h"

PYBIND11_MODULE(hybrid_rcc, m) { AddModules(m); }