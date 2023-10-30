#pragma once

#include <cmath>

inline auto check_norm(double expected, double actual) {
  if (std::abs(expected - actual) <= std::max(std::abs(expected), std::abs(actual))*1E-12) {
    return true;
  }
  return false;
}
