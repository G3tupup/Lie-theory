#pragma once

#include <eigen3/Eigen/Dense>

namespace math {
template <typename T>
constexpr T limit_zero = T(1e-6);

template <typename T>
inline T pow2(const T& value) {
  return value * value;
}

template <typename T>
inline T pow3(const T& value) {
  return value * value * value;
}
