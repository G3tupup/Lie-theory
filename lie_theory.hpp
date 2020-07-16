#pragma once

#include "math.hpp"

template <typename FloatType, int N>
using Matrix = Eigen::Matrix<FloatType, N, N>;
template <typename FloatType, int N>
using Vector = Eigen::Matrix<FloatType, N, 1>;

template <typename FloatType>
using VectorSo3 = Vector<FloatType, 3>;
using VectorSo3d = VectorSo3<double>;
using VectorSo3f = VectorSo3<float>;
template <typename FloatType>
using MatrixSo3 = Matrix<FloatType, 3>;
using MatrixSo3d = MatrixSo3<double>;
using MatrixSo3f = MatrixSo3<float>;

template <typename FloatType>
using VectorSe3 = Vector<FloatType, 6>;
using VectorSe3d = VectorSe3<double>;
using VectorSe3f = VectorSe3<float>;
template <typename FloatType>
using MatrixSe3 = Matrix<FloatType, 4>;
using MatrixSe3d = MatrixSe3<double>;
using MatrixSe3f = MatrixSe3<float>;

template <typename FloatType>
using SO3 = Matrix<FloatType, 3>;
using SO3d = SO3<double>;
using SO3f = SO3<float>;

template <typename FloatType>
using SE3 = Matrix<FloatType, 4>;
using SE3d = SE3<double>;
using SE3f = SE3<float>;

template <typename FloatType>
inline MatrixSo3<FloatType> Hat(const VectorSo3<FloatType>& vector_so3) {
  return (MatrixSo3<FloatType>() << FloatType(0), -vector_so3.z(),
          vector_so3.y(), vector_so3.z(), FloatType(0), -vector_so3.x(),
          -vector_so3.y(), vector_so3.x(), FloatType(0))
      .finished();
}

template <typename FloatType>
inline VectorSo3<FloatType> Vee(const MatrixSo3<FloatType>& matrix_so3) {
  return VectorSo3<FloatType>(matrix_so3(2, 1), matrix_so3(0, 2),
                              matrix_so3(1, 0));
}

template <typename FloatType>
inline SO3<FloatType> Expv(const VectorSo3<FloatType>& vector_so3) {
  using std::cos;
  using std::sin;
  const FloatType angle = vector_so3.norm();
  if (angle < math::limit_zero<FloatType>) {
    return Matrix<FloatType, 3>::Identity() + Hat(vector_so3);
  }
  const Vector<FloatType, 3> axis = vector_so3 / angle;
  const FloatType cos_angle = cos(angle);
  return cos_angle * Matrix<FloatType, 3>::Identity() +
         (FloatType(1) - cos_angle) * axis * axis.transpose() +
         sin(angle) * Hat(axis);
}

template <typename FloatType>
inline SO3<FloatType> Expm(const MatrixSo3<FloatType>& matrix_so3) {
  return Expv(Vee(matrix_so3));
}

template <typename FloatType>
inline MatrixSo3<FloatType> Logm(const SO3<FloatType>& so3) {
  using std::acos;
  using std::sin;
  const FloatType angle = acos((so3.trace() - FloatType(1)) * FloatType(0.5));
  if (angle < math::limit_zero<FloatType>) {
    return so3 - Matrix<FloatType, 3>::Identity();
  }
  return FloatType(0.5) * angle / sin(angle) * (so3 - so3.transpose());
}

template <typename FloatType>
inline VectorSo3<FloatType> Logv(const SO3<FloatType>& so3) {
  return Vee(Logm(so3));
}

template <typename FloatType>
inline Matrix<FloatType, 3> JacobianLeft(
    const VectorSo3<FloatType>& vector_so3) {
  using std::cos;
  using std::sin;
  const FloatType angle = vector_so3.norm();
  if (angle < math::limit_zero<FloatType>) {
    return Matrix<FloatType, 3>::Identity() + FloatType(0.5) * Hat(vector_so3);
  }
  const FloatType angle_inv = FloatType(1) / angle;
  const VectorSo3<FloatType> axis = vector_so3 * angle_inv;
  const FloatType sin_angle_by_angle = sin(angle) * angle_inv;
  return sin_angle_by_angle * Matrix<FloatType, 3>::Identity() +
         (FloatType(1) - sin_angle_by_angle) * axis * axis.transpose() +
         (angle_inv - cos(angle) * angle_inv) * Hat(axis);
}

template <typename FloatType>
inline Matrix<FloatType, 3> JacobianLeftInverse(
    const VectorSo3<FloatType>& vector_so3) {
  using std::tan;
  const FloatType angle = vector_so3.norm();
  if (angle < math::limit_zero<FloatType>) {
    return Matrix<FloatType, 3>::Identity() - FloatType(0.5) * Hat(vector_so3);
  }
  const FloatType half_angle = FloatType(0.5) * angle;
  const VectorSo3<FloatType> axis = vector_so3 / angle;
  const FloatType half_angle_by_tan_half_angle = half_angle / tan(half_angle);
  return half_angle_by_tan_half_angle * Matrix<FloatType, 3>::Identity() +
         (FloatType(1) - half_angle_by_tan_half_angle) * axis *
             axis.transpose() -
         half_angle * Hat(axis);
}

template <typename FloatType>
inline MatrixSe3<FloatType> Hat(const VectorSe3<FloatType>& vector_se3) {
  MatrixSe3<FloatType> matrix_se3;
  matrix_se3.setZero();
  matrix_se3.template block<3, 3>(0, 0) =
      Hat(VectorSo3<FloatType>(vector_se3.template block<3, 1>(3, 0)));
  matrix_se3.template block<3, 1>(0, 3) = vector_se3.template block<3, 1>(0, 0);
  return matrix_se3;
}

template <typename FloatType>
inline VectorSe3<FloatType> Vee(const MatrixSe3<FloatType>& matrix_se3) {
  return (VectorSe3<FloatType>() << matrix_se3.template block<3, 1>(0, 3),
          Vee(MatrixSo3<FloatType>(matrix_se3.template block<3, 3>(0, 0))))
      .finished();
}

template <typename FloatType>
inline SE3<FloatType> Expv(const VectorSe3<FloatType>& vector_se3) {
  using std::cos;
  using std::sin;
  const FloatType angle = vector_se3.template block<3, 1>(3, 0).norm();
  const auto matrix_se3 = Hat(vector_se3);
  if (angle < math::limit_zero<FloatType>) {
    return Matrix<FloatType, 4>::Identity() + matrix_se3;
  }
  const auto matrix_se3_square = matrix_se3 * matrix_se3;
  return Matrix<FloatType, 4>::Identity() + matrix_se3 +
         (FloatType(1) - cos(angle)) / math::pow2(angle) * matrix_se3_square +
         (angle - sin(angle)) / math::pow3(angle) *
             (matrix_se3_square * matrix_se3);
}

template <typename FloatType>
inline SE3<FloatType> Expm(const MatrixSe3<FloatType>& matrix_se3) {
  return Expv(Vee(matrix_se3));
}

template <typename FloatType>
inline VectorSe3<FloatType> Logv(const SE3<FloatType>& se3) {
  VectorSe3<FloatType> vector_se3;
  auto vector_so3 = Logv(SO3<FloatType>(se3.template block<3, 3>(0, 0)));
  vector_se3.template block<3, 1>(0, 0) =
      JacobianLeftInverse(vector_so3) * se3.template block<3, 1>(0, 3);
  vector_se3.template block<3, 1>(3, 0) = vector_so3;
  return vector_se3;
}

template <typename FloatType>
inline MatrixSe3<FloatType> Logm(const SE3<FloatType>& se3) {
  return Hat(Logv(se3));
}
