#include "lie_algebra.hpp"

#include <iostream>

int main() {
  std::cout << "---------------SO3 Test-------------" << std::endl;
  std::cout << "---------------Vee  Hat-------------" << std::endl;
  Eigen::Vector3d phi(0.2, 0.1, -0.6);
  std::cout << "v_so3: " << phi.transpose() << std::endl;
  auto phi_hat = Hat(phi);
  std::cout << "m_so3: " << std::endl << phi_hat << std::endl;
  std::cout << "v_so3 result: " << Vee(phi_hat).transpose() << std::endl;
  std::cout << "---------------Log  Exp-------------" << std::endl;
  auto so3 = Expv(phi);
  std::cout << "Expv SO3: " << std::endl << so3 << std::endl;
  std::cout << "Expm SO3: " << std::endl << Expm(phi_hat) << std::endl;
  std::cout << "Logv v_so3: " << Logv(so3).transpose() << std::endl;
  std::cout << "Logm m_so3: " << std::endl << Logm(so3) << std::endl;
  std::cout << "---------------Jacobian-------------" << std::endl;
  auto jacobian = JacobianLeft(phi);
  std::cout << "Jacobian: " << std::endl << jacobian << std::endl;
  auto jacobian_inv = JacobianLeftInverse(phi);
  std::cout << "Jacobian inverse: " << std::endl << jacobian_inv << std::endl;
  std::cout << "Jacobian * Jacobian inverse: " << std::endl
            << jacobian * jacobian_inv << std::endl;

  std::cout << "---------------SE3 Test-------------" << std::endl;
  std::cout << "---------------Vee  Hat-------------" << std::endl;
  Eigen::Matrix<double, 6, 1> rho_phi;
  rho_phi << 2., -4., 7., 0.2, 0.1, -0.6;
  std::cout << "v_se3: " << rho_phi.transpose() << std::endl;
  auto rho_phi_hat = Hat(rho_phi);
  std::cout << "m_se3: " << std::endl << rho_phi_hat << std::endl;
  std::cout << "v_se3 result: " << Vee(rho_phi_hat).transpose() << std::endl;
  std::cout << "---------------Log  Exp-------------" << std::endl;
  auto se3 = Expv(rho_phi);
  std::cout << "Expv SE3: " << std::endl << se3 << std::endl;
  std::cout << "Expm SE3: " << std::endl << Expm(rho_phi_hat) << std::endl;
  std::cout << "Logv v_se3: " << Logv(se3).transpose() << std::endl;
  std::cout << "Logm m_se3: " << std::endl << Logm(se3) << std::endl;
  std::cout << "--------------limit zero------------" << std::endl;
  rho_phi << 2., -4., 7., 0.0000002, 0.0000001, -0.0000006;
  std::cout << "Log(Exp(v)) result: " << Logv(Expv(rho_phi)).transpose()
            << std::endl;
  return 0;
}
