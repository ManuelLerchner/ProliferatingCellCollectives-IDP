#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <random>

#include "../../hard/src/util/SpherocylinderCell.h"
#include "Constants.h"

namespace soft {

class SoftSpherocylinder {
 public:
  SoftSpherocylinder(const std::array<double, 3>& position,
                     const std::array<double, 4>& orientation,
                     double length,
                     double radius)
      : position_(position),
        orientation_(orientation),
        length_(length),
        radius_(radius),
        velocity_({0.0, 0.0, 0.0}),
        angular_velocity_({0.0, 0.0, 0.0}),
        force_({0.0, 0.0, 0.0}),
        torque_({0.0, 0.0, 0.0}),
        growth_rate_(0.0),
        nutrient_concentration_(1.0),
        contact_stress_(0.0),
        mass_(calculateMass()),
        moment_of_inertia_(calculateMomentOfInertia()) {}

  // Calculate forces between two spherocylinders
  std::pair<std::array<double, 3>, std::array<double, 3>> calculateContactForce(
      const SoftSpherocylinder& other) const {
    using namespace utils::geometry;

    // Create segments for collision detection
    Segment<3, double> seg1 = createSegment();
    Segment<3, double> seg2 = other.createSegment();

    // Get closest points between segments
    DCPSegmentSegment<double, 3> dcpQuery;
    auto result = dcpQuery.ComputeRobust(seg1, seg2);

    double overlap = radius_ + other.radius_ - result.distance;

    if (overlap <= 0) {
      return {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    }

    // Calculate contact geometry
    std::array<double, 3> r1, r2;
    for (int i = 0; i < 3; ++i) {
      r1[i] = result.closest[0][i] - position_[i];
      r2[i] = result.closest[1][i] - other.position_[i];
    }

    // Calculate effective radius based on contact geometry (from CellsMD3D)
    double R_eff = std::sqrt(radius_ * other.radius_ / (radius_ + other.radius_));

    // Hertzian elastic force with CellsMD3D's approach
    double force_magnitude = std::sqrt(R_eff) * k_cc * std::pow(overlap, 1.5);

    // Calculate normal direction
    std::array<double, 3> normal;
    double dist_inv = 1.0 / result.distance;
    for (int i = 0; i < 3; ++i) {
      normal[i] = (result.closest[1][i] - result.closest[0][i]) * dist_inv;
    }

    // Calculate relative velocity at contact point
    std::array<double, 3> va1 = cross(angular_velocity_, r1);
    std::array<double, 3> va2 = cross(other.angular_velocity_, r2);

    std::array<double, 3> v1, v2, dv;
    for (int i = 0; i < 3; ++i) {
      v1[i] = velocity_[i] + va1[i];
      v2[i] = other.velocity_[i] + va2[i];
      dv[i] = v2[i] - v1[i];
    }

    // Calculate normal and tangential velocity components
    double dv_dot_n = dot(dv, normal);
    std::array<double, 3> dv_n, dv_t;
    for (int i = 0; i < 3; ++i) {
      dv_n[i] = normal[i] * dv_dot_n;
      dv_t[i] = dv[i] - dv_n[i];
    }

    // Calculate effective mass (from CellsMD3D)
    double m_eff = mass_ * other.mass_ / (mass_ + other.mass_);

    // Normal damping force
    std::array<double, 3> f_damp_n;
    for (int i = 0; i < 3; ++i) {
      f_damp_n[i] = dv_n[i] * gamma_n * m_eff * overlap;
    }

    // Tangential damping force with friction limit
    std::array<double, 3> f_damp_t = {0.0, 0.0, 0.0};
    double v_t_mag = std::sqrt(dot(dv_t, dv_t));

    if (v_t_mag > 1e-10) {
      double f_damp_t_mag = std::min(gamma_t * m_eff * std::sqrt(overlap),
                                     cell_mu * force_magnitude / v_t_mag);
      for (int i = 0; i < 3; ++i) {
        f_damp_t[i] = -dv_t[i] * f_damp_t_mag;
      }
    }

    // Calculate total force
    std::array<double, 3> total_force;
    for (int i = 0; i < 3; ++i) {
      total_force[i] = -normal[i] * force_magnitude + f_damp_n[i] + f_damp_t[i];
    }

    // Calculate torque
    std::array<double, 3> torque = cross(r1, total_force);

    // Update contact stress for impedance calculation
    contact_stress_ += force_magnitude / (M_PI * radius_ * radius_);

    return {total_force, torque};
  }

  // Growth function with impedance
  void grow(double dt) {
    // Calculate impedance based on contact stress (from CellsMD3D)
    double impedance = std::exp(-impedance_lambda * contact_stress_);

    // Calculate nutrient-dependent base growth rate
    double nutrient_factor = nutrient_concentration_ / (nutrient_concentration_ + nutrient_Kc);
    double base_growth_rate = max_growth_rate * std::max(0.0, nutrient_factor - maintenance_rate);

    // Apply impedance to growth rate
    growth_rate_ = base_growth_rate * impedance;

    // Calculate length increase
    double dL = length_ * dt * growth_rate_ / impedance_tau;
    length_ += dL;

    // Update mass and moment of inertia
    mass_ = calculateMass();
    moment_of_inertia_ = calculateMomentOfInertia();

    // Reset contact stress for next timestep
    contact_stress_ = 0.0;
  }

  // Division function
  SoftSpherocylinder divide() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Random variations in division (from CellsMD3D)
    double dl = dis(gen) * var_length;
    double dangle = dis(gen) * var_angle;

    // Create daughter cell
    SoftSpherocylinder daughter(*this);

    // Adjust lengths (from CellsMD3D)
    double total_length = length_;
    length_ = total_length / 2 - radius_ + dl;
    daughter.length_ = total_length / 2 + radius_ - 2 * daughter.radius_ - dl;

    // Add random orientation changes
    std::array<double, 3> dir = {dis(gen), dis(gen), dis(gen)};
    double norm = std::sqrt(dot(dir, dir));
    for (int i = 0; i < 3; ++i) {
      dir[i] /= norm;
    }

    // Update positions
    std::array<double, 3> axis = getAxis();
    for (int i = 0; i < 3; ++i) {
      daughter.position_[i] = position_[i] + axis[i] * (length_ + radius_);
    }

    // Inherit properties
    daughter.growth_rate_ = growth_rate_;
    daughter.nutrient_concentration_ = nutrient_concentration_;

    // Reset dynamics
    daughter.velocity_ = {0.0, 0.0, 0.0};
    daughter.angular_velocity_ = {0.0, 0.0, 0.0};
    daughter.force_ = {0.0, 0.0, 0.0};
    daughter.torque_ = {0.0, 0.0, 0.0};
    daughter.contact_stress_ = 0.0;

    return daughter;
  }

  // ... (keep other methods the same)

  // Additional getters for growth and impedance
  double getContactStress() const { return contact_stress_; }
  double getImpedance() const { return std::exp(-impedance_lambda * contact_stress_); }

 private:
  double calculateMass() const {
    return M_PI * radius_ * radius_ * (length_ + 4.0 / 3.0 * radius_);
  }

  double calculateMomentOfInertia() const {
    double cylinder_mass = M_PI * radius_ * radius_ * length_;
    double sphere_mass = 4.0 / 3.0 * M_PI * std::pow(radius_, 3);

    // Cylinder contribution
    double I_cylinder = (1.0 / 12.0) * cylinder_mass * (3 * radius_ * radius_ + length_ * length_);

    // Sphere caps contribution
    double I_spheres = (2.0 / 5.0) * sphere_mass * radius_ * radius_;

    return I_cylinder + I_spheres;
  }

  // ... (keep other helper methods the same)

  std::array<double, 3> position_;     // Center position
  std::array<double, 4> orientation_;  // Quaternion orientation
  double length_;                      // Length of cylindrical part
  double radius_;                      // Radius of spherocylinder

  std::array<double, 3> velocity_;
  std::array<double, 3> angular_velocity_;
  std::array<double, 3> force_;
  std::array<double, 3> torque_;

  double growth_rate_;
  double nutrient_concentration_;
  mutable double contact_stress_;  // Accumulated contact stress for impedance
  double mass_;
  double moment_of_inertia_;
};

}  // namespace soft