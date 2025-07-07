#pragma once

#include "logger/ParticleLogger.h"
#include "simulation/ParticleData.h"
#include "simulation/ParticleManager.h"
#include "util/Config.h"

class Domain {
 public:
  Domain(const SimulationConfig& sim_config, const PhysicsConfig& physics_config, const SolverConfig& solver_config);

  void queueNewParticles(std::vector<Particle> particles);
  void commitNewParticles();
  void run();

 private:
  void rebalance();
  void exchangeGhostParticles();
  void resizeDomain();
  void printProgress(int current_iteration, double colony_radius) const;

  std::pair<std::array<double, 3>, std::array<double, 3>> calculateLocalBoundingBox() const;
  void calculateGlobalBounds();
  void padGlobalBounds(double& global_min_x, double& global_max_x, double& global_min_y, double& global_max_y,
                       double& global_min_z, double& global_max_z) const;

  void groupParticlesForExchange(std::vector<std::vector<ParticleData>>& particles_to_send,
                                 std::vector<Particle>& particles_to_keep);
  std::vector<ParticleData> exchangeParticleDataGlobal(const std::vector<std::vector<ParticleData>>& particles_to_send);
  std::vector<ParticleData> exchangeParticleDataWithAdjacentRank(const std::vector<ParticleData>& to_send_left,
                                                                 const std::vector<ParticleData>& to_send_right);
  void updateParticlesAfterExchange(std::vector<Particle>& particles_to_keep,
                                    const std::vector<ParticleData>& received_particles);
  void assignGlobalIDsToNewParticles();

  /**
   * @brief Re-assigns global IDs to all particles in the domain to ensure they are contiguous.
   * This is typically called after a rebalance operation.
   */
  void assignGlobalIDs();

  /**
   * @brief Adjusts the timestep `dt` based on solver performance.
   * @param solver_solution The solution from the physics engine for the last step.
   */
  void adjustDt(const PhysicsEngine::SolverSolution& solver_solution);

  const SimulationConfig& sim_config_;
  const PhysicsConfig& physics_config_;
  const SolverConfig& solver_config_;

  int rank_;
  int size_;

  std::unique_ptr<ParticleManager> particle_manager_;
  std::vector<Particle> new_particle_buffer;

  std::unique_ptr<vtk::ParticleLogger> particle_logger_;
  std::unique_ptr<vtk::ConstraintLogger> constraint_logger_;
  std::unique_ptr<vtk::DomainLogger> domain_logger_;

  MPI_Datatype mpi_particle_type_;

  std::array<double, 3> global_min_bounds_;
  std::array<double, 3> global_max_bounds_;
  std::array<double, 3> min_bounds_;
  std::array<double, 3> max_bounds_;

  double current_dt_;
  double elapsed_time_seconds_ = 0.0;
  double start_time_ = 0.0;
  int global_particle_count = 0;

  // For ETA calculation
  double last_eta_check_time_ = 0.0;
  double last_eta_check_sim_time_ = 0.0;
  double time_last_log_ = 0.0;
};