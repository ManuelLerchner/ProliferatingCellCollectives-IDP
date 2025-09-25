#pragma once

#include <chrono>
#include <memory>
#include <vector>

#include "logger/ConstraintLogger.h"
#include "logger/DomainLogger.h"
#include "logger/ParticleLogger.h"
#include "logger/SimulationLogger.h"
#include "simulation/ParticleManager.h"
#include "util/Config.h"

class Domain {
 public:
  Domain(const SimulationParameters params, bool preserve_existing = false, size_t step = 0, const std::string& output_dir = "./vtk_output");

  void queueNewParticles(std::vector<Particle> particles);
  void commitNewParticles();
  void run();

  // Initialize simulation state from VTK file/directory
  static Domain initializeFromVTK(const SimulationParameters& params, const std::string& vtk_path, const std::string& output_dir);

 private:
  void rebalance();
  void exchangeGhostParticles();
  void resizeDomain();
  void printProgress(int current_iteration, double colony_radius, double cpu_time_s) const;

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
  void assignGlobalIDs();

  void adjustDt(const ParticleManager::SolverSolution& solver_solution);

  SimulationParameters params_;

  std::unique_ptr<ParticleManager> particle_manager_;
  std::unique_ptr<vtk::ParticleLogger> particle_logger_;
  std::unique_ptr<vtk::ConstraintLogger> constraint_logger_;
  std::unique_ptr<vtk::DomainLogger> domain_logger_;
  std::unique_ptr<vtk::SimulationLogger> simulation_logger_;

  std::vector<Particle> new_particle_buffer;
  MPI_Datatype mpi_particle_type_;

  size_t iter = 0;

  int rank_;
  int size_;
  int global_particle_count = 0;

  double simulation_time_seconds_ = 0.0;
  double time_last_log_ = 0.0;

  double start_time_ = 0;
  double step_start_time_ = 0;
  double last_eta_check_time_ = 0;
  double last_eta_check_sim_time_ = 0.0;
  double last_wall_per_sim_seconds_ = 0.0;
  double last_eta_check_radius_ = 0.0;
  double last_growth_rate_wall_ = 0.0;
  std::chrono::steady_clock::time_point sim_start_time_;

  std::array<double, 3> min_bounds_ = {0, 0, 0};
  std::array<double, 3> max_bounds_ = {0, 0, 0};
  std::array<double, 3> global_min_bounds_ = {0, 0, 0};
  std::array<double, 3> global_max_bounds_ = {0, 0, 0};
};