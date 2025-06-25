#pragma once

#include "logger/VTK.h"
#include "simulation/ParticleData.h"
#include "simulation/ParticleManager.h"
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
  void printProgress(int current_iteration, int total_iterations) const;

  std::unique_ptr<vtk::DomainDecompositionState> createDomainDecompositionState() const;
  std::unique_ptr<vtk::ParticleSimulationState> createSimulationState(
      const PhysicsEngine::SolverSolution& solver_solution, const std::vector<Particle>& particles) const;

  std::array<double, 6> calculateLocalBoundingBox() const;
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

  const SimulationConfig& sim_config_;
  const PhysicsConfig& physics_config_;
  const SolverConfig& solver_config_;

  int rank_;
  int size_;

  std::unique_ptr<ParticleManager> particle_manager_;
  std::vector<Particle> new_particle_buffer;

  std::unique_ptr<vtk::SimulationLogger> vtk_logger_;
  std::unique_ptr<vtk::SimulationLogger> ghost_logger_;
  std::unique_ptr<vtk::SimulationLogger> constraint_loggers_;
  std::unique_ptr<vtk::SimulationLogger> domain_decomposition_logger_;

  MPI_Datatype mpi_particle_type_;

  std::array<double, 3> global_min_bounds_;
  std::array<double, 3> global_max_bounds_;
  std::array<double, 3> min_bounds_;
  std::array<double, 3> max_bounds_;

  PetscInt global_particle_count = 0;
};