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
  void exchangeParticles();
  void resizeDomain();
  void redistributeParticles();
  void resetLocalParticles();
  void printProgress(int current_iteration, int total_iterations) const;

  std::unique_ptr<vtk::DomainDecompositionState> createDomainDecompositionState() const;
  std::unique_ptr<vtk::ParticleSimulationState> createSimulationState(
      const PhysicsEngine::SolverSolution& solver_solution, const std::vector<Particle>* particles_for_geometry) const;

  SimulationConfig sim_config_;
  PhysicsConfig physics_config_;
  SolverConfig solver_config_;

  int rank_;
  int size_;

  std::unique_ptr<ParticleManager> particle_manager_;
  std::vector<Particle> new_particle_buffer;

  std::unique_ptr<vtk::SimulationLogger> vtk_logger_;
  std::unique_ptr<vtk::SimulationLogger> constraint_loggers_;
  std::unique_ptr<vtk::SimulationLogger> domain_decomposition_logger_;

  MPI_Datatype mpi_particle_type_;

  std::array<double, 3> global_min_bounds_;
  std::array<double, 3> global_max_bounds_;
  std::array<double, 3> min_bounds_;
  std::array<double, 3> max_bounds_;

  PetscInt global_particle_count = 0;
};