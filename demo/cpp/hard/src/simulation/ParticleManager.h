#pragma once
#include <petsc.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dynamics/PhysicsEngine.h"
#include "logger/VTK.h"
#include "simulation/Particle.h"
#include "spatial/ConstraintGenerator.h"
#include "util/Config.h"

class ParticleManager {
 public:
  ParticleManager(SimulationConfig sim_config, PhysicsConfig physics_config, SolverConfig solver_config);

  void queueNewParticle(Particle p);
  void commitNewParticles();
  void resetLocalParticles();

  void divideParticles();

  void moveLocalParticlesFromSolution(const PhysicsEngine::MovementSolution& solution);
  void growLocalParticlesFromSolution(const PhysicsEngine::GrowthSolution& solution);

  void run(int num_steps);

  void redistributeParticles();
  void updateDomainBounds();

  std::vector<Particle> local_particles;
  PetscInt global_particle_count = 0;

  std::unique_ptr<ConstraintGenerator> constraint_generator;
  std::unique_ptr<PhysicsEngine> physics_engine;

  SimulationConfig sim_config_;
  PhysicsConfig physics_config_;
  SolverConfig solver_config_;

  vtk::SimulationLogger& getVTKLogger() const { return *vtk_logger_; }
  vtk::SimulationLogger& getConstraintLogger() const { return *constraint_loggers_; }

  std::unique_ptr<vtk::ParticleSimulationState> createSimulationState(
      const PhysicsEngine::SolverSolution& solver_solution, const std::vector<Particle>* particles_for_geometry = nullptr) const;

 private:
  std::vector<Particle> new_particle_buffer;

  std::unique_ptr<vtk::SimulationLogger> vtk_logger_;
  std::unique_ptr<vtk::SimulationLogger> constraint_loggers_;
  std::unique_ptr<vtk::SimulationLogger> domain_decomposition_logger_;

  void printProgress(int current_iteration, int total_iterations) const;

  std::array<double, 3> domain_min_ = {0.0, 0.0, 0.0};
  std::array<double, 3> domain_max_ = {0.0, 0.0, 0.0};
  MPI_Datatype mpi_particle_data_type_;

  // Cartesian communicator info
  MPI_Comm cart_comm_;
  int dims_[2] = {0, 0};
  int periods_[2] = {0, 0};
  int coords_[2] = {0, 0};
};
