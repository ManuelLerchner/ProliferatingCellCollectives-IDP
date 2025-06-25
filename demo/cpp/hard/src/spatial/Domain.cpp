#include "Domain.h"

#include <petsc.h>

#include <algorithm>
#include <iostream>

#include "simulation/Particle.h"
#include "simulation/ParticleData.h"
#include "util/ParticleMPI.h"

Domain::Domain(const SimulationConfig& sim_config, const PhysicsConfig& physics_config, const SolverConfig& solver_config)
    : sim_config_(sim_config),
      physics_config_(physics_config),
      solver_config_(solver_config) {
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank_);
  MPI_Comm_size(PETSC_COMM_WORLD, &size_);

  particle_manager_ = std::make_unique<ParticleManager>(sim_config, physics_config, solver_config);

  std::cout << "Domain::run" << std::endl;

  int log_every_n_steps = 1;
  if (sim_config.log_frequency_seconds > 0) {
    log_every_n_steps = std::max(1, static_cast<int>(sim_config.log_frequency_seconds / sim_config.dt));
  }

  vtk_logger_ = vtk::createParticleLogger("./vtk_output", log_every_n_steps);
  constraint_loggers_ = vtk::createConstraintLogger("./vtk_output", log_every_n_steps);
  domain_decomposition_logger_ = vtk::createDomainDecompositionLogger("./vtk_output", log_every_n_steps);
  createParticleMPIType(&mpi_particle_type_);
}

void Domain::queueNewParticles(std::vector<Particle> particles) {
  new_particle_buffer.insert(new_particle_buffer.end(), particles.begin(), particles.end());
}

void Domain::commitNewParticles() {
  assignGlobalIDsToNewParticles();

  particle_manager_->local_particles.insert(
      particle_manager_->local_particles.end(),
      std::make_move_iterator(new_particle_buffer.begin()),
      std::make_move_iterator(new_particle_buffer.end()));

  // Sort particles by ID to maintain ordering across all ranks
  if (new_particle_buffer.size() > 0) {
    std::sort(particle_manager_->local_particles.begin(), particle_manager_->local_particles.end(),
              [](const Particle& a, const Particle& b) { return a.getGID() < b.getGID(); });

    for (int i = 0; i < particle_manager_->local_particles.size(); i++) {
      particle_manager_->local_particles[i].setLocalID(i);
    }
  }

  PetscInt num_added_local = new_particle_buffer.size();
  new_particle_buffer.clear();

  this->global_particle_count += globalReduce(num_added_local, MPI_SUM);
}

void Domain::run() {
  int num_steps = sim_config_.end_time / sim_config_.dt;

  for (int i = 0; i < num_steps; i++) {
    auto new_particles = particle_manager_->divideParticles();
    queueNewParticles(new_particles);
    commitNewParticles();

    if (i % sim_config_.domain_resize_frequency == 0) {
      resizeDomain();
      exchangeParticles();
    }

    auto particles_before_step = particle_manager_->local_particles;
    auto solver_solution = particle_manager_->step(i);

    if (vtk_logger_) {
      auto sim_state = createSimulationState(solver_solution, &particles_before_step);
      auto dd_state = createDomainDecompositionState();

      vtk_logger_->logTimestepComplete(sim_config_.dt, sim_state.get());
      constraint_loggers_->logTimestepComplete(sim_config_.dt, sim_state.get());

      domain_decomposition_logger_->logTimestepComplete(sim_config_.dt, dd_state.get());
      printProgress(i + 1, num_steps);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
  }
}

void Domain::exchangeParticles() {
  // 1. Determine which particles to send to which rank
  std::vector<std::vector<ParticleData>> particles_to_send(size_);
  std::vector<Particle> particles_to_keep;
  groupParticlesForExchange(particles_to_send, particles_to_keep);

  // 2. Exchange particle data
  auto recv_buffer = exchangeParticleData(particles_to_send);

  // 3. Update local particles
  updateParticlesAfterExchange(particles_to_keep, recv_buffer);
}

void Domain::resizeDomain() {
  calculateGlobalBounds();

  // Update local bounds
  double slice_width = (global_max_bounds_[0] - global_min_bounds_[0]) / size_;

  min_bounds_[0] = global_min_bounds_[0] + rank_ * slice_width;
  max_bounds_[0] = global_min_bounds_[0] + (rank_ + 1) * slice_width;
  min_bounds_[1] = global_min_bounds_[1];
  max_bounds_[1] = global_max_bounds_[1];
  min_bounds_[2] = global_min_bounds_[2];
  max_bounds_[2] = global_max_bounds_[2];
}

void Domain::printProgress(int current_iteration, int total_iterations) const {
  PetscPrintf(PETSC_COMM_WORLD, "\rProgress: %3d / %d (%5.1f%%) | Time: %3.1f min / %3.1f min | Particles: %4d",
              current_iteration, total_iterations,
              (double)current_iteration / total_iterations * 100,
              (double)current_iteration * sim_config_.dt / 60,
              (double)total_iterations * sim_config_.dt / 60,
              particle_manager_->global_particle_count);
}

std::array<double, 6> Domain::calculateLocalBoundingBox() const {
  double local_min_x = std::numeric_limits<double>::max();
  double local_max_x = std::numeric_limits<double>::lowest();
  double local_min_y = std::numeric_limits<double>::max();
  double local_max_y = std::numeric_limits<double>::lowest();
  double local_min_z = std::numeric_limits<double>::max();
  double local_max_z = std::numeric_limits<double>::lowest();

  for (const auto& p : particle_manager_->local_particles) {
    const auto& pos = p.getPosition();
    local_min_x = std::min(local_min_x, pos[0]);
    local_max_x = std::max(local_max_x, pos[0]);
    local_min_y = std::min(local_min_y, pos[1]);
    local_max_y = std::max(local_max_y, pos[1]);
    local_min_z = std::min(local_min_z, pos[2]);
    local_max_z = std::max(local_max_z, pos[2]);
  }
  return {local_min_x, local_max_x, local_min_y, local_max_y, local_min_z, local_max_z};
}

void Domain::calculateGlobalBounds() {
  auto local_bounds = calculateLocalBoundingBox();

  double global_min_x, global_max_x, global_min_y, global_max_y, global_min_z, global_max_z;
  getGlobalMinMax(local_bounds[0], local_bounds[1], global_min_x, global_max_x);
  getGlobalMinMax(local_bounds[2], local_bounds[3], global_min_y, global_max_y);
  getGlobalMinMax(local_bounds[4], local_bounds[5], global_min_z, global_max_z);

  padGlobalBounds(global_min_x, global_max_x, global_min_y, global_max_y, global_min_z, global_max_z);

  double padding = 2;
  global_min_bounds_ = {global_min_x - padding, global_min_y - padding, global_min_z};
  global_max_bounds_ = {global_max_x + padding, global_max_y + padding, global_max_z};
}

void Domain::padGlobalBounds(double& global_min_x, double& global_max_x, double& global_min_y, double& global_max_y,
                             double& global_min_z, double& global_max_z) const {
  if (global_max_x - global_min_x < size_ * sim_config_.min_box_size.x) {
    double padding = (size_ * sim_config_.min_box_size.x - (global_max_x - global_min_x)) / 2;
    global_min_x -= padding;
    global_max_x += padding;
  }
  if (global_max_y - global_min_y < size_ * sim_config_.min_box_size.y) {
    double padding = (size_ * sim_config_.min_box_size.y - (global_max_y - global_min_y)) / 2;
    global_min_y -= padding;
    global_max_y += padding;
  }
  if (global_max_z - global_min_z < size_ * sim_config_.min_box_size.z) {
    double padding = (size_ * sim_config_.min_box_size.z - (global_max_z - global_min_z)) / 2;
    global_min_z -= padding;
    global_max_z += padding;
  }
}

std::unique_ptr<vtk::DomainDecompositionState> Domain::createDomainDecompositionState() const {
  std::unique_ptr<vtk::DomainDecompositionState> state = std::make_unique<vtk::DomainDecompositionState>();
  state->domain_min = {min_bounds_[0], min_bounds_[1], min_bounds_[2]};
  state->domain_max = {max_bounds_[0], max_bounds_[1], max_bounds_[2]};
  state->dims[0] = size_;
  state->dims[1] = 1;
  state->dims[2] = 1;
  state->coords[0] = rank_;
  state->coords[1] = 0;
  state->coords[2] = 0;
  state->rank = rank_;
  return state;
}

std::unique_ptr<vtk::ParticleSimulationState> Domain::createSimulationState(
    const PhysicsEngine::SolverSolution& solver_solution, const std::vector<Particle>* particles_for_geometry) const {
  auto state = std::make_unique<vtk::ParticleSimulationState>();

  auto& local_particles = particle_manager_->local_particles;

  // Copy particles
  if (particles_for_geometry) {
    state->particles = *particles_for_geometry;
  } else {
    state->particles = local_particles;
  }
  state->constraints = std::move(solver_solution.constraints);

  // Initialize empty force/torque vectors (would be populated by physics engine in full implementation)
  state->forces.resize(local_particles.size());
  state->torques.resize(local_particles.size());
  state->velocities_linear.resize(local_particles.size());
  state->velocities_angular.resize(local_particles.size());

  for (int i = 0; i < local_particles.size(); i++) {
    state->forces[i] = local_particles[i].getForce();
    state->torques[i] = local_particles[i].getTorque();
    state->velocities_linear[i] = local_particles[i].getVelocityLinear();
    state->velocities_angular[i] = local_particles[i].getVelocityAngular();
  }

  // Calculate max overlap from constraints
  state->residual = solver_solution.residual;

  // Set other state values
  state->constraint_iterations = solver_solution.constraint_iterations;
  state->bbpgd_iterations = solver_solution.bbpgd_iterations;
  state->l0 = physics_config_.l0;

  return state;
}

void Domain::groupParticlesForExchange(std::vector<std::vector<ParticleData>>& particles_to_send,
                                       std::vector<Particle>& particles_to_keep) {
  double slice_width = (global_max_bounds_[0] - global_min_bounds_[0]) / size_;

  for (auto& p : particle_manager_->local_particles) {
    const auto& pos = p.getPosition();
    int target_rank = 0;
    if (slice_width > 1e-9) {  // Avoid division by zero
      target_rank = static_cast<int>(floor((pos[0] - global_min_bounds_[0]) / slice_width));
    }

    // Clamp target_rank to be within [0, size_ - 1]
    target_rank = std::max(0, std::min(size_ - 1, target_rank));

    if (target_rank == rank_) {
      particles_to_keep.push_back(p);
    } else {
      particles_to_send[target_rank].push_back(p.getData());
    }
  }
}

std::vector<ParticleData> Domain::exchangeParticleData(
    const std::vector<std::vector<ParticleData>>& particles_to_send) {
  std::vector<int> send_counts(size_, 0);
  std::vector<int> send_displacements(size_, 0);
  std::vector<ParticleData> send_buffer;

  int total_to_send = 0;
  for (int i = 0; i < size_; ++i) {
    send_counts[i] = particles_to_send[i].size();
    send_displacements[i] = total_to_send;
    total_to_send += send_counts[i];
  }

  send_buffer.reserve(total_to_send);
  for (int i = 0; i < size_; ++i) {
    send_buffer.insert(send_buffer.end(), particles_to_send[i].begin(), particles_to_send[i].end());
  }

  std::vector<int> recv_counts(size_, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);

  std::vector<int> recv_displacements(size_, 0);
  int total_to_receive = 0;
  for (int i = 0; i < size_; ++i) {
    recv_displacements[i] = total_to_receive;
    total_to_receive += recv_counts[i];
  }
  std::vector<ParticleData> recv_buffer(total_to_receive);

  MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displacements.data(), mpi_particle_type_,
                recv_buffer.data(), recv_counts.data(), recv_displacements.data(), mpi_particle_type_,
                PETSC_COMM_WORLD);

  return recv_buffer;
}

void Domain::updateParticlesAfterExchange(std::vector<Particle>& particles_to_keep,
                                          const std::vector<ParticleData>& received_particles) {
  particle_manager_->local_particles = std::move(particles_to_keep);

  for (const auto& pd : received_particles) {
    particle_manager_->local_particles.emplace_back(pd);
  }

  // Sort and re-index to maintain a consistent state.
  auto& local_particles = particle_manager_->local_particles;
  std::sort(local_particles.begin(), local_particles.end(),
            [](const Particle& a, const Particle& b) { return a.getGID() < b.getGID(); });

  for (int i = 0; i < local_particles.size(); ++i) {
    local_particles[i].setLocalID(i);
  }

  // After exchanging particles, we need to update the global particle count.
  int local_particle_count = particle_manager_->local_particles.size();
  MPI_Allreduce(&local_particle_count, &global_particle_count, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
}

void Domain::assignGlobalIDsToNewParticles() {
  PetscInt num_to_add_local = new_particle_buffer.size();

  PetscInt first_id_for_this_rank;
  MPI_Scan(&num_to_add_local, &first_id_for_this_rank, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  first_id_for_this_rank -= num_to_add_local;

  first_id_for_this_rank += this->global_particle_count;

  std::vector<PetscInt> new_ids(num_to_add_local);
  std::iota(new_ids.begin(), new_ids.end(), first_id_for_this_rank);
  for (PetscInt i = 0; i < num_to_add_local; ++i) {
    new_particle_buffer[i].setGID(new_ids[i]);
  }
}
