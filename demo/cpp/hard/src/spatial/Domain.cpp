#include "Domain.h"

#include <mpi.h>
#include <petsc.h>

#include <algorithm>
#include <iostream>
#include <numeric>

#include "simulation/Particle.h"
#include "simulation/ParticleData.h"
#include "util/ArrayMath.h"

Domain::Domain(const SimulationConfig& sim_config, const PhysicsConfig& physics_config, const SolverConfig& solver_config)
    : sim_config_(sim_config),
      physics_config_(physics_config),
      solver_config_(solver_config),
      current_dt_(sim_config.dt) {
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank_);
  MPI_Comm_size(PETSC_COMM_WORLD, &size_);
  start_time_ = MPI_Wtime();
  last_eta_check_time_ = start_time_;

  int log_every_n_steps = 1;
  if (sim_config.log_frequency_seconds > 0) {
    log_every_n_steps = std::max(1, static_cast<int>(sim_config.log_frequency_seconds / current_dt_));
  }

  vtk_logger_ = vtk::createParticleLogger("./vtk_output", log_every_n_steps);
  constraint_loggers_ = vtk::createConstraintLogger("./vtk_output", log_every_n_steps);
  domain_decomposition_logger_ = vtk::createDomainDecompositionLogger("./vtk_output", log_every_n_steps);
  ghost_logger_ = vtk::createParticleLogger("./vtk_output/ghost", log_every_n_steps);
  createParticleMPIType(&mpi_particle_type_);

  particle_manager_ = std::make_unique<ParticleManager>(sim_config, physics_config, solver_config, *vtk_logger_, *constraint_loggers_);
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

  PetscInt num_added_local = new_particle_buffer.size();
  new_particle_buffer.clear();

  this->global_particle_count += globalReduce(num_added_local, MPI_SUM);
}

void Domain::adjustDt(const PhysicsEngine::SolverSolution& solver_solution) {
  if (!sim_config_.enable_adaptive_dt) {
    return;
  }

  auto bbpgd_iterations_per_step = solver_solution.bbpgd_iterations / (double)solver_solution.constraint_iterations;

  // Adjust dt based on the number of BBPGD iterations
  if (bbpgd_iterations_per_step > sim_config_.target_bbpgd_iterations) {
    current_dt_ *= (1.0 - sim_config_.dt_adjust_factor);  // Decrease dt
  } else {
    current_dt_ *= (1.0 + sim_config_.dt_adjust_factor);  // Increase dt
  }

  // Clamp dt to the min/max values
  current_dt_ = std::max(sim_config_.min_dt, std::min(sim_config_.max_dt, current_dt_));
}

void Domain::run() {
  using namespace utils::ArrayMath;

  int i = 0;
  while (elapsed_time_seconds_ < sim_config_.end_time) {
    auto new_particles = particle_manager_->divideParticles();
    queueNewParticles(new_particles);
    commitNewParticles();

    auto [local_min, local_max] = calculateLocalBoundingBox();
    int needs_resize_local = infinity_norm(local_max - min_bounds_) > physics_config_.l0 * 2 || infinity_norm(local_min - min_bounds_) > physics_config_.l0 * 2 ? 1 : 0;
    bool needs_resize_global = globalReduce(needs_resize_local, MPI_SUM) > 0;

    if (needs_resize_global || i % sim_config_.domain_resize_frequency == 0) {
      resizeDomain();
      rebalance();
    }

    auto update_ghosts_fn = [this]() { this->exchangeGhostParticles(); };

    auto particles_before_step = particle_manager_->local_particles;
    auto solver_solution = particle_manager_->step(i, update_ghosts_fn);
    elapsed_time_seconds_ += current_dt_;

    if (sim_config_.enable_adaptive_dt && i % sim_config_.dt_adjust_frequency == 0) {
      adjustDt(solver_solution);
    }

    if (vtk_logger_) {
      auto sim_state = createSimulationState(solver_solution, particles_before_step);
      auto dd_state = createDomainDecompositionState();

      auto ghost_state = createSimulationState(solver_solution, particle_manager_->ghost_particles);

      vtk_logger_->logTimestepComplete(current_dt_, sim_state.get());
      // ghost_logger_->logTimestepComplete(sim_config_.dt, ghost_state.get());
      constraint_loggers_->logTimestepComplete(current_dt_, sim_state.get());

      domain_decomposition_logger_->logTimestepComplete(current_dt_, dd_state.get());

      // Determine colony radius
      double colony_radius_local = 0;
      for (const auto& p : particle_manager_->local_particles) {
        double distance = utils::ArrayMath::magnitude(p.getPosition());
        colony_radius_local = std::max(colony_radius_local, distance);
      }

      double colony_radius_global = globalReduce(colony_radius_local, MPI_MAX);

      printProgress(i + 1, colony_radius_global);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
    i++;
  }
}

void Domain::rebalance() {
  // 1. Determine which particles to send to which rank
  std::vector<std::vector<ParticleData>> particles_to_send(size_);
  std::vector<Particle> particles_to_keep;
  groupParticlesForExchange(particles_to_send, particles_to_keep);

  // 2. Exchange particle data
  auto recv_buffer = exchangeParticleDataGlobal(particles_to_send);

  // 3. Update local particles
  updateParticlesAfterExchange(particles_to_keep, recv_buffer);

  // 4. Re-assign global IDs to all particles to keep them contiguous
  assignGlobalIDs();
}

void Domain::exchangeGhostParticles() {
  std::vector<ParticleData> particles_to_send_left;
  std::vector<ParticleData> particles_to_send_right;
  double buffer = physics_config_.l0 * 0.5;

  for (auto& p : particle_manager_->local_particles) {
    const auto& pos = p.getPosition();
    const auto& length = p.getLength();
    double buffer = physics_config_.l0 * 0.1;

    if (pos[0] - length - buffer < min_bounds_[0] && rank_ > 0) {
      particles_to_send_left.push_back(p.getData());
    }
    if (pos[0] + length + buffer > max_bounds_[0] && rank_ < size_ - 1) {
      particles_to_send_right.push_back(p.getData());
    }
  }

  auto received_particles_data = exchangeParticleDataWithAdjacentRank(particles_to_send_left, particles_to_send_right);

  particle_manager_->ghost_particles.clear();
  for (const auto& pd : received_particles_data) {
    particle_manager_->ghost_particles.emplace_back(pd);
  }
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

  particle_manager_->updateDomainBounds(min_bounds_, max_bounds_);
}

void Domain::printProgress(int current_iteration, double colony_radius) const {
  double time_elapsed_minutes = elapsed_time_seconds_ / 60.0;
  double time_total_minutes = sim_config_.end_time / 60.0;
  double progress_percent = (elapsed_time_seconds_ / sim_config_.end_time) * 100.0;

  std::string eta_str = "N/A";
  if (elapsed_time_seconds_ > 1.0) {
    double current_wall_time = MPI_Wtime();
    double wall_time_since_last_check = current_wall_time - last_eta_check_time_;
    double sim_time_since_last_check = elapsed_time_seconds_ - last_eta_check_sim_time_;

    if (sim_time_since_last_check > 1e-9) {
      double wall_seconds_per_sim_seconds = wall_time_since_last_check / sim_time_since_last_check;
      double estimated_remaining_wall_time_seconds = (sim_config_.end_time - elapsed_time_seconds_) * wall_seconds_per_sim_seconds;

      char buffer[100];
      snprintf(buffer, sizeof(buffer), "%.1f min", estimated_remaining_wall_time_seconds / 60.0);
      eta_str = buffer;

      // Update for next ETA calculation
      const_cast<Domain*>(this)->last_eta_check_time_ = current_wall_time;
      const_cast<Domain*>(this)->last_eta_check_sim_time_ = elapsed_time_seconds_;
    }
  }

  PetscPrintf(PETSC_COMM_WORLD, "\n Time: %3.1f / %3.1f min (%5.1f%%) | ETA: %s | Iter: %d | dt: %.2e | Particles: %d | Colony radius: %.1f",
              time_elapsed_minutes,
              time_total_minutes,
              progress_percent,
              eta_str.c_str(),
              current_iteration,
              current_dt_,
              global_particle_count,
              colony_radius);
}

std::pair<std::array<double, 3>, std::array<double, 3>> Domain::calculateLocalBoundingBox() const {
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
  return {{local_min_x, local_min_y, local_min_z}, {local_max_x, local_max_y, local_max_z}};
}

void Domain::calculateGlobalBounds() {
  auto [local_min, local_max] = calculateLocalBoundingBox();

  double global_min_x, global_max_x, global_min_y, global_max_y, global_min_z, global_max_z;
  getGlobalMinMax(local_min[0], local_max[0], global_min_x, global_max_x);
  getGlobalMinMax(local_min[1], local_max[1], global_min_y, global_max_y);
  getGlobalMinMax(local_min[2], local_max[2], global_min_z, global_max_z);

  // Handle case where no particles exist in the simulation yet
  if (global_max_x < global_min_x) {
    global_min_x = -1.0;
    global_max_x = 1.0;
    global_min_y = -1.0;
    global_max_y = 1.0;
    global_min_z = -1.0;
    global_max_z = 1.0;
  }

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

  auto spatial_grid = particle_manager_->physics_engine->getCollisionDetectorSpatialGrid();

  state->domain_min = {min_bounds_[0], min_bounds_[1], min_bounds_[2]};
  state->domain_max = {max_bounds_[0], max_bounds_[1], max_bounds_[2]};
  state->rank = rank_;

  state->link_cells_min = spatial_grid.getDomainMin();
  state->link_cells_max = spatial_grid.getDomainMax();
  state->link_cells_grid_dims = spatial_grid.getGridDims();

  return state;
}

std::unique_ptr<vtk::ParticleSimulationState> Domain::createSimulationState(
    const PhysicsEngine::SolverSolution& solver_solution, const std::vector<Particle>& particles) const {
  auto state = std::make_unique<vtk::ParticleSimulationState>();

  auto& local_particles = particle_manager_->local_particles;

  state->particles = particles;
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

  state->num_constraints.resize(local_particles.size());
  for (int i = 0; i < local_particles.size(); i++) {
    state->num_constraints[i] = local_particles[i].getNumConstraints();
  }

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

std::vector<ParticleData> Domain::exchangeParticleDataGlobal(
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

std::vector<ParticleData> Domain::exchangeParticleDataWithAdjacentRank(
    const std::vector<ParticleData>& to_send_left, const std::vector<ParticleData>& to_send_right) {
  MPI_Status status;
  int left_neighbor = (rank_ > 0) ? rank_ - 1 : MPI_PROC_NULL;
  int right_neighbor = (rank_ < size_ - 1) ? rank_ + 1 : MPI_PROC_NULL;

  int send_count_left = to_send_left.size();
  int send_count_right = to_send_right.size();
  int recv_count_left = 0;
  int recv_count_right = 0;

  // Exchange counts: send to right, receive from left
  MPI_Sendrecv(&send_count_right, 1, MPI_INT, right_neighbor, 0, &recv_count_left, 1, MPI_INT, left_neighbor, 0,
               PETSC_COMM_WORLD, &status);

  // Exchange counts: send to left, receive from right
  MPI_Sendrecv(&send_count_left, 1, MPI_INT, left_neighbor, 1, &recv_count_right, 1, MPI_INT, right_neighbor, 1,
               PETSC_COMM_WORLD, &status);

  std::vector<ParticleData> received_from_left(recv_count_left);
  std::vector<ParticleData> received_from_right(recv_count_right);

  // Exchange data: send to right, receive from left
  MPI_Sendrecv(to_send_right.data(), send_count_right, mpi_particle_type_, right_neighbor, 2, received_from_left.data(),
               recv_count_left, mpi_particle_type_, left_neighbor, 2, PETSC_COMM_WORLD, &status);
  MPI_Sendrecv(to_send_left.data(), send_count_left, mpi_particle_type_, left_neighbor, 3, received_from_right.data(),
               recv_count_right, mpi_particle_type_, right_neighbor, 3, PETSC_COMM_WORLD, &status);

  std::vector<ParticleData> received_particles;
  received_particles.reserve(recv_count_left + recv_count_right);
  received_particles.insert(received_particles.end(), received_from_left.begin(), received_from_left.end());
  received_particles.insert(received_particles.end(), received_from_right.begin(), received_from_right.end());

  return received_particles;
}

void Domain::updateParticlesAfterExchange(std::vector<Particle>& particles_to_keep,
                                          const std::vector<ParticleData>& received_particles) {
  particle_manager_->local_particles = std::move(particles_to_keep);

  for (const auto& pd : received_particles) {
    particle_manager_->local_particles.emplace_back(pd);
  }

  // After exchanging particles, we need to update the global particle count.
  int local_particle_count = particle_manager_->local_particles.size();
  MPI_Allreduce(&local_particle_count, &global_particle_count, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
}

void Domain::assignGlobalIDs() {
  // Sort particles by their x-position to group close particles.
  std::sort(particle_manager_->local_particles.begin(), particle_manager_->local_particles.end(),
            [](const Particle& a, const Particle& b) {
              return a.getPosition()[1] < b.getPosition()[1];
            });

  PetscInt local_particle_count = particle_manager_->local_particles.size();

  PetscInt first_id_for_this_rank;
  MPI_Scan(&local_particle_count, &first_id_for_this_rank, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  first_id_for_this_rank -= local_particle_count;

  for (PetscInt i = 0; i < local_particle_count; ++i) {
    particle_manager_->local_particles[i].setGID(first_id_for_this_rank + i);
  }
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
