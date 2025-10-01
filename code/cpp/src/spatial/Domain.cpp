#include "Domain.h"

#include <mpi.h>
#include <petsc.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>

#include "loader/VTKStateLoader.h"
#include "logger/ParameterLogger.h"
#include "simulation/Particle.h"
#include "simulation/ParticleData.h"
#include "util/ArrayMath.h"
#include "util/MetricsUtil.h"

Domain::Domain(SimulationParameters& params, bool preserve_existing, size_t iter, const std::string& mode)
    : params_(params),
      iter(iter) {
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank_);
  MPI_Comm_size(PETSC_COMM_WORLD, &size_);
  start_time_ = MPI_Wtime();
  last_eta_check_time_ = start_time_;

  std::string output_dir = "./vtk_output_" + mode;
  particle_logger_ = std::make_unique<vtk::ParticleLogger>(output_dir, "particles", preserve_existing, iter);
  constraint_logger_ = std::make_unique<vtk::ConstraintLogger>(output_dir, "constraints", preserve_existing, iter);
  domain_logger_ = std::make_unique<vtk::DomainLogger>(output_dir, "domain", preserve_existing, iter);
  simulation_logger_ = std::make_unique<vtk::SimulationLogger>(output_dir, "simulation", preserve_existing, iter);

  createParticleMPIType(&mpi_particle_type_);

  particle_manager_ = std::make_unique<ParticleManager>(params, *particle_logger_, *constraint_logger_, mode);
  step_start_time_ = MPI_Wtime();
  sim_start_time_ = std::chrono::steady_clock::now();

  if (rank_ == 0) {
    vtk::ParameterLogger parameter_logger(output_dir, "parameters", true, iter);
    parameter_logger.log(params);
  }
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

void Domain::adaptDt() {
  std::vector<double> local_velocities;

  // Collect velocities
  for (const auto& p : particle_manager_->local_particles) {
    double particle_velocity = utils::ArrayMath::magnitude(p.getVelocityLinear());
    double particle_growth_rate = p.getLdot();
    local_velocities.push_back(particle_velocity + particle_growth_rate);
  }

  // Compute median locally
  double local_median_velocity = 0.0;
  if (!local_velocities.empty()) {
    std::nth_element(local_velocities.begin(),
                     local_velocities.begin() + local_velocities.size() / 2,
                     local_velocities.end());
    local_median_velocity = local_velocities[local_velocities.size() / 2];

    // For even-sized vector, average the two middle values
    if (local_velocities.size() % 2 == 0) {
      auto max_it = std::max_element(local_velocities.begin(),
                                     local_velocities.begin() + local_velocities.size() / 2);
      local_median_velocity = 0.5 * (*max_it + local_median_velocity);
    }
  }

  // Global reduction (median across all MPI ranks is tricky; approximate with average of medians)

  double global_median_velocity = globalReduce(local_median_velocity, MPI_SUM) / size_;

  // CFL condition
  double cfl = 0.25 * params_.solver_config.tolerance;
  double optional_cfl_cap_dt_local = cfl / global_median_velocity;

  // Smooth change to avoid oscillations
  double new_local_dt = params_.sim_config.dt_s * (1.0 - adaptive_dt_smoothing_alpha_) +
                        adaptive_dt_smoothing_alpha_ * optional_cfl_cap_dt_local;
  new_local_dt = std::clamp(new_local_dt, 0.9 * params_.sim_config.dt_s, 1.1 * params_.sim_config.dt_s);

  params_.sim_config.dt_s = new_local_dt;
}

void Domain::run() {
  using namespace utils::ArrayMath;

  double colony_radius = 0;

  while (colony_radius < params_.sim_config.end_radius) {
    auto new_particles = particle_manager_->divideParticles();
    queueNewParticles(new_particles);
    commitNewParticles();

    rebalance();
    assignGlobalIDs();
    resizeDomain();

    auto update_ghosts_fn = [this]() {
      rebalance();
      exchangeGhostParticles();
    };

    double mpi_start_time = MPI_Wtime();
    auto solver_solution = particle_manager_->step(iter, update_ghosts_fn);
    double mpi_end_time = MPI_Wtime();
    double solver_wall_time = mpi_end_time - mpi_start_time;  // wall time for this step (local)

    simulation_time_seconds_ += params_.sim_config.dt_s;

    adaptDt();

    // Determine colony radius
    colony_radius = globalReduce(
        std::accumulate(particle_manager_->local_particles.begin(), particle_manager_->local_particles.end(), 0.0,
                        [this](double acc, const Particle& p) {
                          return std::max(acc, utils::ArrayMath::magnitude(particle_manager_->collision_detector_.getParticleEndpoints(p).second));
                        }),
        MPI_MAX);

    // Log simulation parameters with performance metrics

    // Get performance metrics
    double memory_usage_mb = utils::getCurrentMemoryUsageMB();
    double peak_memory_mb = utils::getPeakMemoryUsageMB();
    double cpu_time_s = utils::getCPUTimeSeconds();
    double load_imbalance = utils::calculateLoadImbalance(particle_manager_->local_particles.size());

    size_t total_constraints = globalReduce(solver_solution.constraints.size(), MPI_SUM);

    vtk::SimulationStep step_data{
        .simulation_time_s = simulation_time_seconds_,
        .time_since_last_log_s = MPI_Wtime() - time_last_log_,
        .step = iter,

        .num_particles = global_particle_count,
        .num_constraints = total_constraints,
        .colony_radius = colony_radius,

        // Solver metrics
        .recursive_iterations = solver_solution.constraint_iterations,
        .bbpgd_iterations = solver_solution.bbpgd_iterations,
        .max_overlap = solver_solution.max_overlap,
        .residual = solver_solution.residual,
        .dt_s = params_.sim_config.dt_s,

        // Add performance metrics
        .memory_usage_mb = memory_usage_mb,
        .peak_memory_mb = peak_memory_mb,
        .cpu_time_s = cpu_time_s,
        .mpi_comm_time_s = solver_wall_time,
        .load_imbalance = load_imbalance};

    bool log_due_to_colony_radius = params_.sim_config.log_every_colony_radius_delta && (colony_radius - colony_radius_last_log_ > params_.sim_config.log_every_colony_radius_delta);
    bool log_due_to_sim_time = params_.sim_config.log_every_sim_time_delta && (simulation_time_seconds_ - simulation_time_last_log_ > params_.sim_config.log_every_sim_time_delta);

    if (log_due_to_colony_radius || log_due_to_sim_time || iter == 0 || colony_radius >= params_.sim_config.end_radius) {
      particle_logger_->log(particle_manager_->local_particles);
      // constraint_logger_->log(solver_solution.constraints);
      domain_logger_->log(std::make_pair(min_bounds_, max_bounds_));
      simulation_logger_->log(step_data);

      colony_radius_last_log_ = colony_radius;
      simulation_time_last_log_ = simulation_time_seconds_;
      time_last_log_ = MPI_Wtime();
    }

    printProgress(iter + 1, colony_radius, std::chrono::duration<double>(std::chrono::steady_clock::now() - sim_start_time_).count(), step_data);
    // PetscPrintf(PETSC_COMM_WORLD, "\n");
    iter++;
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
}

void Domain::exchangeGhostParticles() {
  std::vector<ParticleData> particles_to_send_left;
  std::vector<ParticleData> particles_to_send_right;
  double buffer = params_.physics_config.l0 * 0.5;

  for (auto& p : particle_manager_->local_particles) {
    const auto& pos = p.getPosition();
    const auto& length = p.getLength();
    double buffer = params_.physics_config.l0 * 0.1;

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
  double buffer = params_.physics_config.l0 * 0.5;
  max_bounds_[0] = global_min_bounds_[0] + (rank_ + 1) * slice_width;
  min_bounds_[1] = global_min_bounds_[1];
  max_bounds_[1] = global_max_bounds_[1];
  min_bounds_[2] = global_min_bounds_[2];
  max_bounds_[2] = global_max_bounds_[2];

  particle_manager_->updateDomainBounds(min_bounds_, max_bounds_);
}

void Domain::printProgress(int current_iteration, double colony_radius, double cpu_time_s, vtk::SimulationStep step) {
  double time_elapsed_seconds = simulation_time_seconds_;
  double progress_percent = (colony_radius / params_.sim_config.end_radius) * 100.0;

  std::string eta_str = "N/A";
  double growth_rate_wall = 0.0;  // colony radius growth per real second

  double current_wall_time = MPI_Wtime();
  double wall_time_since_last_check = current_wall_time - last_eta_check_time_;
  double sim_time_since_last_check = simulation_time_seconds_ - last_eta_check_sim_time_;

  if (wall_time_since_last_check > 1 && sim_time_since_last_check > 1e-12) {
    double current_ratio = wall_time_since_last_check / sim_time_since_last_check;

    // Exponential smoothing
    double alpha = 0.1;
    double smoothed_ratio = last_wall_per_sim_seconds_ * (1.0 - alpha) + current_ratio * alpha;

    // Estimate growth rate per sim second
    double delta_radius = colony_radius - last_eta_check_radius_;
    double growth_rate_sim = delta_radius / sim_time_since_last_check;

    // Estimate remaining sim time and convert to wall time
    double remaining_sim_time = (params_.sim_config.end_radius - colony_radius) / growth_rate_sim;
    double estimated_remaining_wall_time_seconds = remaining_sim_time * smoothed_ratio;

    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%.1f min", estimated_remaining_wall_time_seconds / 60.0);
    eta_str = buffer;

    // Colony growth per wall time
    double raw_growth_rate = delta_radius / wall_time_since_last_check;
    double smoothed_growth_rate = last_growth_rate_wall_ * (1.0 - alpha) + raw_growth_rate * alpha;
    growth_rate_wall = smoothed_growth_rate;

    // Update for next call (mark members as mutable to avoid const_cast)
    last_wall_per_sim_seconds_ = smoothed_ratio;
    last_eta_check_radius_ = colony_radius;
    last_growth_rate_wall_ = smoothed_growth_rate;

    last_eta_check_time_ = current_wall_time;
    last_eta_check_sim_time_ = simulation_time_seconds_;

    int total_ranks;
    MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);

    PetscPrintf(PETSC_COMM_WORLD,
                "\n Colony radius: %.3f / %.1f (%4.1f%%) | Time: %3.1f | ETA: %s | "
                "dt: %4.1es | CPU: %3.1fs | Particles: %d | Growth: %.4f/s | BBPGD iters: %ld | Residual: %.2e | Ranks %d\n",
                colony_radius,
                params_.sim_config.end_radius,
                progress_percent,
                time_elapsed_seconds,
                eta_str.c_str(),
                params_.sim_config.dt_s,
                cpu_time_s,
                global_particle_count,
                growth_rate_wall,
                step.bbpgd_iterations,
                step.residual,
                total_ranks);
  }
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
  const auto& sim_config_ = params_.sim_config;

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

  for (auto& pd : received_particles) {
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
              return a.getPosition()[0] < b.getPosition()[0];
            });

  PetscInt local_particle_count = particle_manager_->local_particles.size();

  PetscInt first_id_for_this_rank = 0;
  MPI_Scan(&local_particle_count, &first_id_for_this_rank, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  first_id_for_this_rank -= local_particle_count;

  for (PetscInt i = 0; i < local_particle_count; ++i) {
    particle_manager_->local_particles[i].setGID(first_id_for_this_rank + i);
  }
}

void Domain::assignGlobalIDsToNewParticles() {
  PetscInt num_to_add_local = new_particle_buffer.size();

  PetscInt first_id_for_this_rank = 0;
  MPI_Scan(&num_to_add_local, &first_id_for_this_rank, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  first_id_for_this_rank -= num_to_add_local;

  first_id_for_this_rank += this->global_particle_count;

  std::vector<PetscInt> new_ids(num_to_add_local);
  std::iota(new_ids.begin(), new_ids.end(), first_id_for_this_rank);
  for (PetscInt i = 0; i < num_to_add_local; ++i) {
    new_particle_buffer[i].setGID(new_ids[i]);
  }
}

Domain Domain::initializeFromVTK(SimulationParameters& params, const std::string& vtk_path, const std::string& mode) {
  std::filesystem::path path(vtk_path);
  std::string vtk_dir;

  // If path is a file, use its parent directory
  if (std::filesystem::is_regular_file(path)) {
    vtk_dir = path.parent_path().string();
  } else if (std::filesystem::is_directory(path)) {
    vtk_dir = vtk_path;
  } else {
    throw std::runtime_error("VTK path must be a file or directory");
  }

  // Queue particles for initialization
  vtk::VTKStateLoader loader(vtk_dir);
  auto state = loader.loadLatestState();

  PetscPrintf(PETSC_COMM_WORLD, "Loading %zu particles from VTK state at time %.2f s (iter %u)\n",
              state.particles.size(), state.simulation_time_s, state.step);

  Domain domain(params, true, state.step, mode);

  // Update simulation time
  domain.simulation_time_seconds_ = state.simulation_time_s;

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if (rank == 0) {
    domain.queueNewParticles(state.particles);
  }

  // Commit particles and update state
  domain.commitNewParticles();
  domain.resizeDomain();

  // Exchange ghost particles to ensure proper initialization
  domain.rebalance();
  domain.exchangeGhostParticles();

  return domain;
}
