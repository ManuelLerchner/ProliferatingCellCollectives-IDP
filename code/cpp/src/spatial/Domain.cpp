#include "Domain.h"

#include <mpi.h>
#include <omp.h>
#include <petsc.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
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
  ghost_particle_logger_ = std::make_unique<vtk::ParticleLogger>(output_dir, "ghost_particles", preserve_existing, iter);
  constraint_logger_ = std::make_unique<vtk::ConstraintLogger>(output_dir, "constraints", preserve_existing, iter);
  domain_logger_ = std::make_unique<vtk::DomainLogger>(output_dir, "domain", preserve_existing, iter);
  simulation_logger_ = std::make_unique<vtk::SimulationLogger>(output_dir, "simulation", preserve_existing, iter);

  createParticleMPIType(&mpi_particle_type_);

  particle_manager_ = std::make_unique<ParticleManager>(params, *particle_logger_, *constraint_logger_, mode);
  step_start_time_ = MPI_Wtime();
  time_last_log_ = MPI_Wtime();

  center_radius = 8;

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
  int valid = 0;
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
    valid = 1;
  }

  // Global reduction (median across all MPI ranks is tricky; approximate with average of medians)

  double sum_velocities = globalReduce(local_median_velocity, MPI_SUM);
  int num_valid = globalReduce(valid, MPI_SUM);
  double global_median_velocity = (num_valid > 0) ? (sum_velocities / num_valid) : 0.0;

  if (global_median_velocity == 0.0) {
    // No movement; keep current dt
    return;
  }

  // CFL condition
  double cfl = 0.5 * params_.solver_config.tolerance;
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
    double wall_time = MPI_Wtime() - start_time_;
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
        .cpu_time_s = wall_time,
        .mpi_comm_time_s = solver_wall_time,
        .load_imbalance = load_imbalance};

    bool log_due_to_colony_radius = params_.sim_config.log_every_colony_radius_delta && (colony_radius - colony_radius_last_log_ > params_.sim_config.log_every_colony_radius_delta);
    bool log_due_to_sim_time = params_.sim_config.log_every_sim_time_delta && (simulation_time_seconds_ - simulation_time_last_log_ > params_.sim_config.log_every_sim_time_delta);

    if (log_due_to_colony_radius || log_due_to_sim_time || iter == 0 || colony_radius >= params_.sim_config.end_radius) {
      particle_logger_->log(particle_manager_->local_particles);
      ghost_particle_logger_->log(particle_manager_->ghost_particles);
      // constraint_logger_->log(solver_solution.constraints);
      domain_logger_->log(std::make_pair(min_bounds_, max_bounds_));
      simulation_logger_->log(step_data);

      colony_radius_last_log_ = colony_radius;
      simulation_time_last_log_ = simulation_time_seconds_;
      time_last_log_ = MPI_Wtime();

      PetscPrintf(PETSC_COMM_WORLD, "  Logged data at Colony radius = %.3f, Simulation time = %.3f \n", colony_radius, simulation_time_seconds_);
    }

    printProgress(iter + 1, colony_radius, wall_time, step_data);
    // PetscPrintf(PETSC_COMM_WORLD, "\n");
    iter++;
  }
}

void Domain::groupParticlesForExchange(std::vector<std::vector<ParticleData>>& particles_to_send,
                                       std::vector<Particle>& particles_to_keep) {
  double angle_per_rank = 360.0 / size_;

  for (auto& p : particle_manager_->local_particles) {
    const auto& pos = p.getPosition();

    // Calculate distance from center
    double distance_from_center = std::sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);

    // Center particles (within radius 5) go to rank 0
    if (distance_from_center <= center_radius) {
      if (rank_ == 0) {
        // Rank 0 keeps its center particles locally
        particles_to_keep.push_back(p);
      } else {
        // Other ranks send their center particles to rank 0
        particles_to_send[0].push_back(p.getData());
      }
    } else {
      // Regular angular partitioning for non-center particles
      double angle_rad = std::atan2(pos[1], pos[0]);
      double angle_deg = angle_rad * 180.0 / M_PI;
      if (angle_deg < 0) {
        angle_deg += 360.0;
      }

      int target_rank = static_cast<int>(floor(angle_deg / angle_per_rank));
      target_rank = std::max(0, std::min(size_ - 1, target_rank));

      if (target_rank == rank_) {
        particles_to_keep.push_back(p);
      } else {
        particles_to_send[target_rank].push_back(p.getData());
      }
    }
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

double calculateDistanceToAngularBoundary(const std::array<double, 3>& pos,
                                          double radius, double particle_angle_deg,
                                          double boundary_angle_min,
                                          double boundary_angle_max) {
  // Handle wraparound for angles near 0/360 boundary
  auto normalizeAngleDiff = [](double angle_diff) {
    while (angle_diff > 180.0) angle_diff -= 360.0;
    while (angle_diff < -180.0) angle_diff += 360.0;
    return angle_diff;
  };

  double diff_to_min = normalizeAngleDiff(particle_angle_deg - boundary_angle_min);
  double diff_to_max = normalizeAngleDiff(boundary_angle_max - particle_angle_deg);

  // Particle is between the two boundaries
  if (diff_to_min >= 0 && diff_to_max >= 0) {
    return 0.0;  // Inside the domain
  }

  // Find minimum angular distance to either boundary
  double min_angle_diff = std::min(std::abs(diff_to_min), std::abs(diff_to_max));

  // Convert angular distance to absolute distance (arc length at particle's radius)
  double angle_rad = min_angle_diff * M_PI / 180.0;
  double perpendicular_distance = radius * std::sin(angle_rad);

  return perpendicular_distance;
}

std::vector<ParticleData> Domain::exchangeGhostParticlesAllToAll(
    const std::vector<std::vector<ParticleData>>& particles_to_send) {
  // Prepare send counts for all ranks
  std::vector<int> send_counts(size_);
  for (int i = 0; i < size_; ++i) {
    send_counts[i] = particles_to_send[i].size();
  }

  // Exchange counts with all ranks
  std::vector<int> recv_counts(size_);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               PETSC_COMM_WORLD);

  // Calculate displacements
  std::vector<int> send_displs(size_);
  std::vector<int> recv_displs(size_);
  int send_total = 0;
  int recv_total = 0;

  for (int i = 0; i < size_; ++i) {
    send_displs[i] = send_total;
    recv_displs[i] = recv_total;
    send_total += send_counts[i];
    recv_total += recv_counts[i];
  }

  // Pack send buffer
  std::vector<ParticleData> send_buffer(send_total);
  int offset = 0;
  for (int i = 0; i < size_; ++i) {
    std::copy(particles_to_send[i].begin(), particles_to_send[i].end(),
              send_buffer.begin() + offset);
    offset += send_counts[i];
  }

  // Allocate receive buffer
  std::vector<ParticleData> recv_buffer(recv_total);

  // Exchange particle data with all ranks
  MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), mpi_particle_type_,
                recv_buffer.data(), recv_counts.data(), recv_displs.data(), mpi_particle_type_,
                PETSC_COMM_WORLD);

  return recv_buffer;
}

void Domain::exchangeGhostParticles() {
  double angle_per_rank = 360.0 / size_;
  double buffer = 2.5;

  std::vector<std::vector<ParticleData>> particles_to_send(size_);

  for (auto& p : particle_manager_->local_particles) {
    const auto& pos = p.getPosition();

    double angle_rad = std::atan2(pos[1], pos[0]);
    double angle_deg = angle_rad * 180.0 / M_PI;
    if (angle_deg < 0) angle_deg += 360.0;

    double radius = std::sqrt(pos[0] * pos[0] + pos[1] * pos[1]);

    std::unordered_set<int> target_ranks;

    // Case 1: Center particles near the boundary on rank 0
    if (radius <= center_radius && rank_ == 0 &&
        radius >= center_radius - buffer) {
      for (int r = 0; r < size_; ++r) {
        if (r != rank_) target_ranks.insert(r);
      }
    }
    // Case 2: Particles near center boundary (on other ranks) go to rank 0
    else if (rank_ != 0 && radius <= center_radius + buffer) {
      target_ranks.insert(0);
    }

    // Case 3: Check ±10 neighboring ranks
    if (radius >= center_radius) {
      for (int direction : {1, -1}) {  // 1 = clockwise, -1 = counter-clockwise
        for (int offset = 1; offset <= size_; ++offset) {
          int neighbor_rank = ((rank_ + direction * offset) % size_ + size_) % size_;

          double neighbor_angle_min = neighbor_rank * angle_per_rank;
          double neighbor_angle_max = (neighbor_rank + 1) * angle_per_rank;

          double dist_to_boundary = calculateDistanceToAngularBoundary(
              pos, radius, angle_deg, neighbor_angle_min, neighbor_angle_max);

          if (dist_to_boundary < buffer) {
            target_ranks.insert(neighbor_rank);
          } else {
            // As soon as we find a rank in this direction that is not close, break
            break;
          }
        }
      }
    }

    // Send to all target ranks
    for (int target_rank : target_ranks) {
      particles_to_send[target_rank].push_back(p.getData());
    }
  }

  // Exchange ghost particles with all ranks
  auto received_particles_data = exchangeGhostParticlesAllToAll(particles_to_send);

  particle_manager_->ghost_particles.clear();
  for (const auto& pd : received_particles_data) {
    particle_manager_->ghost_particles.emplace_back(pd);
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
  // Clear current particles and start with the ones we decided to keep
  particle_manager_->local_particles = std::move(particles_to_keep);

  // Add all received particles (including center particles sent to rank 0)
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

void Domain::printProgress(int current_iteration, double colony_radius, double cpu_time_s, vtk::SimulationStep step) {
  bool should_log = MPI_Wtime() - t_last_log > 1.0;
  if (should_log) {
    t_last_log = MPI_Wtime();
  } else {
    return;
  }

  double time_elapsed_seconds = simulation_time_seconds_;
  double progress_percent = (colony_radius / params_.sim_config.end_radius) * 100.0;

  // Exponential model parameters
  constexpr double a = 490;     // seconds
  constexpr double b = 0.0247;  // per radius unit

  double predicted_total_wall_time = a * std::exp(b * params_.sim_config.end_radius);
  double estimated_remaining_wall_time_seconds = predicted_total_wall_time - MPI_Wtime();
  if (estimated_remaining_wall_time_seconds < 0.0)
    estimated_remaining_wall_time_seconds = 0.0;

  auto formatTime = [](double seconds) -> std::string {
    char buffer[100];
    if (seconds < 60.0)
      snprintf(buffer, sizeof(buffer), "%6.1f s", seconds);
    else if (seconds < 3600.0)
      snprintf(buffer, sizeof(buffer), "%6.1f min", seconds / 60.0);
    else if (seconds < 86400.0)
      snprintf(buffer, sizeof(buffer), "%6.1f h", seconds / 3600.0);
    else
      snprintf(buffer, sizeof(buffer), "%6.1f d", seconds / 86400.0);
    return std::string(buffer);
  };

  // Compute ETA and format CPU time
  std::string eta_str = formatTime(estimated_remaining_wall_time_seconds);
  std::string cpu_str = formatTime(cpu_time_s);

  // ANSI colors
  const char* RED = "\033[31m";
  const char* GREEN = "\033[32m";
  const char* YELLOW = "\033[33m";
  const char* BLUE = "\033[34m";
  const char* MAGENTA = "\033[35m";
  const char* CYAN = "\033[36m";
  const char* WHITE = "\033[37m";
  const char* RESET = "\033[0m";

  // Progress color: green → yellow → red
  const char* progress_color = (progress_percent < 50.0 ? GREEN : (progress_percent < 80.0 ? YELLOW : RED));

  int total_ranks;
  MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);

  int nthreads = omp_get_max_threads();

  // current system time

  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);

  std::tm tm = *std::localtime(&t);

  std::ostringstream time_oss;
  time_oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");

  PetscPrintf(PETSC_COMM_WORLD,
              "%s[%s]%s | "
              "Radius: %s%.3f%s / %s%.1f%s (%s%.1f%%%s) | "
              "Runtime: %s%s%s | "
              "ETA: %s%s%s | "
              "dt: %s%4.2e%s | "
              "Particles: %s%d%s | "
              "ReLCP iters: %s%d%s |"
              "BBPGD iters: %s%zu%s | Residual: %s%.2e%s | Ranks: %s%d%s | OMP: %s%d%s\n",
              BLUE, time_oss.str().c_str(), RESET,
              CYAN, colony_radius, RESET,
              CYAN, params_.sim_config.end_radius, RESET,
              progress_color, progress_percent, RESET,
              YELLOW, cpu_str.c_str(), RESET,
              YELLOW, eta_str.c_str(), RESET,
              MAGENTA, params_.sim_config.dt_s, RESET,
              CYAN, global_particle_count, RESET,
              GREEN, step.recursive_iterations, RESET,
              GREEN, step.bbpgd_iterations, RESET,
              RED, step.residual, RESET,
              MAGENTA, total_ranks, RESET,
              CYAN, nthreads, RESET);
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

  // Exchange ghost particles to ensure proper initialization
  domain.rebalance();
  domain.exchangeGhostParticles();

  return domain;
}
