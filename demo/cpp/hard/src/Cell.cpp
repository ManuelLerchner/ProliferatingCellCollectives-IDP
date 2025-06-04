#include <mpi.h>
#include <petsc.h>

#include <array>
#include <chrono>
#include <memory>
#include <type_traits>
#include <vector>

// Common particle properties
struct ParticleProps {
  static constexpr int POS_SIZE = 3;
  static constexpr int QUAT_SIZE = 4;
  static constexpr int FORCE_SIZE = 3;
  static constexpr int TORQUE_SIZE = 3;
  static constexpr int LENGTH_SIZE = 1;
};

// Storage type tags for compile-time dispatch
struct AoSTag {};
struct SoATag {};

// Template-based storage classes (no virtual functions!)
template <typename Tag>
class ParticleStorage;

// Array of Structures specialization
template <>
class ParticleStorage<AoSTag> {
 private:
  struct Particle {
    std::array<PetscReal, ParticleProps::POS_SIZE> position;
    std::array<PetscReal, ParticleProps::QUAT_SIZE> quaternion;
    std::array<PetscReal, ParticleProps::FORCE_SIZE> force;
    std::array<PetscReal, ParticleProps::TORQUE_SIZE> torque;
    PetscInt global_id;
    PetscReal length;
    PetscReal diameter;

    Particle() : global_id(-1), length(0.0), diameter(0.0) {
      position.fill(0.0);
      quaternion.fill(0.0);
      force.fill(0.0);
      torque.fill(0.0);
    }
  };

  std::vector<Particle> particles;
  PetscInt n_local;
  PetscInt n_ghost;

 public:
  using storage_tag = AoSTag;

  void resize(PetscInt n_local_new, PetscInt n_ghost_new = 0) {
    n_local = n_local_new;
    n_ghost = n_ghost_new;
    particles.resize(n_local + n_ghost);
  }

  PetscInt get_local_count() const { return n_local; }
  PetscInt get_ghost_count() const { return n_ghost; }
  PetscInt get_total_count() const { return n_local + n_ghost; }

  // Non-const accessors - inline and fast
  PetscReal* get_position(PetscInt i) { return particles[i].position.data(); }
  PetscReal* get_quaternion(PetscInt i) { return particles[i].quaternion.data(); }
  PetscReal* get_force(PetscInt i) { return particles[i].force.data(); }
  PetscReal* get_torque(PetscInt i) { return particles[i].torque.data(); }
  PetscInt& get_global_id(PetscInt i) { return particles[i].global_id; }
  PetscReal* get_length(PetscInt i) { return &particles[i].length; }
  PetscReal* get_diameter(PetscInt i) { return &particles[i].diameter; }

  // Const accessors
  const PetscReal* get_position(PetscInt i) const { return particles[i].position.data(); }
  const PetscReal* get_quaternion(PetscInt i) const { return particles[i].quaternion.data(); }
  const PetscReal* get_force(PetscInt i) const { return particles[i].force.data(); }
  const PetscReal* get_torque(PetscInt i) const { return particles[i].torque.data(); }
  const PetscInt& get_global_id(PetscInt i) const { return particles[i].global_id; }
  const PetscReal& get_length(PetscInt i) const { return particles[i].length; }
  const PetscReal& get_diameter(PetscInt i) const { return particles[i].diameter; }

  static constexpr const char* storage_type() { return "AoS"; }

  // AoS-specific operations
  void zero_forces() {
    for (auto& p : particles) {
      p.force.fill(0.0);
      p.torque.fill(0.0);
    }
  }

  void copy_particle(PetscInt from, PetscInt to) {
    particles[to] = particles[from];
  }
};

// Structure of Arrays specialization
template <>
class ParticleStorage<SoATag> {
 private:
  std::vector<PetscReal> positions;    // [x1,y1,z1,x2,y2,z2,...]
  std::vector<PetscReal> quaternions;  // [q1w,q1x,q1y,q1z,q2w,...]
  std::vector<PetscReal> forces;
  std::vector<PetscReal> torques;
  std::vector<PetscInt> global_ids;
  std::vector<PetscReal> lengths;
  std::vector<PetscReal> diameters;

  PetscInt n_local;
  PetscInt n_ghost;

 public:
  using storage_tag = SoATag;

  void resize(PetscInt n_local_new, PetscInt n_ghost_new = 0) {
    n_local = n_local_new;
    n_ghost = n_ghost_new;
    PetscInt total = n_local + n_ghost;

    positions.resize(ParticleProps::POS_SIZE * total, 0.0);
    quaternions.resize(ParticleProps::QUAT_SIZE * total, 0.0);
    forces.resize(ParticleProps::FORCE_SIZE * total, 0.0);
    torques.resize(ParticleProps::TORQUE_SIZE * total, 0.0);
    global_ids.resize(total, -1);
    lengths.resize(total, 0.0);
    diameters.resize(total, 0.0);
  }

  PetscInt get_local_count() const { return n_local; }
  PetscInt get_ghost_count() const { return n_ghost; }
  PetscInt get_total_count() const { return n_local + n_ghost; }

  // Non-const accessors - inline and fast
  PetscReal* get_position(PetscInt i) { return &positions[ParticleProps::POS_SIZE * i]; }
  PetscReal* get_quaternion(PetscInt i) { return &quaternions[ParticleProps::QUAT_SIZE * i]; }
  PetscReal* get_force(PetscInt i) { return &forces[ParticleProps::FORCE_SIZE * i]; }
  PetscReal* get_torque(PetscInt i) { return &torques[ParticleProps::TORQUE_SIZE * i]; }
  PetscInt& get_global_id(PetscInt i) { return global_ids[i]; }
  PetscReal* get_length(PetscInt i) { return &lengths[i]; }
  PetscReal* get_diameter(PetscInt i) { return &diameters[i]; }

  // Const accessors
  const PetscReal* get_position(PetscInt i) const { return &positions[ParticleProps::POS_SIZE * i]; }
  const PetscReal* get_quaternion(PetscInt i) const { return &quaternions[ParticleProps::QUAT_SIZE * i]; }
  const PetscReal* get_force(PetscInt i) const { return &forces[ParticleProps::FORCE_SIZE * i]; }
  const PetscReal* get_torque(PetscInt i) const { return &torques[ParticleProps::TORQUE_SIZE * i]; }
  const PetscInt& get_global_id(PetscInt i) const { return global_ids[i]; }
  const PetscReal& get_length(PetscInt i) const { return lengths[i]; }
  const PetscReal& get_diameter(PetscInt i) const { return diameters[i]; }

  static constexpr const char* storage_type() { return "SoA"; }

  // SoA-specific operations - can be heavily optimized
  void zero_forces() {
    std::fill(forces.begin(), forces.end(), 0.0);
    std::fill(torques.begin(), torques.end(), 0.0);
  }

  void copy_particle(PetscInt from, PetscInt to) {
    std::copy(&positions[ParticleProps::POS_SIZE * from],
              &positions[ParticleProps::POS_SIZE * (from + 1)],
              &positions[ParticleProps::POS_SIZE * to]);
    std::copy(&quaternions[ParticleProps::QUAT_SIZE * from],
              &quaternions[ParticleProps::QUAT_SIZE * (from + 1)],
              &quaternions[ParticleProps::QUAT_SIZE * to]);
    std::copy(&forces[ParticleProps::FORCE_SIZE * from],
              &forces[ParticleProps::FORCE_SIZE * (from + 1)],
              &forces[ParticleProps::FORCE_SIZE * to]);
    std::copy(&torques[ParticleProps::TORQUE_SIZE * from],
              &torques[ParticleProps::TORQUE_SIZE * (from + 1)],
              &torques[ParticleProps::TORQUE_SIZE * to]);

    global_ids[to] = global_ids[from];
    lengths[to] = lengths[from];
    diameters[to] = diameters[from];
  }

  // SoA-specific: direct array access for vectorized operations
  std::vector<PetscReal>& get_position_array() { return positions; }
  std::vector<PetscReal>& get_force_array() { return forces; }
  const std::vector<PetscReal>& get_position_array() const { return positions; }
  const std::vector<PetscReal>& get_force_array() const { return forces; }
};

// Type aliases for convenience
using AoSStorage = ParticleStorage<AoSTag>;
using SoAStorage = ParticleStorage<SoATag>;

// Template-based simulation class - zero runtime overhead!
template <typename StorageTag>
class LocalSimulationState {
 private:
  ParticleStorage<StorageTag> particles;

 public:
  using storage_type = ParticleStorage<StorageTag>;

  void initialize(PetscInt n_local_particles, PetscInt n_ghost = 0) {
    particles.resize(n_local_particles, n_ghost);
  }

  // All operations are inlined with no virtual calls
  template <typename ForceFunction>
  void compute_forces(ForceFunction&& force_func) {
    particles.zero_forces();

    PetscInt n = particles.get_local_count();
    for (PetscInt i = 0; i < n; i++) {
      for (PetscInt j = i + 1; j < n; j++) {
        // Direct, inlined access - compiler can optimize heavily
        force_func(particles, i, j);
      }
    }
  }

  storage_type& get_particles() { return particles; }
  const storage_type& get_particles() const { return particles; }
};