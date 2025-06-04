#pragma once
#include <petscksp.h>

#include <memory>

#include "Bacteria.h"

class BacterialSystem {
 public:
  BacterialSystem(int argc, char** argv);

  void run();

 private:
  // PETSc objects
  DM dm_;
  Vec configuration_;  // concatenation of [x,y,z,q0,q1,q2,q3,l] for each particle
  Vec forces_;         // concatenation of [fx,fy,fz,τx,τy,τz] for each particle
  Mat jacobian_;
  Vec lambda_;

  // Simulation parameters
  double time_;
  double dt_;
  int current_step_;

  void initializeSystem();
  void initializeParticles();
  void addParticle(Bacterium p);
  void cleanup();
  void detectContacts();
  void timeStep();

  void redistributeVectors(PetscInt new_local_size);

  int particles_per_rank_;
  int local_estimate_;
  int current_particles_;
};