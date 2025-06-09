#pragma once

#include <vector>

#include "Constraint.h"
#include "Particle.h"
#include "util/PetscRaii.h"

class MappingManager {
 public:
  struct Mappings {
    ISLocalToGlobalMappingWrapper col_map_6d;
    ISLocalToGlobalMappingWrapper row_map_7d;
    ISLocalToGlobalMappingWrapper constraint_map;
  };

  Mappings createMappings(const std::vector<Particle>& particles,
                          const std::vector<Constraint>& constraints);
};