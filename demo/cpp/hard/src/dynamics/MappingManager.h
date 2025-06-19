#pragma once

#include <vector>

#include "Constraint.h"
#include "simulation/Particle.h"
#include "util/PetscRaii.h"

struct Mappings {
  ISLocalToGlobalMappingWrapper velocityL2GMap;
  ISLocalToGlobalMappingWrapper configL2GMap;
  ISLocalToGlobalMappingWrapper constraintL2GMap;
  ISLocalToGlobalMappingWrapper lengthL2GMap;
};

Mappings createMappings(const std::vector<Particle>& particles,
                        const std::vector<Constraint>& constraints);
