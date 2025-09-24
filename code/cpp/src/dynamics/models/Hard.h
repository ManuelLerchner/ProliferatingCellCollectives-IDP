
#include "simulation/ParticleManager.h"
#include "logger/ConstraintLogger.h"
#include "logger/ParticleLogger.h"
#include "simulation/ParticleManager.h"
#include "spatial/CollisionDetector.h"
#include "util/Config.h"

ParticleManager::SolverSolution solveHardModel(ParticleManager& particle_manager, CollisionDetector& collision_detector, SimulationParameters params, double dt, int iter, std::function<void()> exchangeGhostParticles, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger);