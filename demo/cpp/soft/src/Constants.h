#pragma once

namespace soft {

// Physical constants
constexpr double k_cc = 100000.0;    // Cell-cell elastic constant (from CellsMD3D)
constexpr double k_wc = 100000.0;    // Cell-wall elastic constant
constexpr double gamma_n = 100.0;    // Normal damping coefficient
constexpr double gamma_t = 10000.0;  // Tangential damping coefficient (from CellsMD3D)
constexpr double cell_mu = 0.15;     // Cell-cell friction coefficient
constexpr double wall_mu = 0.15;     // Cell-wall friction coefficient
constexpr double viscosity = 1e-3;   // Fluid viscosity

// Growth constants
constexpr double max_growth_rate = 0.5;   // Maximum growth rate
constexpr double maintenance_rate = 0.1;  // Maintenance energy cost
constexpr double nutrient_Kc = 0.1;       // Half-saturation constant
constexpr double division_length = 4.0;   // Length at which cells divide
constexpr double var_length = 0.1;        // Variation in daughter cell lengths
constexpr double var_angle = 0.1;         // Variation in division angles

// Impedance parameters
constexpr double impedance_lambda = 2.44e-3;  // Impedance sensitivity to stress (from CellsMD3D)
constexpr double impedance_tau = 54 * 60;     // Growth timescale (from CellsMD3D)

// Simulation constants
constexpr double dt = 0.001;              // Timestep
constexpr double box_length = 50.0;       // Simulation box size
constexpr bool periodic_boundary = true;  // Use periodic boundaries

}  // namespace soft
