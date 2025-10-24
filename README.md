# Interdisciplinary Project: Proliferating Cell Collectives: A Comparison of Hard and Soft Collision Models

🦠 | Bacteria Growth using Hard Model             |  Bacteria Growth using Soft Model
:-------------------------:|:-------------------------:|:-------------------------:
&nbsp;| $R_{end} = 100$ | $R_{end} = 100$
Bacteria Lengths (dark=short, bright=long)|![Hard Model](latex/figures/growth/hard_e-3/hard_e-3.0198.jpeg) | ![Soft Model](latex/figures/growth/soft_e-3/soft_e-3.0187.jpeg)
Bacteria Orientation (color=angle)|![Hard Model](latex/figures/orientation_comparisons/hard_e-2_orient_big.jpeg) | ![Soft Model](latex/figures/orientation_comparisons/soft_e-2_orient_big.jpeg)

## Abstract

This work extends the hard (constraint-based) collision model introduced by Weady et al.~\cite{Weady2024} by systematically comparing it with a soft (potential-based) approach for simulating proliferating cell collectives. Both models are implemented within a unified computational framework, enabling direct benchmarking of bacterial colony growth.

Our results show that both approaches reproduce key experimentally observed patterns, including concentric rings and microdomain formation, but they involve distinct trade-offs. The hard model permits timesteps roughly 30 times larger than the soft model and achieves runtimes up to 9.36$\times$ faster. In contrast, the soft model suffers from numerical stiffness and allows unphysical cell overlap, producing packing fractions exceeding 5 in colony centers compared to the hard model's realistic 0.9.

We also introduce an adaptive timestepping algorithm based on the Courant-Friedrichs-Lewy condition to dynamically select stable timesteps. Benchmarking reveals the hard model consistently outperforms the soft model across all colony sizes, successfully simulating colonies up to $R \approx 260$ with approximately 301,116 cells while maintaining physical accuracy.

Despite microscopic differences in packing and stress, both models produce similar macroscopic patterns at small scales, indicating that colony-level behavior exhibits some robustness to the collision resolution method. However, the soft model's severe overcrowding and distorted stress distributions make it unsuitable for quantitative analysis or large-scale simulations. We find that the hard model is strongly preferred for nearly all applications, offering both superior computational performance and physically meaningful results.

## Table of Contents

1. **Introduction**
   1. Biological Motivation

2. **Related Work**
   1. Collision Modeling Paradigms
   2. The Benchmarking Gap

3. **Cell Mechanics**
   1. Physical Cell Model
   2. Cell Growth and Division
   3. Rotational Diffusion

4. **Unified Computational Framework**
   1. Colony Representation
   2. Colony Dynamics
   3. State Integration
   4. Soft Collision Model
      1. Force Assembly
      2. Model Parameters and Numerical Stability
   5. Hard Collision Model
      1. Constraint Conditions
      2. Linearization of the Constraint Condition
      3. Nonlinear Complementarity Problem
      4. Energy Minimization Solution
      5. Numerical Solution
      6. Limitations of the Linearization

5. **Implementation**
   1. Distributed Computing Architecture
   2. Collision Handling Pipeline
   3. Adaptive Timestepping
   4. Simulation Output and Availability

6. **Pattern Formation Analysis**
   1. Concentric Ring Patterns
   2. Cell Density and Local Packing Fraction
   3. Colony Growth Dynamics and Number of Cells
   4. Microdomain Formation
   5. Radial Stress Distribution and Growth Rate

7. **Computational Performance**
   1. Critical Time Step
   2. Strong Scaling
   3. Model Efficiency: Walltime and Interaction Overhead
   4. Maximum Attainable Colony Size
   5. Insights into BBPGD Iterations

8. **Discussion and Conclusion**

9. **Future Work**
   1. Adaptive Resource Allocation for Early-Stage Simulations
   2. GPU Acceleration via PETSc
   3. Specialized Molecular Dynamics Libraries for Soft Model Optimization

10. **References**

11. **Appendix**

## Thesis

The thesis is available in LaTeX format in this repository. You can access the rendered version in PDF format by clicking the following link:

[Read the Thesis (PDF)](latex/cell-collectives.pdf)

## Slides

The presentation slides are available in PDF format and can be accessed via the following link:

[View the Slides (PDF)](presentation/slides.pdf)

## Code

### Usage

The code is written in C++ and can be compiled using the following command:

```bash
cd demo/cpp/hard
mkdir -p build
cd build
cmake ..
make -j
```

The executable can be run with the following command to see the available options:

```bash
./cellcollectives -help
```

The executable can also be run with `mpirun` to run the simulation using multiple processes:

```bash
mpirun -n 16 ./cellcollectives -mode "hard"
```

### Results

Running the simulation produces output data in `build/src/vtk_output_{mode}` folder.

This data can be conviniently visualized using Paraview using the provided `ParticleDebugger.pvsm` file in the `visualization` folder.
Further scripts are providied in the `/analysis` folder to analyze the simulation data and produce the figures in the thesis.
