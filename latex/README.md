# Thesis Structure: Computational Modeling of Proliferating Cell Collectives

## 1. Introduction

- **Biological motivation**: Cell collectives and pattern formation in nature
- **The computational challenge**: Modeling proliferating active matter
- **Research questions**: Hard vs. soft collision models, computational efficiency vs. accuracy
- **Key findings preview**: Why constraint-based models may be essential
- **Thesis scope and contributions**

## 2. Background and Literature Review

### 2.1 Proliferating Active Matter

- Theoretical foundations of active matter with growth
- Key differences from static particle systems
- The Weady et al. framework and concentric-ring patterns

### 2.2 Collision Models in Particle Simulations

- Hard collision models: constraint-based approaches
- Soft collision models: potential-based approaches
- Known limitations of soft models for growing systems

### 2.3 Cell Mechanics and Biological Relevance

- Rigid vs. deformable cell assumptions
- Evidence for microbial cell deformation under stress
- Implications for model selection

## 3. Mathematical Formulations

### 3.1 Hard Collision Model

- Constraint formulation and optimization problem
- Growth dynamics and constraint updates
- Time integration schemes
- Advantages for instantaneous conflict resolution

### 3.2 Soft Collision Model

- Potential functions and force calculations
- Growth-induced force imbalances
- Time scale separation challenges
- Stability considerations and adaptive time stepping

## 4. Implementation and Computational Methods

### 4.1 Simulation Framework

- Standalone implementation
- Data structures and algorithmic choices
- Performance considerations

### 4.2 Hard Model Implementation and Validation

- Reproduction of Weady et al. reference patterns
- Parameter calibration strategies
- Verification tests and benchmarks

### 4.3 Soft Model Implementation Challenges

- Parameter tuning attempts and strategies tested
- Numerical stability issues encountered
- Growth event handling approaches

## 5. Results and Analysis

### 5.1 Hard Model Performance

- Successful concentric-ring pattern reproduction
- Quantitative pattern metrics and validation
- Computational performance characteristics

### 5.2 Soft Model Behavior Analysis

- **Clustering instead of ring formation**
- **Particle overlap accumulation over time**
- **Parameter sensitivity analysis and tuning attempts**
- **Comparison of dynamics: soft vs. hard model trajectories**

### 5.3 Fundamental Differences Between Models

- **Force response during growth events**
- **Energy landscape and stability analysis**
- **Time scale requirements and computational implications**
- **Why soft models fail to reproduce experimental patterns**

### 5.4 Computational Performance Comparison

- Runtime scaling with system size
- Memory usage and efficiency analysis

## 6. Discussion

### 6.1 Model Selection for Proliferating Systems

- Synthesis of hard vs. soft model performance
- When constraint-based approaches are essential vs. optional
- Computational trade-offs and practical recommendations

### 6.2 Biological Relevance and Model Assumptions

- Rigid vs. deformable cell assumptions in light of results
- Connection between computational constraints and biological reality
- Implications for understanding real cell collective behavior

### 6.3 Computational Insights and Methodological Contributions

- AutoPas performance and algorithm selection benefits
- Scalability considerations for large systems
- Framework applicability to other active matter problems

### 6.4 Broader Impact on Active Matter Research

- Contribution to proliferating active matter field
- Methodological guidance for future simulations
- Open questions and research directions

## 7. Conclusion and Future Work

- **Key finding: Constraint-based models may be essential for proliferating systems**
- Computational insights and methodological contributions
- **Scientific validation of hard collision approach**
- Directions for future research
- Extensions to deformable systems

## Thesis Summary Statement

**"Investigation reveals fundamental limitations of soft collision models for proliferating particle systems, validating constraint-based approaches for accurate pattern reproduction."**
