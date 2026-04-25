# Feedback from Sam

> **Note:** Extracted from handwritten comments on `feedback_sam.pdf` by Claude (claude-sonnet-4-6) on 2026-04-23.
Colors: **green** = Sam's main comments, **blue** = Sam's secondary/question comments.

| # | Done | Page | Color | Location / Target | Comment |
|---|------|------|-------|-------------------|---------|
| 1 | [ ] | 1 | Green | Top-left / Abstract | "Add a little more intro. Potentially ↓ cut down?" |
| 2 | [ ] | 1 | Blue | Top-right / Title area | "cc?" (corresponding author?) |
| 3 | [ ] | 1 | Green | Left margin / Abstract soft model paragraph | "check" |
| 4 | [ ] | 1 | Green | Bottom / Sec. I Introduction | "I would add more about the different models and the experiments. Plus adaptive timestepping + something about continuum modelling. + unified framework." |
| 5 | [ ] | 2 | Green | Left margin / Soft model, "embarrassingly parallelizable" | "Not ε / using NBL? / maybe 'easily'" |
| 6 | [ ] | 2 | Green | Inline underline / Sec. II-B "better performance remains unresolved" | "direct comparison and organize" (underlined phrase) |
| 7 | [ ] | 2 | Blue | Right margin / Sec. II-B benchmarking gap | "Not sure you can frame this as a negative → Reward" |
| 8 | [ ] | 2 | Green | Right margin / Sec. III heading | "Exactly from [23] unless otherwise specified" |
| 9 | [ ] | 2 | Green | Bottom center / end of page | "ℓ(t)?" |
| 10 | [ ] | 3 | Blue | Top-right / Sec. IV heading | "A, B, C exactly from [23]? Yes unless... little intro" |
| 11 | [ ] | 3 | Blue | Right margin / Eq. 2 (growth rate) | "Time to define ℓ₀ / 1/e×?" |
| 12 | [ ] | 3 | Green | Left margin / non-dimensionalization paragraph | "More careful with non-dim vars we defined → introduced?" |
| 13 | [ ] | 3 | Green | Arrow near daughter division | "Uniformly → Randomly?" |
| 14 | [ ] | 3 | Blue | Right margin / Eq. 5 (mobility matrix) | Annotated entries: "1/ζℓ₀⁻³, 1/ζℓ₀³ I₃" |
| 15 | [ ] | 3 | Green | Bottom-left / Figure 2 | "Is this from anywhere? [15]" |
| 16 | [ ] | 3 | Green | Bottom center | "3D?" |
| 17 | [ ] | 3 | Blue | Right margin bottom | "Maybe more not not?" |
| 18 | [ ] | 4 | Green | Left margin / Sec. D Soft Collision intro | "intra-cell / per chem..." (concerns interpenetration wording) |
| 19 | [ ] | 4 | Green | Right margin / Eq. 9–10 hard model force | "→ why not γ^k? Just compared e.g. like F (so no k)" |
| 20 | [ ] | 4 | Green | Bottom-left / Figure 3 (Hertzian plot) | "→ maybe could be dropped." |
| 21 | [ ] | 4 | Green | Bottom-left / k_cc = 20000 h⁻¹ value | "Is this a value used in these works? Yes" |
| 22 | [ ] | 5 | Blue | Right margin / Eq. 16 energy minimization | "= (" with large bracket — notation question |
| 23 | [ ] | 5 | Green | Left margin / Eq. 14 linearization derivation | "Why is this already captured in it? Because growth is already..." |
| 24 | [ ] | 5 | Green | Right margin / Eq. 17 | "≈ Φ^k(γ)" |
| 25 | [ ] | 5 | Green | Right margin / ref near Eq. 17 | "[15]?" |
| 26 | [ ] | 5 | Blue | Right margin / KKT conditions explanation | "I don't care that this is not explained" |
| 27 | [ ] | 5 | Green | Bottom / Complementarity Condition (3) | "I think (3) is a bit too convoluted" |
| 28 | [ ] | 6 | Green | Left margin / Sec. IV-5 Numerical Solution (BBPGD) | "Maybe BBPGD should be described! It is a key part of one page" |
| 29 | [ ] | 6 | Green | Left margin / Eq. 14 reference | "(14)? ✓" |
| 30 | [ ] | 6 | Blue | Right margin / Sec. V Implementation, PETSc bullet | "I/ PETSc supports 16 CPUs, don't just write it so" [partially illegible] |
| 31 | [ ] | 6 | Blue | Right margin / MPI bullet (ghost cells) | "'ghost cell' not introduced" |
| 32 | [ ] | 6 | Blue | Right margin / Collision Handling Pipeline | "I already introduced. Is this overall really all?" |
| 33 | [ ] | 6 | Blue | Right margin / threshold distance d=0.5 | "not sure about choice" |
| 34 | [ ] | 7 | Green | Left margin / CFL Eq. 18 | "Add more about growth forces" |
| 35 | [ ] | 7 | Green | Near CFL equation / variables | "What is u_m? k_Cmax? Adaptive le" |
| 36 | [ ] | 7 | Green | Algorithm 1 step 4 / smoothing | "Maybe round → Smoother changes" |
| 37 | [ ] | 7 | Green | Left margin / Algorithm 1 steps 4 & 5 | "4&5 feel a bit 'engineered'" |
| 38 | [ ] | 7 | Blue | Left margin / Simulation Output section | "W. Starts, Convey it 'while'" [uncertain reading] |
| 39 | [ ] | 7 | Green | Bottom-left / Algorithm 1 / adaptive timestep | "Incorporating current overlaps in addition to velocity and growth into Alg 1 could result in a smaller Δt that avoids mis-overlap, but would come at the cost of computational efficiency" |
| 40 | [ ] | 7 | Green | Bottom-right / soft model overlap discussion | "Instead, come to conclusion and make statement that whilst a smaller Δt could reduce this, incorporating overlap..." |
| 41 | [ ] | 8 | Blue | Bottom / Figure 4 caption (colony radius units) | "units? Just non-dimensionalized" |
| 42 | [ ] | 9 | Blue | Top-right / Figure 5 caption | "what parameter? λ?" |
| 43 | [ ] | 9 | Green | Near Figure 5 caption / word choice | "→ $? Word choice" |
| 44 | [ ] | 9 | Blue | Right margin / Sec. C "slowen linear growth" phrase | "I don't like this phrasing" |
| 45 | [ ] | 9 | Green | Left margin / Figure 6a packing fraction | "More / Distance from center ↓ / → comparison varies" |
| 46 | [ ] | 9 | Green | Figure 6a x-axis label | "what point in the? R=100" |
| 47 | [ ] | 9 | Blue | Bottom-right / Figure 6a bin width description | "what is this(100) → Binning like in RDF" |
| 48 | [ ] | 9 | Green | Bottom-left / Figure 6b (max overlap vs colony radius) | "→ why less overlap further from centre? Colony Radius = Size of colony at that point in blue" |
| 49 | [ ] | 9 | Green | Bottom-right / Figure 7b growth dynamics | "Maybe: For both models the sensitivity, growth rate of the colony radius increases at — what can we call this? Logistic?" |
| 50 | [ ] | 10 | Blue | Top / Sec. D Microdomain, You et al. [15] | "Read E.Coli / Bac." |
| 51 | [ ] | 10 | Green | Right margin / "do not resemble experimentally observed microdomains" | "→ cite You et al." |
| 52 | [ ] | 12 | Blue | Right margin / Figure 12 (radial growth rate) | "↓ this is it includes in the model" [uncertain reading] |
| 53 | [ ] | 13 | Green | Left margin / Domain Decomposition paragraph | "& mention ⓒ" |
| 54 | [ ] | 13 | Green | Right margin / Strong Scaling, communication cost | "E.g.? MV / V+V / SV / Basic / Snip" |
| 55 | [ ] | 13 | Green | Bottom / load imbalance at high core counts | "→ Isn't this a huge bottleneck? No" |
| 56 | [ ] | 14 | Blue | Right margin / U-shaped runtime curve | "Is this comment?" |
| 57 | [ ] | 14 | Green | Bottom-left / Figure 15 | "Signal Δt" |
| 58 | [ ] | 14 | Green | Bottom-right / Figure 16 | "But Δt is not stable so what is this?" |
| 59 | [ ] | 15 | Blue | Top-right / Figures 17a & 19a comparison | "What? This end 19(a) are basically the same graph" |
| 60 | [ ] | 15 | Blue | Left margin / BBPGD iterations discussion | "Is this happening at N=1.3k? Not much. Sure using..." |
| 61 | [ ] | 15 | Green | Left margin / future improvements paragraph | "Maybe more to base. Any citations? No. Own idea" |
| 62 | [ ] | 15 | Blue | Bottom-center / ReLCP convergence | "But find one?" |
| 63 | [ ] | 15 | Green | Bottom-right / Figure 17c / final phase | "Is this simply the last one?" |
| 64 | [ ] | 16 | Green | Figure 19b / wavelength annotations | "Wavelength Change λ=27, λ=21, λ=32, λ=25, λ=19" |
| 65 | [ ] | 16 | Green | Bottom / stress profile matches analytical prediction | "→ Why? This is not really explained" |
| 66 | [ ] | 17 | Blue | Conclusion / "robustness justifies..." sentence | Bracketed — flag this sentence |
| 67 | [ ] | 17 | Green | Bottom-left / soft model use case | "→ Why? Easier to implement & work at small scales" |
| 68 | [ ] | 17 | Green | Bottom-right / Sec. IX-A PETSc + OpenMP note | "It looks like you can have PETSc w/ OpenMP" |
| 69 | [ ] | 17 | Green | Sec. IX-D Model Extensions | "If not good 'Why', then cut" |
| 70 | [ ] | 20 | Green | Top / Performance section (blank extra page) | "Should have some per-timestep comparison." |
| 71 | [ ] | 20 | Green | Bottom / MPI communication timing | "Where does MPI communication happen? – After every ReLCP step?" |
