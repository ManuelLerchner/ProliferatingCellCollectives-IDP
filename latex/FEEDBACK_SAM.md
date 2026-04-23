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

---

## Triage

### A — Phrasing & Word Choice
*Quick single-word or single-sentence fixes, no new content required.*

**#5** ✅ — "embarrassingly parallelizable"
- *Target:* Sec. II-A, sentence "the calculations are local and embarrassingly parallelizable, enabling large-scale simulations."
- *Fix:* Replace with "easily parallelizable." Also check whether NBL (neighbor list) is actually used; if not, remove that implication.

**#6 + #7** ✅ — Benchmarking gap framed as a negative
- *Target:* Sec. II-B, sentence "the question of which approach offers better performance remains unresolved."
- *Fix:* Reframe positively. E.g. "a direct, systematic comparison has yet to be performed, representing a clear opportunity." Rewording the whole paragraph from "negative gap" to "research opportunity."

**#13** ✅ — "Uniformly → Randomly?"
- *Target:* Sec. III-A, "producing two daughters with lengths sampled from [0.98ℓ₀, 1.02ℓ₀]"
- *Fix:* Add "uniformly" → "lengths uniformly randomly sampled from [0.98ℓ₀, 1.02ℓ₀]."

**#18** ✅ — "intra-cell / per chem..." (interpenetration wording)
- *Target:* Sec. III-D, "preventing significant interpenetration while allowing elastic deformation."
- *Fix:* Clarify → "preventing significant inter-cell penetration while allowing elastic deformation." ("Interpenetration" alone is ambiguous between intra- and inter-cell.)

**#27** ✅ — Complementarity Condition (3) too convoluted
- *Target:* Sec. IV-3 Constraint Conditions, Complementarity Condition explanation (~6 lines starting "This condition enforces that if two cells are in contact…")
- *Fix:* Trim to one physical sentence: "If cells are separated (Φ_α > 0), no force acts (γ_α = 0); if a force acts (γ_α > 0), cells must be touching (Φ_α = 0)."

**#38** ⚠️ TODO — "W. Starts, Convey it 'while'" (illegible — ask Sam)
- *Target:* Sec. V-D Simulation Output opening sentence
- *Fix:* Annotation unreadable; meaning unclear. Ask Sam directly what he intended before editing this sentence.

**#44 + #49** ✅ — "slowen linear growth" / growth curve naming
- *Target:* Sec. VI-C, "both models transition from early exponential growth to slowen linear growth as stress limits expansion"
- *Fix:* Replaced with "stress-limited linear growth." Note: Sam asked "logistic?" but true logistic has a plateau — the colony doesn't plateau, so "logistic-like" would overclaim. Kept as stress-limited linear growth.

---

### B — Missing Content / Additions
*Requires new text, explanations, or significant expansions.*

**#1** ✅ — Abstract needs more intro
- Added 2-sentence biological motivation paragraph before the contribution sentence.

**#4** ✅ — Introduction missing key topics
- Added sentences covering hard vs. soft trade-offs, adaptive timestepping necessity, and unified framework as enabling contribution. Note: continuum modelling relation not added — the paper doesn't position itself as a discrete complement to continuum models, so that framing would be inaccurate.

**#8** ✅ — No attribution sentence at start of Sec. III
- Added "Unless otherwise noted, the mechanics model in this section follows Weady et al. [23,SM]."

**#10** ✅ — No intro to Sec. IV stating what comes from [23]
- Added opening sentence to Sec. IV with subsection references for Weady et al. material vs. novel contributions.

**#11** ✅ — ℓ₀ and the 1/e characteristic not defined at Eq. 2
- Extended "where" clause after Eq. 2 to define τ as 1/e growth timescale and ℓ₀ as reference cell length at birth.

**#12** ✅ — Non-dim variables used before formally introduced
- Added "where ζ is the drag coefficient and d is the cell diameter" to the non-dim paragraph.

**#17** ✅ — Closing paragraph of Sec. IV-C too terse
- Added forward-reference sentence distinguishing hard (constraint forces) vs. soft (repulsive potentials) with section cross-references.

**#28** ✅ — BBPGD not described despite being central
- Added 2 sentences describing gradient-step + projection mechanics and why BB step-size avoids line searches.

**#34** ✅ — CFL discussion omits growth forces
- Rewrote u_m definition to explicitly include ℓ̇ᵢ (growth elongation) alongside mechanical velocity.

**#35** ✅ — Algorithm 1 variables undefined
- Added "where" clause after Eq. 19 defining c, ε, and u_m with their values.

**#36 + #37** ✅ — Steps 4 & 5 of Algorithm 1 feel ad hoc
- Added inline justification comments to Algorithm 1 steps 4 and 5.

**#39 + #40** ✅ — Soft model overlap section needs a conclusion statement
- Replaced vague "beyond the scope" sentence with explicit forward-reference to Sec. IX-C.

**#53** ✅ — Domain decomposition needs a source/credit
- Stated explicitly "We introduce this angular sector decomposition specifically suited to the circular geometry of bacterial colonies."

**#54** ✅ — Strong scaling communication costs: examples missing
- Added "including matrix-vector products, vector additions, and scatter/gather operations."

**#67** ✅ — Conclusion doesn't explain *why* soft model is useful at small scales
- Added "due to its simpler implementation and tolerance for limited packing artifacts... where cell-scale stress resolution is not required."

**#70** ⚠️ TODO — Missing per-timestep cost comparison
- Requires new benchmark data + figure/table. Cannot be done as a text fix. Needs actual timing measurements.

**#71** ✅ — Where MPI communication occurs is never stated
- Added explicit statement of both MPI communication points to Sec. VII-A.

---

### C — Figures & Captions

**#15** ✅ — Figure 2 origin unclear
- Figure is original TikZ drawing — no attribution needed.

**#16** ✅ — 2D vs 3D ambiguity in Figure 2
- Added "Cells are modeled as 3D spherocylinders; the diagram shows a 2D schematic view." to caption.

**#20** ⚠️ TODO — Figure 3 (Hertzian plot) possibly redundant
- Decision required: drop if Eq. 7 + text suffices; keep if nonlinear shape needs visual emphasis. Sam leans toward dropping.

**#41** ✅ — Figure 4 caption missing units
- Added "(non-dimensionalized units, ℓ₀ = 1)" to caption.

**#42** ✅ — Figure 5 caption missing λ value
- Added "at λ = 10⁻³" to caption. ⚠️ Confirm this is the correct λ shown in density_hard/soft.jpeg.

**#43** ✅ — Figure 5 caption word choice flagged
- "overcrowding" → "unphysical crowding".

**#45 + #46 + #47** ✅ — Figure 6a axis and binning unclear
- Added: R=100, non-dim x-axis note, RDF-style binning description to caption.

**#48** ✅ — Figure 6b: no explanation of why overlap decreases with radius
- Added physical explanation to Figure 6b caption.

**#56** ⚠️ TODO — Figure 15 Δt signal not annotated
- Requires editing the matplotlib figure to annotate growth-dominated vs. collision-dominated regimes.

**#57** ✅ — Figure 16 confuses Δt value with CFL parameter
- Caption clarified: x-axis = fixed CFL policy across independent runs; U-shape explained.

**#59** ✅ — Figures 17a and 19a appear identical
- Fig 17a scoped to R≈100 (N≤175k); Fig 19a explicitly contrasted as R=260 large-scale regime.

**#63** ✅ — Figure 17c "final phase" ambiguous
- Added "The rightmost segment (ReLCP Iter 6) shows the final feasible configuration where all constraints are resolved."

**#64** ⚠️ TODO — Figure 19b wavelengths not annotated
- Requires editing the matplotlib figure to add wavelength annotations (λ≈27, 21, 32, 25, 19).

---

### D — Notation & Math

**#9** — ℓ not written as ℓ(t) on first introduction
- *Target:* End of page 2 / start of Sec. III where cell length ℓ first appears
- *Fix:* Write ℓᵢ(t) on first use to make time-dependence explicit before it appears in dynamic equations.

**#14** — Mobility matrix entries: ℓ₀ scaling to verify
- *Target:* Eq. 5: M^k = diag(1/ζℓ₁^k I₃, 12/ζ(ℓ₁^k)³ I₃, ...)
- *Fix:* Sam annotates entries should be "1/ζℓ₀⁻³, 1/ζℓ₀³ I₃." Verify dimensional analysis after non-dimensionalization with ℓ₀ = 1. If ℓ₀ factors cancel, add a note clarifying that ℓ₀ = 1 simplifies the expression.

**#19** — γ missing timestep superscript k (explanation needed)
- *Target:* Eq. 9: F_nα^hard = n̂_α γ_α
- *Fix:* Add a parenthetical: "The multipliers γ are solved fresh each timestep and carry no history; unlike F^k, they require no k superscript."

**#22** — Eq. 16 bracket notation ambiguous
- *Target:* Eq. 16 energy minimization opening "= ("
- *Fix:* Ensure LaTeX uses `\left(` / `\right)` to size brackets correctly and that the equality sign is not visually merged with the opening bracket. Consider a two-line split for readability.

**#23** — Growth term in linearization not explained
- *Target:* Eq. 14, Φ^k_growth = −(∇_ℓ Φ^k)ℓ̇
- *Fix:* Add: "The growth term Φ^k_growth captures how cell elongation reduces separation distances during the timestep, and is already included in the linearized Φ^{k+1} without additional correction."

**#24** — ≈ Φ^k(γ) relation not stated explicitly
- *Target:* Eq. 17 and surrounding text: ∇_γ E = Φ^k + Δt(...)
- *Fix:* Add after Eq. 17: "Noting that ∇_γ E ≈ Φ^{k+1}(γ), minimizing E is equivalent to driving the linearized separation distances to zero."

**#29** — Cross-reference to Eq. 14 verified ✓
- *Target:* Reference to Eq. 14 in the ReLCP/BBPGD section
- *Fix:* No change needed — Sam confirmed correct. Ensure label matches in final LaTeX.

---

### E — Citations

**#25** — [15]? near Eq. 17 energy formulation
- *Target:* Eq. 16–17 energy minimization
- *Fix:* Verify whether the energy reformulation of the NCP originates from [15] (You et al.) or [11, 25]. Add citation if applicable; otherwise cite the NCP literature already referenced.

**#51** — Missing citation to You et al. for microdomains
- *Target:* Sec. VI-D, "...aligned cells that do not resemble experimentally observed microdomains."
- *Fix:* Append [15] directly: "...experimentally observed microdomains [15]."

**#62** — ReLCP convergence needs citation or caveat
- *Target:* Sec. IV-6 or VII-E, claim that ReLCP "typically converges within a small number of iterations [23]"
- *Fix:* Either find a formal convergence result in the ReLCP / LCP literature, or be explicit: "In practice, convergence within ≤6 ReLCP iterations is consistently observed (Figure 17c), though no formal bound is available."

---

### F — Cuts, Decisions & Verification

**#2** — Corresponding author missing
- *Target:* Title/author block
- *Fix:* Add "∗ Corresponding author" marker and email for Manuel Lerchner if applicable.

**#3** — Abstract soft model claims need verification
- *Target:* Abstract: "packing fractions exceeding 5 in colony centers"
- *Fix:* Cross-check against Figure 6a (which does show φ≈5 for λ=10⁻⁴). Confirm "distorted microdomains" and "unrealistic cell bundles" are visually supported by Figure 9/10 before finalising.

**#20** — Figure 3 drop decision
- *Target:* Figure 3 (Hertzian F^elastic vs δ plot)
- *Fix:* Drop if Eq. 7 + text suffices. Keep only if the nonlinear shape needs explicit visualisation. Sam leans toward dropping.

**#30** — PETSc description too vague
- *Target:* PETSc bullet in Sec. V-A
- *Fix:* State actual parallelism capability: "PETSc supports distributed computing across hundreds of CPU cores via MPI, providing scalable sparse matrix operations and iterative solvers." Remove or correct the "16 CPUs" figure if inaccurate.

**#31** — "ghost cell" undefined on first use
- *Target:* MPI bullet: "...including ghost-cell exchanges at domain boundaries..."
- *Fix:* Add parenthetical: "ghost cells (copies of boundary particles replicated to neighbouring MPI ranks for cross-boundary collision detection)."

**#32** — Collision pipeline possibly incomplete
- *Target:* Sec. V-B Collision Handling Pipeline
- *Fix:* Verify the pipeline description covers all steps (broad-phase → narrow-phase → force assembly → stress computation). Sam asks "is this really all?" — add stress computation step if missing.

**#33** — Threshold distance d=0.5 unjustified
- *Target:* "...within a threshold distance d = 0.5"
- *Fix:* Add justification: "The threshold d = 0.5 equals one cell radius, ensuring both overlapping cells and near-contact cells are captured for constraint generation."

**#50** — Microdomain comparison needs literature grounding
- *Target:* Sec. VI-D, comparison with You et al. [15]
- *Fix:* Re-read [15] and related E. coli microdomain literature to ensure the comparison is accurate and appropriately caveated (You et al. uses constant growth rates; this work uses stress-dependent growth).

**#52** — Figure 12 missing analytical growth rate curve
- *Target:* Figure 12 (Radial relative growth rate profiles e^{−λσ})
- *Fix:* Figure 12 currently shows only hard and soft model simulation profiles. Add the analytical growth rate prediction e^{−λσ̄(r)} derived from Weady et al. [23]'s stress formula σ̄(r) ≈ (2/λ)ln(1/(8c) − cλr²) as a reference curve (dashed black line). Add to caption: "Dashed black: analytical prediction from [23]."

**#55** — Load imbalance: confirm it is not a bottleneck
- *Target:* Sec. VII-B, sentence about decreasing efficiency at high core counts
- *Fix:* Add explanation: "Despite increasing load imbalance, the angular decomposition ensures each rank handles a geometrically similar colony slice, limiting imbalance to a modest factor at the core counts tested."

**#58** (was #57) — Δt stability vs. CFL analysis
- *Target:* Figure 16 and surrounding Sec. VII-D text
- *Fix:* Already partially addressed in C-#57. Additionally: explain in text why the U-shaped curve is informative even though Δt is not constant — it reflects the total runtime cost of using a given CFL factor as a policy.

**#60** — BBPGD behaviour at N=1.3k vs. actual colony size
- *Target:* Sec. VII-E, BBPGD iteration discussion
- *Fix:* Confirm and state explicitly which N the Figure 17a data represents (N≈36,244 for R≈100). If Sam questions whether the behaviour is already present at N=1,300, add a note or subplot.

**#61** — Future improvements claimed without citations
- *Target:* Sec. VII-E last paragraph on warm-start and feedback-based Δt
- *Fix:* Add "We propose the following directions as open research problems not yet explored in the literature:" to clearly distinguish original ideas from cited prior work.

**#65** — Stress profile matching not explained
- *Target:* Sec. VII-F, "the simulated stress profile closely follows the analytical prediction of Weady et al. [23]"
- *Fix:* Add: "At R=260 with ≈301k cells, the colony satisfies the continuum limit assumptions underlying [23]'s analytical derivation, explaining the improved agreement relative to smaller colonies."

**#66** — "Robustness justifies…" sentence flagged by Sam
- *Target:* Sec. VIII, "This robustness justifies the use of computationally simpler, soft, models for studies focused on colony-level pattern formation at small radii."
- *Fix:* Revise or remove. If kept, qualify: "...provided packing fidelity and cell-scale accuracy are not required." If this claim is inconsistent with the overall conclusion that the hard model is preferred, remove.

**#68** — PETSc + OpenMP claim may be outdated
- *Target:* Sec. IX-A, "PETSc does not fully support hybrid operations, so alternative libraries or frameworks may need to be considered."
- *Fix:* Verify current PETSc documentation — Sam notes OpenMP support may exist. If PETSc now supports MPI+OpenMP, update to: "PETSc supports limited hybrid MPI+OpenMP parallelism via its VecCUDA and thread-based backends; leveraging this could reduce per-rank memory and communication overhead."

**#69** — Model Extensions: justify or cut each direction
- *Target:* Sec. IX-D Model Extensions paragraph listing external boundaries, nutrient fields, chemotactic signaling
- *Fix:* For each proposed extension, add one sentence of biological motivation. Remove any direction that cannot be motivated in ≤1 sentence.

**[NEW]** — Nutrient depletion incorrectly implied as part of model
- *Target:* Sec. VI-E (p. 12), "...where mechanical feedback and nutrient depletion restrict active expansion to the growing front [5],[31],[32]."
- *Fix:* This sentence describes real microbial biology but may mislead readers into thinking our model captures nutrient depletion. Add a clarifying sentence: "Our model captures only stress-dependent mechanical feedback; nutrient depletion is not included." Alternatively, restructure the sentence to make clear this is a biological observation not reproduced by the model: "...where both mechanical feedback and nutrient depletion drive this transition — our model isolates the mechanical contribution alone."
