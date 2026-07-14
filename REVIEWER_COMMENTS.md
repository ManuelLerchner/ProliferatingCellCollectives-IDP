# Reviewer Comments — COMPHY-D-26-00492

Point-by-point mapping of both reviews to the LaTeX source (`latex/cell-collectives.tex`).
Columns: the reviewer remark, its location in the manuscript, what we currently claim there, and the requested change.

- Manuscript: *Proliferating Cell Collectives: A Comparison of Hard and Soft Collision Models*
- Decision: **Revision** (R1 minor, R2 major) — revised version due **Sep 01, 2026**

Status markers in the `#` column: **✅** implemented (see Progress log for the exact action) · **⏳** partially addressed · no marker = open. All ✅ edits compile (22 pages).

---

## Reviewer #1 (minor revision)

| # | Reviewer remark | LaTeX location | What we currently claim | What reviewer wants |
|---|---|---|---|---|
| R1-1 ✅ | Strengthen discussion of soft-particle approaches; cite recent growing-colony studies using soft models (PRL 2013 **111** 168101; J. R. Soc. Interface 2017 **14** 20170073; 2015 **12** 20141290; Soft Matter **19**(42) 8136–8149 2023; PRE 2017 **96** 052404; Soft Matter **19** 1034–1045 2023; Soft Matter **20** 3823–3835 2024) | §2.1 "Soft (Potential-Based) Models" (L305–308); §4.4 Soft Collision Model (L511) | Soft models described mainly via their limitations (numerical stiffness, elastic-deformation artifacts); only Warren2019 cited as a large-scale soft example (L306). §4.4 cites a generic block of soft-model refs. | Add a brief discussion acknowledging soft-particle frameworks remain widely used and have delivered important insights into colony growth / collective dynamics; cite the listed works to place the study in broader context and give a balanced view. |
| R1-2 ✅ | Discuss expected 3D behavior (biofilms, aggregates are intrinsically 3D); would the hard model's advantage grow, or new challenges arise from contact handling / constraint resolution? No new simulations requested. | §3 Cell Mechanics (L332: "extension to fully three-dimensional simulations is straightforward"); Discussion / Future Work (§Model Extensions, L1171) | We restrict to 2D, keep a 3D vector formulation, and call the 3D extension "straightforward" (L332). No discussion of how the hard-vs-soft trade-offs change in 3D. | Add a short discussion of expected 3D behavior and whether the hard model's computational advantage becomes more significant or whether contact-handling / constraint-resolution challenges arise. |
| R1-3 ✅ | Acknowledge that model choice is dictated not only by efficiency but by which physical processes one wants to capture (soft models naturally accommodate deformability / force-based mechanical interactions). | §2.1 (L305–308); §6 Microdomain discussion (L964–970); Discussion / Conclusion (L1157) | We frame the comparison around accuracy + performance and conclude hard is "strongly preferred for nearly all applications" (L1157). Deformability appears only as a soft-model artifact, not an intended feature. | Add a short perspective noting soft models can naturally represent deformability and force-based interactions, so the appropriate framework depends on the physics of interest — helping readers select a model for their own application. |

---

## Reviewer #2 (major revision)

### Major concerns

| # | Reviewer remark | LaTeX location | What we currently claim | What reviewer wants |
|---|---|---|---|---|
| R2-M1 | Soft model shows very large unphysical overlaps (packing fraction >5 in center). Is the soft model intrinsically inadequate, or does the chosen stiffness / timestep criterion / force law / integration scheme under-resolve soft-contact dynamics? | Abstract (L218); §4.4.2 Model Parameters & Numerical Stability (L530–532); §6.2 (L813–817); Conclusion (L1151) | We report packing fractions "exceeding 5" and attribute overlap to local pairwise forces failing to propagate stress + a too-large step-size (L815); note a smaller overlap-based Δt could help but hurt performance (L817, L1151). k_cc = 20000, Δt ~ 10⁻⁵ stated as fixed choices (L532). | Clarify / analyze whether the inadequacy is intrinsic or a resolution issue — justify or test the stiffness k_cc, timestep criterion, Hertzian force law, and explicit-Euler integration; show the >5 packing fraction is not simply an under-resolved-parameter artifact. |
| R2-M2 | Experimental comparison is largely visual. Add quantitative measures: ring spacing, radial density profiles, colony expansion rate, orientational correlation length, microdomain size distribution, and comparison with experimental data from cited literature. | §6.1 Concentric Rings (L801–807); §6.2 Packing (L811–817, Fig. radial packing); §6.3 Growth Dynamics (L855–860); §6.5 Microdomains (L910–970, Fig. cluster_area_boxplot); §7.7 (L1105 ring spacing ξ≈25) | Ring patterns / microdomains validated "qualitatively" and by visual resemblance to experiments (L803, L807, L915). We already have radial density (Fig. radial packing), microdomain area distribution (cluster_area_boxplot), and ring spacing ξ≈25 at R=260 (L1105); colony expansion is implicit in Fig. sim_time_vs_colony_radius. No orientational correlation length; no direct quantitative experimental comparison. | Add quantitative metrics. Per Sam: RDF and expansion rate may not need new plots (expansion = gradient of radius-vs-time fig; RDF possibly repo/Zenodo only). Orientational correlation length/function and microdomain size distribution are worth actual plots. Ideally compare against experimental values from the cited papers. |
| R2-M3 ✅ | CFL timestep uses a median velocity scale; may miss rare fast-moving cells / local overlaps in dense heterogeneous regions. Justify median over maximum, percentile-based, or local-overlap-based criterion. | §5.3 Adaptive Timestepping (L726, L732, Alg. adaptive_dt); §6.2 (L817) | u_m defined as the **median** of u_i = ‖v_i‖ + ℓ̇_i (L726); we already concede the scheme "bases Δt solely on cell velocities but not on current overlap" and that an overlap-aware scheme could help but cost performance (L817). | Justify why median is preferable to max / percentile / local-overlap criteria — explain the trade-off explicitly rather than only noting the limitation. |
| R2-M4 ✅ | "Hard model strongly preferred for nearly all applications" is too broad: simulations are 2D, one class of rod-like cells, neglect nutrient fields and biochemical feedback, rely on one specific soft-contact implementation. | Abstract (L222); Conclusion (L1157) | "the hard model is strongly preferred for nearly all applications" (L222, L1157). | Qualify / narrow the claim; scope it to the conditions actually tested (2D, single rod-like cell type, no nutrient / biochemical feedback, one soft-contact implementation). |

### Minor comments

| # | Reviewer remark | LaTeX location | What we currently claim | What reviewer wants |
|---|---|---|---|---|
| R2-m1 ✅ | Inconsistent terminology: "step-size," "timestep," "time step" used interchangeably. | Throughout — e.g. "step-size" (L220, L293, L532, L680), "timestep" (L313, L562, L718), "time step" (L1059, L1085) | Mixed usage. | Choose one term and use it consistently throughout. |
| R2-m2 ✅ | Refers to both bacteria and fungi, but later uses "bacteria" exclusively. Clarify scope. | Intro (L280 fungi *Setosphaeria/Exserohilum*); §3 (L328 "bacteria and fungi"); Fig. exserohilum (fungus); but abstract/title/keywords say "bacteria" (L214, L246) | We model rod-like cells and cite both bacteria and fungi (L280, L328), yet the abstract, keywords, and much prose say "bacteria." | State clearly whether the model targets bacterial colonies only or rod-like microbial collectives generally; make terminology consistent. |
| R2-m3 ✅ | The claim that the 3D extension is "straightforward" should be qualified — 3D contact detection, domain decomposition, and solver scalability add substantial challenges. | §3 Cell Mechanics (L332) | "the extension to fully three-dimensional simulations is straightforward" (L332). | Qualify this statement; acknowledge the added 3D challenges (contact detection, decomposition, solver scalability). *(Overlaps R1-2.)* |
| R2-m4 | Define all symbols immediately at first appearance; matrix quantities D, L, G and the stress mapping need clearer explanation or a schematic. | §4.3 (G, L498); §4.5 (D, L544–546; L, L548–551); stress mapping (L549–553, L611–613) | G, D, L defined tersely in-line; their role explained mostly via "mimics … force/stress assembly" and citations to Weady2024SM. | Define every symbol at first use; add a clearer explanation or schematic for D, L, G and the stress mapping. |
| R2-m5 ✅ | Abstract is long and already contains many detailed numerical values; shorten. | Abstract (L213–223) | Four paragraphs with specifics: packing 0.9, ">5", 9.36× speedup, 112 cores, ~30× step-size, CFL (L218–222). | Shorten by focusing on the main comparison, the most important physical difference, and the key performance conclusion; trim detailed numbers. |

---

## Cross-cutting notes

- **R1-2 ≡ R2-m3** — same 3D-"straightforward" issue (L332); address once.
- **R1-3 ↔ R2-M1** — both touch the soft model's deformability framing: R1 wants it credited as a legitimate feature; R2 wants the >5 overlap defended as not merely under-resolution.

---

## Sam's notes (email, 6 July)

General: the reviews seem reasonable; mostly extra discussion is needed. Sam plans to do some of this work (~a week out at time of writing).

| Reviewer point | Sam's take / plan |
|---|---|
| **R2-M2** (quantitative measures) | See if RDF and colony expansion rate really warrant actual plots. **RDF:** unsure how it will look for us; other cell models show it (e.g. Bangalore/BPJ 2015, `doi.org/10.1016/j.bpj.2015.08.003`, Fig. 4). If it looks boring, just add it to the repo/Zenodo and say so in the rebuttal. **Colony expansion rate:** already covered by the gradient of Fig. 6(b) (Sam's figure numbering — the colony-radius-vs-time figure); no new plot needed, one or two sentences in §6.3 suffice, point to this in the rebuttal. **Orientational correlation length:** definitely a good idea — even an actual plot of the orientation correlation *function* could be good (Sam can share implementation details). **Microdomain size distribution:** good idea, probably easy with the current scripts. |
| **R1-3** (soft models capture deformability) | Good point. Add to the discussion that we can reduce the solver tolerance, allowing some cell deformation — this would also *improve the efficiency of the hard model* further — but a proper physical comparison of such an approach would be the subject of future work. |

---

## Progress log

### Done (manuscript compiles clean — 22 pages)

- **R2-m1** (terminology) ✅ — Survey showed the source already distinguishes two genuinely different concepts, each internally consistent: *timestep* (42×, the discrete step) vs *step-size* (40×, the Δt magnitude). Only one true outlier: a lone "time step" (two words) at L1059 → changed to "timestep". Rebuttal line: the two terms denote distinct things (the step vs. its size) and are now spelled consistently.
- **R2-m3 / R1-2 (partial)** (3D "straightforward") ✅ — Rewrote L332. **Important correction:** the implementation is *already* fully 3D-compatible — contact detection runs on 3D line segments (`GeometricTools` [Eberly, *Distance Between Line Segments*] + `Yan2019`), and the 2D runs are just the planar case with out-of-plane force/noise components disabled. New text states the framework is natively 3D, cites both references, mentions the bounded-cube demo (virtual wall constraints) in the supplementary, and gives the one honest 2D-specific caveat: the angular-sector domain decomposition (§7.1) would need generalizing for large-scale 3D. Turns the reviewers' concern into a strengthening point rather than a concession.
- **R2-m2** (bacteria vs fungi scope) ✅ — Existing content (fungus figure, *Setosphaeria*/*Exserohilum* in intro, "bacteria and fungi" in §3) makes bacteria-only untenable, so broadened rather than narrowed. Added a sentence at §3 (after L328): "Although we refer to bacterial colonies throughout for concreteness, the model targets rod-shaped microbial collectives in general, and applies equally to the rod-shaped fungi noted above." Also changed abstract opening "bacterial colonies" → "microbial colonies" for consistency.
- **R2-M4** (claim too broad) ✅ — Per user decision, scoped rather than rebutted. Abstract final sentence and conclusion (L1157) now read "Within the regimes examined here — two-dimensional, mechanically-driven colonies of a single rod-shaped cell type without nutrient or biochemical feedback, and a single soft-contact implementation — the hard model is preferred for most applications…". "strongly preferred for nearly all applications" → "preferred for most applications".
- **R2-m5** (abstract too long) ✅ — Per user decision, shortened now. Cut from 5 paragraphs to 4; removed the specific numbers 0.9, >5, 30×, 112 cores, and the standalone CFL sentence (all still carried in the Highlights); kept one headline performance figure ("roughly 9-fold speedup") and the key physical difference (realistic packing vs. unphysical overlap). Also folds in the M4 scoping.
- **R1-1** (soft-model literature) ✅ — Looked up all 7 reviewer-supplied references online (verified via OpenAlex), added them to `literature.bib` as `Farrell2013`, `Farrell2017`, `Giverso2015`, `GhoshLevine2017`, `Bera2023a` (SM 19:1034), `Bera2023b` (SM 19:8136), `Khandoori2024` (SM 20:3823). Added a balanced passage in §2.1 crediting soft-particle frameworks with concrete insights (front dynamics + mutation surfing, branching instability, phase separation / cell-death morphodynamics, motility–resource–EPS interplay and nematic order) and noting they remain widely used. All citations resolve; no undefined refs.
- **R1-2** (expected 3D behavior) ✅ — Extended the §3 paragraph: hard-model advantage expected to persist/widen in 3D (soft stiffness-limited step-size stays the bottleneck; hard's larger steps keep amortizing cost); main new challenge is solver/decomposition scalability under higher 3D contact density, not contact detection.
- **R1-3** (deformability / model-choice perspective) ✅ — Added a Discussion paragraph: soft models natively capture deformability/force-based interaction (natural when compliance is the object of study); the hard model can reintroduce controlled deformation by relaxing tolerance ε, which *also* speeds it up (Sam's angle); rigorous comparison = future work.
- **R2-M3** (justify median velocity) ✅ — Added a §5.3 paragraph justifying median over max (right-skewed speed distribution; rare outliers would collapse Δt though ReLCP/BBPGD + ε already cap penetration), over percentile (reintroduces tail sensitivity), and over local/per-cell (incompatible with the synchronous global solve); notes overlap-based criteria are possible but costlier. *Argument-based; an optional empirical max-vs-median run (Group C) could strengthen it.*

### Left for later

- **R2-m4** — define symbols / add **schematic for D, L, G**: the symbols are already defined at first use, so the remaining ask is a schematic figure (γ → forces via D, → stress via L, → state update via G). Needs a figure-design pass.
- **R2-M1, R2-M2** — major concerns needing analysis / new plots / possible new runs (see plan groups B & C and Sam's notes). R2-M1 pending the decision on whether to run the convergence/stiffness sweep.

---

*Line numbers refer to `latex/cell-collectives.tex` at the time of review.*
