# Draft rebuttal snippet — quantitative comparison with the literature (R2-M2)

For the response letter (and optionally a short paragraph/table in §6). Addresses
Reviewer 2's request to compare our quantitative measures against published values.

## Unit note (important)

- **Our model & Weady et al. 2024**: lengths nondimensionalized by the birth cell length
  `ℓ₀` (with diameter `d = ℓ₀/2`, aspect ratio 2:1 at birth).
- **You et al. 2018**: lengths nondimensionalized by the cell *diameter* `d₀`
  (= our `d` = `ℓ₀/2`). So `1 d₀ = 0.5 ℓ₀`; a You-et-al. area in `d₀²` is `0.25 ℓ₀²`.

## Comparison table

| Quantity | Our value | Published value | Source |
|---|---|---|---|
| Cell aspect ratio (length/diameter) | 2 → 4 (birth → division) | 2:1 fixed (Weady); 2–5 (You) | Weady2024; You2018 |
| Concentric-ring spacing (λ = 10⁻³) | ξ ≈ 25 ℓ₀ | ≈ 16 ℓ₀ (from 5 ℓ₀ at λ=10⁻² with λ^(−1/2) scaling) | Weady2024 |
| Ring-wavelength scaling with λ | consistent (larger spacing at lower λ) | wavelength ∝ λ^(−1/2) | Weady2024 |
| Microdomain area distribution | broad, right-skewed; broader/larger for soft at low λ | exponential P(A) ∝ exp(−A/A*) | You2018 |
| Microdomain size vs growth | larger domains at lower growth (higher λ) | ⟨A⟩ decreases with growth rate g | You2018 |
| Orientational correlation length ξ_θ | 1.2 → 1.9 ℓ₀ (increasing with λ) | not reported (neither reference quantifies it) | — |

## Talking points for the letter

1. **Ring spacing (Weady2024, same units).** Weady et al. report a ring wavelength of
   ≈ 5 ℓ₀ at λ = 10⁻² that scales as λ^(−1/2); extrapolated to λ = 10⁻³ this gives
   ≈ 16 ℓ₀. Our measured spacing ξ ≈ 25 ℓ₀ (large colony, R ≈ 260) is of the same order.
   The residual factor (~1.5×) is expected: our value is measured at finite colony size,
   whereas the scaling is the asymptotic continuum prediction, and ring spacing still
   coarsens with colony age. We also reproduce the qualitative λ-dependence (wider rings
   at lower stress sensitivity).

2. **Microdomains (You2018).** You et al. find that domain areas follow an exponential
   distribution and that the mean domain size *grows as the growth rate falls* (their
   Fig. 4d) and as cells become more slender. Our microdomain-area distributions
   (Fig. cluster-area) reproduce both trends: broad, right-skewed distributions whose
   size increases at low λ (i.e. reduced growth). The characteristic domain length
   implied by their A* = 54–339 d₀² is √A* ≈ 7–18 d₀ ≈ 3.7–9 ℓ₀, the same scale as the
   domains we observe.

3. **Orientational correlation length is a new metric.** Neither reference reports an
   orientational correlation length — Weady et al. explicitly leave the colony's
   orientational structure to future work. Our ξ_θ ≈ 1.2–1.9 ℓ₀, increasing with stress
   sensitivity, is therefore a new quantitative characterization, consistent in trend
   with You et al.'s finding that slower growth yields larger, more ordered domains.

## Caveats (be upfront in the letter)

- The ring-spacing comparison relies on extrapolating Weady et al.'s λ^(−1/2) law from
  λ=10⁻² to 10⁻³; we should present it as "same order / consistent scaling," not an exact
  match.
- The You et al. A* → domain-length conversion assumes their d₀ = our d; this is a
  geometric estimate, not a like-for-like re-measurement of their data.
- Experimental (as opposed to simulation) values in these papers are limited: You et al.
  state their *experimental* statistics are "not sufficient to make conclusive
  statements," so the strongest quantitative anchors are the two papers' *simulation*
  results, which share our geometry and nondimensionalization.
