# Experiment 24 — Generate matrix with given eigenvalues and sparsity mask

## Goal
Implement `generate(sparsity_mask, eigenvalues)` that produces a complex matrix matching:
1. Given eigenvalues (approximately)
2. Given sparsity mask (or same sparsity level)

Metrics tracked: diagonal ratio, sparsity, eigenvalue diff (mean optimal assignment), mask match.
Preprocessing stage allowed (not timed, cannot use eigenvalues).

---

## Methods

### v1 — Stiefel optimization (direct, no precompute)
Optimize Q ∈ O(n) via Conjugate Gradient on Stiefel manifold to minimize energy at off-mask positions. Cost: `sum |W_ij|^2` for (i,j) outside mask. Hard-zero outside mask after optimization.

| generate() time | diag ratio | sparsity | eigdiff | mask match |
|---|---|---|---|---|
| 16.16s | 0.0930 | 0.9002 | 0.080819 | 1.0000 |

---

### v2 — Generic Stiefel Q + warm-started Stiefel (100 iter)
Precompute: Stiefel over 100 random eigenvalue sets (500 iter). Generate: warm-start + 100 Stiefel iterations with actual eigenvalues.

| generate() time | diag ratio | sparsity | eigdiff | mask match |
|---|---|---|---|---|
| 2.48s | 0.0946 | 0.9002 | 0.124663 | 1.0000 |

Worse eigdiff than v1 — precomputed Q is in a local minimum bad for specific eigenvalues.

---

### v3 — Iterative Schur projection (100 iterations in generate)
Precompute: same Stiefel as v2. Generate: init W from precomputed Q, then 100 Schur iterations. Each iteration: Schur-decompose W, replace diagonal of T with target eigenvalues (optimal assignment), reconstruct W = Z @ T_new @ Z^H, hard-zero outside mask.

| generate() time | diag ratio | sparsity | eigdiff | mask match |
|---|---|---|---|---|
| 16.78s | 0.0472 | 0.9002 | 0.034818 | 1.0000 |

Best eigdiff so far (2.3× better than v1). Diagonal ratio 2× better.

---

### v4 — Schur precompute + O(n²) generate (T_template approach)
Precompute: 100 Schur iterations with template eigenvalues. Store final (T_template, Z_template). Generate: replace diagonal of T_template with actual eigenvalues (optimal assignment), reconstruct W = Z_template @ T_new @ Z_template^H, hard-zero. Pure O(n²).

| generate() time | diag ratio | sparsity | eigdiff | mask match |
|---|---|---|---|---|
| ~1ms | ~0.007 | 0.9002 | ~0.093 | 1.0000 |

Extremely fast generate. eigdiff limited by template specificity — Z_template generalizes imperfectly to new eigenvalues.

---

### v5 — Schur precompute + T_template warm start + 20 Schur refinements (**current best**)
Precompute: 100 Schur iterations with template eigenvalues (complex unitary init). Generate: T_template warm start → O(n²) init → 20 Schur iterations refining with actual eigenvalues.

| generate() time | diag ratio | sparsity | eigdiff | mask match |
|---|---|---|---|---|
| 2.02s | 0.0130 | 0.9002 | 0.042150 | 1.0000 |

8× faster than v1 with eigdiff 2× better. Scaling: O(20n³) — cubic, with 5× lower constant than v1 (100 iters).

**Convergence profile from T_template warm start:**
- 0 iter: 0.093 (1ms), 5 iter: 0.065 (450ms), 10 iter: 0.052 (1.4s), 20 iter: 0.042 (2.0s)
- Each ~95ms per Schur step, diminishing returns after ~20 steps.

---

## Failed approaches

- **Stiefel incoherence objective**: minimize `sum_{(i,j)∉mask} (Z²@Z²ᵀ)_ij` — eigenvalue-agnostic but fourth-order in Z, many bad local minima. eigdiff=0.209.
- **Multi-restart Schur**: restart with different eigenvalues after each block of iterations. Z forgets previous sets, converges to last eigenvalue set. eigdiff=0.391.
- **More precompute iterations**: 300 vs 100 iterations gives same O(n²) eigdiff (~0.093). Bottleneck is Z_template generalization, not T convergence.

---

## Ideas to Explore

- [ ] **Add early stopping** based on eigdiff convergence threshold — save time when method converges quickly
- [ ] **Better precompute init**: multiple random starts, pick best Z_template by held-out validation set
- [ ] **Hybrid O(n²) + few GPU steps**: use GPU torch.linalg.eigvals autograd for fast gradient descent on sparse W entries — may be much faster than CPU Schur
- [ ] **Reduce n_refine to 10** (eigdiff≈0.052, time≈1.4s) if time is more critical than eigdiff
- [ ] **Complex Stiefel (Unitary manifold)** for precompute — more degrees of freedom, potentially better Z
- [ ] **Characterize O(n²) theoretical limit** — is 0.093 close to the minimum for random orthogonal Z?
