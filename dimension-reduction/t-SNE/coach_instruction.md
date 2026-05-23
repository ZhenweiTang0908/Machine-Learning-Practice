# t-SNE Implementation Guide

## Overview

You will implement t-SNE by filling in the TODO blocks in `src/tsne.go`.
`src/data.go` is already complete.

Unlike MDS/PCA (which have closed-form eigendecomposition solutions), t-SNE
is an **iterative optimization algorithm**. The core idea: match a joint
probability distribution P over pairs of points in the high-dimensional
space to a distribution Q in the low-dimensional embedding space, by
minimizing their KL divergence via gradient descent.

## Algorithm Summary (from derivation.pdf)

| Step | Math | Code location |
|------|------|---------------|
| 1. Pairwise squared distances | Dᵢⱼ = ‖xᵢ−xⱼ‖² | `computePairwiseDistances()` (done) |
| 2. High-dim similarities P | p_{j\|i} = softmax(−Dᵢⱼ/2σᵢ²), binary search σᵢ for perplexity, symmetrize Pᵢⱼ = (p_{j\|i}+p_{i\|j})/(2n) | `computeHighDimSimilarities()` + `binarySearchSigma()` |
| 3. Random init Y | Y ~ N(0, 1e-4) | `initEmbedding()` (done) |
| 4a. Low-dim similarities Q | qᵢⱼ ∝ (1+‖yᵢ−yⱼ‖²)⁻¹ | `computeLowDimSimilarities()` |
| 4b. Gradient ∂L/∂Y | ∂L/∂yᵢ = 4 Σⱼ (pᵢⱼ−qᵢⱼ) wᵢⱼ (yᵢ−yⱼ) | `computeGradient()` |
| 4c. Momentum update | v ← αv − η·grad; Y ← Y + v | `Fit()` loop |
| 4d. Early exaggeration | P ← 12·P for first 250 iters | `Fit()` loop (structure done) |
| 5. Return embedding | Y (n×d) | `GetEmbedding()` (done) |

## Math → Code Mapping

### Step 2: High-dim similarities P

This is the most mathematically involved step. The derivation (Appendix A)
uses **perplexity** as a smooth measure of the effective number of neighbors
around each point.

**The chain of reasoning:**

1. For each point i, we define a Gaussian distribution over all other points
   centered at xᵢ with bandwidth σᵢ:

   ```
   p_{j|i} = exp(−D_ij / (2σᵢ²)) / Σ_{k≠i} exp(−D_ik / (2σᵢ²))
   ```

2. The bandwidth σᵢ controls how "wide" the neighborhood is:
   - Small σ → only very close points matter (low entropy)
   - Large σ → many points matter (high entropy)

3. We choose σᵢ so that the entropy of this distribution equals log₂(perplexity):

   ```
   H(Pᵢ) = −Σ_{j≠i} p_{j|i} · log₂(p_{j|i})
   target_H = log₂(perplexity)
   ```

   If perplexity = 30, the distribution's entropy should be log₂(30) ≈ 4.9 bits,
   meaning it effectively "sees" roughly 30 neighbors.

4. Since there's no closed form to go from H → σ, we use **binary search**:
   - If H > target_H: σ is too small (too much entropy) → increase σ
   - If H < target_H: σ is too large (too little entropy) → decrease σ

5. After finding all σᵢ, compute p_{j|i} for all i, j, then symmetrize:

   ```
   P_ij = (p_{j|i} + p_{i|j}) / (2n)
   ```

   The `2n` ensures Σ_{i≠j} P_ij = 1, making P a proper probability distribution.

**Implementation plan for `binarySearchSigma()`:**

```go
targetH := math.Log2(perplexity)
sigmaLo, sigmaHi := 1e-12, max(dists)

for iter := 0; iter < 50; iter++ {
    sigma := (sigmaLo + sigmaHi) / 2

    // Compute p_{j|i} for this sigma
    row := make([]float64, n)
    for j := 0; j < n; j++ {
        if j == i { row[j] = 0; continue }   // source point
        row[j] = math.Exp(-dists[j] / (2 * sigma * sigma))
    }
    // Normalize
    sum := 0.0
    for j := 0; j < n; j++ { sum += row[j] }
    for j := 0; j < n; j++ { row[j] /= sum }

    // Compute entropy H
    H := 0.0
    for j := 0; j < n; j++ {
        if row[j] > 0 {
            H += row[j] * math.Log2(row[j])
        }
    }
    H = -H

    // Binary search decision
    if H > targetH {
        sigmaLo = sigma   // need larger bandwidth
    } else {
        sigmaHi = sigma   // need smaller bandwidth
    }

    if sigmaHi-sigmaLo < 1e-12 { break }
}
```

**Key insight:** You need to know which row index `i` this is, so you can skip
`j == i` in the computation. Pass the source index or handle it in the caller.

### Step 4a: Low-dim similarities Q

Uses the **Student-t kernel** with 1 degree of freedom (Cauchy distribution):

```
q_ij = (1 + ‖y_i − y_j‖²)^(−1)  /  Z
Z = Σ_{k≠l} (1 + ‖y_k − y_l‖²)^(−1)
```

**Why Student-t?** The heavy tail decays as ‖y‖⁻² (vs. exponential for Gaussian).
This solves the **crowding problem** (Appendix B): in 2D, there isn't enough
volume to spread out moderately similar points. The heavy tail allows moderate
distances to still produce meaningful similarities, preventing collapse.

**Implementation:** Two passes:
- Pass 1: Compute w_ij = 1/(1 + ‖y_i−y_j‖²) for all i < j, store in Q (symmetric)
- Pass 2: Sum all w_ij → Z, then divide every Q entry by Z

### Step 4b: Gradient

The elegant closed form derived in the PDF (Step 4):

```
∂L/∂y_i = 4 · Σ_{j≠i} (p_ij − q_ij) · w_ij · (y_i − y_j)
```

**Interpretation as forces:**
- **(p_ij − q_ij) > 0**: P says closer than Q → **attractive** → yᵢ pulled toward yⱼ
- **(p_ij − q_ij) < 0**: P says farther than Q → **repulsive** → yᵢ pushed from yⱼ
- **w_ij**: heavy-tailed weight, decays for distant pairs so only near neighbors
  contribute significantly
- **Factor 4**: emerges from the chain rule (see derivation, Step 4)

**Optimization trick:** For each pair (i, j), the contribution to grad[j] is
the negative of grad[i] (force pairs). So you can compute both in one loop:

```go
for k := 0; k < d; k++ {
    diff := Y.At(i, k) - Y.At(j, k)
    dY.Set(i, k, dY.At(i, k) + coeff*diff)
    dY.Set(j, k, dY.At(j, k) - coeff*diff)
}
```

### Step 4c: Momentum update

```go
// velocity = alpha * velocity − eta * grad
velocity.Scale(alpha, velocity)
for i := 0; i < n; i++ {
    for j := 0; j < d; j++ {
        v := velocity.At(i, j) - t.learningRate*grad.At(i, j)
        velocity.Set(i, j, v)
    }
}
// Y = Y + velocity
Y.Add(Y, velocity)
```

Momentum smooths the trajectory, dampening oscillations and accelerating
convergence (see Appendix D).

## Implementation Order

Fill in the TODOs in this order — each builds on the previous:

### 1. `computeLowDimSimilarities()` — 20 min
Easiest to verify independently. Takes Y, returns normalized Q.
**Test:** Q should be symmetric, Qᵢⱼ ≥ 0, Σᵢ≠ⱼ Qᵢⱼ ≈ 1.

### 2. `computeGradient()` — 15 min
Follow the closed-form formula literally. No understanding of optimization needed.
**Test:** With random P, Q, Y, the gradient should produce reasonable values
(no NaN, no Inf). Gradient should be zero if P = Q.

### 3. `Fit()` momentum update — 20 min
Fill in the `velocity ← α·v − η·grad` and `Y ← Y + v` inside the loop.
At this point you have a working optimizer, just with placeholder P/Q.
**Test:** The code should run without errors (KL will be meaningless until P is correct).

### 4. `binarySearchSigma()` — 30 min
The math-heavy part. Implement the binary search for a single point's bandwidth.
**Test:** For 4 points at distances [0, 1, 4, 9] with perplexity=2, σ should be
small (the two nearest neighbors should dominate). Print H vs target_H to verify.

### 5. `computeHighDimSimilarities()` — 20 min
Wire the binary search into the full P computation. Call `binarySearchSigma()`
for each point, compute rows, symmetrize.
**Test:** P should be symmetric, non-negative, Σᵢ≠ⱼ Pᵢⱼ ≈ 1.

### 6. End-to-end testing — 15 min
```bash
go run main.go test_data.csv 5 2
```
**Verify:** KL divergence decreases, embedding preserves neighborhood structure.

### 7. Hyperparameter exploration — 15 min
Try different perplexity values (2, 5, 8) and observe how the embedding changes.
(Note: perplexity must be < n, so max is 9 for 10 points.)

## Testing

### Quick test with provided data

```bash
cd dimension-reduction/t-SNE
go run main.go test_data.csv 5 2
```

### Expected results for `test_data.csv` (10 2D points, perplexity=5)

- **KL divergence** should decrease monotonically:
  - Initial: ~3–10 (depends on random init)
  - Final: ~0.5–2.0
- **Embedding** should preserve neighborhood structure — the 3×3 grid with one
  offset point should still be recognizable as a grid structure in 2D
- **No NaN/Inf values** anywhere in the output
- **Every Q row should sum to 1** (after normalization)

### Some debugging checks

1. **P matrix sanity:**
   - Σᵢ≠ⱼ Pᵢⱼ ≈ 1.0
   - P is symmetric
   - No NaN values
   - Print a few σᵢ values — they should vary across points

2. **Q matrix sanity:**
   - Σᵢ≠ⱼ Qᵢⱼ ≈ 1.0
   - Q is symmetric
   - Qᵢⱼ should be large for nearby y's, small for distant y's

3. **Gradient sanity:**
   - If you synthetically set Q = P, gradient should be all zeros
   - Gradient magnitude should decrease over iterations

4. **Entropy in binary search:**
   - Print H and target_H at the final sigma to verify the search worked
   - H should be within ±0.01 of target_H

## Common Pitfalls

- **Log base mismatch.** Perplexity = 2^H means you MUST use log₂ (not ln) for
  entropy in the binary search. Using math.Log() will find the wrong σᵢ.

  ```go
  // WRONG:
  H += row[j] * math.Log(row[j])    // natural log

  // RIGHT:
  H += row[j] * math.Log2(row[j])   // log base 2
  ```

- **The 2n symmetrization factor.** It's `(p_{j|i} + p_{i|j}) / (2n)`, not `/ 2`.
  The extra n ensures Σ Pᵢⱼ = 1 after summing over all pairs.

- **Binary search direction.** If H > target_H, the distribution is too
  focused (high entropy = spread out). Actually wait — H is entropy,
  higher entropy means MORE spread out, meaning σ is too LARGE.
  Let me be precise:
  - Large σ → p_{j|i} becomes flat (uniform) → high entropy
  - Small σ → p_{j|i} is peaked (only nearest neighbors matter) → low entropy
  - If H > target_H: distribution is too flat → **decrease σ** (sigmaHi = sigma)
  - If H < target_H: distribution is too peaked → **increase σ** (sigmaLo = sigma)

  This is a common mixup. Double-check your search direction!

- **σ bounds**: Use sigmaLo = 1e-12 (not 0) to avoid division by zero. Use
  sigmaHi = max(dists) as a reasonable upper bound.

- **Momentum order**: Update velocity FIRST, then position. Not the other way.
  ```go
  // RIGHT:
  velocity = alpha*velocity - eta*grad
  Y = Y + velocity

  // WRONG:
  Y = Y - eta*grad + alpha*(previous update)
  ```

- **NaN in Q**: If Z computes to zero (all w_ij = 0), the embedding has
  collapsed to a single point. The random init prevents this, but check.

- **Gradient scaling**: The factor 4 is essential — it comes from the chain
  rule through the Student-t kernel. Omitting it slows convergence noticeably.

- **Integer division**: `1/(2*sigma*sigma)` with int sigma → integer division.
  Use `1.0 / (2.0 * sigma * sigma)` or just ensure sigma is float64.

## What NOT to Implement

- ❌ **Barnes-Hut t-SNE** — the O(n log n) approximation with quad-trees.
  Implement the exact O(n²) version for this exercise.
- ❌ **PCA-based initialization** — initializing Y via PCA is a common
  optimization to improve convergence, but out of scope here.
- ❌ **Adaptive learning rate (gain)** — the original paper uses per-dimension
  adaptive gains. Simple momentum is sufficient for learning.
- ❌ **Euclidean distance computation library** — use the provided
  `computePairwiseDistances()` or write the triple loop yourself.
- ❌ **KL divergence monitoring/early stopping** — the code already stores KL
  per iteration. No need for sophisticated stopping criteria.

## Gonum Quick Reference

| Operation | Code |
|-----------|------|
| Create n×d matrix | `mat.NewDense(n, d, nil)` |
| Get element | `m.At(row, col)` |
| Set element | `m.Set(row, col, val)` |
| Dimensions | `rows, cols := m.Dims()` |
| Scale | `A.Scale(s, B)` → A = s × B |
| Add in-place | `A.Add(A, B)` → A = A + B |
| Subtract in-place | `A.Sub(A, B)` → A = A − B |
| Get row as slice | `row := mat.Row(nil, i, A)` |
| Get col as slice | `col := mat.Col(nil, j, A)` |

| Math function | Go equivalent |
|---------------|---------------|
| eˣ | `math.Exp(x)` |
| ln(x) | `math.Log(x)` |
| log₂(x) | `math.Log2(x)` |
| √x | `math.Sqrt(x)` |

| Random | Code |
|--------|------|
| Seeded PRNG | `rng := rand.New(rand.NewPCG(seed, seed+1))` |
| N(0, σ²) | `rng.NormFloat64() * sigma` |

## Go Design Pattern: Functional Options

The `TSNE` constructor uses Go's **functional options pattern**:

```go
// Use defaults (learning rate 200, 1000 iters, etc.)
tsne := src.NewTSNE(2, 30)

// Or configure only what you need:
tsne := src.NewTSNE(2, 30,
    src.WithLearningRate(500),
    src.WithMaxIter(2000),
    src.WithRandomState(123),
)
```

This is an idiomatic Go pattern used in many libraries (gRPC, AWS SDK, etc.).
Each option is a function that modifies only one field of the struct, avoiding
bloated constructors with dozens of parameters.
