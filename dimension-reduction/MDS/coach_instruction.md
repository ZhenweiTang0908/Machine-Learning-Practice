# MDS Implementation Guide

## Overview

You will implement Classical Multidimensional Scaling (cMDS) by filling in
the TODO blocks in `src/mds.go`. The `src/data.go` file is already complete.

The key insight: cMDS converts pairwise dissimilarities into inner products
(double-centering), then recovers coordinates via eigendecomposition.

## Algorithm Summary (from derivation.pdf)

| Step | Math | Code location |
|------|------|---------------|
| 1. Double-center D | B = −½ H D H | `doubleCenter()` |
| 2. Eigendecompose B | B = V Λ Vᵀ | `Fit()` Step 2 |
| 3. Pick top k eigenpairs | keep λ₁ ≥ … ≥ λₖ | `Fit()` Step 3 |
| 4. Form coordinates | Zₖ = Vₖ √Λₖ | `Fit()` Step 4 |
| (Optional) Compute D from X | Dᵢⱼ = ‖xᵢ − xⱼ‖² | `ComputeSquaredDistances()` |

## Math → Code Mapping

### Step 0: Compute squared distances (if input is feature matrix)

```
math:   D_ij = Σₗ (x_il − x_jl)²
code:   for i, j, l: D.Set(i, j, 差²)
```

### Step 1: Double-centering

```
math:   rowMean_i  = (1/n) Σ_j  D_ij
        colMean_j  = (1/n) Σ_i  D_ij
        grandMean  = (1/n²) Σ_i Σ_j D_ij

        B_ij = -0.5 × (D_ij − rowMean_i − colMean_j + grandMean)

matrix: B = -½ H D H    where H = I − (1/n)𝟙𝟙ᵀ
```

The element-wise formula is the one to implement. The matrix form is
conceptually equivalent but computationally heavier.

**Why this works:** The derivation shows this eliminates the unknown norms
‖zᵢ‖² and tr(B), giving us inner products directly from distances.

### Step 2: Eigendecomposition

```go
var eig mat.EigenSym
sym := mat.NewSymDense(n, nil)
for i := 0; i < n; i++ {
    for j := i; j < n; j++ {
        sym.SetSym(i, j, B.At(i, j))
    }
}
ok := eig.Factorize(sym, true)   // true = compute eigenvectors
// ok == false → matrix is not symmetric enough for EigenSym

vals := eig.Values(nil)           // ASCENDING order!
vecs := mat.NewDense(n, n, nil)
eig.VectorsTo(vecs)               // eigenvectors as columns
```

**Critical:** `eig.Values()` returns **ascending** (smallest first). The
largest eigenvalue is `vals[n-1]`, NOT `vals[0]`.

### Step 3: Sort descending, keep top k

Since eigenvalues are ascending, you need to reorder. Use `sort.Slice`:

```go
order := make([]int, n)
for i := range order { order[i] = i }
sort.Slice(order, func(i, j int) bool {
    return vals[order[i]] > vals[order[j]]
})
// order[0] = index of largest eigenvalue
```

Then take first k indices, clamp negatives to 0, and collect eigenvectors.

### Step 4: Form coordinates

```
math:   Z_k = V_k · √Λ_k
code:   for each dimension j:
          scale = sqrt(eigenvalues[j])
          column j of Z = scale × eigenvector_j   (as column vector)
```

Each row of Z is one object's k-dimensional embedding.

## Gonum Quick Reference

| Operation | Code |
|-----------|------|
| Create n×n matrix | `mat.NewDense(n, n, nil)` |
| Get element | `m.At(row, col)` |
| Set element | `m.Set(row, col, val)` |
| Dimensions | `rows, cols := m.Dims()` |
| Symmetric wrapper | `sym := mat.NewSymDense(n, nil)` + `sym.SetSym(i, j, v)` |
| Eigendecompose symmetric | `eig.Factorize(sym, true)` |
| Get eigenvalues | `eig.Values(nil)` → `[]float64` (ascending!) |
| Get eigenvectors | `eig.VectorsTo(&dst)` → columns of dst |
| Column view | `col := mat.Col(nil, j, A)` → `[]float64` |
| Set column | `m.SetCol(j, col)` |
| Multiply | `C.Mul(A, B)` |

## Implementation Order

Fill in the TODOs in this order:

### 1. `doubleCenter()` (6) — easiest to verify
Compute row means, column means, grand mean, then fill B.
**Test:** B should be symmetric. Row sums of B should be ≈ 0 (since
B = Z Zᵀ with centered Z, each row of Z sums to 0).

### 2. `ComputeSquaredDistances()` (5) — straightforward
Triple loop. Verify: D should be symmetric, diagonal = 0, D_ij ≥ 0.
Test with the provided `test_data.csv`.

### 3. `Fit()` Step 2 (eigendecomposition) — follow the pattern
Use `mat.EigenSym`. Verify: eigenvalues should be ≥ 0 for Euclidean
distances. Print `vals[n-1]` (largest) — should be clearly > 0.

### 4. `Fit()` Step 3 (sort + select) — careful with alignment
Sort descending, pick top k, clamp negatives to 0.
**Verify:** `m.eigenvalues` should be descending and non-negative.

### 5. `Fit()` Step 4 (form coordinates) — final output
Scale eigenvectors by sqrt of eigenvalues.
**Verify:** `m.coords` should be n×k. Row means should be ≈ 0.

### 6. `main.go` — already written, just run

## Testing Your Implementation

### Quick test with provided data

```bash
cd dimension-reduction/MDS
go run main.go test_data.csv 2
```

Expected results for `test_data.csv` (10 2D points, k=2):

- At least one eigenvalue should be clearly positive (the grid is not collinear)
- With k=2, the pairwise distances between recovered points should exactly
  match the original pairwise distances
- The recovered coordinates should form the same shape as the original
  points (up to rotation/reflection — MDS can't recover the absolute
  orientation)

### Manual verification

1. **Double-center check:** All row sums of B should be ~0
2. **Eigenvalue non-negativity:** For Euclidean distances, all λ ≥ 0
3. **Distance preservation:** For k=2 on 2D data, original vs recovered
   pairwise distances should match exactly (within floating-point error)
4. **Zero-centered embedding:** Column means of Zₖ should be ~0
5. **k < true_dim:** With k=1 on the grid data, you'll lose structure

### Test with PCA test data

```bash
go run main.go ../PCA/test_data.csv 2
```

MDS on the Iris-like test data should produce a reasonable 2D embedding.

## Common Pitfalls

- **Using ascending eigenvalues directly.** `eig.Values()` returns ascending!
  Always reorder before taking the top k.

- **Forgetting to clamp negatives.** Floating-point error or non-Euclidean
  distances can produce tiny negative eigenvalues (like −1e⁻¹⁴). Clamp to 0.

- **Eigenvector/eigenvalue misalignment.** When you sort eigenvalues, make
  sure you track their original indices so eigenvectors stay aligned.

- **Integer division.** `sum / n` with ints truncates. Use `sum / float64(n)`.

- **D not symmetric.** Ensure D_ij = D_ji. The distance matrix must be
  symmetric. `ComputeSquaredDistances` naturally produces symmetric output.

- **D not squared.** The input to Fit() must be **squared** dissimilarities,
  not raw distances. The name `ComputeSquaredDistances` is intentional.

## What NOT to implement

- ❌ Eigendecomposition — use `mat.EigenSym`
- ❌ Matrix multiplication — use `mat.Dense.Mul`
- ❌ Vector/matrix containers — use gonum's `mat.Dense`
- ❌ CSV parsing — `LoadCSV` is fully provided
- ❌ The Lagrangian optimization — derivation.pdf proves the result
