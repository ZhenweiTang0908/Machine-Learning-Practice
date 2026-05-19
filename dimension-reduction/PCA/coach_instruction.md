# PCA Implementation Guide

## Overview

You will implement PCA from scratch by filling in the `TODO` blocks in `src/pca.go` and `main.go`. The `src/data.go` file is already complete (it only reads CSV files — no ML logic).

## Algorithm Summary (from derivation.pdf)

| Step | Math | Code location |
|------|------|---------------|
| 1. Compute mean | m = (1/n) Σ xᵢ | `Fit()` Step 1 |
| 2. Center data | Zᵢⱼ = Xᵢⱼ − mⱼ | `Fit()` Step 2 |
| 3. Scatter matrix | S = ZᵀZ | `Fit()` Step 3 |
| 4. Eigendecomposition | Sv = λv | `Fit()` Step 4 |
| 5. Sort & pick top k | λ₁ ≥ λ₂ ≥ ... ≥ λₖ | `Fit()` Step 5 |
| 6. Project | Y = Z × Uₖ | `Transform()` |
| 7. Reconstruct (optional) | X̂ = Y × Uₖᵀ + m | `InverseTransform()` |

## Math → Code Mapping

### Step 1: Column means

```
math:  mⱼ = (1/n) Σᵢ Xᵢⱼ
code:  for each column j:
         col := mat.Col(nil, j, X)   // returns []float64 view of column j
         sum := 0
         for _, v := range col { sum += v }
         mean[j] = sum / float64(n)
```

### Step 2: Center the data

```
math:  Zᵢⱼ = Xᵢⱼ − mⱼ
code:  Z := mat.NewDense(n, d, nil)
       for i := 0; i < n; i++ {
           for j := 0; j < d; j++ {
               Z.Set(i, j, X.At(i, j) - mean[j])
           }
       }
```

### Step 3: Scatter matrix

```
math:  S = ZᵀZ        (d×n × n×d = d×d)
code:  var S mat.Dense
       S.Mul(Z.T(), Z)
```

`mat.Dense.Mul(a, b mat.Matrix)` computes `this = a × b`. `Z.T()` returns a `mat.Matrix` transpose view (no copy).

**Why ZᵀZ, not ZZᵀ?** Because we store data row-wise (n×d). ZᵀZ is d×d — the same scatter matrix as in the derivation. Both formulations have identical non-zero eigenvalues.

### Step 4: Eigendecomposition

```go
var eig mat.EigenSym
ok := eig.Factorize(&S, mat.EigenBoth)  // true = also compute eigenvectors
if !ok {
    return fmt.Errorf("eigendecomposition failed")
}

vals := eig.Values(nil)       // []float64, ASCENDING order (smallest → largest)

var eigVecs mat.Dense
eig.VectorsTo(&eigVecs)       // eigenvectors as columns (aligned with vals)
```

**Important:** `mat.EigenSym` requires a **symmetric** matrix. Our scatter matrix S = ZᵀZ is always symmetric, so this is the right tool. Use `mat.EigenBoth` to compute both eigenvalues and eigenvectors.

**Eigenvalue ordering:** `eig.Values(nil)` returns eigenvalues in **ascending** order. So `vals[0]` is the smallest, `vals[p-1]` is the largest. You need to reverse or sort them for PCA.

### Step 5: Sort descending, pick top k

Two approaches:

**Approach A: Use `sortEigenDescending`** (already provided in `src/pca.go`):

```go
order := sortEigenDescending(vals)
// order[0] = index of largest eigenvalue
// order[1] = index of second largest, etc.

p.eigenvalues = make([]float64, k)
p.components = mat.NewDense(d, k, nil)
for i := 0; i < k; i++ {
    p.eigenvalues[i] = vals[order[i]]           // eigenvalue
    col := mat.Col(nil, order[i], &eigVecs)      // corresponding eigenvector
    p.components.SetCol(i, col)                  // store as column i
}
```

The `order` array tells you which original column index corresponds to which rank. This is safer than reversing because the eigenvalue/vector alignment is preserved explicitly.

**Approach B: Reverse (simpler but less explicit):**

Since eigenvalues are ascending, reverse them, then take first k. The last k columns of `eigVecs` are the corresponding eigenvectors.

```go
reverseFloat64s(vals)                            // now descending
p.eigenvalues = vals[:k]                         // top k
p.components = mat.NewDense(d, k, nil)
for i := 0; i < k; i++ {
    col := mat.Col(nil, d-1-i, &eigVecs)        // last column = largest eigenvalue
    p.components.SetCol(i, col)
}
```

### Step 6: Transform (project)

```
math:  Y = Z × Uₖ      (n×d × d×k = n×k)
code:  1. Center X by subtracting mean
       2. var Y mat.Dense; Y.Mul(Z, p.components)
```

### Step 7: InverseTransform (reconstruct)

```
math:  X̂ = Y × Uₖᵀ + mean
code:  1. var R mat.Dense; R.Mul(Y, p.components.T())
       2. Add mean back manually in a loop
```

**Why manual loop?** `mat.Dense` doesn't have a built-in "add vector to each row". You need:

```go
for i := 0; i < m; i++ {
    for j := 0; j < d; j++ {
        Xhat.Set(i, j, R.At(i, j) + mean[j])
    }
}
```

### ExplainedVariance & ExplainedVarianceRatio

```
math:  variance[i] = λᵢ / (n−1)
       ratio[i]    = λᵢ / Σλⱼ
code:  for each eigenvalue:
         variance[i] = ev / float64(n-1)
         ratio[i]    = ev / total
```

Since ratio uses eigenvalues directly (the `n−1` cancels out), you can compute it from `p.eigenvalues` without needing `n`.

## Gonum Quick Reference

| Operation | Code |
|-----------|------|
| Create n×p matrix | `mat.NewDense(n, p, data)` (data can be nil) |
| Get element | `m.At(row, col)` |
| Set element | `m.Set(row, col, val)` |
| Dimensions | `rows, cols := m.Dims()` |
| Transpose | `m.T()` returns `mat.Matrix` |
| Multiply | `C.Mul(A, B)` (C = A × B) |
| Column view | `col := mat.Col(nil, j, A)` returns `[]float64` |
| Set column | `m.SetCol(j, col)` |
| Row view | `row := mat.Row(nil, i, A)` |
| Subtract matrices | `C.Sub(A, B)` |
| Frobenius norm | `mat.Norm(A, 2)` |

## Implementation Order

Follow this order to build up incrementally:

1. **`src/pca.go` — Step 1 & 2:** Mean + centering. Verify by printing a few rows of Z; the column means of Z should be ~0.

2. **`src/pca.go` — Step 3:** Scatter matrix. Print the matrix — it should be symmetric and d×d.

3. **`src/pca.go` — Step 4:** Eigendecomposition. Print eigenvalues (should be non-negative). Print first eigenvector.

4. **`src/pca.go` — Step 5:** Sorting & selection. Print `p.eigenvalues` (should be descending). Print `p.components` dimensions (should be d×k).

5. **`src/pca.go` — Transform + InverseTransform:** Implement projection and reconstruction.

6. **`src/pca.go` — ExplainedVariance + ExplainedVarianceRatio:** Implement variance metrics.

7. **`main.go` — TODO blocks:** Parse k from CLI, print results, optionally compute reconstruction error.

## Testing Your Implementation

### Create a small test CSV

```csv
x,y
1,2
2,3
3,4
4,5
5,6
```

This is perfectly collinear data. PCA with k=1 should capture ~100% variance, and PC1 should be ~[0.707, 0.707] (the direction of [1,1]).

### Run

```bash
# Install gonum first
cd dimension-reduction/PCA
go get gonum.org/v1/gonum

# Run on test data
go run main.go test.csv 1
```

### Expected output for the collinear test data

- First eigenvalue >> second eigenvalue
- Explained variance ratio for PC1 ≈ 1.0
- PC1 direction ≈ [1/√2, 1/√2] ≈ [0.7071, 0.7071]

### Verify correctness

1. **Column means of Z ≈ 0** (within floating-point error)
2. **Scatter matrix is symmetric:** `S.At(i,j) == S.At(j,i)`
3. **Eigenvalues are non-negative:** S is positive semi-definite
4. **Explained variance ratios sum to 1.0** (within retained components)
5. **Reconstruction error decreases as k increases** (with k=d, error should be ~0)
6. **Components are orthonormal:** `Uₖᵀ Uₖ = Iₖ` (you can check with `mat.Mul`)

## Common Pitfalls

- **Forgetting to center before building the scatter matrix.** If you don't subtract the mean, the first PC will point toward the data centroid, not the direction of maximal variance.

- **Mixing up rows and columns.** The data matrix is n×p (rows=samples, columns=features). The scatter matrix S = ZᵀZ is p×p. Eigenvectors are p-dimensional.

- **Using wrong eigenvalue ordering.** `mat.EigenSym.Values()` returns ascending order. You need descending — the largest eigenvalue corresponds to the first principal component.

- **Eigenvector/eigenvalue misalignment.** Make sure the eigenvalue you pick and the eigenvector you pick come from the same original column index. Both approaches in the guide handle this correctly.

- **Integer division in Go.** `ev / (n-1)` with ints would truncate. Always cast: `ev / float64(n-1)`.

## What NOT to implement

- Do NOT implement your own eigendecomposition, matrix multiplication, or vector operations — use gonum.
- Do NOT implement the Lagrangian optimization — the derivation already proves that eigendecomposition solves it.
- The `LoadCSV` function in `src/data.go` is fully provided — no need to modify it.
