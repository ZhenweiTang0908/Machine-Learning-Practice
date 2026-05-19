package src

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// PCA stores the learned principal components and metadata after fitting.
type PCA struct {
	k           int        // target dimension (number of principal components)
	mean        []float64  // feature means, length = p
	components  *mat.Dense // p×k matrix, each column is a principal component
	eigenvalues []float64  // top-k eigenvalues, sorted descending (largest first)
}

// NewPCA creates a PCA model that will reduce data to k dimensions.
func NewPCA(k int) *PCA {
	return &PCA{k: k}
}

// Fit learns the principal components from data X (n×p).
//
// Algorithm:
//   1. Compute column means
//   2. Center the data (Zᵢⱼ = Xᵢⱼ − meanⱼ)
//   3. Compute scatter matrix S = ZᵀZ  (p×p)
//   4. Eigendecomposition of S (symmetric, use mat.EigenSym)
//   5. Sort eigenvalues descending, pick top k eigenvectors
//   6. Store: p.mean, p.components, p.eigenvalues
func (p *PCA) Fit(X *mat.Dense) error {
	n, d := X.Dims()
	if n < 2 {
		return fmt.Errorf("PCA.Fit: need at least 2 samples, got %d", n)
	}
	if p.k > d {
		return fmt.Errorf("PCA.Fit: k=%d > features=%d", p.k, d)
	}
	if p.k < 1 {
		return fmt.Errorf("PCA.Fit: k must be >= 1, got %d", p.k)
	}

	// ---------------------------------------------------------------
	// Step 1: Compute column means
	// ---------------------------------------------------------------
	//
	// mat.Col(dst, j, X) returns a []float64 view of column j.
	// Loop j=0..d-1, sum the column, divide by n.
	//
	// TODO: p.mean = make([]float64, d)
	// TODO: for j := 0; j < d; j++ { ... }

	// ---------------------------------------------------------------
	// Step 2: Center the data → Z
	// ---------------------------------------------------------------
	//
	// Create Z = mat.NewDense(n, d, nil).
	// For each row i, column j: Z.Set(i, j, X.At(i, j) - p.mean[j])
	//
	// TODO: build centered matrix Z

	// ---------------------------------------------------------------
	// Step 3: Scatter matrix S = Zᵀ Z
	// ---------------------------------------------------------------
	//
	// Z is n×d, Zᵀ is d×n, product is d×d.
	// Use: var S mat.Dense; S.Mul(Z.T(), Z)
	//
	// (Z.T() returns a mat.Matrix, not *mat.Dense — it's fine for Mul.)
	//
	// TODO: var S mat.Dense; S.Mul(Z.T(), Z)

	// ---------------------------------------------------------------
	// Step 4: Eigendecomposition of S
	// ---------------------------------------------------------------
	//
	// S is symmetric and positive semi-definite → use mat.EigenSym.
	//
	//   var eig mat.EigenSym
	//   ok := eig.Factorize(&S, mat.EigenBoth)
	//   if !ok { ... }
	//
	// eig.Values(nil) → []float64, eigenvalues in ASCENDING order.
	// eig.VectorsTo(dst) → eigenvectors as columns of dst (d×d).
	//
	// TODO: factorize, extract values & vectors

	// ---------------------------------------------------------------
	// Step 5: Sort descending, pick top k
	// ---------------------------------------------------------------
	//
	// Approach A (simple — use the helper functions below):
	//   order := sortEigenDescending(vals)    // get indices sorted by value descending
	//   // order[0] = index of largest eigenvalue, order[1] = second, ...
	//   p.eigenvalues = make([]float64, k)
	//   p.components = mat.NewDense(d, k, nil)
	//   for i := 0; i < k; i++ {
	//       p.eigenvalues[i] = vals[order[i]]
	//       col := mat.Col(nil, order[i], eigVecs)
	//       p.components.SetCol(i, col)
	//   }
	//
	// Approach B (reversal):
	//   // eigenvalues are ascending — reverse the slice
	//   // reverseFloat64s(vals)
	//   // then take first k eigenvalues and last k eigenvector columns
	//
	// TODO: select top k eigenvalues & corresponding eigenvectors

	// ---------------------------------------------------------------
	// (Step 6 implicit: results stored in struct fields)
	// ---------------------------------------------------------------

	return nil
}

// Transform projects X (m×p) onto the learned k-dimensional subspace.
//
// Returns Y (m×k) where each row is the low-dimensional representation.
//
//   Y = (X − mean) × Uₖ
//
// Z is m×d (centered), Uₖ is d×k. So Y is m×k.
func (p *PCA) Transform(X *mat.Dense) (*mat.Dense, error) {
	m, d := X.Dims()
	_ = m // will be used when you fill the TODOs below
	if p.components == nil {
		return nil, fmt.Errorf("PCA.Transform: model not fitted")
	}
	if d != len(p.mean) {
		return nil, fmt.Errorf("PCA.Transform: X has %d features, fitted with %d", d, len(p.mean))
	}

	// --- Step 1: Center X ---
	//
	// TODO: Create Z = mat.NewDense(m, d, nil)
	// TODO: Z.Set(i, j, X.At(i, j) - p.mean[j])
	//       (m = number of samples, d = number of features)

	// --- Step 2: Project ---
	//
	// TODO: var Y mat.Dense; Y.Mul(Z, p.components)
	//       Z is m×d, components is d×k → Y is m×k

	return nil, nil // TODO
}

// FitTransform fits the model to X and returns the projection of X.
func (p *PCA) FitTransform(X *mat.Dense) (*mat.Dense, error) {
	if err := p.Fit(X); err != nil {
		return nil, err
	}
	return p.Transform(X)
}

// InverseTransform reconstructs data from the low-dimensional representation Y (m×k)
// back to the original p-dimensional space.
//
//   X̂ = Y × Uₖᵀ + mean
//
// Y is m×k, Uₖᵀ is k×d, so Y × Uₖᵀ is m×d.
func (p *PCA) InverseTransform(Y *mat.Dense) (*mat.Dense, error) {
	if p.components == nil {
		return nil, fmt.Errorf("PCA.InverseTransform: model not fitted")
	}
	m, kY := Y.Dims()
	_ = m // will be used when you fill the TODOs below
	if kY != p.k {
		return nil, fmt.Errorf("PCA.InverseTransform: Y has %d components, fitted with k=%d", kY, p.k)
	}

	// --- Step 1: Y × Uₖᵀ ---
	//
	// TODO: var R mat.Dense; R.Mul(Y, p.components.T())

	// --- Step 2: Add mean back ---
	//
	// TODO: For each row i and column j: Xhat.Set(i, j, R.At(i,j) + p.mean[j])

	return nil, nil // TODO
}

// Components returns the principal components (p×k), each column a PC.
func (p *PCA) Components() *mat.Dense {
	return p.components
}

// Eigenvalues returns the top-k eigenvalues (variances along each PC, up to scale).
func (p *PCA) Eigenvalues() []float64 {
	return p.eigenvalues
}

// ExplainedVariance returns the variance captured by each principal component.
//
// variance[i] = eigenvalue[i] / (n−1)
//
// n must be the number of samples used during Fit. Must be >= 2.
func (p *PCA) ExplainedVariance(n int) []float64 {
	if n < 2 {
		return nil
	}
	// TODO: var out []float64
	// TODO: for _, ev := range p.eigenvalues { out = append(out, ev/float64(n-1)) }
	return nil
}

// ExplainedVarianceRatio returns the fraction of total variance explained
// by each principal component (sums to 1.0 across retained components).
//
// ratio[i] = eigenvalue[i] / sum(all eigenvalues)
//
// Since the denominator cancels the 1/(n−1) factor, eigenvalues directly give
// the proportion. Sum of ratios = 1.0.
func (p *PCA) ExplainedVarianceRatio(n int) []float64 {
	// TODO: compute total = sum(p.eigenvalues)
	// TODO: ratio[i] = p.eigenvalues[i] / total
	return nil
}

// --- Helpers ---

// reverseFloat64s reverses a []float64 slice in place.
//
// Useful because mat.EigenSym.Values() returns eigenvalues in ascending order,
// but PCA needs them descending.
func reverseFloat64s(v []float64) {
	for i, j := 0, len(v)-1; i < j; i, j = i+1, j-1 {
		v[i], v[j] = v[j], v[i]
	}
}

// eigenPair pairs an eigenvalue with its original index (column position).
type eigenPair struct {
	val float64
	idx int
}

// sortEigenDescending returns the indices of eigenvalues sorted from largest to smallest.
//
// Example:
//
//	vals := eig.Values(nil)    // ascending: [0.5, 2.0, 5.0]
//	order := sortEigenDescending(vals)  // order = [2, 1, 0]
//	// order[0]=2 means vals[2]=5.0 is the largest
//	// order[1]=1 means vals[1]=2.0 is the second
//	// order[2]=0 means vals[0]=0.5 is the smallest
func sortEigenDescending(vals []float64) []int {
	pairs := make([]eigenPair, len(vals))
	for i, v := range vals {
		pairs[i] = eigenPair{val: v, idx: i}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].val > pairs[j].val // descending
	})
	order := make([]int, len(vals))
	for i, p := range pairs {
		order[i] = p.idx
	}
	return order
}
