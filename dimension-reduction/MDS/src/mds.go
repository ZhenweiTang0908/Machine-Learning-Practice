// Classical Multidimensional Scaling (cMDS) implementation.
//
// Given n objects with pairwise squared dissimilarities d_ij²,
// cMDS finds a k-dimensional embedding preserving the distances.
//
// Algorithm (see derivation.pdf):
//   1. Double-center D → Gram matrix B = -½ H D H
//   2. Eigendecompose B → B = V Λ Vᵀ
//   3. Keep top k eigenvalues Λ_k and eigenvectors V_k
//   4. Form coordinates Z_k = V_k · √Λ_k
//
// Uses gonum for matrix operations and eigendecomposition.

package src

import (
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// MDS performs Classical Multidimensional Scaling.
type MDS struct {
	k           int         // target embedding dimension
	eigenvalues []float64   // top k eigenvalues (descending)
	coords      *mat.Dense  // n×k coordinate matrix (rows = objects)
}

// NewMDS creates an MDS model targeting k dimensions.
func NewMDS(k int) *MDS {
	return &MDS{k: k}
}

// Fit runs cMDS on a pre-computed squared dissimilarity matrix D (n×n).
// D must be symmetric with zeros on the diagonal.
func (m *MDS) Fit(D *mat.Dense) error {
	n, cols := D.Dims()
	if n != cols {
		return fmt.Errorf("MDS.Fit: D must be square, got %d×%d", n, cols)
	}
	if n < 2 {
		return fmt.Errorf("MDS.Fit: need at least 2 objects, got %d", n)
	}
	if m.k > n {
		return fmt.Errorf("MDS.Fit: k=%d > n=%d", m.k, n)
	}
	if m.k < 1 {
		return fmt.Errorf("MDS.Fit: k must be >= 1, got %d", m.k)
	}

	// Step 1: Double-center D → Gram matrix B
	// B_ij = -0.5 * (D_ij - rowMean_i - colMean_j + grandMean)
	B := doubleCenter(D)

	// Step 2: Eigendecompose B (symmetric → mat.EigenSym)
	// NOTE: eig.Values() returns eigenvalues in ASCENDING order!

	var eig mat.EigenSym
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			sym.SetSym(i, j, B.At(i, j))
		}
	}
	ok := eig.Factorize(sym, true)
	if !ok {
		return fmt.Errorf("MDS.Fit: eigendecomposition failed")
	}
	vals := eig.Values(nil)
	vecs := mat.NewDense(n, n, nil)
	eig.VectorsTo(vecs)

	// Step 3: Sort eigenvalues descending, keep top k, clamp negatives

	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		return vals[order[i]] > vals[order[j]]
	})

	m.eigenvalues = make([]float64, m.k)
	retained := mat.NewDense(n, m.k, nil)
	for i := 0; i < m.k; i++ {
		idx := order[i]
		m.eigenvalues[i] = vals[idx]
		if m.eigenvalues[i] < 0 {
			m.eigenvalues[i] = 0
		}
		col := mat.Col(nil, idx, vecs)
		retained.SetCol(i, col)
	}

	// Step 4: Form coordinates Z_k = V_k · sqrt(Λ_k)

	m.coords = mat.NewDense(n, m.k, nil)
	for j := 0; j < m.k; j++ {
		scale := math.Sqrt(m.eigenvalues[j])
		v := mat.Col(nil, j, retained)
		for i := 0; i < n; i++ {
			m.coords.Set(i, j, v[i]*scale)
		}
	}

	return nil
}

// FitFromFeatures computes squared Euclidean distances from feature
// matrix X (n×d) and then runs Fit.
func (m *MDS) FitFromFeatures(X *mat.Dense) error {
	D := ComputeSquaredDistances(X)
	return m.Fit(D)
}

// GetCoords returns the n×k coordinate matrix.
// Each row is one object's k-dimensional embedding.
func (m *MDS) GetCoords() *mat.Dense {
	return m.coords
}

// Eigenvalues returns the top k eigenvalues in descending order.
func (m *MDS) Eigenvalues() []float64 {
	return m.eigenvalues
}

// ComputeSquaredDistances computes the n×n matrix of squared Euclidean
// distances from an n×d feature matrix X.
// D_ij = ||x_i - x_j||² = Σₗ (x_il - x_jl)²
func ComputeSquaredDistances(X *mat.Dense) *mat.Dense {
	n, d := X.Dims()
	D := mat.NewDense(n, n, nil)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			var sum float64
			for l := 0; l < d; l++ {
				diff := X.At(i, l) - X.At(j, l)
				sum += diff * diff
			}
			D.Set(i, j, sum)
			if i != j {
				D.Set(j, i, sum)
			}
		}
	}

	return D
}

// doubleCenter applies double-centering to squared distance matrix D (n×n).
// Returns B where B_ij = -0.5 * (D_ij - rowMean_i - colMean_j + grandMean).
// This is the element-wise equivalent of B = -½ H D H.
func doubleCenter(D *mat.Dense) *mat.Dense {
	n, _ := D.Dims()

	rowMean := make([]float64, n)
	for i := 0; i < n; i++ {
		var sum float64
		for j := 0; j < n; j++ {
			sum += D.At(i, j)
		}
		rowMean[i] = sum / float64(n)
	}

	colMean := make([]float64, n)
	for j := 0; j < n; j++ {
		var sum float64
		for i := 0; i < n; i++ {
			sum += D.At(i, j)
		}
		colMean[j] = sum / float64(n)
	}

	var grandMean float64
	for _, v := range rowMean {
		grandMean += v
	}
	grandMean /= float64(n)

	B := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := D.At(i, j) - rowMean[i] - colMean[j] + grandMean
			B.Set(i, j, -0.5*v)
		}
	}

	return B
}
