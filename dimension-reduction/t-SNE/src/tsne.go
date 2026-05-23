// t-distributed Stochastic Neighbor Embedding (t-SNE) implementation.
//
// Given n points x_i ∈ ℝᵖ, t-SNE finds a low-dimensional embedding y_i ∈ ℝᵈ
// (typically d=2) that preserves local neighborhood structure by matching
// probability distributions across the two spaces.
//
// Algorithm (see derivation.pdf):
//   1. Compute pairwise squared distances D_ij = ||x_i - x_j||²
//   2. High-dim similarities P: Gaussian kernel, per-point bandwidth σ_i
//      chosen via binary search to achieve target perplexity, symmetrize
//   3. Initialize embedding Y randomly (N(0, 1e-4))
//   4. Gradient descent with momentum + early exaggeration:
//      a. Compute low-dim similarities Q (Student-t kernel, 1 dof)
//      b. Gradient: ∂L/∂y_i = 4 Σ_j (p_ij - q_ij) w_ij (y_i - y_j)
//      c. Update Y with momentum
//      d. Early exaggeration (first ~250 iters): P ← 12×P
//   5. Return final Y

package src

import (
	"fmt"
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

// TSNE performs t-distributed Stochastic Neighbor Embedding.
type TSNE struct {
	nComponents   int
	perplexity    float64
	learningRate  float64
	maxIter       int
	earlyIter     int
	exaggeration  float64
	momentum      float64
	finalMomentum float64
	randomState   uint64

	embedding    *mat.Dense
	klDivergence []float64
}

type TSNEOption func(*TSNE)

func WithLearningRate(eta float64) TSNEOption {
	return func(t *TSNE) { t.learningRate = eta }
}

func WithMaxIter(n int) TSNEOption {
	return func(t *TSNE) { t.maxIter = n }
}

func WithEarlyIter(n int) TSNEOption {
	return func(t *TSNE) { t.earlyIter = n }
}

func WithExaggeration(factor float64) TSNEOption {
	return func(t *TSNE) { t.exaggeration = factor }
}

func WithMomentum(alpha float64) TSNEOption {
	return func(t *TSNE) { t.momentum = alpha }
}

func WithFinalMomentum(alpha float64) TSNEOption {
	return func(t *TSNE) { t.finalMomentum = alpha }
}

func WithRandomState(seed uint64) TSNEOption {
	return func(t *TSNE) { t.randomState = seed }
}

func NewTSNE(nComponents int, perplexity float64, opts ...TSNEOption) *TSNE {
	t := &TSNE{
		nComponents:   nComponents,
		perplexity:    perplexity,
		learningRate:  200,
		maxIter:       1000,
		earlyIter:     250,
		exaggeration:  12,
		momentum:      0.5,
		finalMomentum: 0.8,
		randomState:   42,
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

func (t *TSNE) Fit(X *mat.Dense) error {
	n, p := X.Dims()
	_ = p
	if n < 4 {
		return fmt.Errorf("t-SNE: need at least 4 points, got %d", n)
	}
	if t.perplexity >= float64(n) {
		return fmt.Errorf("t-SNE: perplexity (%.0f) must be < n (%d)", t.perplexity, n)
	}
	if t.nComponents < 1 {
		return fmt.Errorf("t-SNE: nComponents must be >= 1, got %d", t.nComponents)
	}

	D := computePairwiseDistances(X)

	P, err := computeHighDimSimilarities(D, t.perplexity)
	if err != nil {
		return fmt.Errorf("t-SNE: %w", err)
	}

	t.embedding = initEmbedding(n, t.nComponents, t.randomState)
	t.klDivergence = make([]float64, t.maxIter)

	Y := t.embedding
	PExagg := mat.NewDense(n, n, nil)
	PExagg.Scale(t.exaggeration, P)

	velocity := mat.NewDense(n, t.nComponents, nil)
	Pcur := PExagg
	alpha := t.momentum

	for iter := 0; iter < t.maxIter; iter++ {
		Q := computeLowDimSimilarities(Y)
		grad := computeGradient(Pcur, Q, Y)

		velocity.Scale(alpha, velocity)
		for i := 0; i < n; i++ {
			for j := 0; j < t.nComponents; j++ {
				v := velocity.At(i, j) - t.learningRate*grad.At(i, j)
				velocity.Set(i, j, v)
			}
		}
		Y.Add(Y, velocity)

		t.klDivergence[iter] = computeKLDivergence(Pcur, Q)

		if iter == t.earlyIter-1 {
			Pcur = P
			alpha = t.finalMomentum
		}
	}

	return nil
}

func (t *TSNE) GetEmbedding() *mat.Dense {
	return t.embedding
}

func (t *TSNE) GetKLDivergence() []float64 {
	return t.klDivergence
}

// ---------------------------------------------------------------------------
// Pairwise distances
// ---------------------------------------------------------------------------

func computePairwiseDistances(X *mat.Dense) *mat.Dense {
	n, d := X.Dims()
	D := mat.NewDense(n, n, nil)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			var sum float64
			for k := 0; k < d; k++ {
				diff := X.At(i, k) - X.At(j, k)
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

// ---------------------------------------------------------------------------
// High-dim similarities P
// ---------------------------------------------------------------------------

// computeHighDimSimilarities computes the symmetric joint probability P_ij.
//
// For each point i, uses binary search to find σ_i such that the conditional
// distribution P_{·|i} has the target perplexity, then symmetrizes:
//   P_ij = (p_{j|i} + p_{i|j}) / (2n)
func computeHighDimSimilarities(D *mat.Dense, perplexity float64) (*mat.Dense, error) {
	n, _ := D.Dims()
	P := mat.NewDense(n, n, nil)

	for i := 0; i < n; i++ {
		dists := mat.Row(nil, i, D)

		sigma, err := binarySearchSigma(dists, i, perplexity)
		if err != nil {
			return nil, err
		}

		row := make([]float64, n)
		var sum float64
		for j := 0; j < n; j++ {
			if j == i {
				continue
			}
			row[j] = math.Exp(-dists[j] / (2 * sigma * sigma))
			sum += row[j]
		}
		for j := 0; j < n; j++ {
			if j != i {
				row[j] /= sum
			}
		}
		for j := 0; j < n; j++ {
			P.Set(i, j, row[j])
		}
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			pij := (P.At(i, j) + P.At(j, i)) / (2 * float64(n))
			P.Set(i, j, pij)
			P.Set(j, i, pij)
		}
		P.Set(i, i, 0)
	}

	return P, nil
}

// ---------------------------------------------------------------------------
// Binary search for per-point bandwidth σ
// ---------------------------------------------------------------------------

// binarySearchSigma finds σ for one data point via binary search on entropy.
//
// Given distances from point srcIdx to all others, finds σ such that:
//   H(P_{·|srcIdx}) = log₂(perplexity)
// where H = −Σ_{j≠srcIdx} p_j · log₂(p_j) and
//   p_j = exp(−d_j / (2σ²)) / Σ_{k≠srcIdx} exp(−d_k / (2σ²))
func binarySearchSigma(dists []float64, srcIdx int, perplexity float64) (float64, error) {
	n := len(dists)
	targetH := math.Log2(perplexity)

	var sigmaHi float64
	for _, d := range dists {
		if d > sigmaHi {
			sigmaHi = d
		}
	}
	if sigmaHi < 1e-12 {
		sigmaHi = 1
	}

	sigmaLo := 1e-12
	sigma := (sigmaLo + sigmaHi) / 2

	for iter := 0; iter < 100; iter++ {
		sigma = (sigmaLo + sigmaHi) / 2

		row := make([]float64, n)
		var sum float64
		for j := 0; j < n; j++ {
			if j == srcIdx {
				continue
			}
			row[j] = math.Exp(-dists[j] / (2 * sigma * sigma))
			sum += row[j]
		}
		if sum == 0 {
			sigmaLo = sigma
			continue
		}
		for j := 0; j < n; j++ {
			if j != srcIdx {
				row[j] /= sum
			}
		}

		var H float64
		for j := 0; j < n; j++ {
			if row[j] > 1e-20 {
				H += row[j] * math.Log2(row[j])
			}
		}
		H = -H

		if H > targetH {
			sigmaHi = sigma
		} else {
			sigmaLo = sigma
		}

		if sigmaHi-sigmaLo < 1e-12 {
			break
		}
	}

	return sigma, nil
}

// ---------------------------------------------------------------------------
// Low-dim similarities Q
// ---------------------------------------------------------------------------

// computeLowDimSimilarities computes the symmetric joint probability Q_ij
// using the Student-t kernel with 1 degree of freedom:
//   q_ij = (1 + ||y_i − y_j||²)^(−1)  /  Z
//   Z = Σ_{k≠l} (1 + ||y_k − y_l||²)^(−1)
func computeLowDimSimilarities(Y *mat.Dense) *mat.Dense {
	n, d := Y.Dims()
	Q := mat.NewDense(n, n, nil)

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			var dy2 float64
			for k := 0; k < d; k++ {
				diff := Y.At(i, k) - Y.At(j, k)
				dy2 += diff * diff
			}
			w := 1.0 / (1.0 + dy2)
			Q.Set(i, j, w)
			Q.Set(j, i, w)
		}
	}

	var Z float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				Z += Q.At(i, j)
			}
		}
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				Q.Set(i, j, Q.At(i, j)/Z)
			}
		}
	}

	return Q
}

// ---------------------------------------------------------------------------
// Gradient
// ---------------------------------------------------------------------------

// computeGradient computes the KL divergence gradient ∂L/∂Y.
//
// ∂L/∂y_i = 4 · Σ_{j≠i} (p_ij − q_ij) · w_ij · (y_i − y_j)
// where w_ij = 1 / (1 + ||y_i − y_j||²)
func computeGradient(P, Q *mat.Dense, Y *mat.Dense) *mat.Dense {
	n, d := Y.Dims()
	dY := mat.NewDense(n, d, nil)

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			var dy2 float64
			for k := 0; k < d; k++ {
				diff := Y.At(i, k) - Y.At(j, k)
				dy2 += diff * diff
			}
			w := 1.0 / (1.0 + dy2)

			coeff := 4.0 * (P.At(i, j) - Q.At(i, j)) * w

			for k := 0; k < d; k++ {
				diff := Y.At(i, k) - Y.At(j, k)
				dY.Set(i, k, dY.At(i, k)+coeff*diff)
				dY.Set(j, k, dY.At(j, k)-coeff*diff)
			}
		}
	}

	return dY
}

// ---------------------------------------------------------------------------
// KL divergence
// ---------------------------------------------------------------------------

func computeKLDivergence(P, Q *mat.Dense) float64 {
	n, _ := P.Dims()
	var kl float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			p := P.At(i, j)
			q := Q.At(i, j)
			if p > 0 && q > 0 {
				kl += p * math.Log(p/q)
			}
		}
	}
	return kl
}

// ---------------------------------------------------------------------------
// Random initialization
// ---------------------------------------------------------------------------

func initEmbedding(n, d int, seed uint64) *mat.Dense {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	Y := mat.NewDense(n, d, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			Y.Set(i, j, rng.NormFloat64()*1e-4)
		}
	}
	return Y
}
