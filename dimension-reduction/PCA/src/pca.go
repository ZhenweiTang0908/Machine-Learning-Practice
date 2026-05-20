package src

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type PCA struct {
	k           int
	mean        []float64
	components  *mat.Dense
	eigenvalues []float64
}

func NewPCA(k int) *PCA {
	return &PCA{k: k}
}

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

	p.mean = make([]float64, d)
	for j := 0; j < d; j++ {
		col := mat.Col(nil, j, X)
		var sum float64
		for _, v := range col {
			sum += v
		}
		p.mean[j] = sum / float64(n)
	}

	Z := mat.NewDense(n, d, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			Z.Set(i, j, X.At(i, j)-p.mean[j])
		}
	}

	var S mat.Dense
	S.Mul(Z.T(), Z)

	var eig mat.EigenSym
	sym := mat.NewSymDense(d, nil)
	for i := 0; i < d; i++ {
		for j := i; j < d; j++ {
			sym.SetSym(i, j, S.At(i, j))
		}
	}
	ok := eig.Factorize(sym, true)
	if !ok {
		return fmt.Errorf("PCA.Fit: eigendecomposition failed")
	}
	vals := eig.Values(nil)
	eigVecs := mat.NewDense(d, d, nil)
	eig.VectorsTo(eigVecs)

	order := sortEigenDescending(vals)
	p.eigenvalues = make([]float64, p.k)
	p.components = mat.NewDense(d, p.k, nil)
	for i := 0; i < p.k; i++ {
		p.eigenvalues[i] = vals[order[i]]
		col := mat.Col(nil, order[i], eigVecs)
		p.components.SetCol(i, col)
	}

	return nil
}

func (p *PCA) Transform(X *mat.Dense) (*mat.Dense, error) {
	m, d := X.Dims()
	if p.components == nil {
		return nil, fmt.Errorf("PCA.Transform: model not fitted")
	}
	if d != len(p.mean) {
		return nil, fmt.Errorf("PCA.Transform: X has %d features, fitted with %d", d, len(p.mean))
	}

	Z := mat.NewDense(m, d, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < d; j++ {
			Z.Set(i, j, X.At(i, j)-p.mean[j])
		}
	}

	var Y mat.Dense
	Y.Mul(Z, p.components)

	return &Y, nil
}

func (p *PCA) FitTransform(X *mat.Dense) (*mat.Dense, error) {
	if err := p.Fit(X); err != nil {
		return nil, err
	}
	return p.Transform(X)
}

func (p *PCA) InverseTransform(Y *mat.Dense) (*mat.Dense, error) {
	if p.components == nil {
		return nil, fmt.Errorf("PCA.InverseTransform: model not fitted")
	}
	m, kY := Y.Dims()
	if kY != p.k {
		return nil, fmt.Errorf("PCA.InverseTransform: Y has %d components, fitted with k=%d", kY, p.k)
	}

	var R mat.Dense
	R.Mul(Y, p.components.T())

	d := len(p.mean)
	Xhat := mat.NewDense(m, d, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < d; j++ {
			Xhat.Set(i, j, R.At(i, j)+p.mean[j])
		}
	}

	return Xhat, nil
}

func (p *PCA) Components() *mat.Dense {
	return p.components
}

func (p *PCA) Eigenvalues() []float64 {
	return p.eigenvalues
}

func (p *PCA) ExplainedVariance(n int) []float64 {
	if n < 2 {
		return nil
	}
	var out []float64
	for _, ev := range p.eigenvalues {
		out = append(out, ev/float64(n-1))
	}
	return out
}

func (p *PCA) ExplainedVarianceRatio(n int) []float64 {
	total := 0.0
	for _, ev := range p.eigenvalues {
		total += ev
	}
	if total == 0 {
		return nil
	}
	ratio := make([]float64, len(p.eigenvalues))
	for i, ev := range p.eigenvalues {
		ratio[i] = ev / total
	}
	return ratio
}

func reverseFloat64s(v []float64) {
	for i, j := 0, len(v)-1; i < j; i, j = i+1, j-1 {
		v[i], v[j] = v[j], v[i]
	}
}

type eigenPair struct {
	val float64
	idx int
}

func sortEigenDescending(vals []float64) []int {
	pairs := make([]eigenPair, len(vals))
	for i, v := range vals {
		pairs[i] = eigenPair{val: v, idx: i}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].val > pairs[j].val
	})
	order := make([]int, len(vals))
	for i, p := range pairs {
		order[i] = p.idx
	}
	return order
}
