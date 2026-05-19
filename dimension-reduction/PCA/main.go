package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/ZhenweiTang0908/Machine-Learning-Practice/dimension-reduction/PCA/src"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <csv_path> [k]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  csv_path: path to CSV file (first row = header)\n")
		fmt.Fprintf(os.Stderr, "  k:        number of principal components (default: 2)\n")
		os.Exit(1)
	}
	path := os.Args[1]

	k := 2
	if len(os.Args) >= 3 {
		var err error
		k, err = strconv.Atoi(os.Args[2])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid k: %v\n", err)
			os.Exit(1)
		}
		if k < 1 {
			fmt.Fprintf(os.Stderr, "k must be >= 1\n")
			os.Exit(1)
		}
	}

	// 1. Load data
	X, err := src.LoadCSV(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading data: %v\n", err)
		os.Exit(1)
	}
	n, p := X.Dims()
	fmt.Printf("Loaded data: %d samples × %d features\n", n, p)

	// 2. Fit PCA
	pca := src.NewPCA(k)
	if err := pca.Fit(X); err != nil {
		fmt.Fprintf(os.Stderr, "Error fitting PCA: %v\n", err)
		os.Exit(1)
	}

	// 3. Transform (project) the data
	Y, err := pca.Transform(X)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error transforming data: %v\n", err)
		os.Exit(1)
	}
	_, kOut := Y.Dims()

	// 4. Print results

	// --- Eigenvalues ---
	fmt.Printf("\nEigenvalues (top %d):\n", k)
	evals := pca.Eigenvalues()
	for i, v := range evals {
		fmt.Printf("  PC%d: %.4f\n", i+1, v)
	}

	// --- Explained variance ---
	fmt.Printf("\nExplained variance:\n")
	for i, v := range pca.ExplainedVariance(n) {
		fmt.Printf("  PC%d: %.4f\n", i+1, v)
	}

	// --- Explained variance ratio ---
	fmt.Printf("\nExplained variance ratio:\n")
	cumulative := 0.0
	for i, r := range pca.ExplainedVarianceRatio(n) {
		cumulative += r
		fmt.Printf("  PC%d: %.4f  (cumulative: %.4f)\n", i+1, r, cumulative)
	}
	fmt.Printf("  Total retained: %.4f\n", cumulative)

	// --- Projected data (first few rows) ---
	fmt.Printf("\nProjected data (%d samples × %d components) — first 10 rows:\n", n, kOut)
	limit := n
	if limit > 10 {
		limit = 10
	}
	for i := 0; i < limit; i++ {
		fmt.Printf("  row %d: ", i)
		for j := 0; j < kOut; j++ {
			fmt.Printf("%10.4f", Y.At(i, j))
		}
		fmt.Println()
	}
	if n > 10 {
		fmt.Printf("  ... (%d more rows)\n", n-10)
	}

	// 5. (Optional) Reconstruction check — uncomment to verify correctness
	//
	// Xhat, err := pca.InverseTransform(Y)
	// if err != nil {
	// 	fmt.Fprintf(os.Stderr, "Error in inverse transform: %v\n", err)
	// 	os.Exit(1)
	// }
	// // Compute mean squared reconstruction error
	// var diff mat.Dense
	// diff.Sub(X, Xhat)
	// mse := mat.Norm(&diff, 2) / float64(n)
	// // or: mse = mat.Norm(&diff, 2) * mat.Norm(&diff, 2) / float64(n*p)
	// fmt.Printf("\nMean reconstruction error (sqrt): %.6f\n", mse)
}
