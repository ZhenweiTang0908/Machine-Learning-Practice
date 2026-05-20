package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/ZhenweiTang0908/Machine-Learning-Practice/dimension-reduction/MDS/src"
	"gonum.org/v1/gonum/mat"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <csv_path> [k]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  csv_path: path to CSV file (first row = header)\n")
		fmt.Fprintf(os.Stderr, "    - If square (n×n): treated as dissimilarity matrix\n")
		fmt.Fprintf(os.Stderr, "    - If rectangular (n×d): treated as feature matrix, distances computed\n")
		fmt.Fprintf(os.Stderr, "  k:        target embedding dimension (default: 2)\n")
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

	data, err := src.LoadCSV(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading data: %v\n", err)
		os.Exit(1)
	}
	n, d := data.Dims()
	fmt.Printf("Loaded data: %s (%d×%d)\n", path, n, d)

	mds := src.NewMDS(k)

	if n == d {
		fmt.Println("Detected square matrix → treating as dissimilarity matrix")
		err = mds.Fit(data)
	} else {
		fmt.Println("Detected rectangular matrix → computing distances from features")
		err = mds.FitFromFeatures(data)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error fitting MDS: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nEigenvalues (top %d):\n", k)
	for i, v := range mds.Eigenvalues() {
		fmt.Printf("  Dim %d: %.4f\n", i+1, v)
	}

	coords := mds.GetCoords()
	fmt.Printf("\nEmbedding coordinates (%d objects × %d dims) — first 10 objects:\n", n, k)
	limit := n
	if limit > 10 {
		limit = 10
	}
	for i := 0; i < limit; i++ {
		fmt.Printf("  obj %2d:", i)
		for j := 0; j < k; j++ {
			fmt.Printf(" %10.4f", coords.At(i, j))
		}
		fmt.Println()
	}
	if n > 10 {
		fmt.Printf("  ... (%d more objects)\n", n-10)
	}

	// Verify pairwise distances are preserved
	fmt.Printf("\nPairwise distance check (first 5 objects):\n")
	// Get original squared distances (from data or computed from features)
	var origD *mat.Dense
	if n == d {
		origD = data // already a dissimilarity matrix
	} else {
		origD = src.ComputeSquaredDistances(data) // compute from features
	}
	limit = 5
	if n < limit {
		limit = n
	}
	for i := 0; i < limit; i++ {
		for j := i + 1; j < limit; j++ {
			var sumSq float64
			for l := 0; l < k; l++ {
				diff := coords.At(i, l) - coords.At(j, l)
				sumSq += diff * diff
			}
			recovered := fmt.Sprintf("%.4f", sumSq)
			original := fmt.Sprintf("%.4f", origD.At(i, j))
			fmt.Printf("  d(%d,%d) original=%s  recovered=%s\n", i, j, original, recovered)
		}
	}
}
