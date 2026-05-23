package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/ZhenweiTang0908/Machine-Learning-Practice/dimension-reduction/t-SNE/src"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <csv_path> [perplexity] [n_components]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  csv_path:     path to CSV file (first row = header, n rows × d cols)\n")
		fmt.Fprintf(os.Stderr, "  perplexity:   target perplexity (default: 30)\n")
		fmt.Fprintf(os.Stderr, "  n_components: embedding dimension (default: 2)\n")
		os.Exit(1)
	}

	path := os.Args[1]

	perplexity := 30.0
	nComponents := 2

	if len(os.Args) >= 3 {
		var err error
		perplexity, err = strconv.ParseFloat(os.Args[2], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid perplexity: %v\n", err)
			os.Exit(1)
		}
		if perplexity <= 0 {
			fmt.Fprintf(os.Stderr, "perplexity must be > 0\n")
			os.Exit(1)
		}
	}

	if len(os.Args) >= 4 {
		var err error
		nComponents, err = strconv.Atoi(os.Args[3])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid n_components: %v\n", err)
			os.Exit(1)
		}
		if nComponents < 1 {
			fmt.Fprintf(os.Stderr, "n_components must be >= 1\n")
			os.Exit(1)
		}
	}

	data, err := src.LoadCSV(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading data: %v\n", err)
		os.Exit(1)
	}
	n, d := data.Dims()
	fmt.Printf("Loaded data: %s (%d points × %d features)\n", path, n, d)

	tsne := src.NewTSNE(nComponents, perplexity)
	if err = tsne.Fit(data); err != nil {
		fmt.Fprintf(os.Stderr, "Error fitting t-SNE: %v\n", err)
		os.Exit(1)
	}

	embedding := tsne.GetEmbedding()
	fmt.Printf("\nFinal embedding (%d points × %d dims):\n", n, nComponents)
	limit := n
	if limit > 15 {
		limit = 15
	}
	for i := 0; i < limit; i++ {
		fmt.Printf("  pt %2d:", i)
		for j := 0; j < nComponents; j++ {
			fmt.Printf(" %10.4f", embedding.At(i, j))
		}
		fmt.Println()
	}
	if n > 15 {
		fmt.Printf("  ... (%d more points)\n", n-15)
	}

	kl := tsne.GetKLDivergence()
	if len(kl) > 0 {
		fmt.Printf("\nKL divergence: initial=%.4f  final=%.4f\n", kl[0], kl[len(kl)-1])
	}
}
