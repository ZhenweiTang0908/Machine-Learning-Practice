package src

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// LoadCSV reads a CSV file and returns an n×p matrix (rows = samples, cols = features).
// The first line is treated as a header and is skipped.
func LoadCSV(path string) (*mat.Dense, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("LoadCSV: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("LoadCSV: %w", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("LoadCSV: need at least 1 header row + 1 data row, got %d rows", len(records))
	}

	records = records[1:] // skip header
	n := len(records)
	if n == 0 {
		return nil, fmt.Errorf("LoadCSV: no data rows after header")
	}

	p := len(records[0])
	data := make([]float64, n*p)

	for i, row := range records {
		if len(row) != p {
			return nil, fmt.Errorf("LoadCSV: row %d has %d fields, expected %d", i+2, len(row), p)
		}
		for j, val := range row {
			fv, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("LoadCSV: row %d, col %d: %w", i+2, j+1, err)
			}
			data[i*p+j] = fv
		}
	}

	return mat.NewDense(n, p, data), nil
}
