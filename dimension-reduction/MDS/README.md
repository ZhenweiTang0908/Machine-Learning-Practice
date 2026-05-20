# Classical Multidimensional Scaling (cMDS)

## Usage

```bash
go run main.go test_data.csv 2
```

- `test_data.csv` — feature matrix (first row = header, n rows × d cols)
- `2` — target embedding dimension (optional, default=2)

If the CSV is square (n×n), it is treated as a dissimilarity matrix.
Otherwise, pairwise squared Euclidean distances are computed from features.

## Files

| File | Purpose |
|------|---------|
| `main.go` | CLI entry point |
| `src/data.go` | CSV loading (complete) |
| `src/mds.go` | MDS implementation (TODOs to fill) |
| `derivation.pdf` | Mathematical derivation |
| `coach_instruction.md` | Step-by-step implementation guide |
| `test_data.csv` | 10 2D points for testing |
