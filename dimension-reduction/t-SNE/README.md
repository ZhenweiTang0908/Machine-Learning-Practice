# t-SNE (t-distributed Stochastic Neighbor Embedding)

## Usage

```bash
go run main.go test_data.csv 5 2
```

- `test_data.csv` — feature matrix (first row = header, n rows × d cols)
- `5` — target perplexity (optional, default=30; must be < n)
- `2` — target embedding dimension (optional, default=2)

## Files

| File | Purpose |
|------|---------|
| `main.go` | CLI entry point |
| `src/data.go` | CSV loading (complete) |
| `src/tsne.go` | t-SNE implementation (TODOs to fill) |
| `derivation.pdf` | Mathematical derivation |
| `coach_instruction.md` | Step-by-step implementation guide |
| `test_data.csv` | 10 2D points for testing |
