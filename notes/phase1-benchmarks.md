# Phase 1 Benchmarks (M1 Max, fp16)

Measured at end of Task 1.12 with `mx.get_peak_memory()` (non-deprecated API).

| Metric | Value | Budget | Status |
|---|---|---|---|
| Decode latency 1024×1024 | ~100 ms | 200 ms | PASS |
| Peak memory 1024×1024 | 1021.1 MB | 1500 MB | PASS |
| Peak memory 512×512 | 321.2 MB | (informational) | — |

Comparison:
- Full Flux VAE memory at 1024×1024: ~9.6 GB (per mflux issue #407)
- TAEF2 (this library): 1021.1 MB
- Memory win: ~9.4x

Latency comparison vs PyTorch-MPS reference: TBD (manual run with PyTorch's TAESD class needed).
