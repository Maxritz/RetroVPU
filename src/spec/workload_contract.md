# VPU Workload Contract Schema (R3 Compliance)

## Purpose
Every benchmark variant must have a machine-readable workload contract before timing begins.

## Schema

```json
{
  "contract_version": "1.0",
  "workload": {
    "op": "string identifying operation",
    "description": "human-readable description"
  },
  "shape": {
    "M": <batch_size>,
    "N": <hidden_dimension>,
    "K": <intermediate_dimension>,
    "seq_len": <sequence_length>,
    "num_heads": <head_count>,
    "head_dim": <dimension per head>
  },
  "dtypes": {
    "compute": ["fp32", "fp16", "bf16", "int8"],
    "weight": ["fp32", "fp16", "bf16", "int8"],
    "activation": ["fp32", "fp16", "bf16", "int8"]
  },
  "layout": {
    "token": "row-major",
    "weight": "block-raster"
  },
  "dispatch": {
    "count": <number of kernel dispatches>,
    "workgroup_shape": [x, y, z]
  },
  "tolerance": {
    "max_error": <float>,
    "rms_error": <float>
  },
  "io": {
    "inputs": [{"name": "string", "bytes": int, "shape": [...], "dtype": "string"}],
    "outputs": [{"name": "string", "bytes": int, "shape": [...], "dtype": "string"}]
  },
  "source": {
    "paper_citation": "arXiv:XXXXX.XXXXX, section X",
    "spec_entry": "spec/<filename>#anchor"
  },
  "hash": {
    "spirv": "sha256_hex_of_spirv_binary",
    "source": "sha256_hex_of_source_files"
  },
  "device": {
    "gpu": "Intel Arc iGPU (128 EU)",
    "cpu": "Intel Core Ultra 5 135H",
    "memory_mb": 16384
  }
}
```

## Rules
1. Contract must be created BEFORE any timing (R3)
2. Baseline and V12 variants must share the same contract fields except where varying the test axis (R2)
3. Hashes must be computed and recorded in every result (R8)