# TenSys

TenSys estimates the latency, array utilization, and memory traffic of dense matrix multiplication on configurable systolic arrays. The C++20 simulator models weight-stationary and output-stationary execution. The Python suite uses the simulator to compare Transformer prefill and decode projections, memory buffering, and array scaling.

The [evaluation report](docs/evaluation.md) includes five figures, the full 138-case dataset, and reproduction details. The [model guide](docs/model.md) explains how cycles and bytes are counted.

The [HBM writeback study](docs/writeback.md) compares two scheduling policies across 36 fixed-tile cases and shows how to reproduce the results and inspect the DMA timeline.

## Build and test

Use Python 3.11, a C++20 compiler (Clang or GCC), and Make or another CMake-supported build tool. The following setup installs CMake into the project virtual environment. The first configure downloads checksum-pinned copies of nlohmann/json and GoogleTest.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install cmake==3.31.6 -r requirements.txt
cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
TENSYS_BIN=build/release/tensys python -m pytest
```

The `debug` preset builds an unoptimized executable with debug information. Set `-DBUILD_TESTING=OFF` at configure time to build just the simulator and core. The GitHub Actions workflow covers Linux and macOS.

## Simulate and verify

Run all example decode projections with automatic tiling.

```sh
build/release/tensys simulate --hardware examples/hardware.json \
  --workload examples/decode.json --dataflow os
```

Verify an INT8 GEMM whose edges and reductions split across tiles.

```sh
build/release/tensys verify --hardware examples/hardware-small.json \
  --workload examples/verify.json --dataflow ws --tile 5,7,6 --order nmk --seed 42
```

Pass `--trace schedule.ndjson` to `simulate` to stream per-step records. With `--tile Tm,Tn,Tk`, the requested tile reserves its full SRAM footprint and accesses only valid elements at the boundaries. `--order mnk|nmk` fixes traversal for either automatic or manual tiling. Manual tiling defaults to `mnk`. The verification limit is 128 per dimension per layer.

### Input JSON

[Baseline hardware](examples/hardware.json) uses unified SRAM. [Small hardware](examples/hardware-small.json) uses separate activation, weight, and accumulator buffers.

| Section | Fields |
| --- | --- |
| `array` | `rows`, `cols`, `clock_ghz`, `precision` (`INT8`, `BF16`, or `FP16`), and `macs_per_pe_per_cycle` |
| `sram` | `layout` (`unified` or `split`), `read_gbps`, `write_gbps`, and capacity in bytes. Use `capacity_bytes` for unified SRAM or `activation_bytes`, `weight_bytes`, and `accumulator_bytes` for split SRAM. |
| `hbm` | `bandwidth_gbps`, `latency_cycles`, and `buffering` (`single` or `double`) |

A workload contains a `layers` array. Each layer has a unique `name` and a positive `m`, `n`, and `k`. Optional `phase`, `batch`, and `sequence_length` describe the layer. Layer timing depends on its matrix shape and the hardware configuration.

### Output JSON

The CLI writes one `schema_version: 1` JSON document to stdout. It includes normalized hardware values, each layer’s chosen `mapping` and `metrics`, and aggregate `totals`. Verification adds pass status and error details. Exit codes are 0 on success, 2 on invalid input, and 1 on verification or runtime failure. Diagnostics go to stderr.

## Reproduce the evaluation

```sh
PYTHONPATH=python python -m tensys_dse --sim build/release/tensys \
  --suite all --out reports/latest
```

The command writes `raw.json`, `metrics.csv`, `manifest.json`, and five PNG figures. Use `--suite smoke` for a small run through all four experiments, or select `regimes`, `dataflows`, `buffering`, or `scaling` individually. The manifest records the exact cases, program and source fingerprints, and host runtimes.

Use `--suite writeback --out reports/writeback` for the separate matched comparison of blocking and overlapped output writes.
