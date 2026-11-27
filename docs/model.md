# TenSys model

TenSys simulates dense GEMM, C = A × B. A layer starts with its data in HBM and completes when the last output tile is written to HBM. Layers are independent. Timings describe GEMM projections under a synthetic accelerator configuration. The experimental scope covers projection GEMMs, with attention scoring, softmax, KV caches, and serving overhead outside the model.

## Tiling and storage

A logical tile step covers Tm output rows, Tn output columns, and Tk reduction values. A and B use 1 byte per INT8 element and 2 bytes per FP16 or BF16 element. The accumulator and final C use 4 bytes per element. Fit checks reserve Tm × Tk input elements, Tk × Tn weights, and Tm × Tn accumulators. Unified SRAM pools these allocations, while split SRAM checks each against its own capacity. Ping-pong mode reserves two banks of equal size, rounding each capacity down to a whole byte per bank.

Both traversal orders keep K innermost. A and B have independent residency checks. An input survives across consecutive steps when the (M, K) tile coordinates match, and a weight tile survives when (K, N) matches. Within a layer, partial sums move between the array and the accumulator allocation. Each complete C tile is written to HBM once. The arrays and memory transfers use clipped edge extents. Physical array size still determines advertised peak compute.

The auto-tiler searches the full dimension, powers of two, and powers-of-two multiples of the relevant mapping span. It minimizes predicted elapsed cycles with HBM traffic, streaming utilization, and lexicographic order as tie-breakers. Candidate evaluation and summary mode use multiplicities of equal full and edge tile steps, including their transitions. Trace mode enumerates those same steps.

## Array waves

The following formulas use R × C for the physical array, q for MACs per PE per cycle, and lowercase m, n, and k for the valid extents in one wave.

| Component | Output-stationary | Weight-stationary |
| --- | --- | --- |
| Array mapping | M on rows and N on columns | K on rows and N on columns |
| Streaming work | ceil(k / q) cycles | ceil(m / q) cycles |
| Fill and drain | m + n − 2 skew cycles and m drain cycles | R + C − 1 cycles |
| Weight loading | Streamed with activations | k cycles at one weight row per cycle |
| Local data | Each PE accumulates one output across the wave's K values | Each PE holds one weight across the wave's M values |

Output-stationary reads A and B for each wave and writes its output partials. It reads prior partials on later K tiles. Weight-stationary reads B in a separate load phase, then streams A and reads prior partials when resuming a K reduction. It writes the updated partials back after each wave. Increasing q models lanes across K in OS and across M in WS. Numerical verification maintains ascending K reduction order for each output.

## SRAM and HBM timing

Clock GHz and memory GB/s use decimal units. TenSys converts them to the nearest Hz and bytes per second. For an array phase, the duration is the maximum of the nominal cycles, ceil(SRAM read bytes × clock / read bandwidth), and ceil(SRAM write bytes × clock / write bandwidth). The difference from nominal phase time is the SRAM stall. The calculation averages bandwidth demand across the phase.

HBM provides a single DMA lane. Each nonempty fetch bundles its missing A and B panels, and each output write is a separate transaction. Duration is the configured setup latency plus the rounded-up transfer time at the slower of the HBM lane and the SRAM port. The DMA-facing SRAM ports are modeled independently of array-facing ports, with the same configured directional rates. The SRAM traffic counters include both sets of ports.

The first fetch is blocking. Single buffering finishes the array step, writes any completed output, then fetches the next step's operands. Double buffering starts the next fetch with the current array step, leaves any unmatched latency as exposed HBM wait, and writes a completed output only after the prefetch finishes. In `simulate`, completed-output writes are blocking in both modes. The `dma-compare` command also measures writeback through a second accumulator bank. [Writeback scheduler and evaluation](writeback.md)

## Reading the telemetry

- `total_cycles` is the sum of compute, fill/drain, weight load, HBM stall, and SRAM stall cycles. `dma_cycles` is service time, and `dma_hidden_cycles` plus `hbm_stall_cycles` equals that service time.
- `mac_utilization` divides useful MACs by the physical peak MACs per cycle times total cycles. `stream_utilization` uses compute cycles in the denominator and captures idle array lanes and streaming tails.
- Each NDJSON step gives zero-based tile indices, matrix origins, clipped shapes, and step metrics. Fetch traffic and exposed fetch wait are attributed to the preceding step, whose `next_input` identifies the destination. The first step includes the cold fetch. Output writes are attributed to the last reduction step of the completed tile.
- HBM and SRAM counters record bytes in both directions. The roofline uses total HBM bytes and two operations per useful MAC. The HBM side of the ridge is a theoretical bound. A low-intensity case can still spend more time loading weights or filling the array than waiting on HBM.

The dataflow mappings draw on the [SCALE-Sim paper](https://arxiv.org/abs/1811.02883), and the analysis uses the [original roofline formulation](https://escholarship.org/uc/item/5tz795vq).
