#include "sim/tile_cost.hpp"

#include <numeric>

namespace tensys {

void DmaRequest::account(Metrics& result) const {
    if (direction == DmaDirection::input) {
        result.hbm.read.add(bytes, 1);
        result.sram.write.add(bytes, 1);
    } else {
        result.hbm.write.add(bytes, 1);
        result.sram.read.add(bytes, 1);
    }
    result.dma_cycles = checked_add(result.dma_cycles, cycles);
}

TileCost::Rate::Rate(Count clock_hz, Count bytes_per_second) {
    const auto gcd = std::gcd(clock_hz, bytes_per_second);
    clock = clock_hz / gcd;
    bandwidth = bytes_per_second / gcd;
}

Count TileCost::Rate::cycles(Count bytes) const {
    if (bytes <= UINT64_MAX / clock) return ceil_div(bytes * clock, bandwidth);
    return transfer_cycles(bytes, clock, bandwidth);
}

TileCost::TileCost(const Hardware& spec, Dataflow dataflow)
    : hardware(spec), flow(dataflow), input_width(input_bytes(spec.precision)),
      sram_read(spec.clock_hz, spec.sram_read_bps),
      sram_write(spec.clock_hz, spec.sram_write_bps), hbm(spec.clock_hz, spec.hbm_bps) {}

void TileCost::phase(Metrics& result, Count nominal, TensorBytes read, TensorBytes write) const {
    const auto time = std::max({nominal, sram_read.cycles(read.total()),
                                sram_write.cycles(write.total())});
    result.cycles.sram_stall = checked_add(result.cycles.sram_stall, time - nominal);
    result.sram.read.add(read, 1);
    result.sram.write.add(write, 1);
}

Metrics TileCost::array_wave(const Region& wave) const {
    const auto m = wave.m.size;
    const auto n = wave.n.size;
    const auto k = wave.k.size;
    const auto activations = checked_mul(checked_mul(m, k), input_width);
    const auto weights = checked_mul(checked_mul(k, n), input_width);
    const auto outputs = checked_mul(checked_mul(m, n), accumulator_bytes);
    const auto partial = wave.k.start == 0 ? 0 : outputs;
    Metrics result;
    result.waves = 1;
    result.useful_macs = checked_mul(checked_mul(m, n), k);
    if (flow == Dataflow::os) {
        result.cycles.compute = ceil_div(k, hardware.macs_per_pe_per_cycle);
        result.cycles.fill_drain = checked_add(checked_add(m - 1, n - 1), m);
        phase(result, result.cycles.array_time(), {activations, weights, partial},
              {0, 0, outputs});
    } else {
        result.cycles.compute = ceil_div(m, hardware.macs_per_pe_per_cycle);
        result.cycles.fill_drain = checked_add(hardware.rows - 1, hardware.cols);
        result.cycles.weight_load = k;
        phase(result, k, {0, weights, 0}, {});
        const auto stream_time = checked_add(result.cycles.compute, result.cycles.fill_drain);
        phase(result, stream_time, {activations, 0, partial}, {0, 0, outputs});
    }
    return result;
}

Metrics TileCost::array_tile(const Region& tile) const {
    const auto ms = wave_groups(tile.m.size, flow == Dataflow::os ? hardware.rows : tile.m.size,
                                false);
    const auto ns = wave_groups(tile.n.size, hardware.cols, false);
    const auto ks = wave_groups(tile.k.size, flow == Dataflow::ws ? hardware.rows : tile.k.size,
                                tile.k.start == 0);
    Metrics result;
    for (auto k : ks)
        for (auto m : ms)
            for (auto n : ns) {
                const auto wave = wave_at(tile, hardware, flow, {m.index, n.index, k.index});
                const auto repetitions = checked_mul(checked_mul(m.count, n.count), k.count);
                result.add(array_wave(wave), repetitions);
            }
    result.steps = 1;
    return result;
}

DmaRequest TileCost::fetch(const TileStep& tile, const TileStep* previous) const {
    const auto& [m, n, k] = tile.region;
    const bool reuse_a = previous && previous->index.m == tile.index.m &&
                         previous->index.k == tile.index.k;
    const bool reuse_b = previous && previous->index.n == tile.index.n &&
                         previous->index.k == tile.index.k;
    const TensorBytes incoming{reuse_a ? 0 : checked_mul(checked_mul(m.size, k.size), input_width),
                               reuse_b ? 0 : checked_mul(checked_mul(k.size, n.size), input_width),
                               0};
    return {incoming, dma_cycles(incoming.total(), sram_write), DmaDirection::input};
}

DmaRequest TileCost::store(const Region& tile) const {
    const auto bytes = checked_mul(checked_mul(tile.m.size, tile.n.size), accumulator_bytes);
    return {{0, 0, bytes}, dma_cycles(bytes, sram_read), DmaDirection::output};
}

Count TileCost::dma_cycles(Count bytes, const Rate& sram) const {
    if (bytes == 0) return 0;
    return checked_add(hardware.hbm_latency_cycles, std::max(hbm.cycles(bytes), sram.cycles(bytes)));
}

}  // namespace tensys
