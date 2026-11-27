#include "sim/simulator.hpp"
#include "sim/tile_cost.hpp"

#include <tuple>

namespace tensys {
namespace {

void account_dma(Metrics& result, const DmaRequest& request, Count overlap) {
    request.account(result);
    const auto hidden = std::min(request.cycles, overlap);
    result.dma_hidden_cycles = checked_add(result.dma_hidden_cycles, hidden);
    result.cycles.hbm_stall = checked_add(result.cycles.hbm_stall, request.cycles - hidden);
}

struct Baseline {
    TileCost cost;
    bool double_buffered;

    Baseline(const Hardware& hardware, Dataflow flow)
        : cost(hardware, flow), double_buffered(hardware.double_buffered) {}

    TraceStep step(const Schedule& schedule, Shape index) const {
        const auto tile = schedule.at(index);
        auto result = cost.array_tile(tile.region);
        if (index == Shape{0, 0, 0}) account_dma(result, cost.fetch(tile, nullptr), 0);
        const auto next_index = schedule.next(index);
        if (tile.last_k && !double_buffered) account_dma(result, cost.store(tile.region), 0);
        if (next_index) {
            const auto next = schedule.at(*next_index);
            const auto overlap = double_buffered ? result.cycles.array_time() : 0;
            account_dma(result, cost.fetch(next, &tile), overlap);
        }
        if (tile.last_k && double_buffered) account_dma(result, cost.store(tile.region), 0);
        return {tile, next_index, result};
    }

    Metrics summarize(const Schedule& schedule) const {
        Metrics result;
        for (auto m : tile_groups(schedule.grid.m))
            for (auto n : tile_groups(schedule.grid.n))
                for (auto k : tile_groups(schedule.grid.k)) {
                    const auto repetitions = checked_mul(checked_mul(m.count, n.count), k.count);
                    result.add(step(schedule, {m.index, n.index, k.index}).metrics, repetitions);
                }
        return result;
    }
};

void add_scaled(Count& target, Count value, Count repetitions) {
    target = checked_add(target, checked_mul(value, repetitions));
}

}  // namespace

Count Cycles::array_time() const {
    return checked_add(checked_add(compute, fill_drain), checked_add(weight_load, sram_stall));
}

Count Cycles::total() const { return checked_add(array_time(), hbm_stall); }

void Cycles::add(const Cycles& other, Count repetitions) {
    add_scaled(compute, other.compute, repetitions);
    add_scaled(fill_drain, other.fill_drain, repetitions);
    add_scaled(weight_load, other.weight_load, repetitions);
    add_scaled(hbm_stall, other.hbm_stall, repetitions);
    add_scaled(sram_stall, other.sram_stall, repetitions);
}

Count TensorBytes::total() const { return checked_add(checked_add(activations, weights), outputs); }

void TensorBytes::add(const TensorBytes& other, Count repetitions) {
    add_scaled(activations, other.activations, repetitions);
    add_scaled(weights, other.weights, repetitions);
    add_scaled(outputs, other.outputs, repetitions);
}

Count Traffic::total() const { return checked_add(read.total(), write.total()); }

void Traffic::add(const Traffic& other, Count repetitions) {
    read.add(other.read, repetitions);
    write.add(other.write, repetitions);
}

void Metrics::add(const Metrics& other, Count repetitions) {
    cycles.add(other.cycles, repetitions);
    hbm.add(other.hbm, repetitions);
    sram.add(other.sram, repetitions);
    add_scaled(dma_cycles, other.dma_cycles, repetitions);
    add_scaled(dma_hidden_cycles, other.dma_hidden_cycles, repetitions);
    add_scaled(useful_macs, other.useful_macs, repetitions);
    add_scaled(waves, other.waves, repetitions);
    add_scaled(steps, other.steps, repetitions);
}

Mapping select_mapping(const Hardware& hardware, Shape shape, Dataflow dataflow,
                       std::optional<Shape> manual_tile, std::optional<Order> order) {
    if (manual_tile) {
        const auto tile = *manual_tile;
        if (tile.m == 0 || tile.n == 0 || tile.k == 0)
            throw InputError("tile dimensions must be positive");
        if (!fits(tile, hardware)) throw InputError("manual tile does not fit the SRAM bank budget");
        return {tile, order.value_or(Order::mnk)};
    }
    if (!fits({1, 1, 1}, hardware)) throw InputError("SRAM bank cannot fit a 1×1×1 tile");

    const Baseline model(hardware, dataflow);
    const auto ms = tile_candidates(shape.m, dataflow == Dataflow::os ? hardware.rows
                                                                     : hardware.macs_per_pe_per_cycle);
    const auto ns = tile_candidates(shape.n, hardware.cols);
    const auto ks = tile_candidates(shape.k, dataflow == Dataflow::ws ? hardware.rows
                                                                     : hardware.macs_per_pe_per_cycle);
    using Score = std::tuple<Count, Count, Count, Count, Count, Count, Order>;
    std::optional<Score> best_score;
    Mapping best{};
    for (auto m : ms)
        for (auto n : ns)
            for (auto k : ks) {
                const Shape tile{m, n, k};
                if (!fits(tile, hardware)) continue;
                for (auto traversal : {Order::mnk, Order::nmk}) {
                    if (order && traversal != *order) continue;
                    const Mapping candidate{tile, traversal};
                    try {
                        const auto metrics = model.summarize(Schedule(shape, candidate));
                        metrics.sram.total();
                        // Useful MACs and peak rate are fixed, so fewer streaming cycles
                        // mean higher streaming occupancy.
                        const Score score{metrics.cycles.total(), metrics.hbm.total(),
                                          metrics.cycles.compute, m, n, k, traversal};
                        if (!best_score || score < *best_score) {
                            best_score = score;
                            best = candidate;
                        }
                    } catch (const InputError&) {
                        // A different legal tile may have fewer cycles or bytes.
                    }
                }
            }
    if (!best_score) throw InputError("all fitting schedules exceed the counter range");
    return best;
}

Metrics simulate(const Hardware& hardware, const Schedule& schedule, Dataflow dataflow,
                 const TraceSink& trace) {
    const Baseline model(hardware, dataflow);
    if (!trace) return model.summarize(schedule);
    Metrics result;
    for_each_tile(schedule, [&](const TileStep& tile) {
        const auto row = model.step(schedule, tile.index);
        result.add(row.metrics);
        trace(row);
    });
    return result;
}

}  // namespace tensys
