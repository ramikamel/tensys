#include "config/config.hpp"
#include "sim/simulator.hpp"
#include "sim/writeback.hpp"
#include "verify/verify.hpp"

#include <cmath>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace tensys {
namespace {

Hardware tiny(bool double_buffered = false) {
    return {2, 2, 1'000'000'000, 1, Precision::int8, {4096, {}, false},
            64'000'000'000, 64'000'000'000, 4'000'000'000, 3, double_buffered};
}

TEST(Execution, HandCalculatedOutputAndWeightStationary) {
    const Schedule schedule({2, 2, 2}, {{2, 2, 2}, Order::mnk});
    const auto os = simulate(tiny(), schedule, Dataflow::os);
    const auto ws = simulate(tiny(), schedule, Dataflow::ws);
    EXPECT_EQ(os.cycles, (Cycles{2, 4, 0, 12, 0}));
    EXPECT_EQ(ws.cycles, (Cycles{2, 3, 2, 12, 0}));
    EXPECT_EQ(os.cycles.total(), 18);
    EXPECT_EQ(ws.cycles.total(), 19);
    EXPECT_EQ(os.useful_macs, 8);
    EXPECT_EQ(os.hbm, (Traffic{{4, 4, 0}, {0, 0, 16}}));
    EXPECT_EQ(os.sram, (Traffic{{4, 4, 16}, {4, 4, 16}}));
    EXPECT_EQ(os.hbm, ws.hbm);
    EXPECT_EQ(os.sram, ws.sram);
}

TEST(Execution, PrefetchHidesOnlyTheNextInputTransfer) {
    const Schedule schedule({2, 4, 2}, {{2, 2, 2}, Order::mnk});
    const auto single = simulate(tiny(), schedule, Dataflow::os);
    const auto dual = simulate(tiny(true), schedule, Dataflow::os);
    EXPECT_EQ(single.cycles.total(), 35);
    EXPECT_EQ(dual.cycles.total(), 31);
    EXPECT_EQ(single.dma_cycles, 23);
    EXPECT_EQ(dual.dma_cycles, 23);
    EXPECT_EQ(dual.dma_hidden_cycles, 4);
    EXPECT_EQ(dual.cycles.hbm_stall, 19);
    EXPECT_EQ(dual.hbm, single.hbm);
    EXPECT_EQ(dual.hbm.read, (TensorBytes{4, 8, 0}));
    EXPECT_EQ(dual.hbm.write.outputs, 32);
}

TEST(Writeback, OverlapsTheFirstOutputWithTheNextTile) {
    const Schedule schedule({2, 4, 2}, {{2, 2, 2}, Order::mnk});
    const auto hardware = tiny(true);
    const auto baseline = simulate(hardware, schedule, Dataflow::os);
    const auto result = simulate_writeback(hardware, schedule, Dataflow::os);
    EXPECT_EQ(baseline.cycles.total(), 31);
    EXPECT_EQ(result.metrics.cycles.total(), 25);
    EXPECT_EQ(result.metrics.hbm.total(), 44);
    EXPECT_EQ(result.metrics.hbm, baseline.hbm);
    EXPECT_EQ(result.dma.input_service, 9);
    EXPECT_EQ(result.dma.output_service, 14);
    EXPECT_EQ(result.dma.input_hidden, 4);
    EXPECT_EQ(result.dma.output_hidden, 6);
    EXPECT_EQ(result.wait.input, 5);
    EXPECT_EQ(result.wait.accumulator, 0);
    EXPECT_EQ(result.wait.drain, 8);
}

TEST(Writeback, WaitsForAnAccumulatorBankBeforeStartingTheThirdTile) {
    const Schedule schedule({2, 6, 2}, {{2, 2, 2}, Order::mnk});
    std::vector<std::pair<Count, std::size_t>> array_starts;
    std::vector<Count> write_completions;
    const auto result = simulate_writeback(tiny(true), schedule, Dataflow::os,
        [&](const TimelineEvent& event) {
            if (event.operation == OperationKind::array && event.phase == EventPhase::start)
                array_starts.emplace_back(event.cycle, event.accumulator_bank.value());
            if (event.operation == OperationKind::output_write && event.phase == EventPhase::finish)
                write_completions.push_back(event.cycle);
        });
    EXPECT_EQ(result.metrics.cycles.total(), 36);
    EXPECT_EQ(result.metrics.hbm.total(), 64);
    EXPECT_EQ(result.wait.accumulator, 5);
    EXPECT_EQ(array_starts, (std::vector<std::pair<Count, std::size_t>>{{5, 0}, {11, 1}, {22, 0}}));
    EXPECT_EQ(write_completions, (std::vector<Count>{22, 29, 36}));
}

TEST(Writeback, KeepsOneBankThroughAllReductionSteps) {
    const Schedule schedule({2, 2, 6}, {{2, 2, 2}, Order::mnk});
    Count reductions{};
    const auto result = simulate_writeback(tiny(true), schedule, Dataflow::ws,
        [&](const TimelineEvent& event) {
            if (event.operation == OperationKind::array && event.phase == EventPhase::finish) {
                EXPECT_EQ(event.accumulator_bank, 0);
                ++reductions;
            }
            if (event.operation == OperationKind::output_write && event.phase == EventPhase::start)
                EXPECT_EQ(reductions, 3);
        });
    EXPECT_EQ(reductions, 3);
    EXPECT_EQ(result.metrics, simulate(tiny(true), schedule, Dataflow::ws));
}

TEST(Writeback, PreservesWorkAndTrafficAcrossLayoutsAndDataflows) {
    for (auto flow : {Dataflow::os, Dataflow::ws}) {
        for (auto order : {Order::mnk, Order::nmk}) {
            for (bool split : {false, true}) {
                auto hardware = tiny(true);
                hardware.capacity = {4096, {512, 512, 1024}, split};
                hardware.rows = 3;
                hardware.precision = Precision::bf16;
                hardware.sram_read_bps = 900'000'000;
                for (Count k : {1, 6, 13}) {
                    const Schedule schedule({7, 11, k}, {{4, 5, 6}, order});
                    const auto baseline = simulate(hardware, schedule, flow);
                    const auto result = simulate_writeback(hardware, schedule, flow);
                    const auto& accelerated = result.metrics;
                    EXPECT_EQ(accelerated.hbm, baseline.hbm);
                    EXPECT_EQ(accelerated.sram, baseline.sram);
                    EXPECT_EQ(accelerated.useful_macs, baseline.useful_macs);
                    EXPECT_EQ(accelerated.cycles.array_time(), baseline.cycles.array_time());
                    EXPECT_EQ(accelerated.dma_cycles, baseline.dma_cycles);
                    EXPECT_EQ(result.dma.input_service + result.dma.output_service,
                              accelerated.dma_cycles);
                    EXPECT_EQ(result.dma.input_hidden + result.dma.output_hidden,
                              accelerated.dma_hidden_cycles);
                    EXPECT_EQ(accelerated.dma_hidden_cycles + accelerated.cycles.hbm_stall,
                              accelerated.dma_cycles);
                    EXPECT_EQ(result.wait.input + result.wait.accumulator + result.wait.drain,
                              accelerated.cycles.hbm_stall);
                }
            }
        }
    }
}

TEST(Execution, SramCanLimitArrayAndDmaSeparately) {
    auto hardware = tiny();
    hardware.sram_read_bps = 4'000'000'000;
    hardware.sram_write_bps = 1'000'000'000;
    const Schedule schedule({2, 2, 2}, {{2, 2, 2}, Order::mnk});
    const auto result = simulate(hardware, schedule, Dataflow::os);
    EXPECT_EQ(result.cycles.sram_stall, 10);
    EXPECT_EQ(result.cycles.hbm_stall, 18);
    EXPECT_EQ(result.cycles.total(), 34);
}

TEST(Execution, DataflowsHaveDifferentWaveReuseAndPartialTraffic) {
    auto hardware = tiny();
    hardware.cols = 3;
    const Schedule schedule({3, 4, 5}, {{3, 4, 5}, Order::mnk});
    const auto os = simulate(hardware, schedule, Dataflow::os);
    const auto ws = simulate(hardware, schedule, Dataflow::ws);
    EXPECT_EQ(os.waves, 4);
    EXPECT_EQ(ws.waves, 6);
    EXPECT_EQ(os.sram, (Traffic{{30, 40, 48}, {15, 20, 48}}));
    EXPECT_EQ(ws.sram, (Traffic{{30, 20, 144}, {15, 20, 144}}));
    EXPECT_EQ(os.hbm, (Traffic{{15, 20, 0}, {0, 0, 48}}));
    EXPECT_EQ(ws.hbm, os.hbm);
    EXPECT_EQ(os.useful_macs, 60);
    EXPECT_EQ(ws.useful_macs, 60);
}

TEST(Execution, GroupedSummariesMatchEnumeratedTransitions) {
    for (const auto flow : {Dataflow::os, Dataflow::ws}) {
        for (const auto order : {Order::mnk, Order::nmk}) {
            for (const auto buffering : {false, true}) {
                auto hardware = tiny(buffering);
                hardware.rows = 3;
                hardware.macs_per_pe_per_cycle = 2;
                hardware.sram_read_bps = 500'000'000;
                hardware.sram_write_bps = 1'500'000'000;
                for (Count m : {1, 8, 23}) {
                    for (Count n : {1, 7, 27}) {
                        for (Count k : {1, 6, 31}) {
                            const Shape extent{m, n, k};
                            const Shape tile{std::min(m, Count{4}), std::min(n, Count{5}),
                                             std::min(k, Count{6})};
                            const Schedule schedule(extent, {tile, order});
                            const auto grouped = simulate(hardware, schedule, flow);
                            Metrics traced;
                            const auto enumerated = simulate(hardware, schedule, flow,
                                [&](const TraceStep& step) { traced.add(step.metrics); });
                            EXPECT_EQ(grouped, enumerated);
                            EXPECT_EQ(grouped, traced);
                            EXPECT_EQ(grouped.hbm.write.outputs, m * n * 4);
                            EXPECT_EQ(grouped.useful_macs, m * n * k);
                            EXPECT_GE(grouped.hbm.total(), m * k + n * k + 4 * m * n);
                            EXPECT_EQ(grouped.dma_cycles,
                                      grouped.dma_hidden_cycles + grouped.cycles.hbm_stall);
                            EXPECT_LE(grouped.useful_macs,
                                      peak_macs_per_cycle(hardware) * grouped.cycles.compute);
                        }
                    }
                }
            }
        }
    }
}

TEST(Execution, SummarizesOneBillionSteps) {
    constexpr Count count = 1'000'000'000;
    const Schedule schedule({1, 1, count}, {{1, 1, 1}, Order::mnk});
    const auto result = simulate(tiny(), schedule, Dataflow::os);
    EXPECT_EQ(result.steps, count);
    EXPECT_EQ(result.useful_macs, count);
    EXPECT_EQ(result.cycles.total(), 6 * count + 4);
}

TEST(Tiling, SingleAndDoubleBufferCapacityBoundaries) {
    auto hardware = tiny();
    hardware.capacity.unified = 19;
    const Shape tile{2, 2, 2};
    EXPECT_FALSE(fits(tile, hardware));
    hardware.capacity.unified = 24;
    EXPECT_TRUE(fits(tile, hardware));
    hardware.double_buffered = true;
    hardware.capacity.unified = 47;
    EXPECT_FALSE(fits(tile, hardware));
    hardware.capacity.unified = 48;
    EXPECT_TRUE(fits(tile, hardware));
    hardware.capacity = {0, {8, 8, 31}, true};
    EXPECT_FALSE(fits(tile, hardware));
    hardware.capacity.split[2] = 32;
    EXPECT_TRUE(fits(tile, hardware));
    hardware.precision = Precision::bf16;
    EXPECT_FALSE(fits(tile, hardware));
}

TEST(Tiling, AutoFitsAndTradesFullReductionForCapacity) {
    auto hardware = tiny();
    hardware.rows = hardware.cols = 4;
    hardware.capacity.unified = 896;
    const Shape shape{16, 16, 32};
    const auto manual = select_mapping(hardware, shape, Dataflow::os, Shape{16, 4, 32}, {});
    const auto single = simulate(hardware, Schedule(shape, manual), Dataflow::os);
    EXPECT_EQ(single.hbm.total(), 2048);
    hardware.double_buffered = true;
    EXPECT_THROW(select_mapping(hardware, shape, Dataflow::os, manual.tile, {}), InputError);
    const auto automatic = select_mapping(hardware, shape, Dataflow::os, {}, {});
    EXPECT_TRUE(fits(automatic.tile, hardware));
    const auto dual = simulate(hardware, Schedule(shape, automatic), Dataflow::os);
    EXPECT_GT(dual.hbm.total(), single.hbm.total());
    const auto fixed_order = select_mapping(hardware, shape, Dataflow::os, {}, Order::nmk);
    EXPECT_EQ(fixed_order.order, Order::nmk);
}

TEST(Verification, HandlesTailTilesAndSplitReductionsAcrossPrecisions) {
    for (const auto precision : {Precision::int8, Precision::fp16, Precision::bf16}) {
        auto hardware = tiny(true);
        hardware.rows = 4;
        hardware.cols = 5;
        hardware.precision = precision;
        hardware.macs_per_pe_per_cycle = 2;
        for (const auto order : {Order::mnk, Order::nmk}) {
            for (const auto flow : {Dataflow::os, Dataflow::ws}) {
                const Schedule schedule({7, 11, 13}, {{5, 7, 6}, order});
                const auto result = verify(hardware, schedule, flow, 101);
                EXPECT_FALSE(result.first_failure);
                EXPECT_EQ(result.checked_elements, 77);
                EXPECT_EQ(result.seed, 101);
                if (precision == Precision::int8) EXPECT_EQ(result.max_abs_error, 0);
            }
        }
    }
}

TEST(Verification, RoundsTiesToEvenAndPreservesSubnormals) {
    EXPECT_EQ(encode_float16(1.0f + 0x1p-11f, Precision::fp16), 0x3c00);
    EXPECT_EQ(encode_float16(1.0f + 0x1.8p-10f, Precision::fp16), 0x3c02);
    EXPECT_EQ(encode_float16(0x1p-25f, Precision::fp16), 0);
    EXPECT_EQ(encode_float16(0x1p-24f, Precision::fp16), 1);
    EXPECT_EQ(decode_float16(1, Precision::fp16), 0x1p-24f);
    EXPECT_EQ(decode_float16(0x7bff, Precision::fp16), 65504.0f);
    EXPECT_TRUE(std::signbit(decode_float16(0x8000, Precision::fp16)));
    EXPECT_EQ(encode_float16(1.0f + 0x1p-8f, Precision::bf16), 0x3f80);
    EXPECT_EQ(encode_float16(1.0f + 0x1.8p-7f, Precision::bf16), 0x3f82);
    EXPECT_EQ(decode_float16(0x3f82, Precision::bf16), 1.015625f);
}

TEST(Configuration, ExactTransferCeilingsAndOverflow) {
    EXPECT_EQ(transfer_cycles(2, 3, 4), 2);
    EXPECT_EQ(transfer_cycles(0, 7, 5), 0);
    EXPECT_EQ(transfer_cycles(UINT64_MAX, UINT64_MAX, UINT64_MAX), UINT64_MAX);
    EXPECT_THROW(transfer_cycles(UINT64_MAX, UINT64_MAX, 1), InputError);
    EXPECT_THROW(checked_mul(UINT64_MAX, 2), InputError);
}

TEST(Configuration, RejectsInvalidShapesAndNames) {
    const auto json = nlohmann::json::parse(R"({"layers":[{"name":"x","m":1,"n":2,"k":3}]})");
    EXPECT_EQ(parse_workload(json)[0].shape, (Shape{1, 2, 3}));
    auto invalid = json;
    invalid["layers"][0]["k"] = 0;
    EXPECT_THROW(parse_workload(invalid), InputError);
    invalid["layers"][0]["k"] = 1.5;
    EXPECT_THROW(parse_workload(invalid), InputError);
    invalid = json;
    invalid["layers"].push_back(json["layers"][0]);
    EXPECT_THROW(parse_workload(invalid), InputError);
}

}  // namespace
}  // namespace tensys
