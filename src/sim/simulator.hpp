#pragma once

#include "tiler/schedule.hpp"

#include <functional>

namespace tensys {

struct Cycles {
    Count compute{}, fill_drain{}, weight_load{}, hbm_stall{}, sram_stall{};

    Count array_time() const;
    Count total() const;
    void add(const Cycles& other, Count repetitions);
    bool operator==(const Cycles&) const = default;
};

struct TensorBytes {
    Count activations{}, weights{}, outputs{};

    Count total() const;
    void add(const TensorBytes& other, Count repetitions);
    bool operator==(const TensorBytes&) const = default;
};

struct Traffic {
    TensorBytes read, write;

    Count total() const;
    void add(const Traffic& other, Count repetitions);
    bool operator==(const Traffic&) const = default;
};

struct Metrics {
    Cycles cycles;
    Traffic hbm, sram;
    Count dma_cycles{}, dma_hidden_cycles{}, useful_macs{}, waves{}, steps{};

    void add(const Metrics& other, Count repetitions = 1);
    bool operator==(const Metrics&) const = default;
};

struct TraceStep {
    TileStep tile;
    std::optional<Shape> next_input;
    Metrics metrics;
};

using TraceSink = std::function<void(const TraceStep&)>;

Mapping select_mapping(const Hardware& hardware, Shape shape, Dataflow dataflow,
                       std::optional<Shape> manual_tile, std::optional<Order> order);
Metrics simulate(const Hardware& hardware, const Schedule& schedule, Dataflow dataflow,
                 const TraceSink& trace = {});

}  // namespace tensys
