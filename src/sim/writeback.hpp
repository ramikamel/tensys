#pragma once

#include "sim/simulator.hpp"

namespace tensys {

struct DmaTiming {
    Count input_service{}, output_service{}, input_hidden{}, output_hidden{};
};

struct ExposedWait {
    Count input{}, accumulator{}, drain{};
};

struct WritebackResult {
    Metrics metrics;
    DmaTiming dma;
    ExposedWait wait;

    void add(const WritebackResult& other);
};

enum class OperationKind { array, input_fetch, output_write };
enum class EventPhase { start, finish };

struct TimelineEvent {
    Count cycle;
    OperationKind operation;
    EventPhase phase;
    Shape tile;
    std::optional<std::size_t> accumulator_bank;
};

using TimelineSink = std::function<void(const TimelineEvent&)>;

WritebackResult simulate_writeback(const Hardware& hardware, const Schedule& schedule,
                                   Dataflow dataflow, const TimelineSink& timeline = {});

}  // namespace tensys
