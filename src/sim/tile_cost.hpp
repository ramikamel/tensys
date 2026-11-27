#pragma once

#include "sim/simulator.hpp"

namespace tensys {

enum class DmaDirection { input, output };

struct DmaRequest {
    TensorBytes bytes;
    Count cycles{};
    DmaDirection direction;

    void account(Metrics& result) const;
};

class TileCost {
public:
    TileCost(const Hardware& hardware, Dataflow dataflow);
    Metrics array_tile(const Region& tile) const;
    DmaRequest fetch(const TileStep& tile, const TileStep* previous) const;
    DmaRequest store(const Region& tile) const;

private:
    struct Rate {
        Count clock, bandwidth;
        Rate(Count clock_hz, Count bytes_per_second);
        Count cycles(Count bytes) const;
    };

    void phase(Metrics& result, Count nominal, TensorBytes read, TensorBytes write) const;
    Metrics array_wave(const Region& wave) const;
    Count dma_cycles(Count bytes, const Rate& sram) const;

    const Hardware& hardware;
    Dataflow flow;
    Count input_width;
    Rate sram_read, sram_write, hbm;
};

}  // namespace tensys
