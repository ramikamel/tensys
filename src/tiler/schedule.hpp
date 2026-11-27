#pragma once

#include "config/config.hpp"

#include <algorithm>
#include <optional>

namespace tensys {

struct Mapping {
    Shape tile;
    Order order;
};

struct Range {
    Count start, size;
};

struct Region {
    Range m, n, k;
};

struct TileStep {
    Shape index;
    Region region;
    bool last_k;
};

struct Schedule {
    Shape shape;
    Mapping mapping;
    Shape grid;

    Schedule(Shape problem, Mapping selection);
    TileStep at(Shape index) const;
    std::optional<Shape> next(Shape index) const;
};

struct Group {
    Count index, count;
};

struct Groups {
    std::array<Group, 4> storage{};
    std::size_t size{};

    void add(Count index, Count count) {
        if (count != 0) storage.at(size++) = {index, count};
    }
    const Group* begin() const { return storage.data(); }
    const Group* end() const { return storage.data() + size; }
};

bool fits(Shape tile, const Hardware& hardware);
std::vector<Count> tile_candidates(Count extent, Count alignment);
Groups tile_groups(Count grid_extent);
Groups wave_groups(Count extent, Count span, bool distinguish_first);
Shape wave_grid(const Region& tile, const Hardware& hardware, Dataflow dataflow);
Region wave_at(const Region& tile, const Hardware& hardware, Dataflow dataflow, Shape index);

template<class Callback>
void for_each_tile(const Schedule& schedule, Callback&& callback) {
    auto index = std::optional<Shape>{{0, 0, 0}};
    while (index) {
        callback(schedule.at(*index));
        index = schedule.next(*index);
    }
}

template<class Callback>
void for_each_wave(const Region& tile, const Hardware& hardware, Dataflow dataflow,
                   Callback&& callback) {
    const auto grid = wave_grid(tile, hardware, dataflow);
    for (Count k = 0; k < grid.k; ++k)
        for (Count m = 0; m < grid.m; ++m)
            for (Count n = 0; n < grid.n; ++n)
                callback(wave_at(tile, hardware, dataflow, {m, n, k}));
}

}  // namespace tensys
