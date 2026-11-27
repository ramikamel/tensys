#include "tiler/schedule.hpp"

namespace tensys {
namespace {

Count tile_count(Count extent, Count span) {
    if (extent == 0 || span == 0) throw InputError("layer and tile dimensions must be positive");
    return ceil_div(extent, span);
}

Range span_at(Count extent, Count span, Count index) {
    const auto start = index * span;
    return {start, std::min(span, extent - start)};
}

Range subrange(Range outer, Count span, Count index) {
    auto inner = span_at(outer.size, span, index);
    inner.start += outer.start;
    return inner;
}

bool fits_matrix(Count rows, Count cols, Count width, Count budget) {
    return cols <= budget / width / rows;
}

}  // namespace

Schedule::Schedule(Shape problem, Mapping selection)
    : shape(problem), mapping(selection),
      grid{tile_count(problem.m, selection.tile.m),
           tile_count(problem.n, selection.tile.n),
           tile_count(problem.k, selection.tile.k)} {}

TileStep Schedule::at(Shape index) const {
    return {index,
            {span_at(shape.m, mapping.tile.m, index.m), span_at(shape.n, mapping.tile.n, index.n),
             span_at(shape.k, mapping.tile.k, index.k)},
            index.k == grid.k - 1};
}

std::optional<Shape> Schedule::next(Shape index) const {
    if (++index.k < grid.k) return index;
    index.k = 0;
    auto& outer = mapping.order == Order::mnk ? index.m : index.n;
    auto& inner = mapping.order == Order::mnk ? index.n : index.m;
    const auto outer_count = mapping.order == Order::mnk ? grid.m : grid.n;
    const auto inner_count = mapping.order == Order::mnk ? grid.n : grid.m;
    if (++inner < inner_count) return index;
    inner = 0;
    if (++outer < outer_count) return index;
    return std::nullopt;
}

bool fits(Shape tile, const Hardware& hardware) {
    if (tile.m == 0 || tile.n == 0 || tile.k == 0) return false;
    const auto width = input_bytes(hardware.precision);
    const Count banks = hardware.double_buffered ? 2 : 1;
    const auto& capacity = hardware.capacity;
    if (capacity.is_split) {
        return fits_matrix(tile.m, tile.k, width, capacity.split[0] / banks) &&
               fits_matrix(tile.k, tile.n, width, capacity.split[1] / banks) &&
               fits_matrix(tile.m, tile.n, accumulator_bytes, capacity.split[2] / banks);
    }
    auto remaining = capacity.unified / banks;
    if (!fits_matrix(tile.m, tile.k, width, remaining)) return false;
    remaining -= tile.m * tile.k * width;
    if (!fits_matrix(tile.k, tile.n, width, remaining)) return false;
    remaining -= tile.k * tile.n * width;
    return fits_matrix(tile.m, tile.n, accumulator_bytes, remaining);
}

std::vector<Count> tile_candidates(Count extent, Count alignment) {
    std::vector<Count> result{extent};
    for (const auto first : {Count{1}, alignment}) {
        for (auto value = first; value <= extent;) {
            result.push_back(value);
            if (value > extent / 2) break;
            value *= 2;
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

Groups tile_groups(Count grid_extent) {
    // Split positions whose current or next tile may have a special extent or K state.
    const auto last = grid_extent - 1;
    const auto before_last = last == 0 ? 0 : last - 1;
    const std::array bounds{Count{0}, std::min(Count{1}, before_last), before_last,
                            last, grid_extent};
    Groups groups;
    for (std::size_t i = 1; i < bounds.size(); ++i) {
        groups.add(bounds[i - 1], bounds[i] - bounds[i - 1]);
    }
    return groups;
}

Groups wave_groups(Count extent, Count span, bool distinguish_first) {
    const auto full = extent / span;
    Groups groups;
    const Count first = distinguish_first && full != 0 ? 1 : 0;
    groups.add(0, first);
    groups.add(first, full - first);
    groups.add(full, extent % span != 0 ? 1 : 0);
    return groups;
}

Shape wave_grid(const Region& tile, const Hardware& hardware, Dataflow dataflow) {
    return {dataflow == Dataflow::os ? ceil_div(tile.m.size, hardware.rows) : 1,
            ceil_div(tile.n.size, hardware.cols),
            dataflow == Dataflow::ws ? ceil_div(tile.k.size, hardware.rows) : 1};
}

Region wave_at(const Region& tile, const Hardware& hardware, Dataflow dataflow, Shape index) {
    return {dataflow == Dataflow::os ? subrange(tile.m, hardware.rows, index.m) : tile.m,
            subrange(tile.n, hardware.cols, index.n),
            dataflow == Dataflow::ws ? subrange(tile.k, hardware.rows, index.k) : tile.k};
}

}  // namespace tensys
