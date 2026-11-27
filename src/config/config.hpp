#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace tensys {

using Count = std::uint64_t;

class InputError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class Precision { int8, fp16, bf16 };
enum class Dataflow { os, ws };
enum class Order { mnk, nmk };

struct Shape {
    Count m, n, k;
    bool operator==(const Shape&) const = default;
};

struct Capacity {
    Count unified{};
    std::array<Count, 3> split{};
    bool is_split{};
};

struct Hardware {
    Count rows, cols, clock_hz, macs_per_pe_per_cycle;
    Precision precision;
    Capacity capacity;
    Count sram_read_bps, sram_write_bps, hbm_bps, hbm_latency_cycles;
    bool double_buffered;
};

struct Layer {
    std::string name;
    Shape shape;
    std::string phase;
    Count batch{}, sequence_length{};
};

constexpr Count input_bytes(Precision precision) {
    return precision == Precision::int8 ? 1 : 2;
}

constexpr Count accumulator_bytes = 4;

inline Count checked_add(Count a, Count b) {
    if (b > UINT64_MAX - a) {
        throw InputError("simulation counter exceeds 64 bits");
    }
    return a + b;
}

inline Count checked_mul(Count a, Count b) {
    if (a != 0 && b > UINT64_MAX / a) {
        throw InputError("simulation counter exceeds 64 bits");
    }
    return a * b;
}

constexpr Count ceil_div(Count numerator, Count denominator) {
    return numerator / denominator + (numerator % denominator != 0);
}

Count peak_macs_per_cycle(const Hardware& hardware);
Count transfer_cycles(Count bytes, Count clock_hz, Count bytes_per_second);
nlohmann::json read_json(const std::filesystem::path& path);
Hardware parse_hardware(const nlohmann::json& value);
std::vector<Layer> parse_workload(const nlohmann::json& value);
const char* name(Precision precision);
const char* name(Dataflow dataflow);
const char* name(Order order);

}  // namespace tensys
