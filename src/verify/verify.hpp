#pragma once

#include "tiler/schedule.hpp"

namespace tensys {

struct Verification {
    Count seed{}, checked_elements{};
    double max_abs_error{};
    std::optional<std::array<Count, 2>> first_failure;
};

constexpr Count verification_dimension_limit = 128;

std::uint16_t encode_float16(float value, Precision precision);
float decode_float16(std::uint16_t bits, Precision precision);
Verification verify(const Hardware& hardware, const Schedule& schedule, Dataflow dataflow,
                    Count seed);

}  // namespace tensys
