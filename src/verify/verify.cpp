#include "verify/verify.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <random>

namespace tensys {
namespace {

std::uint32_t round_shift(std::uint32_t value, unsigned shift) {
    const auto retained = value >> shift;
    const auto lost = value & ((1U << shift) - 1);
    const auto midpoint = 1U << (shift - 1);
    return retained + (lost > midpoint || (lost == midpoint && (retained & 1U) != 0));
}

struct IntegerGemm {
    using Input = std::int8_t;
    using Sum = std::uint32_t;
    using Reference = std::int64_t;

    static Input sample(std::mt19937_64& random) {
        return static_cast<Input>(static_cast<int>(random() & 255) - 128);
    }
    static void accumulate(Sum& sum, Input a, Input b) {
        sum += static_cast<Sum>(static_cast<int>(a) * static_cast<int>(b));
    }
    static void reference(Reference& sum, double&, Input a, Input b) {
        sum += static_cast<Reference>(a) * b;
    }
    static double error(Sum result, Reference expected) {
        return std::abs(static_cast<double>(std::bit_cast<std::int32_t>(result)) -
                        std::bit_cast<std::int32_t>(static_cast<Sum>(expected)));
    }
    static double tolerance(double) { return 0; }
};

struct FloatingGemm {
    using Input = std::uint16_t;
    using Sum = float;
    using Reference = double;

    Precision precision;

    Input sample(std::mt19937_64& random) const {
        const auto value = static_cast<int>(random() % 65'537) - 32'768;
        return encode_float16(static_cast<float>(value) / 10'003.0f, precision);
    }
    void accumulate(Sum& sum, Input a, Input b) const {
        sum = std::fma(decode_float16(a, precision), decode_float16(b, precision), sum);
    }
    void reference(Reference& sum, double& magnitude, Input a, Input b) const {
        const double product =
            static_cast<double>(decode_float16(a, precision)) * decode_float16(b, precision);
        sum += product;
        magnitude += std::abs(product);
    }
    static double error(Sum result, Reference expected) {
        return std::abs(static_cast<double>(result) - expected);
    }
    static double tolerance(double magnitude) { return 1e-6 + 1e-5 * magnitude; }
};

template<class Arithmetic>
Verification run_verification(const Hardware& hardware, const Schedule& schedule, Dataflow dataflow,
                              Count seed, Arithmetic arithmetic) {
    const auto [m, n, k] = schedule.shape;
    std::mt19937_64 random(seed);
    std::vector<typename Arithmetic::Input> a(m * k), b(k * n);
    for (auto& value : a) value = arithmetic.sample(random);
    for (auto& value : b) value = arithmetic.sample(random);
    std::vector<typename Arithmetic::Sum> output(m * n);
    for_each_tile(schedule, [&](const TileStep& step) {
        for_each_wave(step.region, hardware, dataflow, [&](const Region& wave) {
            for (Count i = wave.m.start; i < wave.m.start + wave.m.size; ++i) {
                for (Count j = wave.n.start; j < wave.n.start + wave.n.size; ++j) {
                    auto partial = wave.k.start == 0 ? typename Arithmetic::Sum{} : output[i * n + j];
                    for (Count p = wave.k.start; p < wave.k.start + wave.k.size; ++p) {
                        arithmetic.accumulate(partial, a[i * k + p], b[p * n + j]);
                    }
                    output[i * n + j] = partial;
                }
            }
        });
    });
    Verification result{seed, m * n, 0, std::nullopt};
    for (Count i = 0; i < m; ++i) {
        for (Count j = 0; j < n; ++j) {
            typename Arithmetic::Reference expected{};
            double magnitude = 0;
            for (Count p = 0; p < k; ++p) {
                arithmetic.reference(expected, magnitude, a[i * k + p], b[p * n + j]);
            }
            const auto error = arithmetic.error(output[i * n + j], expected);
            result.max_abs_error = std::max(result.max_abs_error, error);
            if ((!std::isfinite(error) || error > arithmetic.tolerance(magnitude)) &&
                !result.first_failure) {
                result.first_failure = std::array{i, j};
            }
        }
    }
    return result;
}

}  // namespace

std::uint16_t encode_float16(float value, Precision precision) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto exponent = (bits >> 23) & 255U;
    const auto fraction = bits & 0x7fffffU;
    if (precision == Precision::bf16) {
        if (exponent == 255 && fraction != 0)
            return static_cast<std::uint16_t>((bits >> 16) | 0x40U);
        return static_cast<std::uint16_t>(round_shift(bits, 16));
    }
    const auto sign = (bits >> 16) & 0x8000U;
    if (exponent == 255 && fraction != 0) return static_cast<std::uint16_t>(sign | 0x7e00U);
    const auto unbiased = static_cast<int>(exponent) - 127;
    if (unbiased > 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (unbiased >= -14) {
        const auto biased = static_cast<std::uint32_t>(unbiased + 15);
        return static_cast<std::uint16_t>(sign | ((biased << 10) + round_shift(fraction, 13)));
    }
    if (unbiased < -25) return static_cast<std::uint16_t>(sign);
    return static_cast<std::uint16_t>(
        sign | round_shift(fraction | 0x800000U, static_cast<unsigned>(-unbiased - 1)));
}

float decode_float16(std::uint16_t bits, Precision precision) {
    if (precision == Precision::bf16) {
        return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
    }
    const auto exponent = (bits >> 10) & 31;
    const auto fraction = bits & 1023;
    float value;
    if (exponent == 31) {
        value = fraction == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    } else if (exponent == 0) {
        value = std::ldexp(static_cast<float>(fraction), -24);
    } else {
        value = std::ldexp(static_cast<float>(1024 + fraction), exponent - 25);
    }
    return (bits & 0x8000U) ? -value : value;
}

Verification verify(const Hardware& hardware, const Schedule& schedule, Dataflow dataflow,
                    Count seed) {
    const auto [m, n, k] = schedule.shape;
    if (std::max({m, n, k}) > verification_dimension_limit) {
        throw InputError("verification requires each dimension to be at most 128");
    }
    if (hardware.precision == Precision::int8)
        return run_verification(hardware, schedule, dataflow, seed, IntegerGemm{});
    return run_verification(hardware, schedule, dataflow, seed, FloatingGemm{hardware.precision});
}

}  // namespace tensys
