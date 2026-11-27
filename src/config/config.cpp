#include "config/config.hpp"

#include <cmath>
#include <fstream>
#include <set>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace tensys {
namespace {

using Json = nlohmann::json;

const Json& field(const Json& object, const char* key) {
    if (!object.is_object() || !object.contains(key)) {
        throw InputError(std::string("missing field: ") + key);
    }
    return object.at(key);
}

std::string string_field(const Json& object, const char* key) {
    const auto& value = field(object, key);
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        throw InputError(std::string(key) + " must be a nonempty string");
    }
    return value.get<std::string>();
}

Count integer_field(const Json& object, const char* key, Count min = 1) {
    const auto& value = field(object, key);
    if ((!value.is_number_unsigned() && !value.is_number_integer()) ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)) {
        throw InputError(std::string(key) + " must be a nonnegative integer");
    }
    const auto number = value.get<Count>();
    if (number < min) {
        throw InputError(std::string(key) + " must be positive");
    }
    return number;
}

Count giga_field(const Json& object, const char* key) {
    const auto& value = field(object, key);
    if (!value.is_number()) {
        throw InputError(std::string(key) + " must be a number");
    }
    const auto scaled = std::round(value.get<long double>() * 1'000'000'000.0L);
    if (!std::isfinite(scaled) || scaled < 1 || scaled >= 0x1p64L) {
        throw InputError(std::string(key) + " is outside the supported rate range");
    }
    return static_cast<Count>(scaled);
}

Precision parse_precision(std::string_view text) {
    if (text == "INT8") return Precision::int8;
    if (text == "FP16") return Precision::fp16;
    if (text == "BF16") return Precision::bf16;
    throw InputError("precision must be INT8, FP16, or BF16");
}

}  // namespace

Count peak_macs_per_cycle(const Hardware& hardware) {
    return checked_mul(checked_mul(hardware.rows, hardware.cols), hardware.macs_per_pe_per_cycle);
}

Count transfer_cycles(Count bytes, Count clock_hz, Count bytes_per_second) {
    // A full-width product keeps rate conversion exact even for large transfers.
    __extension__ using Wide = unsigned __int128;
    const auto product = static_cast<Wide>(bytes) * clock_hz;
    const auto cycles = product / bytes_per_second + (product % bytes_per_second != 0);
    if (cycles > UINT64_MAX) throw InputError("transfer cycle count exceeds 64 bits");
    return static_cast<Count>(cycles);
}

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw InputError("cannot open JSON file: " + path.string());
    try {
        return Json::parse(file);
    } catch (const Json::exception& error) {
        throw InputError(path.string() + ": " + error.what());
    }
}

Hardware parse_hardware(const nlohmann::json& value) {
    const auto& array = field(value, "array");
    const auto& sram = field(value, "sram");
    const auto& hbm = field(value, "hbm");
    Capacity capacity;
    const auto layout = string_field(sram, "layout");
    if (layout == "unified") {
        capacity.unified = integer_field(sram, "capacity_bytes");
    } else if (layout == "split") {
        capacity.is_split = true;
        capacity.split = {integer_field(sram, "activation_bytes"),
                          integer_field(sram, "weight_bytes"),
                          integer_field(sram, "accumulator_bytes")};
    } else {
        throw InputError("SRAM layout must be unified or split");
    }
    const auto buffering = string_field(hbm, "buffering");
    if (buffering != "single" && buffering != "double") {
        throw InputError("HBM buffering must be single or double");
    }
    Hardware result{integer_field(array, "rows"),
                    integer_field(array, "cols"),
                    giga_field(array, "clock_ghz"),
                    integer_field(array, "macs_per_pe_per_cycle"),
                    parse_precision(string_field(array, "precision")),
                    capacity,
                    giga_field(sram, "read_gbps"),
                    giga_field(sram, "write_gbps"),
                    giga_field(hbm, "bandwidth_gbps"),
                    integer_field(hbm, "latency_cycles", 0),
                    buffering == "double"};
    peak_macs_per_cycle(result);
    return result;
}

std::vector<Layer> parse_workload(const nlohmann::json& value) {
    const auto& layers = field(value, "layers");
    if (!layers.is_array() || layers.empty()) throw InputError("layers must be a nonempty array");
    std::set<std::string> names;
    std::vector<Layer> result;
    for (const auto& layer : layers) {
        Layer item{string_field(layer, "name"),
                   {integer_field(layer, "m"), integer_field(layer, "n"), integer_field(layer, "k")},
                   layer.contains("phase") ? string_field(layer, "phase") : std::string{},
                   layer.contains("batch") ? integer_field(layer, "batch") : 0,
                   layer.contains("sequence_length") ? integer_field(layer, "sequence_length") : 0};
        if (!names.insert(item.name).second) throw InputError("duplicate layer name: " + item.name);
        checked_mul(checked_mul(item.shape.m, item.shape.n), item.shape.k);
        result.push_back(std::move(item));
    }
    return result;
}

const char* name(Precision precision) {
    switch (precision) {
    case Precision::int8: return "INT8";
    case Precision::fp16: return "FP16";
    case Precision::bf16: return "BF16";
    }
    throw std::logic_error("invalid precision");
}

const char* name(Dataflow dataflow) { return dataflow == Dataflow::os ? "os" : "ws"; }
const char* name(Order order) { return order == Order::mnk ? "mnk" : "nmk"; }

}  // namespace tensys
