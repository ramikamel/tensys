#include "config/config.hpp"
#include "sim/simulator.hpp"
#include "sim/writeback.hpp"
#include "verify/verify.hpp"

#include <charconv>
#include <cstdio>
#include <iostream>
#include <memory>
#include <set>
#include <string_view>
#include <utility>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace {
using namespace tensys;
using Json = nlohmann::json;
using Path = std::filesystem::path;

constexpr std::string_view usage = R"(Usage:
  tensys simulate --hardware FILE --workload FILE --dataflow os|ws
                  [--tile Tm,Tn,Tk] [--order mnk|nmk] [--trace FILE]
  tensys verify   --hardware FILE --workload FILE --dataflow os|ws
                  [--tile Tm,Tn,Tk] [--order mnk|nmk] [--seed INTEGER]
  tensys dma-compare --hardware FILE --workload FILE --dataflow os|ws
                     [--tile Tm,Tn,Tk] [--order mnk|nmk] [--timeline FILE]
)";

enum class Command { simulate, verify, dma_compare };

struct Arguments {
    Command command{Command::simulate};
    Path hardware, workload;
    std::optional<Path> trace;
    std::optional<Dataflow> dataflow;
    std::optional<Shape> tile;
    std::optional<Order> order;
    Count seed{42};
};

Count parse_count(std::string_view value, const char* label) {
    Count result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw InputError(std::string(label) + " must be an unsigned integer");
    }
    return result;
}

Shape parse_tile(std::string_view value) {
    const auto first = value.find(',');
    const auto second = first == value.npos ? value.npos : value.find(',', first + 1);
    if (second == value.npos) throw InputError("--tile expects Tm,Tn,Tk");
    return {parse_count(value.substr(0, first), "Tm"),
            parse_count(value.substr(first + 1, second - first - 1), "Tn"),
            parse_count(value.substr(second + 1), "Tk")};
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    constexpr auto command_error = "expected simulate, verify, or dma-compare (see --help)";
    if (argc < 2) throw InputError(command_error);
    const std::string_view command(argv[1]);
    if (command == "verify") args.command = Command::verify;
    else if (command == "dma-compare") args.command = Command::dma_compare;
    else if (command != "simulate") throw InputError(command_error);

    std::set<std::string_view> seen;
    for (int i = 2; i < argc; i += 2) {
        const std::string_view flag(argv[i]);
        if (i + 1 == argc) throw InputError("missing value for " + std::string(flag));
        if (!seen.insert(flag).second) throw InputError("duplicate option: " + std::string(flag));
        const std::string_view value(argv[i + 1]);
        if (flag == "--hardware") args.hardware = value;
        else if (flag == "--workload") args.workload = value;
        else if (flag == "--tile") args.tile = parse_tile(value);
        else if (flag == "--dataflow") {
            if (value == "os") args.dataflow = Dataflow::os;
            else if (value == "ws") args.dataflow = Dataflow::ws;
            else throw InputError("--dataflow must be os or ws");
        } else if (flag == "--order") {
            if (value == "mnk") args.order = Order::mnk;
            else if (value == "nmk") args.order = Order::nmk;
            else throw InputError("--order must be mnk or nmk");
        } else if (!value.empty() &&
                   ((flag == "--trace" && args.command == Command::simulate) ||
                    (flag == "--timeline" && args.command == Command::dma_compare))) {
            args.trace = value;
        } else if (flag == "--seed" && args.command == Command::verify) {
            args.seed = parse_count(value, "--seed");
        } else {
            throw InputError("invalid option: " + std::string(flag));
        }
    }
    if (args.hardware.empty() || args.workload.empty() || !args.dataflow) {
        throw InputError("--hardware, --workload, and --dataflow are required");
    }
    return args;
}

class TraceFile {
public:
    explicit TraceFile(Path target) : destination(std::move(target)) {
        auto pattern = destination.string() + ".tmp.XXXXXX";
        const auto descriptor = mkstemp(pattern.data());
        if (descriptor == -1) throw std::runtime_error("cannot create trace: " + destination.string());
        temp = pattern;
        file = fdopen(descriptor, "w");
        if (!file) {
            close(descriptor);
            std::filesystem::remove(temp);
            throw std::runtime_error("cannot open trace: " + destination.string());
        }
    }
    TraceFile(const TraceFile&) = delete;
    TraceFile& operator=(const TraceFile&) = delete;
    ~TraceFile() {
        if (file) std::fclose(file);
        std::error_code ignored;
        if (!temp.empty()) std::filesystem::remove(temp, ignored);
    }
    void write(const Json& row) {
        const auto line = row.dump() + '\n';
        if (std::fwrite(line.data(), 1, line.size(), file) != line.size()) {
            throw std::runtime_error("cannot write trace: " + destination.string());
        }
    }
    void publish() {
        const auto status = std::fclose(file);
        file = nullptr;
        if (status != 0) throw std::runtime_error("cannot write trace: " + destination.string());
        std::filesystem::rename(temp, destination);
        temp.clear();
    }

private:
    Path destination, temp;
    std::FILE* file{};
};

Json shape_json(Shape shape) { return {{"m", shape.m}, {"n", shape.n}, {"k", shape.k}}; }

Json tensor_json(const TensorBytes& bytes) {
    return {{"activations", bytes.activations}, {"weights", bytes.weights}, {"outputs", bytes.outputs}};
}

Json traffic_json(const Traffic& traffic) {
    return {{"read", tensor_json(traffic.read)}, {"write", tensor_json(traffic.write)},
            {"total_bytes", traffic.total()}};
}

double utilization(Count useful_macs, Count peak, Count time) {
    return static_cast<double>(static_cast<long double>(useful_macs) / peak / time);
}

Json metrics_json(const Metrics& metrics, const Hardware& hardware) {
    const auto& cycles = metrics.cycles;
    const auto total = cycles.total();
    const auto peak = peak_macs_per_cycle(hardware);
    return {{"total_cycles", total},
            {"compute_cycles", cycles.compute},
            {"array_fill_drain_cycles", cycles.fill_drain},
            {"weight_load_cycles", cycles.weight_load},
            {"hbm_stall_cycles", cycles.hbm_stall},
            {"sram_stall_cycles", cycles.sram_stall},
            {"dma_cycles", metrics.dma_cycles},
            {"dma_hidden_cycles", metrics.dma_hidden_cycles},
            {"hbm", traffic_json(metrics.hbm)},
            {"sram", traffic_json(metrics.sram)},
            {"useful_macs", metrics.useful_macs},
            {"mac_utilization", utilization(metrics.useful_macs, peak, total)},
            {"stream_utilization", utilization(metrics.useful_macs, peak, cycles.compute)},
            {"array_waves", metrics.waves},
            {"tile_steps", metrics.steps}};
}

Json hardware_json(const Hardware& hardware) {
    const auto& capacity = hardware.capacity;
    Json sram = {{"read_bytes_per_second", hardware.sram_read_bps},
                 {"write_bytes_per_second", hardware.sram_write_bps}};
    if (capacity.is_split) {
        sram["layout"] = "split";
        sram["activation_bytes"] = capacity.split[0];
        sram["weight_bytes"] = capacity.split[1];
        sram["accumulator_bytes"] = capacity.split[2];
    } else {
        sram["layout"] = "unified";
        sram["capacity_bytes"] = capacity.unified;
    }
    return {{"array", {{"rows", hardware.rows}, {"cols", hardware.cols},
                       {"clock_hz", hardware.clock_hz}, {"precision", name(hardware.precision)},
                       {"macs_per_pe_per_cycle", hardware.macs_per_pe_per_cycle},
                       {"peak_macs_per_cycle", peak_macs_per_cycle(hardware)},
                       {"input_bytes_per_element", input_bytes(hardware.precision)},
                       {"accumulator_bytes_per_element", accumulator_bytes}}},
            {"sram", std::move(sram)},
            {"hbm", {{"bandwidth_bytes_per_second", hardware.hbm_bps},
                     {"latency_cycles", hardware.hbm_latency_cycles},
                     {"buffering", hardware.double_buffered ? "double" : "single"}}}};
}

Json layer_json(const Layer& layer, Mapping mapping) {
    Json result{{"name", layer.name},
                {"shape", shape_json(layer.shape)},
                {"mapping", {{"tile", shape_json(mapping.tile)}, {"order", name(mapping.order)}}}};
    if (!layer.phase.empty()) result["phase"] = layer.phase;
    if (layer.batch != 0) result["batch"] = layer.batch;
    if (layer.sequence_length != 0) result["sequence_length"] = layer.sequence_length;
    return result;
}

const char* operation_name(OperationKind operation) {
    switch (operation) {
    case OperationKind::array: return "array";
    case OperationKind::input_fetch: return "input_fetch";
    case OperationKind::output_write: return "output_write";
    }
    throw std::logic_error("unknown timeline operation");
}

Json dma_json(const DmaTiming& dma) {
    return {{"input_service_cycles", dma.input_service},
            {"output_service_cycles", dma.output_service},
            {"input_hidden_cycles", dma.input_hidden},
            {"output_hidden_cycles", dma.output_hidden}};
}

Json comparison_json(const Hardware& hardware, const Metrics& baseline, const WritebackResult& writeback) {
    const auto before = baseline.cycles.total();
    const auto after = writeback.metrics.cycles.total();
    const auto delta = after >= before ? static_cast<double>(after - before)
                                       : -static_cast<double>(before - after);
    auto prefetch_metrics = metrics_json(baseline, hardware);
    // Transfer costs are shared, and the baseline only overlaps input fetches.
    prefetch_metrics["dma_timing"] = dma_json(
        {writeback.dma.input_service, writeback.dma.output_service, baseline.dma_hidden_cycles, 0});
    auto writeback_metrics = metrics_json(writeback.metrics, hardware);
    writeback_metrics["dma_timing"] = dma_json(writeback.dma);
    writeback_metrics["exposed_wait"] = {
        {"input_cycles", writeback.wait.input},
        {"accumulator_bank_cycles", writeback.wait.accumulator},
        {"final_writeback_cycles", writeback.wait.drain}};
    return {{"baseline", std::move(prefetch_metrics)}, {"writeback", std::move(writeback_metrics)},
            {"latency_change_pct", 100.0 * delta / static_cast<double>(before)}};
}

Json run_comparison(const Arguments& args, const Hardware& hardware, const std::vector<Layer>& layers,
                    TraceFile* timeline) {
    const auto dataflow = *args.dataflow;
    Metrics baseline_totals;
    WritebackResult writeback_totals;
    Json results = Json::array();
    for (const auto& layer : layers) {
        const auto mapping = select_mapping(hardware, layer.shape, dataflow, args.tile, args.order);
        const Schedule schedule(layer.shape, mapping);
        TimelineSink sink;
        if (timeline) {
            sink = [&](const TimelineEvent& event) {
                const auto bank = event.accumulator_bank;
                timeline->write({{"schema_version", 1}, {"layer", layer.name}, {"cycle", event.cycle},
                                 {"operation", operation_name(event.operation)},
                                 {"phase", event.phase == EventPhase::start ? "start" : "finish"},
                                 {"index", shape_json(event.tile)},
                                 {"accumulator_bank", bank ? Json(*bank) : Json(nullptr)}});
            };
        }
        const auto baseline = simulate(hardware, schedule, dataflow);
        const auto writeback = simulate_writeback(hardware, schedule, dataflow, sink);
        baseline_totals.add(baseline);
        writeback_totals.add(writeback);
        auto result = layer_json(layer, mapping);
        result.update(comparison_json(hardware, baseline, writeback));
        results.push_back(std::move(result));
    }
    return {{"schema_version", 1}, {"command", "dma-compare"}, {"dataflow", name(dataflow)},
            {"hardware", hardware_json(hardware)}, {"layers", std::move(results)},
            {"totals", comparison_json(hardware, baseline_totals, writeback_totals)}};
}

int run(const Arguments& args) {
    const auto hardware = parse_hardware(read_json(args.hardware));
    const auto layers = parse_workload(read_json(args.workload));
    const auto dataflow = *args.dataflow;
    if (args.command == Command::dma_compare && !hardware.double_buffered)
        throw InputError("dma-compare requires double buffering");
    std::unique_ptr<TraceFile> trace;
    if (args.trace) {
        if (std::filesystem::exists(*args.trace) &&
            (std::filesystem::equivalent(*args.trace, args.hardware) ||
             std::filesystem::equivalent(*args.trace, args.workload))) {
            throw InputError("trace must be a separate file from the hardware and workload");
        }
        trace = std::make_unique<TraceFile>(*args.trace);
    }

    if (args.command == Command::dma_compare) {
        const auto serialized = run_comparison(args, hardware, layers, trace.get()).dump();
        if (trace) trace->publish();
        std::cout << serialized << '\n';
        return 0;
    }

    const bool verifying = args.command == Command::verify;
    Json results = Json::array();
    Metrics totals;
    bool passed = true;
    for (const auto& layer : layers) {
        if (verifying && std::max({layer.shape.m, layer.shape.n, layer.shape.k}) >
                               verification_dimension_limit) {
            throw InputError(layer.name + ": verification requires each dimension to be at most 128");
        }
        const auto mapping = select_mapping(hardware, layer.shape, dataflow, args.tile, args.order);
        const Schedule schedule(layer.shape, mapping);
        TraceSink sink;
        if (trace) {
            sink = [&](const TraceStep& step) {
                const auto& [m, n, k] = step.tile.region;
                trace->write({{"schema_version", 1}, {"layer", layer.name},
                              {"index", shape_json(step.tile.index)},
                              {"origin", shape_json({m.start, n.start, k.start})},
                              {"shape", shape_json({m.size, n.size, k.size})},
                              {"next_input", step.next_input ? shape_json(*step.next_input) : Json(nullptr)},
                              {"writes_output", step.tile.last_k},
                              {"metrics", metrics_json(step.metrics, hardware)}});
            };
        }
        const auto metrics = simulate(hardware, schedule, dataflow, sink);
        totals.add(metrics);
        auto result = layer_json(layer, mapping);
        result["metrics"] = metrics_json(metrics, hardware);
        if (verifying) {
            const auto check = verify(hardware, schedule, dataflow, args.seed);
            passed &= !check.first_failure;
            result["verification"] = {
                {"passed", !check.first_failure},
                {"seed", check.seed},
                {"checked_elements", check.checked_elements},
                {"max_abs_error", check.max_abs_error},
                {"first_failure", check.first_failure ? Json(*check.first_failure) : Json(nullptr)}};
        }
        results.push_back(std::move(result));
    }
    Json output{{"schema_version", 1}, {"command", verifying ? "verify" : "simulate"},
                {"dataflow", name(dataflow)}, {"hardware", hardware_json(hardware)},
                {"layers", std::move(results)}, {"totals", metrics_json(totals, hardware)}};
    if (verifying) output["passed"] = passed;
    const auto serialized = output.dump();
    if (trace) trace->publish();
    std::cout << serialized << '\n';
    return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "TenSys " << TENSYS_VERSION << '\n' << usage;
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "tensys " << TENSYS_VERSION << '\n';
        return 0;
    }
    try {
        return run(parse_arguments(argc, argv));
    } catch (const InputError& error) {
        std::cerr << "tensys: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "tensys: " << error.what() << '\n';
        return 1;
    }
}
