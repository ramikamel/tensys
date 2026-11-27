from dataclasses import dataclass


CYCLE_FIELDS = (
    "compute_cycles", "array_fill_drain_cycles", "weight_load_cycles",
    "sram_stall_cycles", "hbm_stall_cycles",
)


@dataclass(frozen=True)
class Roofline:
    peak: float
    bandwidth: float

    @classmethod
    def from_hardware(cls, hardware: dict) -> "Roofline":
        array = hardware["array"]
        return cls(
            2 * array["peak_macs_per_cycle"] * array["clock_hz"],
            hardware["hbm"]["bandwidth_bytes_per_second"],
        )

    @property
    def ridge(self) -> float:
        return self.peak / self.bandwidth

    def limit(self, operational_intensity: float) -> float:
        return min(self.peak, self.bandwidth * operational_intensity)


def flatten(entry: dict) -> dict:
    hardware = entry["hardware"]
    layer = entry["layer"]
    array = hardware["array"]
    sram = hardware["sram"]
    metrics = layer["metrics"]
    roofline = Roofline.from_hardware(hardware)
    latency = metrics["total_cycles"] / array["clock_hz"]
    useful_ops = 2 * metrics["useful_macs"]
    intensity = useful_ops / metrics["hbm"]["total_bytes"]
    throughput = useful_ops / latency
    if intensity == roofline.ridge:
        side = "ridge"
    else:
        side = "bandwidth" if intensity < roofline.ridge else "compute"
    return {
        "case": layer["name"],
        "experiment": entry["experiment"],
        "workload": entry["workload"],
        "variant": entry["variant"],
        "phase": layer.get("phase", ""),
        "batch": layer.get("batch", ""),
        "sequence_length": layer.get("sequence_length", ""),
        "dataflow": entry["dataflow"],
        "precision": array["precision"],
        "array_rows": array["rows"],
        "array_cols": array["cols"],
        "clock_hz": array["clock_hz"],
        "sram_capacity_bytes": sram["capacity_bytes"] if sram["layout"] == "unified" else sum(
            sram[key] for key in ("activation_bytes", "weight_bytes", "accumulator_bytes")
        ),
        "hbm_bandwidth_bps": roofline.bandwidth,
        "buffering": hardware["hbm"]["buffering"],
        **layer["shape"],
        "tile_m": layer["mapping"]["tile"]["m"],
        "tile_n": layer["mapping"]["tile"]["n"],
        "tile_k": layer["mapping"]["tile"]["k"],
        "order": layer["mapping"]["order"],
        "tile_steps": metrics["tile_steps"],
        "array_waves": metrics["array_waves"],
        "useful_macs": metrics["useful_macs"],
        "total_cycles": metrics["total_cycles"],
        **{field: metrics[field] for field in CYCLE_FIELDS},
        "dma_cycles": metrics["dma_cycles"],
        "dma_hidden_cycles": metrics["dma_hidden_cycles"],
        "hbm_bytes": metrics["hbm"]["total_bytes"],
        "sram_bytes": metrics["sram"]["total_bytes"],
        "latency_s": latency,
        "mac_utilization": metrics["mac_utilization"],
        "stream_utilization": metrics["stream_utilization"],
        "peak_ops_per_s": roofline.peak,
        "operational_intensity_ops_per_byte": intensity,
        "throughput_ops_per_s": throughput,
        "ridge_ops_per_byte": roofline.ridge,
        "roofline_limit_ops_per_s": roofline.limit(intensity),
        "roofline_side": side,
        "dominant_cycle_component": max(CYCLE_FIELDS, key=metrics.__getitem__),
    }


def flatten_writeback(entry: dict) -> dict:
    hardware = entry["hardware"]
    layer = entry["layer"]
    baseline = layer["baseline"]
    writeback = layer["writeback"]
    tile = layer["mapping"]["tile"]
    return {
        "case": layer["name"],
        "experiment": entry["experiment"],
        "workload": entry["workload"],
        "dataflow": entry["dataflow"],
        **layer["shape"],
        "tile_m": tile["m"],
        "tile_n": tile["n"],
        "tile_k": tile["k"],
        "order": layer["mapping"]["order"],
        "hbm_bandwidth_bps": hardware["hbm"]["bandwidth_bytes_per_second"],
        "hbm_bytes": baseline["hbm"]["total_bytes"],
        "sram_bytes": baseline["sram"]["total_bytes"],
        "baseline_total_cycles": baseline["total_cycles"],
        "writeback_total_cycles": writeback["total_cycles"],
        "latency_change_pct": layer["latency_change_pct"],
        "baseline_hbm_stall_cycles": baseline["hbm_stall_cycles"],
        "writeback_hbm_stall_cycles": writeback["hbm_stall_cycles"],
        "baseline_input_hidden_cycles": baseline["dma_timing"]["input_hidden_cycles"],
        **writeback["dma_timing"],
        **writeback["exposed_wait"],
    }
