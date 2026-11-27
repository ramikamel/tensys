from dataclasses import dataclass, replace
from typing import Literal

Flow = Literal["os", "ws"]
Shape = tuple[int, int, int]
FLOWS: tuple[Flow, ...] = ("os", "ws")
EXPERIMENTS = ("regimes", "dataflows", "buffering", "scaling")
KIB = 1024
MIB = KIB * KIB


@dataclass(frozen=True)
class Hardware:
    rows: int = 64
    cols: int = 64
    precision: str = "BF16"
    capacity_bytes: int = 8 * MIB
    sram_gbps: int = 2048
    hbm_gbps: int = 64
    hbm_latency_cycles: int = 200
    buffering: str = "single"

    def config(self) -> dict:
        return {
            "array": {
                "rows": self.rows,
                "cols": self.cols,
                "clock_ghz": 1,
                "macs_per_pe_per_cycle": 1,
                "precision": self.precision,
            },
            "sram": {
                "layout": "unified",
                "capacity_bytes": self.capacity_bytes,
                "read_gbps": self.sram_gbps,
                "write_gbps": self.sram_gbps,
            },
            "hbm": {
                "bandwidth_gbps": self.hbm_gbps,
                "latency_cycles": self.hbm_latency_cycles,
                "buffering": self.buffering,
            },
        }


@dataclass(frozen=True)
class Case:
    identifier: str
    experiment: str
    workload: str
    variant: str
    shape: Shape
    dataflow: Flow
    hardware: Hardware = Hardware()
    tile: Shape | None = None
    order: str | None = None
    phase: str | None = None
    batch: int | None = None
    sequence_length: int | None = None

    def layer(self) -> dict:
        m, n, k = self.shape
        result = {"name": self.identifier, "m": m, "n": n, "k": k}
        if self.phase is not None:
            result.update(
                phase=self.phase, batch=self.batch, sequence_length=self.sequence_length
            )
        return result


PROJECTIONS = {
    "fused_qkv": (6144, 4096),
    "attention_output": (4096, 4096),
    "ffn_gate": (14336, 4096),
    "ffn_up": (14336, 4096),
    "ffn_down": (4096, 14336),
}
TOKEN_COUNTS = (1, 8, 32, 128, 512, 2048, 4096)


def regime_cases() -> list[Case]:
    cases = []
    for m in TOKEN_COUNTS:
        phase = "decode" if m <= 32 else "prefill"
        batch, sequence_length = (m, 1) if phase == "decode" else (1, m)
        for workload, (n, k) in PROJECTIONS.items():
            for flow in FLOWS:
                cases.append(
                    Case(
                        f"regime_{workload}_{m}_{flow}", "regimes", workload, "auto",
                        (m, n, k), flow, phase=phase, batch=batch,
                        sequence_length=sequence_length,
                    )
                )
    return cases


def dataflow_cases() -> list[Case]:
    shapes = {"tall_skinny": (2048, 128, 2048), "square": (512, 512, 512)}
    return [
        Case(
            f"dataflow_{workload}_{variant}_{flow}", "dataflows", workload, variant,
            shape, flow, tile=(64, 64, 64) if variant == "fixed" else None,
            order="mnk" if variant == "fixed" else None,
        )
        for workload, shape in shapes.items()
        for variant in ("fixed", "auto")
        for flow in FLOWS
    ]


def buffering_cases() -> list[Case]:
    tiny = Hardware(rows=2, cols=2, precision="INT8", capacity_bytes=4096,
                    sram_gbps=64, hbm_gbps=4, hbm_latency_cycles=3)
    controls = [
        Case(
            f"buffer_overlap_{mode}", "buffering", "fixed_tile_control", mode,
            (2, 4, 2), "os", replace(tiny, buffering=mode), (2, 2, 2), "mnk",
        )
        for mode in ("single", "double")
    ]
    capacity_control = replace(tiny, rows=4, cols=4, capacity_bytes=896)
    controls += [
        Case(
            f"buffer_capacity_{mode}", "buffering", "capacity_control", mode,
            (16, 16, 32), "os", replace(capacity_control, buffering=mode),
            (16, 4, 32) if mode == "single" else None,
            "mnk" if mode == "single" else None,
        )
        for mode in ("single", "double")
    ]
    return controls + [
        Case(
            f"buffer_projection_{m}_{capacity}_{flow}_{mode}", "buffering",
            "attention_output", mode, (m, *PROJECTIONS["attention_output"]), flow,
            Hardware(capacity_bytes=capacity, buffering=mode),
        )
        for m in (128, 512)
        for capacity in (256 * KIB, MIB, 8 * MIB)
        for flow in FLOWS
        for mode in ("single", "double")
    ]


def scaling_cases() -> list[Case]:
    return [
        Case(
            f"scaling_{m}_{width}_{flow}", "scaling", "attention_output", "auto",
            (m, *PROJECTIONS["attention_output"]), flow, Hardware(rows=width, cols=width),
        )
        for m in (1, 32, 512, 2048)
        for width in (16, 32, 64, 128)
        for flow in FLOWS
    ]


def writeback_cases() -> list[Case]:
    shapes = [
        ("attention_output", (m, *PROJECTIONS["attention_output"]), (min(m, 128), 256, 512))
        for m in (1, 32, 128, 512, 2048)
    ] + [("output_heavy", (512, 1024, 128), (128, 256, 128))]
    return [
        Case(
            f"writeback_{workload}_{shape[0]}_{bandwidth}_{flow}",
            "writeback", workload, "fixed", shape, flow,
            Hardware(hbm_gbps=bandwidth, buffering="double"), tile, "mnk",
        )
        for workload, shape, tile in shapes
        for bandwidth in (32, 64, 128)
        for flow in FLOWS
    ]


def cases_for(suite: str) -> list[Case]:
    subsets = dict(zip(
        EXPERIMENTS, (regime_cases, dataflow_cases, buffering_cases, scaling_cases), strict=True
    ))
    if suite in subsets:
        return subsets[suite]()
    if suite == "all":
        return [case for build_cases in subsets.values() for case in build_cases()]
    if suite == "writeback":
        return writeback_cases()
    if suite == "smoke":
        return (
            [c for c in regime_cases() if c.workload == "attention_output" and c.shape[0] in (1, 512)]
            + [c for c in dataflow_cases() if c.workload == "tall_skinny"]
            + [c for c in buffering_cases() if c.workload.endswith("_control")
               or (c.shape[0] == 128 and c.hardware.capacity_bytes == MIB)]
            + [c for c in scaling_cases() if c.shape[0] in (1, 512) and c.hardware.rows in (16, 64)]
        )
    raise ValueError(f"unknown suite: {suite}")
