import json
from pathlib import Path

import pytest

from tensys_dse.experiments import Hardware, cases_for
from tensys_dse.metrics import Roofline, flatten


ROOT = Path(__file__).resolve().parents[2]


def test_experiments_have_unique_cases_and_reproducible_baseline():
    cases = cases_for("all")
    assert len(cases) == len({c.identifier for c in cases}) == 138
    assert {c.experiment for c in cases} == {"regimes", "dataflows", "buffering", "scaling"}
    assert {c.identifier for c in cases_for("smoke")} <= {c.identifier for c in cases}
    assert Hardware().config() == json.loads((ROOT / "examples/hardware.json").read_text())
    for case in cases_for("regimes"):
        assert case.shape[0] == case.batch * case.sequence_length
        assert (case.sequence_length == 1) == (case.phase == "decode")


def test_roofline_peak_ridge_and_both_slopes():
    hardware = {
        "array": {"peak_macs_per_cycle": 4096, "clock_hz": 10**9},
        "hbm": {"bandwidth_bytes_per_second": 64 * 10**9},
    }
    roofline = Roofline.from_hardware(hardware)
    assert roofline.peak == 8.192e12
    assert roofline.ridge == 128
    assert roofline.limit(10) == 6.4e11
    assert roofline.limit(512) == roofline.peak


def test_uses_actual_traffic_and_counts_both_mac_operations():
    metrics = {
        "useful_macs": 16, "total_cycles": 32, "compute_cycles": 8,
        "array_fill_drain_cycles": 4, "weight_load_cycles": 0, "sram_stall_cycles": 0,
        "hbm_stall_cycles": 20, "dma_cycles": 24, "dma_hidden_cycles": 4,
        "hbm": {"total_bytes": 20}, "sram": {"total_bytes": 80},
        "mac_utilization": 0.125, "stream_utilization": 0.5,
        "array_waves": 2, "tile_steps": 2,
    }
    row = flatten({
        "dataflow": "os", "experiment": "buffering", "workload": "control", "variant": "double",
        "hardware": {
            "array": {"rows": 2, "cols": 2, "precision": "INT8", "peak_macs_per_cycle": 4,
                      "clock_hz": 10**9},
            "hbm": {"bandwidth_bytes_per_second": 4 * 10**9, "buffering": "double"},
            "sram": {"layout": "split", "activation_bytes": 16, "weight_bytes": 16,
                     "accumulator_bytes": 32},
        },
        "layer": {
            "name": "control", "shape": {"m": 2, "n": 4, "k": 2},
            "mapping": {"tile": {"m": 2, "n": 2, "k": 2}, "order": "mnk"}, "metrics": metrics,
        },
    })
    assert row["sram_capacity_bytes"] == 64
    assert row["operational_intensity_ops_per_byte"] == 1.6
    assert row["throughput_ops_per_s"] == pytest.approx(1e9)
    assert row["dominant_cycle_component"] == "hbm_stall_cycles"
    assert row["roofline_side"] == "bandwidth"
