import json
import os
import subprocess
from pathlib import Path

import pytest

from tensys_dse.experiments import Case, Hardware, cases_for
from tensys_dse.metrics import CYCLE_FIELDS, flatten, flatten_writeback
from tensys_dse.runner import run_cli, simulate_cases


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def simulator() -> Path:
    configured = os.environ.get("TENSYS_BIN")
    path = Path(configured).resolve() if configured else ROOT / "build/release/tensys"
    if not path.is_file():
        if configured:
            pytest.fail(f"TENSYS_BIN does not exist: {path}")
        pytest.skip("build the Release simulator or set TENSYS_BIN")
    return path


def test_verifier_and_trace_match_summary(simulator, tmp_path):
    common = [
        "--hardware", str(ROOT / "examples/hardware.json"),
        "--workload", str(ROOT / "examples/verify.json"),
        "--dataflow", "ws", "--tile", "5,7,6", "--order", "nmk",
    ]
    checked = subprocess.run([simulator, "verify", *common, "--seed", "123"],
                             capture_output=True, text=True, check=True)
    verified = json.loads(checked.stdout)
    assert verified["passed"] and verified["layers"][0]["verification"]["seed"] == 123

    trace = tmp_path / "schedule.ndjson"
    completed = subprocess.run([simulator, "simulate", *common, "--trace", trace],
                               capture_output=True, text=True, check=True)
    simulated = json.loads(completed.stdout)
    assert simulated["schema_version"] == 1
    assert simulated["totals"] == verified["totals"]
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    totals = simulated["totals"]
    assert len(rows) == totals["tile_steps"] == 12
    for field in (*CYCLE_FIELDS, "total_cycles", "dma_cycles", "dma_hidden_cycles", "useful_macs"):
        assert sum(row["metrics"][field] for row in rows) == totals[field]
    for memory in ("hbm", "sram"):
        assert sum(row["metrics"][memory]["total_bytes"] for row in rows) == totals[memory]["total_bytes"]
    assert all(row["next_input"] == next_row["index"] for row, next_row in zip(rows, rows[1:]))
    assert rows[-1]["next_input"] is None and rows[-1]["writes_output"]


def test_failed_trace_request_keeps_previous_output(simulator, tmp_path):
    hardware = tmp_path / "hardware.json"
    spec = Hardware(rows=2, cols=2, precision="INT8").config()
    spec["sram"]["capacity_bytes"] = 24
    hardware.write_text(json.dumps(spec))
    workload = tmp_path / "workload.json"
    workload.write_text('{"layers":[{"name":"tiny","m":2,"n":2,"k":2}]}')
    trace = tmp_path / "schedule.ndjson"
    trace.write_text("previous contents\n")

    result = subprocess.run([
        simulator, "simulate", "--hardware", hardware, "--workload", workload,
        "--dataflow", "os", "--tile", "4,4,4", "--trace", trace,
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout and "does not fit" in result.stderr
    assert trace.read_text() == "previous contents\n"
    assert not list(tmp_path.glob("*.tmp.*"))


def test_split_sram_and_clipped_manual_tile(simulator, tmp_path):
    hardware = tmp_path / "hardware.json"
    spec = Hardware(rows=2, cols=2, precision="INT8", buffering="double").config()
    spec["sram"] = {
        "layout": "split", "activation_bytes": 16, "weight_bytes": 8, "accumulator_bytes": 64,
        "read_gbps": 64, "write_gbps": 64,
    }
    hardware.write_text(json.dumps(spec))
    workload = tmp_path / "workload.json"
    workload.write_text('{"layers":[{"name":"tiny","m":2,"n":2,"k":2}]}')
    result = subprocess.run([
        simulator, "verify", "--hardware", hardware, "--workload", workload,
        "--dataflow", "os", "--tile", "4,2,2",
    ], capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["passed"]
    assert payload["layers"][0]["mapping"]["tile"] == {"m": 4, "n": 2, "k": 2}
    assert payload["totals"]["hbm"]["total_bytes"] == 24


def test_python_batching_preserves_case_order_and_units(simulator, tmp_path):
    cases = cases_for("smoke")[:4]
    raw = simulate_cases(simulator, cases, tmp_path)
    assert [case.identifier for case in cases] == [item["layer"]["name"] for item in raw]
    for item in raw:
        row = flatten(item)
        assert row["throughput_ops_per_s"] <= row["roofline_limit_ops_per_s"] * (1 + 1e-12)
        assert sum(row[field] for field in CYCLE_FIELDS) == row["total_cycles"]
        assert 0 < row["mac_utilization"] <= row["stream_utilization"] <= 1.0


def test_dma_compare_reports_overlap_and_streams_timeline(simulator, tmp_path):
    hardware = Hardware(rows=2, cols=2, precision="INT8", sram_gbps=64,
                        hbm_gbps=4, hbm_latency_cycles=3, buffering="double")
    case = Case("tiny", "writeback", "control", "fixed", (2, 4, 2), "os",
                hardware, (2, 2, 2), "mnk")
    timeline = tmp_path / "clock.ndjson"
    response = run_cli(simulator, [case], tmp_path, operation="dma-compare", timeline=timeline)
    layer = response["layers"][0]
    assert layer["baseline"]["total_cycles"] == 31
    assert layer["writeback"]["total_cycles"] == 25
    assert layer["baseline"]["hbm"] == layer["writeback"]["hbm"]
    assert response["totals"]["latency_change_pct"] == pytest.approx(-600 / 31)
    events = [json.loads(line) for line in timeline.read_text().splitlines()]
    assert all(e["schema_version"] == 1 and e["layer"] == "tiny" for e in events)
    assert [(e["cycle"], e["operation"], e["phase"], e["accumulator_bank"]) for e in events] == [
        (0, "input_fetch", "start", None),
        (5, "input_fetch", "finish", None),
        (5, "array", "start", 0),
        (5, "input_fetch", "start", None),
        (9, "input_fetch", "finish", None),
        (11, "array", "finish", 0),
        (11, "array", "start", 1),
        (11, "output_write", "start", 0),
        (17, "array", "finish", 1),
        (18, "output_write", "finish", 0),
        (18, "output_write", "start", 1),
        (25, "output_write", "finish", 1),
    ]


def test_dma_compare_requires_double_buffering(simulator):
    result = subprocess.run([
        simulator, "dma-compare", "--hardware", ROOT / "examples/hardware.json",
        "--workload", ROOT / "examples/verify.json", "--dataflow", "os",
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout and "requires double buffering" in result.stderr


def test_writeback_cases_export_matched_schedules(simulator, tmp_path):
    cases = cases_for("writeback")
    assert len(cases) == len({c.identifier for c in cases}) == 36
    assert all(c.tile and c.order == "mnk" and c.hardware.buffering == "double" for c in cases)
    sample = cases[:6]
    entries = simulate_cases(simulator, sample, tmp_path, operation="dma-compare")
    for entry, case in zip(entries, sample, strict=True):
        row = flatten_writeback(entry)
        assert row["case"] == case.identifier
        assert entry["layer"]["baseline"]["hbm"] == entry["layer"]["writeback"]["hbm"]
        assert row["writeback_hbm_stall_cycles"] == (
            row["input_cycles"] + row["accumulator_bank_cycles"] + row["final_writeback_cycles"]
        )
