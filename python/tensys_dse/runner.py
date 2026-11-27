import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from importlib.metadata import version
from math import prod
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from .experiments import Case, cases_for
from .metrics import flatten, flatten_writeback

ROOT = Path(__file__).resolve().parents[2]


def run_cli(simulator: Path, cases: list[Case], folder: Path, *,
            operation: str = "simulate", timeline: Path | None = None) -> dict:
    case = cases[0]
    hardware = folder / "hardware.json"
    workload = folder / "workload.json"
    hardware.write_text(json.dumps(case.hardware.config()), encoding="utf-8")
    workload.write_text(json.dumps({"layers": [item.layer() for item in cases]}), encoding="utf-8")
    command = [
        str(simulator), operation, "--hardware", str(hardware), "--workload",
        str(workload), "--dataflow", case.dataflow,
    ]
    if case.tile is not None:
        command.extend(("--tile", ",".join(map(str, case.tile))))
    if case.order is not None:
        command.extend(("--order", case.order))
    if timeline is not None:
        command.extend(("--timeline", str(timeline)))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        raise RuntimeError(f"{case.identifier}: simulator exited {completed.returncode}: "
                           f"{completed.stderr.strip()}")
    response = json.loads(completed.stdout)
    if (response["schema_version"] != 1 or response["dataflow"] != case.dataflow
            or response["command"] != operation):
        raise RuntimeError("unexpected simulator response version, command, or dataflow")
    expected = [item.identifier for item in cases]
    actual = [item["name"] for item in response["layers"]]
    if actual != expected:
        raise RuntimeError(f"{case.identifier}: layers in response differ from the request")
    return response


def simulate_cases(simulator: Path, cases: list[Case], folder: Path, *,
                   operation: str = "simulate") -> list[dict]:
    groups = defaultdict(list)
    for case in cases:
        groups[case.hardware, case.dataflow, case.tile, case.order].append(case)

    entries = {}
    for group in groups.values():
        response = run_cli(simulator, group, folder, operation=operation)
        for case, layer in zip(group, response["layers"], strict=True):
            entries[case.identifier] = {
                "experiment": case.experiment,
                "workload": case.workload,
                "variant": case.variant,
                "dataflow": case.dataflow,
                "hardware": response["hardware"],
                "layer": layer,
            }
    return [entries[case.identifier] for case in cases]


def file_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def source_digest() -> str:
    sources = set((ROOT / "src").rglob("*.cpp")) | set((ROOT / "src").rglob("*.hpp"))
    sources.update((ROOT / "python" / "tensys_dse").glob("*.py"))
    sources.update(ROOT / name for name in ("CMakeLists.txt", "CMakePresets.json", "requirements.txt"))
    digest = hashlib.sha256()
    for source in sorted(sources):
        digest.update(f"{source.relative_to(ROOT)}\0{file_digest(source)}\n".encode())
    return digest.hexdigest()


def git_state() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()

    try:
        return {"head": git("rev-parse", "HEAD"), "working_tree_dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "working_tree_dirty": None}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(simulator: Path, suite: str, output: Path) -> dict:
    started = time.perf_counter()
    from . import plots

    simulator = simulator.resolve(strict=True)
    cases = cases_for(suite)
    comparing = suite == "writeback"
    operation = "dma-compare" if comparing else "simulate"
    with TemporaryDirectory(prefix="tensys-") as temporary:
        folder = Path(temporary)
        simulation_start = time.perf_counter()
        entries = simulate_cases(simulator, cases, folder, operation=operation)
        simulation_seconds = time.perf_counter() - simulation_start

        largest = max(cases, key=lambda case: (prod(case.shape), case.identifier))
        case_start = time.perf_counter()
        run_cli(simulator, [largest], folder, operation=operation)
        largest_case_seconds = time.perf_counter() - case_start

    format_row = flatten_writeback if comparing else flatten
    rows = [format_row(entry) for entry in entries]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "raw.json", {"schema_version": 1, "cases": entries})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    artifacts = plots.render(rows, output)
    manifest = {
        "schema_version": 1,
        "suite": suite,
        "case_count": len(cases),
        "simulator": {
            "version": subprocess.check_output([str(simulator), "--version"], text=True).strip(),
            "binary_sha256": file_digest(simulator),
            "source_sha256": source_digest(),
            "git": git_state(),
        },
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": sys.version.split()[0], "numpy": version("numpy"),
                 "matplotlib": version("matplotlib")},
        "seconds": {
            "end_to_end": time.perf_counter() - started,
            "simulate_all_cases": simulation_seconds,
            "largest_case": largest_case_seconds,
        },
        "largest_case": {"identifier": largest.identifier, "shape": largest.shape},
        "artifacts": ["raw.json", "metrics.csv", *artifacts],
        "cases": [asdict(case) for case in cases],
    }
    if comparing:
        changes = [row["latency_change_pct"] for row in rows]
        manifest["summary"] = {
            "median_latency_change_pct": median(changes),
            "improved_cases": sum(change < 0 for change in changes),
            "regressed_cases": sum(change > 0 for change in changes),
            "tied_cases": changes.count(0),
        }
    write_json(output / "manifest.json", manifest)
    return manifest
