from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from .experiments import FLOWS, KIB
from .metrics import CYCLE_FIELDS, Roofline

FLOW_COLORS = {"os": "#007c83", "ws": "#a94a1e"}
MODE_COLORS = {"single": "#64748b", "double": "#007c83"}
TOKEN_COLORS = ("#007c83", "#3c58a3", "#943c75", "#a94a1e")
STAGE_COLORS = ("#007c83", "#8ba9ae", "#7d54a4", "#d2a337", "#bc5040")
STAGE_LABELS = ("Compute", "Fill / drain", "Weight load", "SRAM wait", "HBM wait")


def save(fig, folder: Path, name: str) -> str:
    fig.savefig(folder / name, dpi=160, facecolor="white")
    plt.close(fig)
    return name


def roofline(rows: list[dict], folder: Path) -> str:
    bandwidth = rows[0]["hbm_bandwidth_bps"]
    ceiling = Roofline(rows[0]["peak_ops_per_s"], bandwidth)
    intensity = [row["operational_intensity_ops_per_byte"] for row in rows]
    xs = np.geomspace(min(intensity) / 1.35, max(max(intensity) * 1.4, 2 * ceiling.ridge), 256)
    colors = plt.get_cmap("viridis")
    tokens = sorted({row["m"] for row in rows})
    norm = LogNorm(vmin=tokens[0], vmax=tokens[-1])
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10.8, 4.7),
                             layout="constrained")
    fig.suptitle("Projection roofline · 64 × 64 array, BF16")
    for flow, axis in zip(FLOWS, axes, strict=True):
        axis.loglog(xs, np.minimum(ceiling.peak, xs * bandwidth) / 1e12,
                    color="#172b42", linewidth=1.8)
        axis.axvline(ceiling.ridge, linewidth=1, linestyle=":", color="#64748b")
        axis.text(ceiling.ridge * 1.07, 0.36, f"Ridge  {ceiling.ridge:g} FLOPs/B",
                  transform=axis.get_xaxis_transform(), fontsize=8, color="#475569")
        for phase, marker in (("decode", "o"), ("prefill", "^")):
            points = [row for row in rows if row["dataflow"] == flow and row["phase"] == phase]
            axis.scatter(
                [p["operational_intensity_ops_per_byte"] for p in points],
                [p["throughput_ops_per_s"] / 1e12 for p in points],
                c=[p["m"] for p in points], cmap=colors, norm=norm, marker=marker, s=33,
                edgecolors="white", linewidths=0.35, label=phase.capitalize(), zorder=3,
            )
        axis.set_title("Output-stationary" if flow == "os" else "Weight-stationary")
        axis.set_xlabel("Operational intensity (FLOPs / HBM byte)")
        axis.grid(True, which="major")
    axes[0].set_ylabel("Achieved performance (TFLOP/s)")
    axes[0].legend(loc="lower right")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colors), ax=axes, shrink=0.82,
                       ticks=tokens, format="%d")
    bar.set_label("Rows processed together (M)")
    return save(fig, folder, "01_roofline.png")


def regime_cycles(rows: list[dict], folder: Path) -> str:
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10.6, 4.4), layout="constrained")
    fig.suptitle("Where cycles go · attention output projection")
    for flow, axis in zip(FLOWS, axes, strict=True):
        points = sorted((r for r in rows if r["workload"] == "attention_output"
                         and r["dataflow"] == flow), key=lambda row: row["m"])
        xs = np.arange(len(points))
        bottom = np.zeros(len(points))
        for field, label, color in zip(CYCLE_FIELDS, STAGE_LABELS, STAGE_COLORS, strict=True):
            share = np.array([r[field] / r["total_cycles"] for r in points])
            axis.bar(xs, share, bottom=bottom, color=color, width=0.72, label=label)
            bottom += share
        axis.set_title(flow.upper())
        axis.set_xticks(xs, [str(row["m"]) for row in points])
        axis.set_xlabel("Rows processed together (M)")
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[0].set_ylabel("Share of modeled cycles")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=5)
    return save(fig, folder, "02_regime_cycles.png")


def dataflows(rows: list[dict], folder: Path) -> str:
    keys = list(dict.fromkeys((row["workload"], row["variant"]) for row in rows))
    labels = [f"{shape.replace('_', ' ').capitalize()}\n{variant}" for shape, variant in keys]
    by_key = {(r["workload"], r["variant"], r["dataflow"]): r for r in rows}
    columns = (
        ("latency_s", 1e3, "Latency (ms)"),
        ("mac_utilization", 1, "End-to-end MAC utilization"),
        ("hbm_bytes", 2**-20, "HBM traffic (MiB)"),
        ("sram_bytes", 2**-20, "SRAM traffic (MiB)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), layout="constrained")
    fig.suptitle("Dataflow comparison · matched and auto-selected tiles")
    xs = np.arange(len(keys))
    for axis, (field, multiplier, title) in zip(axes.flat, columns, strict=True):
        for index, flow in enumerate(FLOWS):
            vs = [by_key[*key, flow][field] * multiplier for key in keys]
            axis.bar(xs + (index - 0.5) * 0.34, vs, 0.34, color=FLOW_COLORS[flow], label=flow.upper())
        axis.set_title(title)
        axis.set_xticks(xs, labels)
        axis.grid(axis="y")
        if field == "mac_utilization":
            axis.set_ylim(0, 1)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    return save(fig, folder, "03_dataflows.png")


def buffering(rows: list[dict], folder: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8), layout="constrained")
    fig.suptitle("Double buffering · overlap and SRAM capacity")
    modes = ("single", "double")
    controls = (
        ("fixed_tile_control", "total_cycles", "Cycles · same tile"),
        ("capacity_control", "hbm_bytes", "HBM bytes · tight capacity"),
    )
    for axis, (workload, counter, title) in zip(axes[0], controls, strict=True):
        by_mode = {r["buffering"]: r for r in rows if r["workload"] == workload}
        values = [by_mode[mode][counter] for mode in modes]
        bars = axis.bar(modes, values, color=[MODE_COLORS[mode] for mode in modes], width=0.55)
        axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=4)
        axis.set_ylim(0, max(values) * 1.18)
        axis.set_title(title)
        axis.grid(axis="y")
    projections = [r for r in rows if r["workload"] == "attention_output"]
    lookup = {(r["m"], r["sram_capacity_bytes"], r["dataflow"], r["buffering"]): r
              for r in projections}
    sizes = sorted({r["sram_capacity_bytes"] for r in projections})
    tokens = sorted({r["m"] for r in projections})
    for axis, (metric, title) in zip(
        axes[1], (("total_cycles", "Latency ratio"), ("hbm_bytes", "HBM traffic ratio")), strict=True
    ):
        for token_index, m in enumerate(tokens):
            for flow in FLOWS:
                ratios = [
                    lookup[m, capacity, flow, "double"][metric]
                    / lookup[m, capacity, flow, "single"][metric]
                    for capacity in sizes
                ]
                axis.plot(np.arange(len(sizes)), ratios, "o-" if flow == "os" else "s--",
                          color=TOKEN_COLORS[token_index], label=f"M={m} · {flow.upper()}")
        axis.axhline(1, color="#64748b", linestyle=":", linewidth=1)
        axis.set_xticks(np.arange(len(sizes)), [f"{capacity // KIB:,}" for capacity in sizes])
        axis.set_xlabel("Total SRAM capacity (KiB)")
        axis.set_ylabel("Double / single")
        axis.set_title(f"{title} · auto-tiling")
        axis.grid(axis="y")
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
    return save(fig, folder, "04_buffering.png")


def scaling(rows: list[dict], folder: Path) -> str:
    widths = sorted({r["array_rows"] for r in rows})
    tokens = sorted({r["m"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 7.3), layout="constrained")
    fig.suptitle("Array scaling · fixed clock, SRAM and HBM")
    for index, m in enumerate(tokens):
        for flow in FLOWS:
            points = sorted((r for r in rows if r["m"] == m and r["dataflow"] == flow),
                            key=lambda r: r["array_rows"])
            series = (
                [r["latency_s"] * 1e3 for r in points],
                [points[0]["total_cycles"] / r["total_cycles"] for r in points],
                [r["mac_utilization"] for r in points],
                [r["hbm_stall_cycles"] / r["total_cycles"] for r in points],
            )
            for axis, values in zip(axes.flat, series, strict=True):
                axis.plot(widths, values, "o-" if flow == "os" else "s--",
                          color=TOKEN_COLORS[index], markersize=4)
    titles = ("Latency (ms)", "Speedup vs. 16 × 16", "End-to-end MAC utilization", "HBM wait share")
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_xticks(widths, [f"{width} × {width}" for width in widths])
        axis.set_xlabel("Array size")
        axis.grid()
    axes[0, 0].set_yscale("log")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    handles = [Line2D([], [], color=TOKEN_COLORS[i], label=f"M={m}") for i, m in enumerate(tokens)]
    handles += [Line2D([], [], color="#334155", marker="o", label="OS"),
                Line2D([], [], color="#334155", marker="s", linestyle="--", label="WS")]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    return save(fig, folder, "05_array_scaling.png")


def writeback(rows: list[dict], folder: Path) -> str:
    workloads = list(dict.fromkeys((row["workload"], row["m"]) for row in rows))
    bandwidths = sorted({row["hbm_bandwidth_bps"] for row in rows})
    lookup = {(r["dataflow"], r["workload"], r["m"], r["hbm_bandwidth_bps"]):
              -r["latency_change_pct"] for r in rows}
    limit = max(1, max(abs(change) for change in lookup.values()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("BrBG")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9.6, 4.6), layout="constrained")
    fig.suptitle("Output writeback overlap at identical tile sizes and HBM traffic")
    for flow, axis in zip(FLOWS, axes, strict=True):
        values = np.array([[lookup[flow, workload, m, bw] for bw in bandwidths]
                           for workload, m in workloads])
        axis.imshow(values, aspect="auto", cmap=cmap, norm=norm)
        for (i, j), value in np.ndenumerate(values):
            color = "white" if abs(value) > limit * 0.64 else "#172b42"
            axis.text(j, i, f"{value:+.2f}" if value else "0.00", ha="center", va="center",
                      color=color)
        axis.set_title("Output-stationary" if flow == "os" else "Weight-stationary")
        axis.set_xticks(range(len(bandwidths)), [f"{bw / 1e9:g}" for bw in bandwidths])
        axis.set_xlabel("HBM bandwidth (GB/s)")
        axis.axhline(len(workloads) - 1.5, color="#64748b", linewidth=0.8)
    axes[0].set_yticks(range(len(workloads)), [
        f"Attention · M={m:,}" if workload == "attention_output" else "Output-heavy control"
        for workload, m in workloads
    ])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.85)
    bar.set_label("Modeled latency reduction (%)")
    return save(fig, folder, "writeback_latency.png")


def render(rows: list[dict], folder: Path) -> list[str]:
    artifacts = []
    style = {
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "axes.titleweight": "semibold",
        "figure.titlesize": 15,
        "font.size": 9,
        "legend.frameon": False,
    }
    with plt.rc_context(style):
        for experiment, charts in (
            ("regimes", (roofline, regime_cycles)),
            ("dataflows", (dataflows,)),
            ("buffering", (buffering,)),
            ("scaling", (scaling,)),
            ("writeback", (writeback,)),
        ):
            points = [row for row in rows if row["experiment"] == experiment]
            if points:
                artifacts.extend(chart(points, folder) for chart in charts)
    return artifacts
