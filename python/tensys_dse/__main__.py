import argparse
import subprocess
import sys
from pathlib import Path

from .experiments import EXPERIMENTS
from .runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TenSys design-space experiments.")
    parser.add_argument("--sim", type=Path, default=Path("build/release/tensys"),
                        help="Path to the TenSys executable")
    parser.add_argument("--suite", choices=("all", "smoke", *EXPERIMENTS, "writeback"), default="all")
    parser.add_argument("--out", type=Path, default=Path("reports/latest"),
                        help="Directory for raw JSON, CSV, plots, and the run manifest")
    args = parser.parse_args()
    try:
        manifest = run(args.sim, args.suite, args.out)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"tensys_dse: {error}", file=sys.stderr)
        return 1
    seconds = manifest["seconds"]["end_to_end"]
    print(f"Wrote {manifest['case_count']} cases and plots to {args.out} in {seconds:.2f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
