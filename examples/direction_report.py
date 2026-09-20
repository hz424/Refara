"""Classify synthetic supplied intervals; no interval estimation is performed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_design.direction_report import Comparison, ReportedDirection, summarize_directions
from reference_design.cli import _destination, _publish, _tsv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory for JSON and TSV")
    args = parser.parse_args()
    try:
        if args.output.is_symlink():
            raise ValueError("--output must not be a symbolic link")
        _destination(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    roster = [Comparison(task, "model_1", "model_2") for task in
              ("B_cell/stimulus_1/6h", "B_cell/stimulus_2/6h",
               "T_cell/stimulus_1/6h", "T_cell/stimulus_2/6h")]
    reports = [ReportedDirection(roster[0], "a", (0.1, 0.3)),
               ReportedDirection(roster[1], "b", (0.1, 0.4)),
               ReportedDirection(roster[2], "a", (0.0, 0.2)),
               ReportedDirection(roster[3], "hold")]
    result = summarize_directions(roster, reports)
    result["example"] = "Synthetic intervals and decisions; no biological or reliability claim"
    rows = []
    for comparison in result["comparisons"]:
        interval = comparison["interval"]
        rows.append({key: comparison[key] for key in ("task", "model_a", "model_b", "direction", "status")} |
                    {"interval_lower": "" if interval is None else interval[0],
                     "interval_upper": "" if interval is None else interval[1]})
    contents = {
        "direction_summary.json": (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8"),
        "direction_comparisons.tsv": _tsv(list(rows[0]), rows),
    }
    try:
        _publish(args.output, contents)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in ("counts", "coverage", "reversal_fraction", "unresolved_fraction")}))


if __name__ == "__main__":
    main()
