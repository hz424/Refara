#!/usr/bin/env python3
"""Plot organizer comparison tables without refitting or selecting thresholds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9, "axes.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.width": .6, "ytick.major.width": .6,
    "legend.fontsize": 8, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
}
RANKINGS = (
    ("anchor_margin", "Anchor margin", "#B64342", "-"),
    ("calibrated_margin", "Calibrated margin", "#0F4D92", (0, (4, 2))),
)
RULES = (
    ("single_split", "Single split", "#767676", ""),
    ("repeated_primary", "Repeated primary", "#3775BA", "///"),
    ("development_margin", "Development margin", "#B64342", "..."),
)
GROUP_LABELS = {"original8": "8-cell training", "cap48": "48-cell training cap"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_table(path, columns):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(set(columns) <= set(reader.fieldnames or ()),
                f"Missing columns in {Path(path).name}: {sorted(set(columns) - set(reader.fieldnames or ()))}")
        rows = list(reader)
    require(rows, f"No rows in {Path(path).name}")
    return rows


def number(value, name):
    result = float(value)
    require(math.isfinite(result), f"Nonfinite {name}")
    return result


def at_gap(rows, gap):
    selected = [row for row in rows if math.isclose(number(row["minimum_gap"], "gap"),
                                                   gap, rel_tol=0, abs_tol=1e-12)]
    require(selected, f"No results at reporting gap {gap:g}")
    return selected


def risk_coverage_figure(rows, summary, gap):
    """Rows contain the assessment result for each retained-count threshold."""
    rows = at_gap(rows, gap)
    budgets = sorted((int(row["control_cells_available"]), row["id"])
                     for row in summary["costs"]["budgets_per_realization"])
    require(len(budgets) == 6, "The Norman figure requires exactly six budgets")
    require(len({budget for _, budget in budgets}) == 6, "Inconsistent budget cell counts")
    require({row["budget"] for row in rows} == {budget for _, budget in budgets},
            "Risk table budgets differ from the comparison metadata")
    maximum = max(number(row["risk"], "unsupported fraction") for row in rows)
    upper = min(100, max(10, math.ceil(100 * maximum / 10) * 10))
    figure, axes = plt.subplots(2, 3, figsize=(7.1, 4.5), sharex=True, sharey=True)
    figure.subplots_adjust(left=.10, right=.98, bottom=.19, top=.80, hspace=.43, wspace=.25)
    for panel, (axis, (cells, budget)) in enumerate(zip(axes.flat, budgets)):
        for ranking, label, color, linestyle in RANKINGS:
            selected = sorted((row for row in rows if row["budget"] == budget
                               and row["ranking"] == ranking), key=lambda row: int(row["selected"]))
            require(selected, f"Missing {ranking} results at {budget}")
            require(len({int(row["selected"]) for row in selected}) == len(selected),
                    "Duplicate retained counts")
            require([int(row["selected"]) for row in selected]
                    == list(range(1, int(selected[-1]["comparisons"]) + 1)),
                    "Risk curve must include every retained count through full coverage")
            for row in selected:
                count, total, unsupported = (int(row[key]) for key in ("selected", "comparisons", "unsupported"))
                require(0 <= unsupported <= count <= total and count > 0,
                        "Invalid retained-comparison counts")
                require(math.isclose(number(row["coverage"], "coverage"), count / total, abs_tol=1e-12)
                        and math.isclose(number(row["risk"], "risk"), unsupported / count, abs_tol=1e-12),
                        "Risk or coverage differs from its count denominator")
            coverage = [100 * number(row["coverage"], "coverage") for row in selected]
            risk = [100 * number(row["risk"], "unsupported fraction") for row in selected]
            require(all(0 < x <= 100 for x in coverage) and all(0 <= y <= 100 for y in risk),
                    "Coverage and unsupported fractions must lie in [0, 1]")
            axis.plot(coverage, risk, color=color, linestyle=linestyle, linewidth=1.3, label=label)
        axis.set_title(f"{cells:,} controls", loc="left", pad=6)
        axis.text(-.18, 1.07, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=10, fontweight="bold", va="bottom")
        axis.set(xlim=(0, 100), ylim=(-.02 * upper, 1.02 * upper))
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.tick_params(length=3)
    for axis in axes[-1]:
        axis.set_xlabel("Retained comparisons")
    figure.text(.02, .505, "Unsupported among retained (%)", rotation=90,
                va="center", ha="center", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(.52, .94))
    figure.suptitle(f"Norman reporting rules · MSE gap {gap:g}", y=.99, fontsize=10)
    cases = int(summary["assessment_cases"])
    comparisons = {int(row["comparisons"]) for row in rows}
    require(len(comparisons) == 1 and cases > 0 and next(iter(comparisons)) % cases == 0,
            "Inconsistent assessment comparison counts")
    pairs = comparisons.pop() // cases
    realizations = len(summary["assessment_realizations"])
    figure.text(.10, .045,
                f"{cases} assessment tasks × {pairs} model pairs; {realizations} recorded reference realizations.\n"
                "Tasks share a control pool. Curves describe these finite realizations.",
                fontsize=7, va="bottom", linespacing=1.5)
    return figure


def transfer_figure(groups, gap):
    """Keep training groups separate; their assessment donors are shared."""
    require(groups, "No donor-transfer result tables supplied")
    figure, axes = plt.subplots(len(groups), 2, figsize=(5.8, 1.45 * len(groups) + 1.15),
                                squeeze=False, gridspec_kw={"width_ratios": [3, 1]})
    figure.subplots_adjust(left=.30, right=.96, bottom=.23, top=.78,
                           hspace=.80, wspace=.50)
    controls, units, cases, realizations = set(), set(), set(), set()
    for index, (group, all_rows, summary) in enumerate(groups):
        rows = at_gap(all_rows, gap)
        controls.update(int(row["control_cells_available"])
                        for row in summary["costs"]["budgets_per_realization"])
        units.add(int(summary["assessment_units"]))
        cases.add(int(summary["assessment_cases"]))
        realizations.add(len(summary["assessment_realizations"]))
        require(len({row["budget"] for row in rows}) == 1,
                "Donor-transfer figure requires one budget per training group")
        selected = {}
        for rule, _, _, _ in RULES:
            matches = [row for row in rows if row["rule"] == rule]
            require(len(matches) == 1, f"Expected one {rule} row for {group}")
            selected[rule] = matches[0]
        totals = {int(row["comparisons"]) for row in selected.values()}
        require(len(totals) == 1, "Comparison denominator differs between rules")
        total = totals.pop()
        require(total > 0, "No assessment comparisons")
        left, right = axes[index]
        for position, (rule, label, color, hatch) in enumerate(RULES):
            row = selected[rule]
            released, unsupported = int(row["selected"]), int(row["unsupported"])
            coverage = number(row["coverage"], "coverage")
            require(0 <= unsupported <= released <= total, "Invalid reported-comparison counts")
            require(math.isclose(coverage, released / total, abs_tol=1e-12),
                    "Coverage differs from its count denominator")
            left.barh(position, 100 * coverage, height=.57, color=color, hatch=hatch,
                      edgecolor="white", linewidth=.5)
            left.annotate(f"{released}/{total}", (100 * coverage, position),
                          xytext=(4, 0), textcoords="offset points", va="center",
                          fontsize=7, annotation_clip=False)
            right.text(.5, position, str(unsupported), transform=right.get_yaxis_transform(),
                       va="center", ha="center", fontsize=8)
        left.set_yticks(range(len(RULES)), [item[1] for item in RULES])
        left.set_xlim(0, 100)
        left.set_xticks([0, 25, 50, 75, 100])
        left.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        left.invert_yaxis()
        left.tick_params(axis="y", length=0)
        left.spines["left"].set_visible(False)
        right.set_ylim(left.get_ylim())
        right.set_axis_off()
        left.set_title(GROUP_LABELS.get(group, group), loc="left", pad=8)
        left.set_xlabel("Reported comparisons (%)")
        right.set_title("Unsupported\ncomparisons (n)", fontsize=8, pad=8)
    require(all(len(values) == 1 for values in (controls, units, cases, realizations)),
            "Training groups have different control or assessment designs")
    cells, donors, case_count, realization_count = (next(iter(values)) for values in
                                                   (controls, units, cases, realizations))
    require(donors > 0 and case_count % donors == 0, "Unequal cell-type count per donor")
    figure.suptitle(f"Influenza donor transfer · {cells} controls · MSE gap {gap:g}",
                   y=.97, fontsize=10)
    figure.text(.07, .045,
                f"{donors} assessment donors × {case_count // donors} cell types; shared by both training groups.\n"
                f"Support evaluated across {realization_count} recorded reference realizations.",
                fontsize=7, va="bottom", linespacing=1.5)
    return figure


def save_figure(figure, output, name):
    paths = []
    for suffix in ("pdf", "svg", "png"):
        path = output / f"{name}.{suffix}"
        metadata = {"Creator": "Refara"}
        if suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        elif suffix == "svg":
            metadata["Date"] = None
        figure.savefig(path, dpi=600, metadata=metadata)
        paths.append(path)
    plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path,
                        help="Output directory from evaluate.py")
    parser.add_argument("--gap", type=float, help="Defaults to the recorded primary reporting gap")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "Output already exists; choose a fresh directory")
        norman = args.results / "datasets" / "norman" / "check"
        risk_path, summary_path = norman / "risk_coverage.tsv", norman / "check.json"
        summary = json.loads(summary_path.read_text())
        gap = args.gap if args.gap is not None else float(summary["protocol"]["primary_minimum_gap"])
        require(math.isfinite(gap) and gap >= 0, "Gap must be finite and nonnegative")
        inputs = [risk_path, summary_path]
        risk = read_table(risk_path, ("minimum_gap", "budget", "ranking", "selected", "comparisons",
                                     "unsupported", "coverage", "risk"))
        groups = []
        for group in GROUP_LABELS:
            directory = args.results / "datasets" / group / "check"
            path, meta_path = directory / "rule_summary.tsv", directory / "check.json"
            inputs.extend((path, meta_path))
            metadata = json.loads(meta_path.read_text())
            groups.append((group, read_table(path, ("minimum_gap", "budget", "rule", "comparisons",
                                                   "selected", "unsupported", "coverage")), metadata))
        with plt.rc_context(STYLE):
            figures = [("norman_risk_coverage", risk_coverage_figure(risk, summary, gap)),
                       ("donor_transfer_fixed_rules", transfer_figure(groups, gap))]
            args.output.mkdir(parents=True)
            outputs = [path for name, figure in figures for path in save_figure(figure, args.output, name)]
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = {"schema_version": 1, "minimum_gap": gap,
                   "inputs": [{"name": str(path), "sha256": digest(path)} for path in inputs],
                   "outputs": {path.name: digest(path) for path in outputs}}
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"figures": [name for name, _ in figures], "formats": ["pdf", "svg", "png"]}))
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
