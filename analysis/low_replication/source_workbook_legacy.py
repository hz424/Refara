#!/usr/bin/env python3
"""Build the Source Data workbook for manuscript Figure 5.

The four input TSV files are verified byte for byte. The workbook contains all
rows used in panels a–c and links the full 160-row stress analysis to Extended
Data Figure 5. Outputs are written below ``outputs/``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import openpyxl
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]

SOURCE_FILES = {
    "F2A_BASE_RECOVERY_POOLED_V1.tsv": "3e89e55ab97e3c7e9fc09493829b2082a42b3f8347958e6330fc9e987b84c67d",
    "F2B_GLOBAL_NULL_CALIBRATION_V1.tsv": "edb701c3117fc74483502e3f82236690f8d8dd04c00cc7519333f205444a708c",
    "F2C_STRESS_RECOVERY_ABSTENTION_POOLED_V1.tsv": "62bc66b13de78e4273ebf80a2b4e10a0e816a250da07dc70b68021eab0dbadbe",
    "F2D_PREEXISTING_NEGATIVE_CONTROLS_V1.tsv": "e414ea9827e3bceda6f003f6720ab97605805cfb97dc123068d50b807af5c629",
    "FIGURE_2_SOURCE_DATA_MANIFEST_V1.json": "b3dc6cb4106a4f41f164a1cd929f6b0085584710e343c88a9def26d06df9fc49",
}

PANEL_MAP = (
    {
        "panel": "Fig. 5a",
        "sheet": "Fig2a_null_40",
        "source": "F2B_GLOBAL_NULL_CALIBRATION_V1.tsv",
        "rows": 40,
        "source_panel_id": "F2B_GLOBAL_NULL_CALIBRATION (historical identifier)",
        "display": "All 40 global-null cells: 10 specified states × 4 root counts",
        "denominator": "N=5,000 Monte Carlo runs per cell",
        "interval": "Pointwise two-sided 95% Clopper-Pearson",
    },
    {
        "panel": "Fig. 5b",
        "sheet": "Fig2b_controls_20",
        "source": "F2D_PREEXISTING_NEGATIVE_CONTROLS_V1.tsv",
        "rows": 20,
        "source_panel_id": "F2D historical file; current main panel b",
        "display": "All 20 hierarchical-null procedure × root-count rows",
        "denominator": "N=2,500 Monte Carlo runs per row",
        "interval": "Original pointwise two-sided 95% Clopper-Pearson",
    },
    {
        "panel": "Fig. 5c",
        "sheet": "Fig2c_recovery_24",
        "source": "F2A_BASE_RECOVERY_POOLED_V1.tsv",
        "rows": 24,
        "source_panel_id": "F2A_BASELINE_RECOVERY (historical identifier)",
        "display": "All 24 endpoint × separation × root-count recovery summaries",
        "denominator": "N=10,000 after pooling the prespecified positive and negative separation conditions",
        "interval": "Pointwise 95% Clopper-Pearson (exact family) or Hoeffding (true edge)",
    },
)

CROSSWALK_COLUMNS = (
    "panel",
    "workbook_sheet",
    "frozen_source_file",
    "source_panel_id_note",
    "displayed_rows",
    "displayed_observations",
    "denominator",
    "interval_definition",
    "current_disposition",
)


class Figure2WorkbookError(RuntimeError):
    """Raised when frozen inputs or generated workbook geometry changes."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise Figure2WorkbookError(f"missing header: {path.name}")
        return list(reader.fieldnames), list(reader)


def validate_frozen_sources(package: Path) -> dict[str, str]:
    source = package / "data" / "derived" / "figure2"
    observed: dict[str, str] = {}
    for filename, expected in SOURCE_FILES.items():
        path = source / filename
        if not path.is_file() or path.is_symlink():
            raise Figure2WorkbookError(f"missing or unsafe frozen source: {filename}")
        digest = sha256(path)
        if digest != expected:
            raise Figure2WorkbookError(f"frozen source hash mismatch: {filename}")
        observed[filename] = digest
    expected_rows = {
        "F2A_BASE_RECOVERY_POOLED_V1.tsv": 24,
        "F2B_GLOBAL_NULL_CALIBRATION_V1.tsv": 40,
        "F2C_STRESS_RECOVERY_ABSTENTION_POOLED_V1.tsv": 160,
        "F2D_PREEXISTING_NEGATIVE_CONTROLS_V1.tsv": 20,
    }
    for filename, expected in expected_rows.items():
        _fields, rows = read_tsv(source / filename)
        if len(rows) != expected:
            raise Figure2WorkbookError(
                f"frozen source row count changed: {filename}: {len(rows)}"
            )
    return observed


def typed_value(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if not any(character in value for character in (".", "e", "E")):
            return int(value)
        return float(value)
    except ValueError:
        return value


def style_table_sheet(sheet: Any) -> None:
    header_fill = PatternFill("solid", fgColor="DDE9EC")
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="222426")
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, column_cells in enumerate(sheet.columns, start=1):
        lengths = [len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells]
        sheet.column_dimensions[get_column_letter(index)].width = min(42, max(10, max(lengths) + 2))


def crosswalk_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in PANEL_MAP:
        rows.append(
            {
                "panel": item["panel"],
                "workbook_sheet": item["sheet"],
                "frozen_source_file": item["source"],
                "source_panel_id_note": item["source_panel_id"],
                "displayed_rows": item["rows"],
                "displayed_observations": item["display"],
                "denominator": item["denominator"],
                "interval_definition": item["interval"],
                "current_disposition": "Main Figure 5",
            }
        )
    rows.append(
        {
            "panel": "Extended Data Fig. 5",
            "workbook_sheet": "Not included in this main-figure workbook",
            "frozen_source_file": "F2C_STRESS_RECOVERY_ABSTENTION_POOLED_V1.tsv",
            "source_panel_id_note": "F2C historical file; current ED7",
            "displayed_rows": 160,
            "displayed_observations": "All 160 stress-analysis rows: four endpoints × ten states × four root counts",
            "denominator": "N=10,000 after pooling the prespecified +0.5 and −0.5 conditions within each row",
            "interval_definition": "Pointwise two-sided 95% Clopper-Pearson",
            "current_disposition": "Extended Data Figure 7; retained in frozen TSV",
        }
    )
    return rows


def write_crosswalk(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=CROSSWALK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_workbook(package: Path, destination: Path, rows: list[dict[str, Any]]) -> None:
    source = package / "data" / "derived" / "figure2"
    workbook = Workbook()
    readme = workbook.active
    readme.title = "README"
    readme_rows = [
        ("Source Data for manuscript Figure 5",),
        ("Scope", "Rows used in panels a-c."),
        ("Observation type", "Monte Carlo aggregate counts or bounded-score summaries; not biological observations."),
        ("Decision rule", "One-sided exact sign tests on non-tied roots; Holm over 12 ordered directions at familywise alpha 0.05."),
        ("Withholding", "Structural withholding applies when fewer than eight non-tied roots remain."),
        ("Intervals", "Pointwise Monte Carlo intervals only; not simultaneous or biological confidence intervals."),
        ("Inputs", "Numerical values come from the verified TSV files."),
        ("Related analysis", "The complete 160-row stress analysis is provided in the verified TSV and summarized in Extended Data Figure 7."),
    ]
    for row in readme_rows:
        readme.append(row)
    readme["A1"].font = Font(bold=True, size=12)
    readme.column_dimensions["A"].width = 24
    readme.column_dimensions["B"].width = 110
    for row in readme.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    crosswalk = workbook.create_sheet("Panel_crosswalk")
    crosswalk.append(list(CROSSWALK_COLUMNS))
    for row in rows:
        crosswalk.append([row[column] for column in CROSSWALK_COLUMNS])
    style_table_sheet(crosswalk)

    for item in PANEL_MAP:
        fields, table_rows = read_tsv(source / item["source"])
        if len(table_rows) != item["rows"]:
            raise Figure2WorkbookError(f"panel row count changed: {item['panel']}")
        sheet = workbook.create_sheet(item["sheet"])
        sheet.append(fields)
        for table_row in table_rows:
            sheet.append([typed_value(table_row[field]) for field in fields])
        style_table_sheet(sheet)

    workbook.properties.title = "Source Data for manuscript Figure 5"
    workbook.properties.subject = "Panels a-c and the Extended Data Figure 7 cross-reference"
    workbook.properties.creator = "Python/openpyxl"
    # Excel stores naive datetimes.  Fixing both fields makes the workbook
    # reproducible without asking openpyxl to serialize an unsupported tzinfo.
    fixed_time = datetime(2000, 1, 1)
    workbook.properties.created = fixed_time
    workbook.properties.modified = fixed_time
    temporary = destination.with_name(f".{destination.stem}.tmp.xlsx")
    workbook.save(temporary)
    temporary.replace(destination)
    normalize_xlsx_container(destination)

    reopened = load_workbook(destination, read_only=True, data_only=False)
    expected_sheet_rows = {
        "Fig2a_null_40": 41,
        "Fig2b_controls_20": 21,
        "Fig2c_recovery_24": 25,
        "Panel_crosswalk": 5,
    }
    for sheet, expected in expected_sheet_rows.items():
        if reopened[sheet].max_row != expected:
            raise Figure2WorkbookError(
                f"generated workbook geometry changed: {sheet}: {reopened[sheet].max_row}"
            )
    reopened.close()


def normalize_xlsx_container(path: Path) -> None:
    """Make XLSX metadata and ZIP member timestamps byte-stable."""

    temporary = path.with_name(f".{path.name}.normalized.tmp")
    with ZipFile(path, "r") as source, ZipFile(
        temporary, "w", compression=ZIP_DEFLATED, compresslevel=9
    ) as destination:
        for name in sorted(source.namelist()):
            original = source.getinfo(name)
            payload = source.read(name)
            if name == "docProps/core.xml":
                payload = re.sub(
                    br"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)",
                    br"\g<1>2000-01-01T00:00:00Z\g<2>",
                    payload,
                )
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = original.external_attr
            info.internal_attr = original.internal_attr
            info.create_system = original.create_system
            destination.writestr(
                info,
                payload,
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )
    os.replace(temporary, path)


def build(package: Path) -> dict[str, Any]:
    package = package.resolve(strict=True)
    receipt_dir = package / "outputs" / "provenance"
    source_data_dir = package / "outputs" / "source_data" / "figure2"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir.mkdir(parents=True, exist_ok=True)
    frozen_before = validate_frozen_sources(package)
    rows = crosswalk_rows()
    crosswalk = source_data_dir / "FIGURE_2_V7_PANEL_CROSSWALK.tsv"
    workbook = source_data_dir / "SOURCE_DATA_FIGURE_2_V7.xlsx"
    write_crosswalk(crosswalk, rows)
    build_workbook(package, workbook, rows)
    frozen_after = validate_frozen_sources(package)
    if frozen_before != frozen_after:
        raise Figure2WorkbookError("frozen Figure 2 inputs changed during workbook build")

    payload = {
        "schema_version": "1.0.0",
        "status": "PASS_V7_FIGURE_2_SOURCE_BINDING",
        "figure_id": "Figure 5",
        "active_panel_rows": {"a": 40, "b": 20, "c": 24},
        "active_panel_source": {
            item["panel"]: {
                "workbook_sheet": item["sheet"],
                "frozen_source_file": item["source"],
                "rows": item["rows"],
            }
            for item in PANEL_MAP
        },
        "extended_data_disposition": {
            "F2C_STRESS_RECOVERY_ABSTENTION_POOLED_V1.tsv": {
                "rows": 160,
                "destination": "Extended Data Figure 7",
            }
        },
        "frozen_source_sha256": frozen_after,
        "frozen_sources_rewritten": False,
        "crosswalk": {
            "path": str(crosswalk.relative_to(package)),
            "rows": len(rows),
            "sha256": sha256(crosswalk),
        },
        "workbook": {
            "path": str(workbook.relative_to(package)),
            "panel_rows": 84,
            "sha256": sha256(workbook),
        },
        "builder_sha256": sha256(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "openpyxl": openpyxl.__version__,
        },
    }
    binding = receipt_dir / "FIGURE_2_SOURCE_BINDING.json"
    binding.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["binding_path"] = str(binding)
    payload["binding_sha256"] = sha256(binding)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    package = args.package_root.resolve(strict=True)
    frozen = validate_frozen_sources(package)
    if args.validate_only:
        print(json.dumps({"status": "PASS_V7_F2_FROZEN_SOURCE_AUDIT", "sha256": frozen}, sort_keys=True))
        return 0
    payload = build(package)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
