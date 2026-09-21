"""Guard the corrected display direction without altering historical numbers."""
import csv
from pathlib import Path


def test_e3_label_matches_raw_direction(source_data_root: Path) -> None:
    authority = source_data_root / "Figure_4_attribution"
    for filename, expected_rows in (
        ("KANG_V22_PRIMARY_ENDPOINTS_PUBLIC_V1.tsv", 1),
        ("KANG_V22_DONOR_ROOT_ENDPOINTS_PUBLIC_V1.tsv", 8),
    ):
        with (authority / filename).open() as handle:
            rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["endpoint"] == "E3"]
        assert len(rows) == expected_rows
        assert all("Q(A)-Q(C)" in r["endpoint_label"] for r in rows)
        assert all(r["raw_direction"] == "-1" for r in rows)
