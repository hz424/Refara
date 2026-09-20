from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
REPLAY = REPO / "capsules" / "gse162632_allocation" / "replay.py"


def test_gse162632_label_level_allocation_capsule() -> None:
    command = [sys.executable, "-B", str(REPLAY)]
    source_data_root = os.environ.get("REFERENCE_CELL_SOURCE_DATA_ROOT")
    if source_data_root:
        command.extend(("--source-data-root", source_data_root))
    completed = subprocess.run(
        command,
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "PASS"
    assert receipt["unit_utility_shape"] == [1000, 3, 27, 8, 8]
    assert receipt["pattern_utility_shape"] == [1000, 3, 5, 8, 8]
    assert receipt["interaction_shape"] == [1000, 3, 4, 8, 28]
    assert receipt["reversal_counts_by_depth_M_P_O_D"] == [
        [0, 5, 5, 5],
        [0, 5, 5, 5],
        [0, 5, 5, 5],
    ]
    assert receipt["partitions_are_biological_replicates"] is False
    if source_data_root:
        assert receipt["figure_source_data_check"]["status"] == "PASS"
        assert receipt["figure_source_data_check"]["rows_checked"] == 136
    else:
        assert receipt["figure_source_data_check"]["status"] == "NOT_REQUESTED"
