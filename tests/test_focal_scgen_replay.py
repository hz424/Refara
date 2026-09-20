from __future__ import annotations

from pathlib import Path

from perturb_nuisance_focal.replay import reproduce_focal


REPOSITORY = Path(__file__).resolve().parents[1]
CAPSULE = REPOSITORY / "capsules/gse162632_scgen"


def test_bundled_focal_replay() -> None:
    report = reproduce_focal(
        CAPSULE / "replay", CAPSULE / "expected/focal_expected.tsv"
    )
    assert report["status"] == "PASS"
    assert report["all_five_direction_patterns_passed"] is True
    assert report["q1_reversed_in_all_eight_roots"] is True

