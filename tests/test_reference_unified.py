"""Regression checks for coexistence of the two accepted reference-design branches."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from reference_design import allocate_controls
from reference_design import cli, vcc_cli


@pytest.mark.parametrize("seed", [0, 17, 29])
@pytest.mark.parametrize("depth", [1, 3])
def test_default_three_matches_independent_legacy_assignment(seed, depth):
    # Historical contract: sorted strata and cell IDs; interleaved groups of three
    # from each stratum's seeded permutation; pooled equal-depth block means.
    cell_ids = [f"cell_{i:02d}" for i in range(24)][::-1]
    strata = ["donor_b" if int(cell.split("_")[1]) % 2 else "donor_a" for cell in cell_ids]
    values = np.asarray([[i / 7, math.sin(i), i * i / 19] for i in range(24)])
    rng = np.random.default_rng(seed)
    selected = [[] for _ in range(3)]
    membership = []
    for stratum in sorted(set(strata)):
        candidates = sorted([i for i, label in enumerate(strata) if label == stratum],
                            key=lambda i: cell_ids[i])
        ordering = rng.permutation(candidates).tolist()
        for block in range(3):
            chosen = [ordering[3 * j + block] for j in range(depth)]
            selected[block].extend(chosen)
            for j, index in enumerate(chosen):
                membership.append(dict(cell_id=cell_ids[index], cell_index=index, stratum=stratum,
                                       block=block, within_block_index=j))
    expected = np.asarray([[math.fsum(float(values[i, g]) for i in indices) / len(indices)
                            for g in range(values.shape[1])] for indices in selected])
    default = allocate_controls(values, cell_ids, depth, seed, strata)
    explicit = allocate_controls(values, cell_ids, depth, seed, strata, blocks=3)
    assert default.membership == explicit.membership == membership
    np.testing.assert_array_equal(default.means, explicit.means)
    np.testing.assert_allclose(default.means, expected, rtol=1e-15, atol=1e-15)


def test_two_block_minimum_support_is_distinct_from_three_blocks():
    values = np.arange(12, dtype=float).reshape(4, 3)
    ids = ["d", "b", "a", "c"]
    result = allocate_controls(values, ids, depth=2, seed=7, blocks=2)
    assert result.means.shape == (2, 3)
    assert len({row["cell_id"] for row in result.membership}) == 4
    with pytest.raises(ValueError, match="3 blocks need 6"):
        allocate_controls(values, ids, depth=2, seed=7)


@pytest.mark.parametrize("command", ["allocate", "score", "prepare", "run", "design", "check-design"])
def test_ordinary_cli_routes_remain_reachable(command, capsys):
    with pytest.raises(SystemExit) as result:
        cli.main([command, "--help"])
    assert result.value.code == 0
    assert "--output" in capsys.readouterr().out


@pytest.mark.parametrize("implicit_argv", [False, True])
def test_nested_vcc_dispatch_preserves_complete_unavailable_exit(monkeypatch, tmp_path, implicit_argv):
    arguments = ["vcc", "summarize", str(tmp_path)]
    calls = []
    def independent_return(argv):
        calls.append(argv)
        return 2
    monkeypatch.setattr(vcc_cli, "main", independent_return)
    if implicit_argv:
        monkeypatch.setattr(sys, "argv", ["reference-design", *arguments])
        result = cli.main()
    else:
        result = cli.main(arguments)
    assert result == 2
    assert calls == [arguments[1:]]


def test_merged_help_lists_both_protocol_and_vcc(capsys):
    with pytest.raises(SystemExit) as result:
        cli.main(["--help"])
    assert result.value.code == 0
    text = capsys.readouterr().out
    for command in ("allocate", "score", "prepare", "run", "design", "check-design", "vcc"):
        assert command in text

