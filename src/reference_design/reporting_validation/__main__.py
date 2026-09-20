"""Run the portable historical replay or an invented end-to-end fixture."""
import argparse
import json
from pathlib import Path

from . import integration_driver
from .replay import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    empirical = commands.add_parser("replay", help="Recompute the historical five-rule comparison from saved scores")
    empirical.add_argument("--input", "--capsule", dest="input", type=Path, required=True)
    empirical.add_argument("--output", type=Path, required=True)
    synthetic = commands.add_parser("synthetic", help="Invented freeze/interval/classification example; no scientific validation")
    synthetic.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        result = replay(args.input, args.output)
        print(json.dumps({"status": result["status"], "scientific_confirmation": False,
            "primary_M": result["panels"]["primary"]["primary_M"],
            "primary_natural": [r for r in result["panels"]["primary"]["summary"] if r["scope"] == "natural"],
            "scope": result["scope"], "native_unit_caveat": result["native_unit_caveat"]}, indent=2))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        bundle, protocol, admission, plan = integration_driver.integration_fixture()
        seal = integration_driver.freeze_for_matrix(bundle, protocol, admission, plan, args.output / "publication")
        matrix = args.output / "invented_matrix.json"
        integration_driver._save(matrix, integration_driver.matrix_fixture(args.output / "publication", seal))
        result = integration_driver.assess_matrix(args.output / "publication", seal, matrix, args.output / "assessment")
        print(json.dumps({"status": result["status"], "mode": result["mode"], "scientific_validation": False,
                          "scope": result["scope"]}, indent=2))


if __name__ == "__main__":
    main()
