"""Run a reference-sensitivity example from synthetic cell measurements."""

import argparse
from pathlib import Path
import shutil

from reference_design.cli import main


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    source = Path(__file__).resolve().parent / "data"
    inputs = output / "input"
    shutil.copytree(source, inputs)
    main(["allocate", str(inputs / "controls.tsv"), "--depth", "2",
          "--allocations", "6", "--seed", "7",
          "--output", str(inputs / "references")])
    main(["score", str(inputs / "config.json"),
          "--output", str(output / "results")])
    print(f"Results: {output / 'results'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
