"""Run the installed small operation demo; no empirical inputs are required."""
import argparse
from pathlib import Path
from reference_design.audit_cli import run_demo

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run_demo(args.output)
