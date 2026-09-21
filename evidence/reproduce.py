"""Reproduce the paper's main numerical conclusions from supplied inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Keys describe conclusions; historical component names remain valid paths.
CLAIMS = {
    'reference_roles': ('Reference sharing and equivalent representations', 'evidence/reference_roles/replay.py', 'REPLAY.json'),
    'model_comparisons': ('Reference sharing changes model comparisons', 'evidence/model_comparisons/replay.py', 'REPLAY.json'),
    'input_effects': ('Model-input controls change predictions and scores', 'evidence/figure3/replay.py', 'REPLAY.json'),
    'program_responses': ('Reference choices change program-error comparisons', 'evidence/program_responses/replay.py', 'REPLAY.json'),
    'matching_sensitivity': ('Matching choice changes directional conclusions and leaders', 'evidence/matching_sensitivity/replay.py', 'SUMMARY.json'),
    'role_sensitivity': ('Input updates and rescoring have different effects', 'evidence/replay_table1.py', 'CURRENT_TABLE1.json'),
}


def reproduce(output, claims):
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Use a new output directory')
    if output.resolve().is_relative_to(ROOT / 'evidence'):
        raise ValueError('Keep generated outputs outside evidence/')
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    report = {'status': 'RUNNING', 'claims': {}}
    receipt = output / 'RESULTS.json'
    try:
        for key in claims:
            title, script, report_name = CLAIMS[key]
            destination = output / key
            command = [sys.executable, '-B', str(ROOT / script), '--output', str(destination)]
            print(f'{key}: {title}', flush=True)
            with (output / (key + '.log')).open('x') as log:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode != 0:
                raise RuntimeError(f'{key} failed; see {output / (key + ".log")}')
            path = destination / report_name
            result = json.loads(path.read_text())
            accepted = result.get('status', '').startswith('PASS')
            if key == 'matching_sensitivity':
                accepted = result.get('status') == 'COMPLETE_MATCHING_SENSITIVITY_AGGREGATE_REPLAY'
            if not accepted:
                raise ValueError(f'{key} did not report a passed numerical check')
            report['claims'][key] = {'title': title, 'result': result,
                                     'receipt': path.relative_to(output).as_posix(),
                                     'receipt_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        report['status'] = 'PASS_IMPORTANT_CONCLUSIONS'
    except Exception as error:
        report.update(status='FAIL_IMPORTANT_CONCLUSIONS', error=str(error))
        raise
    finally:
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Passed {len(claims)} numerical replays: {receipt}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New directory for numerical tables, logs and RESULTS.json')
    parser.add_argument('--claim', choices=CLAIMS, action='append', help='Run only this conclusion; repeat to select several')
    parser.add_argument('--list', action='store_true', help='List the available conclusions')
    args = parser.parse_args()
    if args.list:
        for key, (title, _, _) in CLAIMS.items():
            print(f'{key}: {title}')
        return
    if args.output is None:
        parser.error('--output is required unless --list is used')
    try:
        reproduce(args.output, list(dict.fromkeys(args.claim or CLAIMS)))
    except (ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
