"""Run the current Figure 2, Figure 3 and Extended Data Figure 9 numerical replays."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or out.is_relative_to(ROOT):raise ValueError('Use a new output directory outside the capsule')
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,pin in manifest['files'].items():
        file=ROOT/name
        if file.is_symlink() or not file.resolve().is_relative_to(ROOT) or hashlib.sha256(file.read_bytes()).hexdigest()!=pin['sha256']:
            raise ValueError('Capsule file differs: '+name)
    out.mkdir(parents=True);reports={};started=time.monotonic()
    for component in ['model_comparisons','figure3','ed9']:
        run=subprocess.run([sys.executable,'-B',str(ROOT/component/'replay.py'),'--output',str(out/component)],text=True,capture_output=True)
        (out/(component+'.log')).write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError(component+' failed; see '+str(out/(component+'.log')))
        reports[component]=json.loads((out/component/'REPLAY.json').read_text())
    result={'status':'PASS_CURRENT_SUBMISSION_NUMERICAL_REPLAY','capsule_id':manifest['capsule_id'],
            'seconds':time.monotonic()-started,'components':reports}
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
if __name__=='__main__':main()
