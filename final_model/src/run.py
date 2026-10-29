"""One entry point for the retained, fixed 20-path model and its verification."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
PRICE_HASH='7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-features',action='store_true',help='Recompute all 77 months from raw prices and retained selections')
    parser.add_argument('--rebuild-selection',action='store_true',help='Recompute all monthly candidate graphs and selections, then features; may be slow')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    price=ROOT/'data/prices.parquet.gz'
    if hashlib.sha256(price.read_bytes()).hexdigest()!=PRICE_HASH:
        raise ValueError('The fixed model price input has changed')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for folder in ['results','reports','audit']:(ROOT/folder).mkdir(exist_ok=True)
    logpath=ROOT/'audit/reproduction.log'
    stages=[]
    with logpath.open('w') as log:
        def execute(script,*extra):
            print('RUN',script,flush=True)
            log.write('\nRUN '+script+' '+' '.join(extra)+'\n');log.flush()
            subprocess.run([sys.executable,str(ROOT/'src'/script),*extra],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            stages.append(script)
        if args.rebuild_selection:
            execute('prepare.py','--stage','all','--force','--workers',str(args.workers))
        elif args.rebuild_features:
            execute('prepare.py','--stage','features','--force','--workers',str(args.workers))
        else:
            execute('prepare.py','--stage','all','--workers',str(args.workers))
        execute('risk.py','--scope','all')
        execute('statistics.py')
        execute('risk_intervals.py')
        execute('test_prepare.py')
        execute('verify.py')
        execute('diagnostics.py')
    records=[]
    for folder in ['src','data','config','results']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                records.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    record=dict(completed_at=datetime.now(timezone.utc).isoformat(),python=sys.version,
                stages=stages,rebuilt_selection=args.rebuild_selection,
                rebuilt_features=args.rebuild_features or args.rebuild_selection,
                primary_paths=4,auxiliary_paths=16,all_checks_passed=True,files=records)
    (ROOT/'audit/reproduction_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print('COMPLETE: 20 paths, 100 metrics, independent checks and diagnostic charts',flush=True)

if __name__=='__main__':main()
