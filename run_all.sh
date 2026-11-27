#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -f predictions.json
mkdir -p rerun
python - <<'PY'
import json, subprocess, sys, pathlib, os
for c in json.load(open('predictions.json'))['configurations']:
    assert c['page_bytes']==os.sysconf('SC_PAGE_SIZE')
    prefix='rerun/'+c['experiment']
    if list(pathlib.Path('rerun').glob(c['experiment']+'_*')): raise SystemExit('Existing output: '+prefix)
    subprocess.run([sys.executable,'cow_ml_memory_v2.py','--strategy','both','--workers',str(c['workers']),'--modify-workers',str(c['modify_workers']),'--modify',str(c['modify']),'--width',str(c['width']),'--epochs',str(c['epochs']),'--repeats',str(c['repeats']),'--output',prefix],check=True)
PY
