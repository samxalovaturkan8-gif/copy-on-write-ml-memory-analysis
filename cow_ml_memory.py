#!/usr/bin/env python3
"""Linux CoW experiment: actual trained MLP weights, isolated VMA PSS, repeated trials.
Install numpy and scikit-learn. Run --help. No ML libraries imported in fork parent.
Read/write phases are page-access workloads, not inference or training benchmarks.
"""
import argparse, csv, ctypes, json, math, mmap, os, platform, re, resource, statistics, subprocess, sys, tempfile, time
from pathlib import Path

PAGE = os.sysconf('SC_PAGE_SIZE')
MIB = 1024**2
PHASES = ('init', 'read', 'modify')


def train(path, width, epochs):
    # Runs in a separate executable process, which exits before any worker fork.
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    model = MLPClassifier(hidden_layer_sizes=(width, width), max_iter=epochs, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        model.fit(Xtr, ytr)
    arrays = model.coefs_ + model.intercepts_
    with open(path, 'wb') as f:
        for a in arrays:
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    print(json.dumps(dict(model='MLPClassifier', dataset='sklearn digits', width=width,
        epochs=model.n_iter_, accuracy=float(model.score(Xte,yte)), parameters=sum(a.size for a in arrays),
        shapes=[list(a.shape) for a in arrays], dtype='float32')))


class ModelBuffer:
    def __init__(self, path):
        self.payload = os.path.getsize(path)
        self.size = math.ceil(self.payload / PAGE)*PAGE
        self.pages = self.size // PAGE
        # Guard pages isolate the measured mapping from adjacent anonymous VMAs.
        self.mm = mmap.mmap(-1, self.size+2*PAGE, flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,
                            prot=mmap.PROT_READ|mmap.PROT_WRITE)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        self.addr = base + PAGE
        libc = ctypes.CDLL(None, use_errno=True)
        for addr in (base, self.addr+self.size):
            if libc.mprotect(ctypes.c_void_p(addr), ctypes.c_size_t(PAGE), 0):
                raise OSError(ctypes.get_errno(), 'mprotect guard failed')
        # Fail rather than silently assume base-page CoW granularity.
        self.mm.madvise(mmap.MADV_NOHUGEPAGE, PAGE, self.size)
        with open(path,'rb') as f:
            with memoryview(self.mm)[PAGE:PAGE+self.payload] as view:
                if f.readinto(view) != self.payload:
                    raise IOError('Incomplete weights file')
        # Last page is touched by payload; padding is only alignment, not model data.
        self.indices = ()

    def prepare(self, fraction):
        k = round(fraction*self.pages)
        self.indices = tuple(PAGE+(i*self.pages//k)*PAGE for i in range(k)) if k else ()

    def read(self):
        return sum(self.mm[i] for i in range(PAGE, PAGE+self.size, PAGE))

    def write(self):
        for i in self.indices:
            self.mm[i] ^= 1
        return len(self.indices)

    def close(self):
        self.mm.close()


def mapping_stats(pid, addr, size):
    if addr is None:
        return dict(rss=0, pss=0, private_dirty=0, shared=0)
    header = re.compile(r'^([0-9a-f]+)-([0-9a-f]+)\s')
    found = False
    values = {}
    with open(f'/proc/{pid}/smaps') as f:
        for line in f:
            match = header.match(line)
            if match:
                if found: break
                lo, hi = (int(x,16) for x in match.groups())
                if lo == addr and hi == addr+size: found = True
            elif found and ':' in line:
                key, val = line.split(':',1)
                if val.strip().endswith('kB'): values[key] = int(val.split()[0])
    required = ('Rss','Pss','Private_Dirty','Shared_Clean','Shared_Dirty','AnonHugePages','KernelPageSize')
    if not found or any(k not in values for k in required):
        raise RuntimeError(f'Exact model mapping/PSS unavailable for PID {pid}; trial invalid')
    if values['AnonHugePages'] or values['KernelPageSize']*1024 != PAGE:
        raise RuntimeError('Unexpected huge page mapping; trial invalid')
    return dict(rss=values['Rss'], pss=values['Pss'], private_dirty=values['Private_Dirty'],
                shared=values['Shared_Clean']+values['Shared_Dirty'])


def total_pss(pid):
    with open(f'/proc/{pid}/smaps_rollup') as f:
        for line in f:
            if line.startswith('Pss:'): return int(line.split()[1])
    raise RuntimeError('Process PSS unavailable; RSS is not a substitute')


def faults():
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


def worker(cmd, reply, idx, strategy, model, args, path):
    for line in cmd:
        phase = line.strip()
        if phase == 'exit': break
        # Warm measurement calls before the measured interval.
        faults(); time.perf_counter()
        before = faults(); start = time.perf_counter()
        touched = 0
        if phase == 'init':
            if model is None: model = ModelBuffer(path)
            model.prepare(args.modify if idx < args.modify_workers else 0)
        elif phase == 'read': model.read()
        elif phase == 'modify': touched = model.write()
        elapsed = (time.perf_counter()-start)*1000
        delta = faults()-before
        reply.write(json.dumps(dict(addr=model.addr, size=model.size, minor_faults=delta,
                                    written_pages=touched, ms=elapsed))+'\n'); reply.flush()


def run(strategy, args, path, trial):
    if len(os.listdir('/proc/self/task')) != 1:
        raise RuntimeError('Fork parent is not single-threaded')
    model = ModelBuffer(path) if strategy == 'cow' else None
    workers=[]; rows=[]
    size=math.ceil(os.path.getsize(path)/PAGE)*PAGE
    try:
        for idx in range(args.workers):
            cr,cw=os.pipe(); rr,rw=os.pipe()
            pid=os.fork()
            if pid==0:
                code=1
                try:
                    os.close(cw); os.close(rr)
                    for _, c, r in workers: c.close(); r.close()
                    with os.fdopen(cr) as c, os.fdopen(rw,'w') as r:
                        worker(c,r,idx,strategy,model,args,path)
                    code=0
                finally: os._exit(code)
            os.close(cr); os.close(rw)
            workers.append((pid,os.fdopen(cw,'w'),os.fdopen(rr)))
        for phase in PHASES:
            for _,c,_ in workers: c.write(phase+'\n'); c.flush()
            replies=[]
            for pid,_,r in workers:
                line=r.readline()
                if not line: raise RuntimeError(f'Worker {pid} failed')
                replies.append(json.loads(line))
            pids=[os.getpid()]+[w[0] for w in workers]
            maps=[mapping_stats(pids[0],model.addr if model else None,size)]
            maps += [mapping_stats(pid,rep['addr'],rep['size']) for pid,rep in zip(pids[1:],replies)]
            totals={k:sum(m[k] for m in maps)/1024 for k in maps[0]}
            written=sum(r['written_pages'] for r in replies)
            predicted=(args.workers*size if strategy=='independent' else size+written*PAGE)/MIB
            row=dict(trial=trial,strategy=strategy,phase=phase,workers=args.workers,modify=args.modify,
                     modify_workers=args.modify_workers,model_mib=size/MIB,predicted_mib=predicted,
                     model_pss_mib=totals['pss'],model_rss_mib=totals['rss'],
                     model_private_dirty_mib=totals['private_dirty'],model_shared_sum_mib=totals['shared'],
                     process_pss_mib=sum(total_pss(pid) for pid in pids)/1024,
                     minor_faults=sum(r['minor_faults'] for r in replies),written_pages=written,
                     expected_cow_pages=written if strategy=='cow' else 0,
                     mean_worker_ms=statistics.mean(r['ms'] for r in replies))
            rows.append(row)
            print(f"{trial:2} {strategy:11} {phase:6} PSS={row['model_pss_mib']:8.3f} MiB predicted={predicted:8.3f} private={totals['private_dirty']:8.3f} faults={row['minor_faults']:6} writes={written}",flush=True)
    finally:
        for _,c,r in workers:
            try: c.write('exit\n'); c.flush()
            except OSError: pass
            c.close(); r.close()
        for pid,_,_ in workers: os.waitpid(pid,0)
        if model: model.close()
    return rows


def save_csv(path, rows):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strategy',choices=['both','cow','independent'],default='both')
    p.add_argument('--workers',type=int,default=4); p.add_argument('--modify',type=float,default=.25)
    p.add_argument('--modify-workers',type=int); p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--width',type=int,default=1024,help='Actual hidden-layer width; 4096 gives ~65 MiB weights')
    p.add_argument('--epochs',type=int,default=10); p.add_argument('--output',default='cow_results')
    p.add_argument('--train-only',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.train_only: train(a.train_only,a.width,a.epochs); return
    if a.modify_workers is None: a.modify_workers=a.workers
    if not (a.workers>=1 and 0<=a.modify<=1 and 0<=a.modify_workers<=a.workers and a.repeats>=2 and a.width>=1 and a.epochs>=1): p.error('Invalid parameters; repeats must be >=2')
    print(f'Linux {platform.release()} | Python {platform.python_version()} | page {PAGE} bytes',flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        path=str(Path(tmp)/'weights.bin')
        env=os.environ.copy()
        for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'): env[k]='1'
        metadata=json.loads(subprocess.check_output([sys.executable,__file__,'--train-only',path,'--width',str(a.width),'--epochs',str(a.epochs)],env=env,text=True))
        metadata.update(payload_bytes=os.path.getsize(path),kernel=platform.release(),python=platform.python_version(),page_bytes=PAGE)
        print(json.dumps(metadata,indent=2),flush=True)
        rows=[]
        for trial in range(1,a.repeats+1):
            strategies=['independent','cow'] if a.strategy=='both' else [a.strategy]
            if trial%2==0: strategies.reverse()
            for strategy in strategies: rows.extend(run(strategy,a,path,trial))
        summary=[]
        metrics=['model_pss_mib','model_rss_mib','model_private_dirty_mib','model_shared_sum_mib','process_pss_mib','minor_faults','mean_worker_ms']
        for strategy in sorted({r['strategy'] for r in rows}):
            for phase in PHASES:
                group=[r for r in rows if r['strategy']==strategy and r['phase']==phase]
                s=dict(strategy=strategy,phase=phase,repeats=len(group),predicted_mib=group[0]['predicted_mib'])
                for metric in metrics:
                    vals=[r[metric] for r in group]; s[metric+'_mean']=statistics.mean(vals); s[metric+'_sd']=statistics.stdev(vals)
                summary.append(s)
        save_csv(a.output+'_raw.csv',rows); save_csv(a.output+'_summary.csv',summary)
        Path(a.output+'_metadata.json').write_text(json.dumps(metadata|dict(config=vars(a)),indent=2))
        print('Saved raw CSV, summary CSV and metadata JSON. Minor faults are NOT exact CoW faults. Shared sum and RSS double-count shared pages.')

if __name__=='__main__':
    if sys.platform!='linux': sys.exit('Run in Linux / WSL2 Ubuntu')
    # Avoid site startup hooks that may silently create native threads.
    if '--train-only' not in sys.argv and not sys.flags.no_site:
        os.execv(sys.executable, [sys.executable, '-S', os.path.abspath(__file__), *sys.argv[1:]])
    main()