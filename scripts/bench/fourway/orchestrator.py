"""Two-lane run queue for the four-way speed study (A6000 GPU1 + GPU2).

plan.json: list of {name, kind: verl|hf, env: {...}, cmd: [...] (hf only), lane: 0|1|null,
timeout_s}. The plan is re-read before each pick, so runs can be appended while
the queue is running. A run is skipped when runs/<name>/result.json exists.
An external nvidia-smi sampler (500 ms) records memory/utilization for every run.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
IMAGE = 'sha256:38667c913cd21afba9af63ab428a226b32f74308bdbcb4d7a82bacd9f3e4f2ca'
GPUS = ['GPU-a1be328a-16c1-3546-64c1-a84d652e97cc', 'GPU-de6b16f1-d52f-652a-65d0-533561179530']
GPU_INDEX = ['1', '2']
CPUSETS = ['0-23', '24-47']
LOCK = ROOT / 'queue.lock'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def pick(lane):
    with LOCK.open('a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        plan = json.loads((ROOT / 'plan.json').read_text())
        for run in plan:
            out = ROOT / 'runs' / run['name']
            if run.get('lane') not in (None, lane):
                continue
            if run['name'].startswith(('orig', 'zero')) and lane != 0:
                continue  # never two CPU-offload-heavy orig runs at once
            if (out / 'claimed').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            (out / 'claimed').write_text(json.dumps({'lane': lane, 'at': time.time()}))
            return run
    return None


def gpu_busy(lane):
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    return GPUS[lane] in apps


def run_one(lane, run):
    name = run['name']
    out = ROOT / 'runs' / name
    (out / 'tmp').mkdir(exist_ok=True)
    env = {k: str(v) for k, v in run.get('env', {}).items()}
    env.update(RUN_NAME=name)
    # Never share a GPU with another workload: wait until the lane's GPU is idle.
    waited = 0
    while gpu_busy(lane):
        if waited % 600 == 0:
            log(f'lane{lane}: GPU busy with a foreign process, waiting before {name}')
        time.sleep(30)
        waited += 30
    sampler = subprocess.Popen(
        ['nvidia-smi', '-i', GPUS[lane], '--query-gpu=timestamp,memory.used,utilization.gpu,power.draw',
         '--format=csv,noheader,nounits', '-lms', '500'],
        stdout=(out / 'gpu-samples.csv').open('w'), stderr=subprocess.DEVNULL)
    t0 = time.time()
    container = None
    if run['kind'] == 'verl':
        container = 'fourway-' + name.replace('_', '-').replace('.', '-')
        subprocess.run(['docker', 'rm', '-f', container], capture_output=True)
        args = ['docker', 'run', '-d', '--init', '--pull=never', '--name', container,
                '--label', 'experiment=fourway-speed-a6000-20261118', '--user', '1015:1015',
                '--gpus', 'device=' + GPUS[lane], '--shm-size=80g', '--memory=170g', '--memory-swap=170g',
                '--cpuset-cpus=' + CPUSETS[lane], '--cpuset-mems=0,1',
                '--mount', f'type=bind,src={ROOT},dst=/study',
                '--mount', f'type=bind,src={ROOT}/{run.get("source", "source")},dst=/repo,readonly',
                '--mount', 'type=bind,src=/home/xxf/models/OPD,dst=/models,readonly',
                '--mount', f'type=bind,src={ROOT}/cache{lane},dst=/cache',
                '--mount', f'type=bind,src={out}/tmp,dst=/tmp',
                '--workdir', '/study/runs/' + name]
        for k, v in env.items():
            args += ['-e', f'{k}={v}']
        args += ['--entrypoint', 'bash', IMAGE, '/study/harness/run_verl_case.sh']
        (out / 'launch-argv.json').write_text(json.dumps(args, indent=2))
        cid = subprocess.check_output(args, text=True).strip()
        follower = subprocess.Popen(['docker', 'logs', '-f', cid], stdout=(out / 'train.log').open('w'),
                                    stderr=subprocess.STDOUT)
        def alive():
            st = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}} {{.State.ExitCode}}', cid],
                                capture_output=True, text=True).stdout.split()
            return st and st[0] == 'true', (int(st[1]) if st else -1)
    else:
        penv = dict(os.environ)
        penv.update(env)
        penv.update(CUDA_VISIBLE_DEVICES=GPU_INDEX[lane], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    RUN_DIR=str(out), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                    HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TMPDIR=str(out / 'tmp'),
                    OMP_NUM_THREADS='8')
        cmd = ['taskset', '-c', CPUSETS[lane]] + run['cmd']
        (out / 'launch-argv.json').write_text(json.dumps({'cmd': cmd, 'env': env}, indent=2))
        proc = subprocess.Popen(cmd, stdout=(out / 'train.log').open('w'), stderr=subprocess.STDOUT,
                                env=penv, cwd=str(out), start_new_session=True)
        follower = None
        def alive():
            rc = proc.poll()
            return rc is None, rc
    timeout = float(run.get('timeout_s', 5400))
    timed_out = False
    while True:
        running, rc = alive()
        if not running:
            break
        steps_file = out / 'steps.jsonl'
        last_progress = steps_file.stat().st_mtime if steps_file.exists() else t0
        stalled = time.time() - last_progress > float(run.get('stall_s', 1800 if not steps_file.exists() else 1200))
        if time.time() - t0 > timeout or stalled:
            if stalled:
                log(f'{name}: no step progress, treating as stalled')
            timed_out = True
            log(f'{name}: timeout, killing')
            if container:
                subprocess.run(['docker', 'kill', container], capture_output=True)
            else:
                os.killpg(proc.pid, 9)
            time.sleep(5)
            running, rc = alive()
            break
        time.sleep(5)
    t1 = time.time()
    time.sleep(2)
    sampler.terminate()
    if follower:
        try:
            follower.wait(timeout=30)
        except Exception:
            follower.terminate()
    if container:
        subprocess.run(['docker', 'rm', '-f', container], capture_output=True)
    result = dict(name=name, lane=lane, gpu=GPUS[lane], started_at=t0, finished_at=t1,
                  exit_code=rc, timed_out=timed_out, run=run)
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    log(f'{name}: exit={rc} timed_out={timed_out} wall={t1 - t0:.0f}s')


def lane_worker(lane):
    while True:
        if (ROOT / 'STOP').exists():
            return
        run = pick(lane)
        if run is None:
            # idle: wait for more plan entries unless told to finish
            if (ROOT / 'FINISH_WHEN_EMPTY').exists():
                return
            time.sleep(30)
            continue
        log(f'lane{lane}: start {run["name"]}')
        try:
            run_one(lane, run)
        except Exception:
            log(f'lane{lane}: {run["name"]} crashed\n' + traceback.format_exc())


if __name__ == '__main__':
    lanes = [int(x) for x in sys.argv[1:]] or [0, 1]
    threads = [threading.Thread(target=lane_worker, args=(l,)) for l in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log('queue finished')
