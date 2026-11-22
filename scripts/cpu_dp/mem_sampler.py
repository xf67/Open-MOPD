"""Sample per-process host memory of a training run (Rss/Pss/private anon) and /dev/shm usage.

python scripts/cpu_dp/mem_sampler.py logs/mem.log [interval_s]
"""
import os
import sys
import time


def rollup(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    out[parts[0].rstrip(":")] = int(parts[1]) / 2**20  # GiB
    except OSError:
        return None
    return out


def name(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            cmd = handle.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return None
    for key in ("WorkerDict", "TaskRunner", "openmopd-dp-adam", "main_ppo", "raylet", "multiprocessing.spawn"):
        if key in cmd:
            return key
    return None


def main():
    path, interval = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    with open(path, "a") as log:
        while True:
            rows, total_priv = [], 0.0
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                label = name(pid)
                if label is None:
                    continue
                info = rollup(pid)
                if not info:
                    continue
                priv = info.get("Private_Dirty", 0) + info.get("Private_Clean", 0) - info.get("Shmem", 0) * 0
                anon = info.get("Anonymous", 0)
                total_priv += anon
                if info.get("Rss", 0) > 0.5:
                    rows.append(f"{label}:{pid} rss={info.get('Rss',0):.1f} pss={info.get('Pss',0):.1f} anon={anon:.1f}")
            st = os.statvfs("/dev/shm")
            shm = (st.f_blocks - st.f_bfree) * st.f_frsize / 2**30
            with open("/proc/meminfo") as handle:
                mem = {l.split(":")[0]: int(l.split()[1]) / 2**20 for l in handle}
            log.write(f"{time.strftime('%H:%M:%S')} avail={mem['MemAvailable']:.1f} shm={shm:.1f} "
                      f"anon_total={total_priv:.1f} | " + " ; ".join(sorted(rows)) + "\n")
            log.flush()
            time.sleep(interval)


if __name__ == "__main__":
    main()
