# Copyright 2026 Open-MOPD contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in JSONL memory telemetry. No CUDA synchronization or allocator mutation."""

import argparse
import functools
import json
import os
from pathlib import Path
import sys
import threading
import time

_monitor = None
_monitor_pid = None
_start_lock = threading.Lock()


def read_fields(path):
    """Read proc status/meminfo fields; kB values become bytes."""
    result = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if parts and parts[0].isdigit():
            result[key] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
            if key == "NSpid":
                result["namespace_pid"] = int(parts[-1])
    return result


def cgroup_snapshot(path):
    root = Path(path)
    result = {"path": str(root)}
    for name in (
        "memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes",
        "memory.failcnt", "memory.current", "memory.peak", "memory.max",
    ):
        file = root / name
        if file.exists():
            value = file.read_text().strip()
            result[name] = int(value) if value.isdigit() else value
    for name in ("memory.stat", "memory.events", "memory.oom_control"):
        file = root / name
        if file.exists():
            result[name] = {k: int(v) for k, v in (line.split() for line in file.read_text().splitlines())}
    return result


def current_memory_cgroup():
    entries = [line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()]
    v1 = next((entry[2] for entry in entries if "memory" in entry[1].split(",")), None)
    wanted = v1 if v1 is not None else next((entry[2] for entry in entries if entry[0] == "0"), None)
    if wanted is None:
        return None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, options = left.split(), right.split()
        if (v1 is not None and options[0] == "cgroup" and "memory" in options[2].split(",")) or (
            v1 is None and options[0] == "cgroup2"
        ):
            relative = os.path.relpath(wanted, fields[3])
            if relative == ".." or relative.startswith("../"):
                continue
            return str(Path(fields[4]) / relative)
    return None


def _append(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


class _ProcessMonitor:
    def __init__(self, directory, interval):
        self.pid = os.getpid()
        self.path = Path(directory) / f"process-{self.pid}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.lock = threading.Lock()
        self.stages = {}
        self.warned = False
        self.last_pss = 0.0
        self.cgroup = current_memory_cgroup()
        self._write({"kind": "start", "cgroup_membership": Path("/proc/self/cgroup").read_text()})
        threading.Thread(target=self._loop, name="memory-telemetry", daemon=True).start()

    def _write(self, record):
        record.update(time=time.time(), monotonic=time.monotonic(), pid=self.pid)
        with self.lock:
            record["stages"] = {str(k): list(v) for k, v in self.stages.items()}
            _append(self.path, record)

    def event(self, name, entering, step):
        tid = threading.get_ident()
        with self.lock:
            stack = self.stages.setdefault(tid, [])
            if entering:
                stack.append({"name": name, "step": step})
            elif stack:
                stack.pop()
            if not stack:
                self.stages.pop(tid, None)
        self._write({"kind": "stage", "name": name, "edge": "begin" if entering else "end", "step": step})

    def sample(self):
        record = {"kind": "sample", "status": read_fields("/proc/self/status")}
        record["host"] = read_fields("/proc/meminfo")
        if time.monotonic() - self.last_pss >= 10:
            try:
                record["smaps"] = read_fields("/proc/self/smaps_rollup")
            except OSError as exc:
                record["smaps_error"] = type(exc).__name__
            self.last_pss = time.monotonic()
        if self.cgroup:
            record["cgroup"] = cgroup_snapshot(self.cgroup)
        torch = sys.modules.get("torch")
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_initialized():
            record["cuda_allocated"] = torch.cuda.memory_allocated()
            record["cuda_reserved"] = torch.cuda.memory_reserved()
            stats = getattr(torch.cuda, "host_memory_stats", None)
            if callable(stats):
                values = stats()
                record["pinned"] = {k: values.get(k) for k in (
                    "allocated_bytes.current", "allocated_bytes.peak", "num_host_alloc", "num_host_free",
                    "active_bytes.current",
                )}
                record["pinned_active_caveat"] = "active_bytes counter is unreliable on deployed torch 2.11"
        self._write(record)

    def _loop(self):
        while True:
            try:
                self.sample()
            except Exception as exc:
                self.warn(exc)
            time.sleep(self.interval)

    def warn(self, exc):
        if not self.warned:
            self.warned = True
            print(f"Memory telemetry warning: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def start_memory_monitor():
    """Start once per process only when OPENMOPD_MEMORY_MONITOR_DIR is set."""
    global _monitor, _monitor_pid
    directory = os.environ.get("OPENMOPD_MEMORY_MONITOR_DIR")
    if not directory:
        return None
    if _monitor_pid == os.getpid():
        return _monitor
    with _start_lock:
        if _monitor_pid != os.getpid():
            try:
                interval = max(0.5, float(os.environ.get("OPENMOPD_MEMORY_MONITOR_INTERVAL", "2")))
                _monitor = _ProcessMonitor(directory, interval)
            except Exception as exc:
                print(f"Cannot start memory telemetry: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                _monitor = None
            _monitor_pid = os.getpid()
    return _monitor


def memory_stage(name):
    """Preserve call results/exceptions. Telemetry failures must never stop training."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            monitor = start_memory_monitor()
            step = None
            if monitor is not None:
                for value in (*args, *kwargs.values()):
                    info = getattr(value, "meta_info", None)
                    if isinstance(info, dict) and isinstance(info.get("global_steps"), int):
                        step = info["global_steps"]
                        break
                try:
                    monitor.event(name, True, step)
                except Exception as exc:
                    monitor.warn(exc)
            try:
                return function(*args, **kwargs)
            finally:
                if monitor is not None:
                    try:
                        monitor.event(name, False, step)
                    except Exception as exc:
                        monitor.warn(exc)
        return wrapped
    return decorate


def host_snapshot(container_id, group):
    result = {"time": time.time(), "monotonic": time.monotonic(), "host": read_fields("/proc/meminfo")}
    result["target_cgroup"] = cgroup_snapshot(group)
    processes = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            status = read_fields(directory / "status")
            membership = (directory / "cgroup").read_text()
            inside = container_id in membership
            if status.get("VmRSS", 0) < 32 * 1024**2 and not inside:
                continue
            entry = {
                "pid": int(directory.name), "name": (directory / "comm").read_text().strip(),
                "container": inside,
                **{key: status.get(key) for key in (
                    "PPid", "Uid", "VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSwap", "namespace_pid",
                )},
            }
            stat_text = (directory / "stat").read_text()
            entry["start_ticks"] = int(stat_text[stat_text.rfind(")") + 2:].split()[19])
            processes.append(entry)
        except (OSError, ValueError):
            continue
    result["processes"] = sorted(processes, key=lambda item: item.get("VmRSS") or 0, reverse=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--cgroup-path", required=True)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--samples", type=int, default=0, help="0 means until --stop-file exists")
    parser.add_argument("--stop-file")
    args = parser.parse_args()
    if args.interval < 0.5 or args.samples < 0:
        parser.error("interval must be >=0.5 and samples >=0")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        begin = time.monotonic()
        try:
            record = host_snapshot(args.container_id, args.cgroup_path)
            record["sample_seconds"] = time.monotonic() - begin
        except Exception as exc:
            record = {"time": time.time(), "error": f"{type(exc).__name__}: {exc}"}
        _append(args.output, record)
        index += 1
        if (args.samples and index >= args.samples) or (args.stop_file and Path(args.stop_file).exists()):
            break
        time.sleep(max(0.01, args.interval - (time.monotonic() - begin)))


if __name__ == "__main__":
    main()
