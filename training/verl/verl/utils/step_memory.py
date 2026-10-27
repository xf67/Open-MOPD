
"""Per-step allocator peaks and sampled host memory, without CUDA synchronization.

The window is generation through actor update (including offload), excluding
validation/checkpoints. Asynchronous work is attributed to its wall-clock window;
we never flush the optimizer or synchronize CUDA to improve attribution.
"""

import functools
import logging
import math
import threading
import time
from contextlib import contextmanager

import psutil

logger = logging.getLogger(__name__)
GIB = 1024**3
_active_monitor = None
PHASE_VALUES = {"gpu_allocated", "gpu_reserved", "gpu_device_used", "worker_rss", "pinned_pool"}


def worker_tree_pss(process):
    """PSS, not summed RSS: CPUAdam shares large parameter/gradient mappings.

    Covers this worker and descendants only, not Ray's driver or sibling actors.
    A denied/incomplete census is omitted instead of reported as a small value.
    """
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_full_info().pss
        except psutil.NoSuchProcess:
            continue
    return total


class StepMemoryMonitor:
    def __init__(self, device=None, interval_s=0.5, pss_interval_s=10.0):
        if not math.isfinite(interval_s) or interval_s < 0.1:
            raise ValueError("step_memory.interval_s must be finite and >= 0.1")
        if not math.isfinite(pss_interval_s) or pss_interval_s < 0:
            raise ValueError("step_memory.pss_interval_s must be finite and >= 0 (0 disables PSS)")
        self.device = device
        self.device_id = device.current_device() if device is not None else None
        self.interval_s = interval_s
        self.pss_interval_s = pss_interval_s
        self.process = psutil.Process()
        self.step = None
        self.thread = None
        self.stop = threading.Event()
        self.warned = False
        self.lock = threading.RLock()
        self.previous_end = {}
        self.owner = None
        self.pss_active = threading.Event()
        self.pss_thread = None
        self.window = 0

    def _error(self, exc):
        self.errors += 1
        if not self.warned:
            logger.warning("Step memory sampling incomplete: %s", exc)
            self.warned = True

    def _peak(self, key, value):
        self.peaks[key] = max(self.peaks.get(key, 0), value / GIB)
        if self.phase is not None and key.removesuffix("_peak_gib") in PHASE_VALUES:
            phase = self.phases[self.phase]
            phase[key] = max(phase.get(key, 0), value / GIB)

    def _current(self, name, value, values):
        values[name] = value / GIB
        self._peak(f"{name}_peak_gib", value)

    def _sample(self):
        with self.lock:
            return self._sample_locked()

    def _sample_locked(self):
        self.samples += 1
        values = {}
        try:
            self._current("worker_rss", self.process.memory_info().rss, values)
            host = psutil.virtual_memory()
            self._peak("host_used_peak_gib", host.total - host.available)
        except Exception as exc:
            self._error(exc)
        if self.device is not None:
            try:
                self._current("gpu_allocated", self.device.memory_allocated(self.device_id), values)
                self._current("gpu_reserved", self.device.memory_reserved(self.device_id), values)
            except Exception as exc:
                self._error(exc)
            try:
                # Query physical residency independently of allocator accounting:
                # vLLM sleep can unmap CuMem storage while tensors remain alive.
                free, total = self.device.mem_get_info(self.device_id)
                self._current("gpu_device_used", total - free, values)
            except Exception as exc:
                self._error(exc)
            try:
                host_stats = getattr(self.device, "host_memory_stats", None)
                if host_stats is not None:
                    # allocated_bytes includes active + cached pinned blocks.
                    # active_bytes is unreliable on the deployed torch build.
                    self._current("pinned_pool", host_stats()["allocated_bytes.current"], values)
            except Exception as exc:
                self._error(exc)
        return values

    def _pss_loop(self):
        # Reading smaps for a ~100 GiB worker can take over a second. Never
        # perform it on the training thread or while holding the phase lock.
        while True:
            self.pss_active.wait()
            with self.lock:
                window = self.window
                active = self.step is not None
            if not active:
                continue
            started = time.monotonic()
            value, error = None, None
            try:
                value = worker_tree_pss(self.process)
            except Exception as exc:
                error = exc
            finished = time.monotonic()
            with self.lock:
                # Discard a sample that crossed a logical step boundary. In
                # particular, never let an old PSS read contaminate a new step.
                if self.step is not None and self.window == window:
                    if error is not None:
                        self._error(error)
                    else:
                        self.pss_samples += 1
                        self._peak("worker_tree_pss_sampled_peak_gib", value)
                        self.latest_pss = (value / GIB, finished, finished - started)
            time.sleep(self.pss_interval_s)

    def _capture_gpu_peak(self, reset=False):
        if self.gpu_peak_valid:
            try:
                self._peak("gpu_allocated_peak_gib", self.device.max_memory_allocated(self.device_id))
                self._peak("gpu_reserved_peak_gib", self.device.max_memory_reserved(self.device_id))
            except Exception as exc:
                self._error(exc)
        if reset and self.device is not None:
            try:
                self.device.reset_peak_memory_stats(self.device_id)
                self.gpu_peak_valid = True
            except Exception as exc:
                self.gpu_peak_valid = False
                self._error(exc)

    def enter_phase(self, name):
        # CPUAdam background threads must not reset the main thread's counters.
        if self.step is None or threading.get_ident() != self.owner:
            return False
        with self.lock:
            if self.phase is not None:  # joint scoring includes nested teachers
                return False
            self._capture_gpu_peak(reset=True)
            self.phase = name
            phase = self.phases.setdefault(name, {})
            values = self._sample_locked()
            for key, value in values.items():
                phase.setdefault(f"{key}_start_gib", value)
            return True

    def exit_phase(self):
        with self.lock:
            values = self._sample_locked()
            self._capture_gpu_peak(reset=True)
            phase = self.phases[self.phase]
            phase.update({f"{key}_end_gib": value for key, value in values.items()})
            self.phase = None

    def _loop(self):
        while not self.stop.wait(self.interval_s):
            self._sample()

    def begin_transition(self):
        """Sample immediately before sleep/wake, without resetting GPU peaks."""
        with self.lock:
            if self.step is None or self.device is None or threading.get_ident() != self.owner:
                return None
            return self.window, self._sample_locked().get("gpu_device_used")

    def end_transition(self, name, token):
        with self.lock:
            window, before = token
            if self.step is None or self.window != window or threading.get_ident() != self.owner:
                return
            after = self._sample_locked().get("gpu_device_used")
            # Keep one matched pair (the last call), not unrelated maxima.
            values = {"call_count": self.transitions.get(name, {}).get("call_count", 0) + 1}
            if before is not None:
                values["gpu_device_used_before_gib"] = before
            if after is not None:
                values["gpu_device_used_after_gib"] = after
            if before is not None and after is not None:
                values["gpu_device_used_delta_gib"] = after - before
            self.transitions[name] = values

    def begin(self, step):
        # Dynamic filtering may generate multiple batches for one logical step.
        if self.step == step:
            return
        if self.step is not None:
            self.finish(self.step)
        with self.lock:
            self._begin_window(step)

    def _begin_window(self, step):
        global _active_monitor
        self.step = step
        self.peaks = {}
        self.phases = {}
        self.transitions = {}
        self.phase = None
        self.samples = self.pss_samples = self.errors = 0
        self.latest_pss = None
        self.window += 1
        self.gpu_peak_valid = False
        self.stop.clear()
        self.owner = threading.get_ident()
        _active_monitor = self
        self._capture_gpu_peak(reset=True)
        self._sample()
        self.thread = threading.Thread(target=self._loop, name="step-memory", daemon=True)
        self.thread.start()
        if self.pss_interval_s:
            self.pss_active.set()
            if self.pss_thread is None:
                self.pss_thread = threading.Thread(target=self._pss_loop, name="step-pss", daemon=True)
                self.pss_thread.start()

    def finish(self, step):
        global _active_monitor
        if self.step is None:
            return {}
        if self.step != step:
            raise ValueError(f"Memory window step {self.step} does not match {step}")
        self.stop.set()
        self.thread.join()
        with self.lock:
            end = self._sample_locked()
            self._capture_gpu_peak()
            self.step = None
            self.pss_active.clear()
            result = {f"memory/step_{key}": value for key, value in self.peaks.items()}
            if self.latest_pss is not None:
                value, timestamp, duration = self.latest_pss
                result.update({
                    "memory/step_worker_tree_pss_latest_gib": value,
                    "memory/step_pss_latest_age_s": time.monotonic() - timestamp,
                    "memory/step_pss_read_duration_s": duration,
                })
        for key, value in end.items():
            result[f"memory/step_{key}_end_gib"] = value
            if key in self.previous_end:
                result[f"memory/step_{key}_end_delta_gib"] = value - self.previous_end[key]
        self.previous_end = end
        for phase, values in self.phases.items():
            result.update({f"memory/{phase}_{key}": value for key, value in values.items()})
        for transition, values in self.transitions.items():
            result.update({f"memory/{transition}_{key}": value for key, value in values.items()})
        result.update({
            "memory/step_sample_count": self.samples,
            "memory/step_pss_sample_count": self.pss_samples,
            "memory/step_sample_interval_s": self.interval_s,
            "memory/step_pss_interval_s": self.pss_interval_s,
            "memory/step_sampling_errors": self.errors,
        })
        self.thread = None
        if _active_monitor is self:
            _active_monitor = None
        return result


def memory_phase(name):
    """Annotate top-level worker stages; keep results/exceptions unchanged."""
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            monitor = _active_monitor
            entered = False
            if monitor is not None:
                try:
                    entered = monitor.enter_phase(name)
                except Exception as exc:
                    monitor._error(exc)
            try:
                return function(*args, **kwargs)
            finally:
                if entered:
                    try:
                        monitor.exit_phase()
                    except Exception as exc:
                        monitor.phase = None
                        monitor._error(exc)
        return wrapped
    return decorate


@contextmanager
def memory_transition(name):
    """Record physical usage around a vLLM call, including inside a phase.

    Samples also contribute to phase/step peaks. A disabled monitor is a
    pass-through, and instrumentation failures never replace model exceptions.
    """
    monitor = _active_monitor
    token = None
    if monitor is not None:
        try:
            token = monitor.begin_transition()
        except Exception as exc:
            monitor._error(exc)
    try:
        yield
    finally:
        if token is not None:
            try:
                monitor.end_transition(name, token)
            except Exception as exc:
                monitor._error(exc)


def reduce_step_memory_metrics(worker_metrics):
    """Use max across ranks, never mean or sums of overlapping shared memory.

    On multiple ranks these are worst-worker/worst-device peaks, not job totals.
    """
    result = {}
    for metrics in worker_metrics:
        for key, value in metrics.items():
            result[key] = max(result.get(key, value), value)
    return result
