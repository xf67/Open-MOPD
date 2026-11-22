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
"""One host copy shared by all ranks of a single node (CPU data parallel).

Tensors live in named files under ``/dev/shm/openmopd-<namespace>/`` mapped with
``MAP_SHARED``. Every process that copies a tensor to or from its GPU registers
its own mapping with ``cudaHostRegister``; the physical pages are pinned once and
shared, so asynchronous DMA works without a private pinned replica. Never call
``pin_memory()`` on these tensors: that would create a private copy.

Page placement is set with ``mbind`` before the first touch: ``interleave`` for
data that every GPU reads (teachers, student BF16 weights), ``node:<n>`` for data
written or reduced by one socket (per-rank gradient staging, optimizer state).

Arenas are created collectively: the creator allocates, places and fills the file,
every rank meets at a CPU (gloo) barrier, and the remaining ranks attach. Layouts
are a pure function of the specs, so attachers compute the same offsets.
"""

import ctypes
import functools
import mmap
import os
import shutil
import sys
import threading
import time
import uuid
import warnings

import torch

NAMESPACE_ENV = "OPENMOPD_SHM_NAMESPACE"
_ROOT = "/dev/shm"
_ALIGN = 2 * 2**20  # file capacity; per-tensor offsets are 4 KiB aligned
_TENSOR_ALIGN = 4096
_MPOL_PREFERRED = 1
_MPOL_INTERLEAVE = 3
_SYS_MBIND = {"x86_64": 237, "aarch64": 235}
_lock = threading.Lock()
_stats = {"mapped_bytes": 0, "registered_bytes": 0}


# --------------------------------------------------------------------------- namespace
def new_namespace():
    """Driver side: create a fresh namespace directory and export it."""
    name = f"openmopd-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    os.makedirs(os.path.join(_ROOT, name), mode=0o700)
    os.environ[NAMESPACE_ENV] = name
    return name


def remove_namespace(name):
    """Unlink a namespace. Live mappings stay valid until their last reference dies."""
    if name and name.startswith("openmopd-") and "/" not in name:
        shutil.rmtree(os.path.join(_ROOT, name), ignore_errors=True)


def namespace_dir():
    name = os.environ.get(NAMESPACE_ENV)
    if not name:
        raise RuntimeError(f"CPU data parallel requires {NAMESPACE_ENV}; it is created by the trainer driver")
    path = os.path.join(_ROOT, name)
    if not os.path.isdir(path):
        raise RuntimeError(f"shared host namespace {path} does not exist")
    return path


def stats():
    with _lock:
        return dict(_stats)


# --------------------------------------------------------------------------- NUMA
@functools.cache
def numa_nodes():
    """{node: [cpu, ...]} for online NUMA nodes; {0: all cpus} without sysfs."""
    base = "/sys/devices/system/node"
    nodes = {}
    try:
        for entry in sorted(os.listdir(base)):
            if entry.startswith("node") and entry[4:].isdigit():
                with open(os.path.join(base, entry, "cpulist")) as handle:
                    nodes[int(entry[4:])] = _parse_cpulist(handle.read())
    except OSError:
        pass
    return nodes or {0: sorted(os.sched_getaffinity(0))}


def _parse_cpulist(text):
    cpus = []
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def gpu_numa_node(device=None):
    """NUMA node of a CUDA device from its PCI address (0 when unknown)."""
    device = torch.cuda.current_device() if device is None else device
    props = torch.cuda.get_device_properties(device)
    try:
        address = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{address}/numa_node") as handle:
            node = int(handle.read().strip())
    except (AttributeError, OSError, ValueError):
        return 0
    return node if node in numa_nodes() else 0


def bind_process_to_node(node):
    """Restrict this process's CPUs to a NUMA node (no-op on single-node hosts)."""
    nodes = numa_nodes()
    if len(nodes) > 1 and node in nodes:
        os.sched_setaffinity(0, nodes[node])


def _mbind(address, length, policy):
    if policy is None or len(numa_nodes()) < 2:
        return
    number = _SYS_MBIND.get(os.uname().machine)
    if number is None:
        return
    if policy == "interleave":
        mode, nodes = _MPOL_INTERLEAVE, list(numa_nodes())
    elif policy.startswith("node:"):
        mode, nodes = _MPOL_PREFERRED, [int(policy[5:])]
    else:
        raise ValueError(f"unknown NUMA policy {policy!r}")
    maxnode = max(nodes) + 2
    mask = (ctypes.c_ulong * ((maxnode + 63) // 64))()
    for node in nodes:
        mask[node // 64] |= 1 << (node % 64)
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(number, ctypes.c_void_p(address), ctypes.c_ulong(length), mode, mask,
                          ctypes.c_ulong(maxnode), 0)
    if result != 0:
        warnings.warn(f"mbind({policy}) failed: errno {ctypes.get_errno()}", RuntimeWarning, stacklevel=3)


# --------------------------------------------------------------------------- mapping
class _SharedMapping(mmap.mmap):
    """MAP_SHARED view of a /dev/shm file, optionally cudaHostRegister-ed by this process."""

    def __new__(cls, path, capacity, create, policy, register):
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)
        try:
            if create:
                os.ftruncate(fd, capacity)
            elif os.fstat(fd).st_size != capacity:
                raise RuntimeError(f"{path}: size {os.fstat(fd).st_size} != expected {capacity}")
            self = super().__new__(cls, fd, capacity, flags=mmap.MAP_SHARED,
                                   prot=mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        return self

    def __init__(self, path, capacity, create, policy, register):
        self.path = path
        self.capacity = capacity
        self._registered = False
        self._pid = os.getpid()
        self._ptr = ctypes.addressof(ctypes.c_char.from_buffer(self))
        if create:
            _mbind(self._ptr, capacity, policy)
            _prefault(self._ptr, capacity)
        with _lock:
            _stats["mapped_bytes"] += capacity
        if register:
            self.register()

    def register(self):
        if self._registered:
            return
        cuda = torch.cuda
        self._device = cuda.current_device()
        self._runtime = cuda.cudart()
        cuda.check_error(self._runtime.cudaHostRegister(self._ptr, self.capacity, 0))
        self._registered = True
        with _lock:
            _stats["registered_bytes"] += self.capacity

    def __del__(self):
        try:
            if getattr(self, "_registered", False) and os.getpid() == self._pid and not sys.is_finalizing():
                with torch.cuda.device(self._device):
                    torch.cuda.synchronize(self._device)
                    torch.cuda.check_error(self._runtime.cudaHostUnregister(self._ptr))
                with _lock:
                    _stats["registered_bytes"] -= self.capacity
            self._registered = False
            if getattr(self, "capacity", None) is not None and os.getpid() == getattr(self, "_pid", None):
                with _lock:
                    _stats["mapped_bytes"] -= self.capacity
                self.capacity = None
            self.close()
        except Exception as exc:  # never unmap a range a DMA may still use
            warnings.warn(f"Keeping shared mapping after release failure: {exc}", RuntimeWarning, stacklevel=2)


def _prefault(address, length):
    """Touch every page under the mbind policy (memset is multi-GB/s, single call)."""
    ctypes.memset(address, 0, length)


def _layout(specs):
    offsets, cursor = {}, 0
    for key, shape, dtype in specs:
        numel = 1
        for size in shape:
            numel *= int(size)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        offsets[key] = (cursor, numel, tuple(int(s) for s in shape), dtype)
        cursor += (nbytes + _TENSOR_ALIGN - 1) // _TENSOR_ALIGN * _TENSOR_ALIGN
    capacity = max(_ALIGN, (cursor + _ALIGN - 1) // _ALIGN * _ALIGN)
    return offsets, capacity


def _tensors(mapping, offsets):
    out = {}
    for key, (offset, numel, shape, dtype) in offsets.items():
        if numel == 0:
            out[key] = torch.empty(shape, dtype=dtype)
        else:
            out[key] = torch.frombuffer(mapping, dtype=dtype, count=numel, offset=offset).view(shape)
    return out


def open_arena(name, specs, create, policy=None, register=False):
    """Create or attach one arena locally (no collective). Returns {key: tensor}."""
    if "/" in name:
        raise ValueError("arena names must not contain '/'")
    offsets, capacity = _layout(specs)
    mapping = _SharedMapping(os.path.join(namespace_dir(), name), capacity, create, policy, register)
    return _tensors(mapping, offsets)


def cpu_barrier(group=None):
    """Barrier over gloo (CPU tensors), never NCCL."""
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(torch.zeros(1), group=group)


def shared_arena(name, specs, creator=0, policy=None, register=True, attach=True, fill=None, group=None):
    """Collective: ``creator`` allocates and runs ``fill(tensors)``; others attach after a barrier.

    ``register`` and ``attach`` may be booleans or callables of the global rank.
    Ranks that do not attach still take part in the barrier and receive ``None``.
    """
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    pick = lambda value: value(rank) if callable(value) else value
    tensors = None
    if rank == creator:
        tensors = open_arena(name, specs, create=True, policy=policy, register=pick(register))
        if fill is not None:
            fill(tensors)
    cpu_barrier(group)
    if rank != creator and pick(attach):
        tensors = open_arena(name, specs, create=False, register=pick(register))
    return tensors


def wait_until(predicate, timeout=3600.0, what="shared host condition", poll=5e-5):
    """Spin briefly, then sleep-poll; shared flags are plain aligned int64 stores (x86 TSO)."""
    deadline = time.monotonic() + timeout
    spins = 0
    while not predicate():
        spins += 1
        if spins > 2000:
            time.sleep(poll if spins < 20000 else 1e-3)
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what}")
