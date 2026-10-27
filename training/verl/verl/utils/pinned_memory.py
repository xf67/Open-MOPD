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
"""Long-lived, single-device pipeline buffers outside PyTorch's pinned cache.

Round capacity, not tensor shape. Allocate/register once at pipeline startup;
reuse throughout training. Owners of CUDA graphs must retain their host tensors
until the graphs are destroyed. Do not send these buffers to other processes or
use them on another CUDA device. Ordinary transient tensors still use PyTorch.
"""

import ctypes
import mmap
import os
import sys
import threading
import warnings

_lock = threading.Lock()
_stats = {"allocated_bytes": 0, "requested_bytes": 0, "allocation_count": 0}
# If CUDA reports an error on release, retain the mapping until process exit.
# Unmapping a possibly in-flight DMA buffer would be unsafe.
_failed_releases = []


def pinned_alignment_bytes():
    value = int(os.environ.get("OPENMOPD_PINNED_ALIGNMENT_MB", "0"))
    if value < 0:
        raise ValueError("OPENMOPD_PINNED_ALIGNMENT_MB must be a nonnegative integer (MiB)")
    return value * 2**20


def registered_pinned_stats():
    """Live custom allocations only; disjoint from torch.cuda.host_memory_stats."""
    with _lock:
        return dict(_stats)


class _RegisteredMapping(mmap.mmap):
    def __new__(cls, capacity, requested, cuda):
        return super().__new__(
            cls, -1, capacity, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

    def __init__(self, capacity, requested, cuda):
        self._registered = False
        self._cuda = cuda
        self._device = cuda.current_device()
        self._pid = os.getpid()
        self._capacity = capacity
        self._requested = requested
        self._ptr = ctypes.addressof(ctypes.c_char.from_buffer(self))
        self._runtime = cuda.cudart()
        cuda.check_error(self._runtime.cudaHostRegister(self._ptr, capacity, 0))
        self._registered = True
        with _lock:
            _stats["allocated_bytes"] += capacity
            _stats["requested_bytes"] += requested
            _stats["allocation_count"] += 1

    def __del__(self):
        if not getattr(self, "_registered", False):
            return
        # A forked child cannot use the parent's CUDA context. At interpreter
        # shutdown the OS/driver reclaim mappings and registrations together.
        if os.getpid() != self._pid or sys.is_finalizing():
            return
        try:
            # Unlike the caching host allocator, these buffers have no stream
            # event tracking. Synchronize only when the LAST storage alias dies.
            with self._cuda.device(self._device):
                self._cuda.synchronize(self._device)
                self._cuda.check_error(self._runtime.cudaHostUnregister(self._ptr))
            self._registered = False
            with _lock:
                _stats["allocated_bytes"] -= self._capacity
                _stats["requested_bytes"] -= self._requested
                _stats["allocation_count"] -= 1
            self.close()
        except Exception as exc:
            _failed_releases.append(self)
            warnings.warn(f"Keeping pinned mapping after CUDA release failure: {exc}", RuntimeWarning, stacklevel=2)


def empty_pinned_like(tensor):
    """Allocate a contiguous CPU buffer; 0 keeps the original PyTorch allocator.

    frombuffer retains the mmap owner through the underlying tensor storage,
    including .data, detach(), views and FSDP aliases. A tensor-only finalizer
    would incorrectly free the region while one of those aliases was still live.
    """
    import torch

    alignment = pinned_alignment_bytes()
    if not alignment or tensor.numel() == 0:
        return torch.empty_like(tensor, device="cpu", pin_memory=True)
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Aligned pinned pipeline buffers require Linux and CUDA; set alignment to 0 for fallback")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError("Aligned pinned pipeline buffers require a contiguous dense tensor")
    requested = tensor.numel() * tensor.element_size()
    capacity = (requested + alignment - 1) // alignment * alignment
    region = _RegisteredMapping(capacity, requested, torch.cuda)
    return torch.frombuffer(region, dtype=tensor.dtype, count=tensor.numel()).view(tensor.shape)
