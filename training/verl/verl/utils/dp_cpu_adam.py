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
"""CPU data parallel CPUAdamPipeline: CPU gradient reduction and one CPU AdamW.

Every rank keeps a full BF16 student replica on its GPU (FSDP over a rank-local
process group, so FSDP never communicates). Gradients never cross GPUs:

* each rank's existing early-offload hooks copy its local layer gradients into its
  own staging file, placed on the rank's NUMA node (D2H stays on-socket);
* the optimizer owner runs one worker process per NUMA node. A worker owns a
  contiguous block of parameters whose FP32 master, moments and reduced gradient
  live on its node; it sums the ranks' staging buffers (in rank order, so the
  result is deterministic), returns its partial squared norm, and after the owner
  computes the global clip coefficient runs DeepSpeed CPUAdam on its block and
  writes BF16 weights into the single shared host copy;
* every rank restores/flushes its GPU weights from that shared copy, so replicas
  stay bitwise identical by construction.

Losses are expected to be summed across ranks (see core_algos.global_loss_denominator):
the reduced gradient is the sum of the ranks' gradients, not their mean.

Ordering: a rank sets ready[rank]=seq after its gradient DMA completes; the owner
waits for all ranks, updates, then sets published=seq. A rank reads the shared
weights only after published>=seq, and writes its staging for seq+1 only during
the next backward, which requires restored (published) weights. The owner writes
the shared weights only after every rank's ready flag, i.e. after each rank's
previous restore finished (restore precedes forward precedes backward).
"""

import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from verl.utils import shared_host
from verl.utils.cpu_adam_pipeline import CPUAdamPipeline
from verl.utils.memory_monitor import memory_stage, start_memory_monitor

_PREFIX = "student_"
# 1M elements: the sum stays in L2 between the four adds and the norm (16M chunks with an
# FP64 norm measured ~10x slower: 6.5 s vs 0.6 s for 1.5e9 elements on a Xeon 8336C node).
_CHUNK = 1 << 20


def _node_worker(connection, node, cpus, threads, world, stage_dtype, total, part, host_specs, opt_specs, meta,
                 group_count, stage_name="stage"):
    """One NUMA node's share of reduction and AdamW. CPUAdam holds the GIL, hence a process."""
    start_memory_monitor()
    try:
        if cpus:
            os.sched_setaffinity(0, cpus)
        torch.set_num_threads(threads)
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        lo, hi = part
        stages = [
            shared_host.open_arena(f"{_PREFIX}{stage_name}{r}", [("g", (total,), stage_dtype)],
                                   create=False)["g"][lo:hi]
            for r in range(world)
        ]
        host = shared_host.open_arena(f"{_PREFIX}host", host_specs, create=False)
        opt = shared_host.open_arena(f"{_PREFIX}opt{node}", opt_specs, create=False)
        grad_flat = opt["grad"]
        masters, outputs, grads = [], [], []
        groups = [[] for _ in range(group_count)]
        for _index, offset, numel, shape, host_key, host_offset, group in meta:
            master = torch.nn.Parameter(opt["master"][offset:offset + numel].view(shape), requires_grad=False)
            masters.append(master)
            grads.append(grad_flat[offset:offset + numel].view(shape))
            outputs.append(host[host_key][host_offset:host_offset + numel].view(shape))
            groups[group].append(master)
        used = [g for g in range(group_count) if groups[g]]
        optimizer = None
        if used:
            optimizer = DeepSpeedCPUAdam([{"params": groups[g]} for g in used], adamw_mode=True,
                                         fp32_optimizer_states=True)
            for (_i, offset, numel, shape, *_), master in zip(meta, masters, strict=True):
                optimizer.state[master] = {
                    "step": 0,
                    "exp_avg": opt["exp_avg"][offset:offset + numel].view(shape),
                    "exp_avg_sq": opt["exp_avg_sq"][offset:offset + numel].view(shape),
                }
        connection.send((True, None))
        while True:
            message = connection.recv()
            if message is None:
                break
            if message[0] == "reduce":
                start = time.perf_counter()
                sumsq = 0.0
                with torch.no_grad():
                    for s in range(0, hi - lo, _CHUNK):
                        e = min(s + _CHUNK, hi - lo)
                        out = grad_flat[s:e]
                        out.copy_(stages[0][s:e])
                        for stage in stages[1:]:
                            out.add_(stage[s:e])
                        sumsq += float(torch.dot(out, out))  # FP32 per chunk, FP64 across chunks
                connection.send((True, (sumsq, time.perf_counter() - start)))
            elif message[0] == "adam":
                _, scale, options, steps, live = message
                start = time.perf_counter()
                if optimizer is not None:
                    with torch.no_grad():
                        if scale < 1.0:
                            grad_flat.mul_(scale)
                    for group, g in zip(optimizer.param_groups, used, strict=True):
                        group.update(options[g])
                    for master, grad, step, is_live in zip(masters, grads, steps, live, strict=True):
                        master.grad = grad if is_live else None
                        optimizer.state[master]["step"] = step
                    with torch.cuda.nvtx.range("openmopd::cpu::adam_native"):
                        optimizer.step()
                    with torch.no_grad():
                        for master, output, is_live in zip(masters, outputs, live, strict=True):
                            if is_live:
                                output.copy_(master)
                connection.send((True, time.perf_counter() - start))
            else:
                raise ValueError(f"unknown node worker command {message[0]!r}")
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


class SharedCPUAdamPipeline(CPUAdamPipeline):
    """CPUAdamPipeline for CPU data parallel: identical GPU-facing protocol, one shared CPU optimizer."""

    global_step_decision = True  # the actor must call step() on every rank; skips are decided globally

    def __init__(self, optimizer, model, overlap=True, threads=8, gradient_offload=True,
                 async_update=None, fp32_gradients=False, owner_rank=0, node_threads=None,
                 accumulate=False, ring_slots=4):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp._runtime_utils import _lazy_init

        if not gradient_offload:
            raise ValueError("CPU data parallel requires pipeline_gradient_offload=true")
        if async_update is not None and not isinstance(async_update, bool):
            raise ValueError("cpu_adam_async must be true, false, or null")
        if optimizer.state:
            raise ValueError("SharedCPUAdamPipeline must be constructed before any optimizer updates")
        if not isinstance(model, FSDP) or threads < 1:
            raise ValueError("SharedCPUAdamPipeline requires FSDP1 and a positive CPU thread count")
        if accumulate and async_update is not False:
            raise ValueError("gradient accumulation (large_batch) requires cpu_adam_async=false")
        self.backend = "cpu"
        self.accumulate = accumulate
        self.fp32_gradients = fp32_gradients
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.owner = owner_rank
        self.is_owner = self.rank == owner_rank
        if not 0 <= owner_rank < self.world:
            raise ValueError("cpu_dp owner rank must be in [0, world_size)")
        torch.set_num_threads(threads)
        _lazy_init(model, model)
        self.modules = {
            n: m for n, m in model.named_modules() if isinstance(m, FSDP) and m._handle is not None
        }
        self.device = model.compute_device
        self.node = shared_host.gpu_numa_node(self.device)
        if fp32_gradients:
            for module in self.modules.values():
                precision = module.mixed_precision
                if not precision.keep_low_precision_grads or precision.reduce_dtype != torch.float32:
                    raise ValueError("FP32 staging requires keep_low_precision_grads=True and reduce_dtype=float32")
            for parameter in model.parameters():
                parameter.grad_dtype = None
            for module in self.modules.values():
                module._handle.flat_param.grad_dtype = torch.float32

        # --- shared BF16 host weights: one copy, interleaved, registered by every rank
        host_specs, flats = [], {}
        for name, module in self.modules.items():
            handle = module._handle
            flat = handle.flat_param._local_shard
            if (handle.world_size != 1 or handle.uses_sharded_strategy or handle._offload_params
                    or not handle._use_orig_params or handle._uses_param_mixed_precision
                    or flat.dtype != torch.bfloat16):
                raise ValueError("CPU data parallel requires rank-local BF16 NO_SHARD original parameters")
            host_specs.append((name, tuple(flat.shape), flat.dtype))
            flats[name] = flat

        def fill_host(tensors):
            for name, flat in flats.items():
                tensors[name].copy_(flat)

        self.host_weights = shared_host.shared_arena(
            f"{_PREFIX}host", host_specs, creator=owner_rank, policy="interleave", fill=fill_host,
        )
        self.gpu_weights = {name: flat.detach() for name, flat in flats.items()}
        storage = {flat.untyped_storage().data_ptr(): (name, flat) for name, flat in flats.items()}

        torch.optim.Optimizer.__init__(self, optimizer.param_groups, optimizer.defaults)
        # --- parameter table (order defines staging and optimizer layouts)
        table = []  # (param, group, host_key, host_offset)
        for gi, group in enumerate(self.param_groups):
            if group.get("maximize", False) or group.get("amsgrad", False):
                raise ValueError("CPU pipeline supports ordinary AdamW only")
            for param in group["params"]:
                name, flat = storage[param.untyped_storage().data_ptr()]
                table.append((param, gi, name, param.storage_offset() - flat.storage_offset()))
        self.total = sum(p.numel() for p, *_ in table)
        grad_dtype = torch.float32 if fp32_gradients else torch.bfloat16
        # Accumulation: the per-rank stage is an FP32 accumulator written by the CPU (not
        # CUDA-registered); gradients land in a small pinned ring first. Otherwise the stage
        # receives the gradient DMA directly in the gradient dtype.
        stage_dtype = torch.float32 if accumulate else grad_dtype
        self.stage_name = "accum" if accumulate else "stage"

        # --- per-rank gradient staging on the rank's NUMA node; only its writer registers it
        stage = None
        for r in range(self.world):
            arena = shared_host.shared_arena(
                f"{_PREFIX}{self.stage_name}{r}", [("g", (self.total,), stage_dtype)], creator=r,
                policy=f"node:{self.node}" if r == self.rank else None, attach=False,
                register=not accumulate,
            )
            if r == self.rank:
                stage = arena["g"]

        # --- control block (owner-created, registered nowhere)
        self.ctrl = shared_host.shared_arena(
            f"{_PREFIX}ctrl",
            [("ready", (self.world,), torch.int64), ("published", (1,), torch.int64),
             ("floats", (8,), torch.float64), ("live", (self.world, len(table)), torch.uint8)],
            creator=owner_rank, register=False,
        )

        # --- pairs/state with the base class's shapes
        self.pairs, offset = [], 0
        for param, _gi, name, host_offset in table:
            numel = param.numel()
            staging = stage[offset:offset + numel].view(param.shape)
            output = self.host_weights[name][host_offset:host_offset + numel].view(param.shape)
            self.pairs.append((param, None, staging, output))
            self.state[param] = {"step": 0}
            offset += numel
        self.shared_gradients = []
        self.stage_flat = stage

        # Start from identical weights everywhere (all ranks loaded the same checkpoint;
        # this also guards against nondeterministic initialization).
        with torch.no_grad():
            for name, destination in self.gpu_weights.items():
                destination.copy_(self.host_weights[name], non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()

        self.workers = []
        if self.is_owner:
            if node_threads is None:  # reduction/AdamW are bandwidth bound; half a node's cores saturate it
                per_node = min(len(cpus) for cpus in shared_host.numa_nodes().values())
                node_threads = max(threads, min(16, per_node // 2))
            self._start_node_workers(table, host_specs, stage_dtype, node_threads)
        shared_host.cpu_barrier()

        self.overlap = overlap
        self.async_update = async_update
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-adam")
        self.signal_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-dp-signal")
        self.signal_future = None
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.future = None
        self.pending = None
        self.submit_lock = threading.Lock()
        self.version = 0
        self.seq = 0
        self.max_norm = None
        self.last_metrics = {}
        self.gradient_offload = True
        self.gradient_norms = {}
        self.gradient_scale = None
        self.gradient_layers = set()
        self.backward_hooks = {}
        self.forward_hooks = []
        indices = {id(p): index for index, (p, *_) in enumerate(self.pairs)}
        self.layer_indices = {
            name: [indices[id(p)] for p in module._handle.flat_param._params if id(p) in indices]
            for name, module in self.modules.items()
        }
        from functools import partial

        for name, module in self.modules.items():
            self.forward_hooks.append(module.register_forward_hook(partial(self._arm_gradient_offload, name)))
        if accumulate:
            self._init_accumulation(grad_dtype, ring_slots)

    # ------------------------------------------------------------------ FP32 accumulation
    def _init_accumulation(self, grad_dtype, ring_slots):
        """Pinned landing ring + one accumulate thread per rank (large-batch mode).

        Backward hook: layer gradient -> free ring slot (D2H on copy_stream) -> queue. The
        accumulate thread waits for the copy, then copies (first micro-batch of the
        mini-batch) or adds the slot into this rank's FP32 accumulator and frees the slot.
        """
        import queue

        if ring_slots < 1:
            raise ValueError("accum_ring_slots must be positive")
        starts, acc = [], 0
        for param, *_ in self.pairs:
            starts.append(acc)
            acc += param.numel()
        self.param_starts = starts
        largest = max(sum(self.pairs[i][0].numel() for i in indices) for indices in self.layer_indices.values())
        template = torch.empty((largest,), dtype=grad_dtype, device="meta")
        from verl.utils.pinned_memory import empty_pinned_like

        self.ring = [empty_pinned_like(template) for _ in range(ring_slots)]
        self.free_slots = queue.Queue()
        for slot in range(ring_slots):
            self.free_slots.put(slot)
        self.accum_queue = queue.Queue()
        self.accum_written = set()  # parameter indices written in the current mini-batch
        self.accum_error = None
        self.accum_seconds = 0.0
        self.accum_wait_slot_seconds = 0.0
        self.local_sumsq = None
        self.microbatches = 0
        self.accum_thread = threading.Thread(target=self._accumulate_loop, name="cpu-dp-accumulate", daemon=True)
        self.accum_thread.start()

    def begin_microbatch(self):
        """Call before each training micro-batch forward: one hook firing per layer per backward."""
        if not self.accumulate:
            return
        self._check_accumulate()
        self.gradient_layers.clear()
        self.microbatches += 1

    def _check_accumulate(self):
        if self.accum_error is not None:
            raise RuntimeError(f"CPU DP gradient accumulation failed:\n{self.accum_error}")

    @torch.no_grad()
    def _accumulate_loop(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(threads)
        while True:
            item = self.accum_queue.get()
            try:
                if item is None:
                    return
                slot, ready, entries = item
                if self.accum_error is None:
                    ready.synchronize()
                    start = time.perf_counter()
                    buffer = self.ring[slot]
                    for index, offset, numel in entries:
                        begin = self.param_starts[index]
                        destination = self.stage_flat[begin:begin + numel]
                        source = buffer[offset:offset + numel]
                        if index in self.accum_written:
                            destination.add_(source)
                        else:
                            destination.copy_(source)
                            self.accum_written.add(index)
                    self.accum_seconds += time.perf_counter() - start
            except BaseException:
                self.accum_error = traceback.format_exc()
            finally:
                if item is not None:
                    self.free_slots.put(item[0])
                self.accum_queue.task_done()

    @torch.no_grad()
    def _offload_layer_gradient(self, name, module, *unused):
        if not self.accumulate:
            return super()._offload_layer_gradient(name, module, *unused)
        self.backward_hooks.pop(name).remove()
        if name in self.gradient_layers:
            raise RuntimeError(f"layer {name or 'root'}: more than one AccumulateGrad per backward")
        if self.pending is not None or self.future is not None or not module._sync_gradients:
            raise RuntimeError("publish the previous CPU update before a synchronized backward")
        self._check_accumulate()
        self.gradient_layers.add(name)
        handle = module._handle
        live = [(i, self.pairs[i][0].grad) for i in self.layer_indices[name] if self.pairs[i][0].grad is not None]
        if live:
            with torch.cuda.stream(module._post_backward_stream):
                norms = torch._foreach_norm([g for _, g in live], 2.0)
                sumsq = torch.stack(norms).float().square().sum()
                self.local_sumsq = sumsq if self.local_sumsq is None else self.local_sumsq + sumsq
            start = time.perf_counter()
            slot = self.free_slots.get()  # blocks while every landing slot is still being accumulated
            self.accum_wait_slot_seconds += time.perf_counter() - start
            buffer = self.ring[slot]
            self.copy_stream.wait_stream(module._post_backward_stream)
            entries, offset = [], 0
            with torch.cuda.stream(self.copy_stream), torch.cuda.nvtx.range(
                f"openmopd::io::d2h::actor_gradients::{name or 'root'}"
            ):
                for i, gradient in live:
                    numel = gradient.numel()
                    buffer[offset:offset + numel].view(gradient.shape).copy_(gradient, non_blocking=True)
                    gradient.record_stream(self.copy_stream)
                    entries.append((i, offset, numel))
                    offset += numel
                ready = torch.cuda.Event()
                ready.record(self.copy_stream)
            self.accum_queue.put((slot, ready, entries))
        handle.flat_param.grad = None
        handle._use_unsharded_grad_views()

    def _seal_accumulation(self):
        """Drain the accumulate thread; returns the parameter indices this rank wrote."""
        with torch.cuda.nvtx.range("openmopd::cpu::accumulate_drain"):
            self.accum_queue.join()
        self._check_accumulate()
        live = sorted(self.accum_written)
        self.accum_written = set()
        return live

    # ------------------------------------------------------------------ owner setup
    def _start_node_workers(self, table, host_specs, stage_dtype, threads):
        nodes = shared_host.numa_nodes()
        node_ids = sorted(nodes)
        # Contiguous parameter blocks of roughly equal size, one per node.
        bounds, target, acc, cut = [0], self.total / len(node_ids), 0, 1
        for index, (param, *_) in enumerate(table):
            acc += param.numel()
            if cut < len(node_ids) and acc >= target * cut and index + 1 < len(table):
                bounds.append(index + 1)
                cut += 1
        while len(bounds) < len(node_ids):
            bounds.append(len(table))
        bounds.append(len(table))
        starts, acc = [], 0
        for param, *_ in table:
            starts.append(acc)
            acc += param.numel()
        starts.append(acc)
        self.partitions = []
        context = torch.multiprocessing.get_context("spawn")
        group_count = len(self.param_groups)
        for slot, node in enumerate(node_ids):
            first, last = bounds[slot], bounds[slot + 1]
            lo, hi = starts[first], starts[last]
            numel = hi - lo
            opt_specs = [("master", (numel,), torch.float32), ("exp_avg", (numel,), torch.float32),
                         ("exp_avg_sq", (numel,), torch.float32), ("grad", (numel,), torch.float32)]
            opt = shared_host.open_arena(f"{_PREFIX}opt{node}", opt_specs, create=True, policy=f"node:{node}")
            meta = []
            for index in range(first, last):
                param, gi, name, host_offset = table[index]
                offset = starts[index] - lo
                meta.append((index, offset, param.numel(), tuple(param.shape), name, host_offset, gi))
                view = lambda key: opt[key][offset:offset + param.numel()].view(param.shape)
                master = view("master")
                master.copy_(self.pairs[index][3])
                self.state[param] = {
                    "step": 0, "exp_avg": view("exp_avg"), "exp_avg_sq": view("exp_avg_sq"), "cpu_master": master,
                }
            self.partitions.append((node, first, last))
            parent, child = context.Pipe()
            process = context.Process(
                target=_node_worker,
                args=(child, node, nodes[node] if len(nodes) > 1 else None, threads, self.world, stage_dtype,
                      self.total, (lo, hi), host_specs, opt_specs, meta, group_count, self.stage_name),
                name=f"openmopd-dp-adam-node{node}", daemon=True,
            )
            process.start()
            child.close()
            self.workers.append((process, parent))
        for _process, connection in self.workers:
            self._receive_from(connection)

    @staticmethod
    def _receive_from(connection):
        ok, result = connection.recv()
        if not ok:
            raise RuntimeError(f"CPU data parallel node worker failed:\n{result}")
        return result

    # ------------------------------------------------------------------ GPU-facing protocol
    @torch.no_grad()
    def clip_grad_norm_(self, max_norm):
        """Return this rank's local norm; clipping uses the reduced gradient's norm on the owner."""
        self.max_norm = float(max_norm)
        for module in self.modules.values():
            torch.cuda.current_stream(self.device).wait_stream(module._post_backward_stream)
        if self.accumulate:
            # Root-sum-square of the micro-batch gradient norms (diagnostic only; the
            # clip coefficient is computed by the owner from the reduced FP32 gradient).
            if self.local_sumsq is None:
                return torch.zeros((), device=self.device)
            return self.local_sumsq.sqrt()
        norms = [self.gradient_norms[i] for i in sorted(self.gradient_norms)]
        if not norms:
            return torch.zeros((), device=self.device)
        return torch.linalg.vector_norm(torch.stack(norms), 2.0)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("CPU pipeline does not support optimizer closures")
        if self.future is not None or self.pending is not None:
            raise RuntimeError("previous CPU update has not been published")
        if self.max_norm is None:
            raise RuntimeError("call clip_grad_norm_ before step")
        if self.accumulate:
            produced = None
            live = self._seal_accumulation()
            self.local_sumsq = None
            self.last_accumulate_metrics = {
                "cpu_dp/accumulate_seconds": self.accum_seconds,
                "cpu_dp/accumulate_wait_slot_seconds": self.accum_wait_slot_seconds,
                "cpu_dp/microbatches_per_update": float(self.microbatches),
            }
            self.accum_seconds = self.accum_wait_slot_seconds = 0.0
            self.microbatches = 0
        else:
            produced = torch.cuda.Event()
            produced.record(self.copy_stream)
            live = sorted(self.gradient_norms)
        self.seq += 1
        options = [self._options(group) for group in self.param_groups]
        self.pending = (None, produced, live, self.max_norm, options)
        self.signal_future = self.signal_executor.submit(self._signal_ready, produced, live, self.seq)
        self.gradient_layers.clear()
        self.gradient_norms.clear()
        self.gradient_scale = None
        if self.async_update is False:
            self.flush()
        elif self.async_update is None and not self.overlap:
            self.wait()

    @torch.no_grad()
    def _signal_ready(self, produced, live, seq):
        if produced is not None:
            produced.synchronize()
        flags = torch.zeros(len(self.pairs), dtype=torch.uint8)
        if live:
            flags[torch.tensor(live, dtype=torch.long)] = 1
        for index in torch.nonzero(flags == 0).flatten().tolist():
            self.pairs[index][2].zero_()  # a parameter without a local gradient contributes zero
        self.ctrl["live"][self.rank].copy_(flags)
        self.ctrl["ready"][self.rank] = seq

    @torch.no_grad()
    @memory_stage("cpu_dp_update")
    def _update(self, ready, live, options, max_norm):
        seq = self.seq
        start = time.perf_counter()
        self.signal_future.result()
        if not self.is_owner:
            shared_host.wait_until(lambda: int(self.ctrl["published"][0]) >= seq, what=f"CPU DP update {seq}")
            floats = self.ctrl["floats"]
            self.last_metrics = {
                "pipeline/cpu_optimizer_seconds": time.perf_counter() - start,
                "cpu_dp/global_grad_norm": float(floats[0]),
                "cpu_dp/skipped": float(floats[1]),
                **getattr(self, "last_accumulate_metrics", {}),
            }
            return
        ready_flags = self.ctrl["ready"]
        shared_host.wait_until(lambda: bool(torch.all(ready_flags >= seq)), what=f"CPU DP gradients {seq}")
        wait_seconds = time.perf_counter() - start
        live_any = self.ctrl["live"].amax(dim=0).bool().tolist()
        for _process, connection in self.workers:
            connection.send(("reduce",))
        results = [self._receive_from(connection) for _process, connection in self.workers]
        reduce_seconds = max(seconds for _sumsq, seconds in results)
        total_norm = math.sqrt(sum(sumsq for sumsq, _seconds in results))
        finite = math.isfinite(total_norm)
        adam_seconds = 0.0
        if finite:
            scale = min(1.0, max_norm / (total_norm + 1e-6))
            for (node, first, last), (_process, connection) in zip(self.partitions, self.workers, strict=True):
                steps = [int(self.state[self.pairs[i][0]]["step"]) for i in range(first, last)]
                connection.send(("adam", scale, options, steps, live_any[first:last]))
            adam_seconds = max(self._receive_from(connection) for _process, connection in self.workers)
            for index, is_live in enumerate(live_any):
                if is_live:
                    self.state[self.pairs[index][0]]["step"] += 1
        else:
            print(f"WARN: CPU DP update {seq}: reduced gradient norm is not finite ({total_norm}); skipping")
        floats = self.ctrl["floats"]
        floats[0] = total_norm
        floats[1] = 0.0 if finite else 1.0
        self.ctrl["published"][0] = seq
        self.last_metrics = {
            "pipeline/cpu_optimizer_seconds": time.perf_counter() - start,
            "cpu_dp/wait_ranks_seconds": wait_seconds,
            "cpu_dp/reduce_seconds": reduce_seconds,
            "pipeline/cpu_adam_native_seconds": adam_seconds,
            "cpu_dp/global_grad_norm": total_norm,
            "cpu_dp/skipped": 0.0 if finite else 1.0,
            **getattr(self, "last_accumulate_metrics", {}),
        }

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self):
        if self.future is not None or self.pending is not None:
            raise RuntimeError("flush CPU pipeline before saving optimizer state")
        result = torch.optim.Optimizer.state_dict(self)
        result["pipeline_version"] = self.version
        result["cpu_dp_seq"] = self.seq
        result["cpu_dp_owner"] = self.is_owner
        return result

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        if self.future is not None or self.pending is not None:
            raise RuntimeError("cannot load an optimizer with a pending CPU update")
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("CPU optimizer checkpoint parameter groups do not match")
        for group, saved in zip(self.param_groups, saved_groups, strict=True):
            group.update({key: value for key, value in saved.items() if key != "params"})
        if self.is_owner:
            if not state_dict.get("cpu_dp_owner", "cpu_master" in next(iter(state_dict["state"].values()), {})):
                raise ValueError("CPU DP owner must load the owner's optimizer checkpoint")
            indices = [index for group in saved_groups for index in group["params"]]
            for index, (param, _master, _staging, output) in zip(indices, self.pairs, strict=True):
                saved = state_dict["state"][index]
                state = self.state[param]
                state["cpu_master"].copy_(saved["cpu_master"])
                for key in ("exp_avg", "exp_avg_sq"):
                    state[key].copy_(saved[key])
                state["step"] = int(saved["step"])
                output.copy_(state["cpu_master"])
        self.version = state_dict.get("pipeline_version", 0)
        self.seq = state_dict.get("cpu_dp_seq", self.seq)
        if self.is_owner:
            self.ctrl["published"][0] = self.seq
            self.ctrl["ready"].fill_(self.seq)
            steps = [int(self.state[p]["step"]) for p, *_ in self.pairs]
            print(f"[cpu_dp] resumed optimizer: version {self.version}, seq {self.seq}, "
                  f"adam steps {min(steps)}..{max(steps)}, lr {self.param_groups[0]['lr']}", flush=True)
        shared_host.cpu_barrier()
        # Every replica restarts from the restored shared weights.
        for name, destination in self.gpu_weights.items():
            destination.copy_(self.host_weights[name], non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()

    def close(self):
        try:
            self.flush()
        finally:
            for hook in self.forward_hooks + list(self.backward_hooks.values()):
                hook.remove()
            self.executor.shutdown(wait=True)
            self.signal_executor.shutdown(wait=True)
            if getattr(self, "accumulate", False):
                self.accum_queue.put(None)
                self.accum_thread.join(timeout=60)
            for process, connection in self.workers:
                if process.is_alive():
                    connection.send(None)
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join()
                connection.close()
