#!/usr/bin/env python3
"""Export local PyTorch CUDA snapshots to HTML and locate retained-history peaks."""

import argparse
import csv
import json
import pickle
from pathlib import Path


def analyze_device(snapshot, device, max_entries=0):
    """Replay requested active bytes, including allocations awaiting a CUDA free.

    Start from the final allocator state and undo events to recover the baseline.
    Starting at zero is incorrect when recording starts late or the ring wraps.
    Event sizes are requested bytes, not allocator block sizes rounded to 512 B.
    """
    events = snapshot["device_traces"][device]
    segments = [segment for segment in snapshot["segments"] if segment["device"] == device]
    live = {}
    for segment in segments:
        address = segment["address"]
        for block in segment["blocks"]:
            if block["state"] != "inactive":
                live[block.get("address", address)] = {
                    "size": block["requested_size"],
                    "frames": block.get("frames", []),
                    "allocation_event": None,
                }
            address += block["size"]

    reserved = sum(segment["total_size"] for segment in segments)
    reserve_signs = {"segment_alloc": 1, "segment_map": 1, "segment_free": -1, "segment_unmap": -1}
    for event in reversed(events):
        action = event["action"]
        if action == "alloc":
            live.pop(event["addr"], None)
        elif action == "free_completed":
            # A free's stack is not the original allocation's stack.
            live[event["addr"]] = {"size": event["size"], "frames": [], "allocation_event": None}
        reserved -= reserve_signs.get(action, 0) * event.get("size", 0)

    active = sum(block["size"] for block in live.values())
    if reserved < 0:
        raise ValueError(f"Inconsistent initial reserved memory for cuda:{device}")
    peak_active, peak_reserved = active, reserved
    peak_event = reserved_peak_event = -1
    initial_active = active
    initial_reserved = reserved
    # One pass to locate the peak, then one to reconstruct its live allocations.
    for index, event in enumerate(events):
        action, size = event["action"], event.get("size", 0)
        if action == "alloc":
            active += size
        elif action == "free_completed":
            active -= size
        reserved += reserve_signs.get(action, 0) * size
        if active < 0 or reserved < 0:
            raise ValueError(f"Inconsistent allocator history at cuda:{device} event {index}")
        if active > peak_active:
            peak_active, peak_event = active, index
        if reserved > peak_reserved:
            peak_reserved, reserved_peak_event = reserved, index

    for index, event in enumerate(events[: peak_event + 1]):
        if event["action"] == "alloc":
            live[event["addr"]] = {"size": event["size"], "frames": event.get("frames", []), "allocation_event": index}
        elif event["action"] == "free_completed":
            live.pop(event["addr"], None)
    top_allocations = [
        {"address": hex(address), **block}
        for address, block in sorted(live.items(), key=lambda item: item[1]["size"], reverse=True)[:10]
    ]
    return {
        "device": device,
        "event_count": len(events),
        "history_at_capacity": bool(max_entries and len(events) >= max_entries),
        "initial_active_requested_bytes": initial_active,
        "initial_reserved_bytes": initial_reserved,
        "peak_active_requested_bytes": peak_active,
        "peak_event_index": peak_event,
        "peak_event": events[peak_event] if peak_event >= 0 else None,
        "peak_reserved_bytes": peak_reserved,
        "reserved_peak_event_index": reserved_peak_event,
        "oom_count": sum(event["action"] == "oom" for event in events),
        "largest_allocations_at_peak": top_allocations,
    }


def device_peaks(path):
    """Read nvidia-smi's whole-device samples, keeping physical GPU identities."""
    peaks = {}
    if path and path.is_file():
        with path.open() as handle:
            for raw in csv.DictReader(handle):
                row = {key.strip(): value.strip() for key, value in raw.items() if key and value}
                try:
                    used = float(row["memory.used [MiB]"])
                    uuid = row["uuid"]
                except (KeyError, ValueError):
                    continue  # Ignore truncated final samples and driver messages.
                if uuid not in peaks or used > peaks[uuid]["used_mib"]:
                    peaks[uuid] = {**row, "used_mib": used}
    return list(peaks.values())


def format_frames(frames):
    if not frames:
        return "Allocation stack unavailable (possibly allocated before retained history)."
    # Prefer Python frames in the short text report; full Python/C++ stacks remain in HTML/JSON.
    python_frames = [frame for frame in frames if frame.get("filename", "").endswith(".py")]
    return "\n".join(
        f"{frame.get('filename', '?')}:{frame.get('line', '?')} in {frame.get('name', '?')}"
        for frame in (python_frames or frames)[:16]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Local .pickle file or snapshots directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=0, help="Recording limit, for ring-buffer warnings")
    parser.add_argument("--device-csv", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.snapshot.is_dir():
        # The legacy post_update dump uses the NEXT step's number and duplicates
        # the regular dump. Only use canonical end-of-step snapshots here.
        files = sorted(args.snapshot.glob("step*/torch_memory*.pickle"))
    else:
        files = [args.snapshot] if args.snapshot.is_file() else []

    report = [
        "# CUDA memory peak report",
        "",
        "Snapshots contain cumulative worker history, including initialization/validation, up to each export.",
        "Peaks below cover only retained events, not necessarily the entire run or just the export step.",
        "Active = requested bytes of live allocations (including pending frees); reserved = allocator segments.",
        "Allocator rounding is excluded from active bytes. These are not whole-device nvidia-smi totals.",
        "vLLM sleep can unmap CUDA memory below PyTorch: allocator totals may include non-resident pools",
        "and can exceed physical GPU capacity. Use whole-device samples for resident VRAM usage.",
        "Event indices are zero-based within each device's history; -1 means the beginning of retained history.",
        "The trigger stack identifies the allocation reaching the peak; large resident tensors also contribute.",
        "HTML embeds the snapshot but loads PyTorch's viewer JavaScript from a CDN (internet required).",
        "",
    ]
    samples = device_peaks(args.device_csv)
    if samples:
        report += [
            "## Whole-device sampled peaks",
            "",
            "Includes all processes and non-PyTorch allocations. Sampling can miss short peaks.",
            "Match the GPU UUID/bus ID to the training GPU; physical indices may differ from CUDA ordinals.",
            "",
            "| Physical GPU | UUID | PCI bus | Peak GiB | Time |",
            "|---|---|---|---:|---|",
        ]
        for sample in samples:
            report.append(
                f"| {sample.get('index', '?')} | {sample['uuid']} | {sample.get('pci.bus_id', '?')} | "
                f"{sample['used_mib'] / 1024:.3f} | {sample.get('timestamp', '?')} |"
            )
        report.append("")
    else:
        report += ["Whole-device samples unavailable or disabled; check device_memory_monitor.log if enabled.", ""]

    results, errors = [], []
    if files:
        from torch.cuda import _memory_viz

    for path in files:
        try:
            # Input is a locally generated, trusted PyTorch snapshot.
            with path.open("rb") as handle:
                snapshot = pickle.load(handle)
            devices = [
                device
                for device, events in enumerate(snapshot["device_traces"])
                if events or any(segment["device"] == device for segment in snapshot["segments"])
            ]
            if not any(snapshot["device_traces"]):
                raise ValueError("No CUDA allocation history: memory recording may have failed")
            name = f"{path.parent.name}_{path.stem}"
            timeline = f"{name}_timeline.html"
            allocator = f"{name}_allocator.html"
            (args.output_dir / timeline).write_text(_memory_viz.trace_plot(snapshot), encoding="utf-8")
            (args.output_dir / allocator).write_text(_memory_viz.segment_plot(snapshot), encoding="utf-8")
            report += [
                f"## {path.parent.name}/{path.name}",
                "",
                f"[Active Memory Timeline]({timeline}) · [Allocator State History]({allocator})",
                "",
            ]
            for device in devices:
                result = {"snapshot": str(path), **analyze_device(snapshot, device, args.max_entries)}
                results.append(result)
                report += [
                    f"### cuda:{device}: {result['event_count']:,} events",
                    "",
                    f"Active peak: **{result['peak_active_requested_bytes'] / 1024**3:.4f} GiB**, "
                    f"event **{result['peak_event_index']}**. "
                    f"Reserved peak: **{result['peak_reserved_bytes'] / 1024**3:.4f} GiB**, "
                    f"event **{result['reserved_peak_event_index']}**. OOM events: {result['oom_count']}.",
                    "",
                ]
                if result["history_at_capacity"]:
                    report += [
                        "**History reached its capacity; older events may be lost. Increase "
                        "OPENMOPD_MEMORY_MAX_ENTRIES to capture earlier peaks.**",
                        "",
                    ]
                event = result["peak_event"]
                if event:
                    report += [
                        f"Peak trigger: `{event['action']}`, address `{hex(event['addr'])}`, "
                        f"allocation {event.get('size', 0) / 1024**2:.3f} MiB.",
                        "",
                        "```text",
                        format_frames(event.get("frames", [])),
                        "```",
                        "",
                    ]
                else:
                    report += ["Peak already present at the start of retained history; trigger stack unavailable.", ""]
                report += ["Largest live allocations at this peak (addresses can be searched in the viewer):", ""]
                for allocation in result["largest_allocations_at_peak"]:
                    report += [
                        f"- `{allocation['address']}`: {allocation['size'] / 1024**2:.3f} MiB, "
                        f"allocation event {allocation['allocation_event']}"
                    ]
                report.append("")
        except Exception as exc:
            message = f"{path}: {exc}"
            errors.append(message)
            report += [f"**Could not analyze snapshot:** {message}", ""]
    if not files:
        errors.append(
            "No canonical torch_memory snapshots found. Check run.log; training may have failed before export."
        )
        report += [errors[-1], ""]
    summary = args.output_dir / "summary.md"
    summary.write_text("\n".join(report), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps({"snapshots": results, "device_peaks": samples, "errors": errors}, indent=2), encoding="utf-8"
    )
    print(summary)
    for result in results:
        print(
            f"{Path(result['snapshot']).parent.name} cuda:{result['device']}: "
            f"active peak {result['peak_active_requested_bytes'] / 1024**3:.4f} GiB "
            f"at event {result['peak_event_index']}"
        )
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
