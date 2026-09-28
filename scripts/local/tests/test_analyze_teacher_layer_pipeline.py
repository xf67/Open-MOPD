# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_teacher_forward_overlap import analyze  # noqa: E402


def test_layer_copy_target_is_distinct_from_submitting_teacher(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    with sqlite3.connect(traces / "worker.sqlite") as db:
        db.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, streamId INTEGER, contextId INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                start INTEGER, end INTEGER, streamId INTEGER, contextId INTEGER,
                correlationId INTEGER, copyKind INTEGER, bytes INTEGER);
        """)
        db.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            [(1, "cudaLaunchKernel"), (2, "cudaMemcpyAsync")],
        )
        ranges = [
            (0, 400, 1, "step::2"),
            (10, 50, 1, "compute::student_log_prob"),
            (15, 60, 2, "compute::teacher_model_forward::Math"),
            (15, 60, 2, "compute::teacher_layer_model_forward::rm-Math"),
            (60, 70, 1, "compute::teacher_postprocess::Math"),
            (70, 110, 1, "compute::teacher_forward::Code"),
            (70, 100, 1, "compute::teacher_layer_model_forward::mt_rm_1-Code"),
            (30, 32, 2, "io::h2d::teacher_layer_prefetch::mt_rm_1-Code::layer0"),
            (80, 82, 1, "io::h2d::teacher_layer_prefetch::rm-Math::layer0"),
        ]
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",
            [(a, b, tid, "openmopd::" + name) for a, b, tid, name in ranges],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
            [
                (20, 21, 1, 1, 1),
                (20, 21, 2, 1, 2),
                (75, 76, 1, 1, 3),
                (30, 31, 2, 2, 4),
                (80, 81, 1, 2, 5),
            ],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",
            [
                (100, 200, 1, 0, 1),
                (120, 210, 2, 0, 2),
                (240, 300, 1, 0, 3),
            ],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?)",
            [
                (170, 230, 3, 0, 4, 1, 2**20),
                (280, 320, 3, 0, 5, 1, 2**20),
            ],
        )
    (result,) = analyze(tmp_path)
    assert (
        result["teacher_htod_MiB"] == 0
    )  # Math's range submitted Code's load, not Math's.
    assert result["prefetch_MiB"] == 1
    math, code = result["layer_pipeline"]
    assert math["previous_teacher"] == "mt_rm_1-Code"
    assert math["previous_teacher_kernel_overlap_percent"] == 50
    assert code["previous_teacher"] == "rm-Math"
    assert code["htod_count"] == 1 and code["htod_MiB"] == 1
    assert code["previous_teacher_kernel_overlap_percent"] == pytest.approx(
        100 * 40 / 60
    )
