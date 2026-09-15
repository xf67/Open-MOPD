"""合并 AIME、LiveCodeBench、IFEval，保留全部样本及评分 metadata。"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SOURCES = [
    ("math/aime24.parquet", "math"),
    ("math/aime25.parquet", "math"),
    ("code/livecodebench_v5.parquet", "code"),
    ("code/livecodebench_v6.parquet", "code"),
    ("if/ifeval_aligned.parquet", "if"),
]

SCHEMA = pa.schema([
    *[(name, pa.string()) for name in ("data_source", "dataset", "domain", "ability")],
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
    ("extra_info", pa.struct([
        ("metadata", pa.large_string()),
        *[(name, pa.string()) for name in
          ("dataset", "sample_id", "request_id", "raw_prompt", "split", "index")],
    ])),
])


def convert_row(row, domain):
    dataset, sample_id = row["dataset"], str(row["sample_id"])
    metadata = json.loads(row["metadata"])
    return {
        "data_source": dataset,
        "dataset": dataset,
        "domain": domain,
        "ability": domain,
        "prompt": row["prompt"],
        "reward_model": {
            "style": "rule",
            "ground_truth": str(row["answer"]) if domain == "math" else "",
        },
        "extra_info": {
            "metadata": row["metadata"],
            "dataset": dataset,
            "sample_id": sample_id,
            "request_id": str(row["request_id"]),
            "raw_prompt": (
                row.get("original_prompt") or metadata.get("question_content") or metadata.get("prompt") or ""
            ),
            "split": "val",
            "index": f"{dataset}:{sample_id}",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.output.open("wb" if args.overwrite else "xb") as output:
        with pq.ParquetWriter(output, SCHEMA, compression="zstd") as writer:
            for filename, domain in SOURCES:
                for batch in pq.ParquetFile(args.source_dir / filename).iter_batches(batch_size=8):
                    rows = [convert_row(row, domain) for row in batch.to_pylist()]
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                    total += len(rows)
    print(f"已生成 {args.output}，共 {total} 条")


if __name__ == "__main__":
    main()


# python build_training_eval.py --source-dir data/eval --output data/rl_prompt_mix/eval.parquet