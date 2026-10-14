"""Convert your own rubric-graded tasks into the held-out validation parquet the RRI
trainer reads (the same row layout as preprocess_healthbench_professional.py).

Input: JSON lines, one task per line.

    {"id": "case-001",                                   # optional
     "prompt": "A 64-year-old on apixaban ...",          # or "conversation": [{"role", "content"}, ...]
     "rubric": [{"criterion": "States that ...", "points": 8},
                {"criterion": "Recommends stopping apixaban without bridging", "points": -6}],
     "category": "treatment"}                            # optional, reported separately in validation

* `conversation` is a list of chat turns ending on a user turn; `prompt` is shorthand
  for a single user turn. No system message: the policy sees the conversation as-is.
* `rubric` items are graded one at a time by the judge. Positive points reward content
  the answer should contain; negative points describe an error ("Recommends X") and
  are deducted when the answer makes it. The task score is the sum of the points of
  the criteria met divided by the sum of the positive points.
* `points` follow HealthBench's scale (-10 to +10); integers are kept as integers.

    python scripts/self_evolving/eval/prepare_rubric_dataset.py \\
        --input my_tasks.jsonl --name my_bench --out data/my_bench_val.parquet

Validation then reports `val-core/<name>/<category>/...` per category and the overall
mean. Pass the file with VAL_PARQUET=data/my_bench_val.parquet. Training never reads
these tasks: the proposer only sees the domain description (SE_DOMAIN).
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

import pandas as pd


def _messages(rec: dict, lineno: int) -> list[dict]:
    conv = rec.get("conversation")
    if conv is None and rec.get("prompt"):
        conv = [{"role": "user", "content": str(rec["prompt"])}]
    if not conv:
        raise SystemExit(f"line {lineno}: needs `prompt` or `conversation`")
    msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in conv]
    if msgs[-1]["role"] != "user":
        raise SystemExit(f"line {lineno}: the conversation must end on a user turn")
    return msgs


def _rubric(rec: dict, lineno: int) -> list[dict]:
    items = []
    for it in rec.get("rubric") or rec.get("rubric_items") or []:
        text = it.get("criterion") or it.get("criterion_text")
        if not text or it.get("points") is None:
            raise SystemExit(f"line {lineno}: every rubric item needs `criterion` and `points`")
        pts = float(it["points"])
        items.append({"criterion_text": str(text), "points": int(pts) if pts.is_integer() else pts})
    if not any(it["points"] > 0 for it in items):
        raise SystemExit(f"line {lineno}: the rubric needs at least one positive criterion")
    return items


def convert(path: str, name: str, split: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            msgs = _messages(rec, lineno)
            qid = str(rec.get("id") or uuid.uuid4().hex)
            category = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rec.get("category") or "all"))
            rows.append({
                "data_source": f"{name}/{category}",
                "prompt": msgs,
                "reward_model": {"style": "rubric", "ground_truth": ""},
                "extra_info": {
                    "question_id": qid,
                    "index": qid,
                    "split": split,
                    "rubric_items": _rubric(rec, lineno),
                    "conversation": msgs,
                    "use_case": rec.get("category"),
                    "type": rec.get("type"),
                    "difficulty": rec.get("difficulty"),
                    "specialty": rec.get("specialty"),
                    "source": name,
                },
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSON lines, one task per line")
    ap.add_argument("--name", required=True, help="benchmark name, the data_source prefix")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    if "/" in args.name:
        ap.error("--name must not contain '/'")
    rows = convert(args.input, args.name, args.split)
    if not rows:
        raise SystemExit(f"no tasks in {args.input}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    cats = pd.Series([r["data_source"] for r in rows]).value_counts()
    print(f"wrote {len(rows)} tasks to {args.out}")
    print(cats.to_string())


if __name__ == "__main__":
    main()
