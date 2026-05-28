#!/usr/bin/env python3
"""Merge a partial re-run into a saved full AgentHarm results.json.

Use case: Gemini (or any flaky API) leaves N transient-fail cases in a
176-record run. You re-run only those N via `--behavior_ids ...`, which
overwrites results.json with N records. This script restores the full set
by overlaying the N new records on top of the prior full snapshot.

Workflow:
    # one-time, before re-running fails
    cp results.json results_full.json

    # re-run only the failed ids (agentharm/run_eval.py overwrites results.json)
    python run_eval.py --behavior_ids 24-2 24-3 ... --task_name harmful ...

    # merge — overlays the N new records onto the 176-record snapshot
    python merge_results.py outputs/agentharm_harmful_test_public/gemini-2.5-pro/no_defense

    # if failed_ids.txt is non-empty, repeat re-run with that id list,
    # then re-run merge_results.py (results_full.json is now the latest full set)

Failure criterion (same as the diagnostic):
    any role=assistant message with empty content AND no tool_calls → fail
    (covers both "0 assistant turn" hard-fail and "mid-run empty turn" cases)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def is_failed_record(record: dict) -> bool:
    """True if any assistant turn in the record's messages has no content
    and no tool_calls. Matches both hard-fail (no assistant turn at all,
    so the empty placeholder set is empty -> covered by the 0-turn branch)
    and mid-run-fail (last assistant turn is empty)."""
    messages = record.get("messages") or []
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return True
    for m in assistant_msgs:
        content = (m.get("content") or "").strip()
        tool_calls = m.get("tool_calls")
        if not content and not tool_calls:
            return True
    return False


def load_json(path: Path) -> list | dict:
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge(full_records: list[dict], partial_records: list[dict]) -> list[dict]:
    """Overlay partial onto full by behavior_id. Partial wins unconditionally
    — caller is responsible for only re-running the ids that need updating."""
    by_id = {r["behavior_id"]: r for r in full_records}
    overlaid = 0
    added = 0
    for r in partial_records:
        bid = r["behavior_id"]
        if bid in by_id:
            overlaid += 1
        else:
            added += 1
        by_id[bid] = r
    print(f"  overlaid existing: {overlaid}  added new: {added}")
    return list(by_id.values())


def compute_summary(records: list[dict], template: dict | None) -> dict:
    """Re-run the same stats block from run_eval.py:944-977."""
    n = len(records)
    total_score = sum(r.get("score") or 0 for r in records)
    total_refusal = sum(1 for r in records if r.get("refusal", False))
    avg_score = total_score / n if n else 0.0
    refusal_rate = total_refusal / n if n else 0.0
    non_ref_scores = [r["score"] for r in records if not r.get("refusal", False) and r.get("score") is not None]
    avg_score_non_ref = sum(non_ref_scores) / len(non_ref_scores) if non_ref_scores else 0.0

    summary = dict(template) if template else {}
    summary.update({
        "total_behaviors": n,
        "average_score": avg_score,
        "average_score_non_refusals": avg_score_non_ref,
        "refusal_rate": refusal_rate,
        "timestamp": datetime.now().isoformat(),
    })
    return summary


def main(output_dir: Path) -> int:
    results_path = output_dir / "results.json"
    full_path = output_dir / "results_full.json"
    summary_path = output_dir / "summary.json"

    if not results_path.exists():
        print(f"ERROR: {results_path} not found (expected partial re-run output)")
        return 1
    if not full_path.exists():
        print(f"ERROR: {full_path} not found")
        print("       Before the partial re-run, run: cp results.json results_full.json")
        return 1

    partial = load_json(results_path)
    full = load_json(full_path)
    print(f"Loaded partial: {len(partial)} records  (from {results_path.name})")
    print(f"Loaded full:    {len(full)} records  (from {full_path.name})")

    # Audit copy of the partial before we overwrite results.json.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_path = output_dir / f"results_patch_{ts}.json"
    save_json(patch_path, partial)
    print(f"Saved patch snapshot -> {patch_path.name}")

    print("Merging (partial overlays full by behavior_id):")
    merged = merge(full, partial)
    print(f"Merged total: {len(merged)} records")

    save_json(results_path, merged)
    save_json(full_path, merged)
    print(f"Wrote merged set -> results.json + results_full.json (the latter becomes the new snapshot)")

    template = load_json(summary_path) if summary_path.exists() else None
    summary = compute_summary(merged, template)
    save_json(summary_path, summary)
    print(f"Recomputed summary -> {summary_path.name}")
    print(
        f"  total={summary['total_behaviors']}  "
        f"avg={summary['average_score']:.4f}  "
        f"avg_non_ref={summary['average_score_non_refusals']:.4f}  "
        f"refusal={summary['refusal_rate']:.2%}"
    )

    failed = [r["behavior_id"] for r in merged if is_failed_record(r)]
    failed_path = output_dir / "failed_ids.txt"
    if failed:
        with failed_path.open("w") as f:
            f.write(" ".join(failed) + "\n")
        print(f"\n{len(failed)} cases STILL fail. ids written to {failed_path.name}:")
        print(" ", " ".join(failed))
        print("\nNext round:")
        print(f"  python run_eval.py --llm_path <model> --task_name <task> --split <split> "
              f"--num_workers 5 --max_new_tokens 4096 --behavior_ids $(cat {failed_path})")
        print(f"  python merge_results.py {output_dir}")
    else:
        if failed_path.exists():
            failed_path.unlink()
        print("\nAll records pass the failure check. Done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Directory containing results.json + results_full.json + summary.json")
    args = parser.parse_args()
    sys.exit(main(args.output_dir.resolve()))
