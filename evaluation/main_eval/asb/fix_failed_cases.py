"""Remove failed cases from ASB result JSON files so resume re-runs them.

Failed cases are detected by (all signal that no scorable trajectory exists):
  1. `error` field set ("Execution returned None" — guardrail path tags this
     explicitly; no_defense ReAct path does NOT, so rules 2-4 catch the same
     symptom as empty/missing assistant turns).
  2. messages length < 3 (round-1 agent call died; no assistant turn ever produced)
  3. all assistant turns have empty content AND empty tool_calls
  4. trajectory ends with role=assistant whose content + tool_calls are both
     empty (mid-run API exhaustion — earlier turns may carry tool_calls but
     the agent died on a later turn before scoring closure).

NOT a failure: trajectory ending with role=tool. This signals max_rounds
termination — empirically verified that these records carry valid
attack_successful / refused / original_successful scores. ASB scoring is
based on tool calls within the trajectory, not on a final assistant answer.
Earlier versions of this script flagged these as failures; for no_defense
ReAct (which legitimately hits max_rounds often, 22-141 cases per cell)
the over-count was ~30% of total fail tallies and led to needless re-runs.

After filtering, the file is rewritten in place (with a .preretry.bak backup),
so the next `bash run_*.sh` will see those (agent, attack_tool, task) triples
as missing and re-execute only those cases via the existing resume logic.

Usage:
    python fix_failed_cases.py <path1.json> [<path2.json> ...]

    # discover candidates first (no modification):
    python fix_failed_cases.py --dry-run outputs/**/*-all_advance.json
"""

import argparse
import glob
import json
import shutil
import sys


def is_failed(r):
    if r.get("error"):
        return True
    msgs = r.get("messages", [])
    if len(msgs) < 3:
        return True
    has_substantive_assistant = any(
        m.get("role") == "assistant" and (m.get("content") or m.get("tool_calls"))
        for m in msgs
    )
    if not has_substantive_assistant:
        return True
    last = msgs[-1]
    if last.get("role") == "assistant" and not last.get("content") and not last.get("tool_calls"):
        return True
    return False


def patch(path, dry_run=False):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    failed = [r for r in d if is_failed(r)]
    kept = [r for r in d if not is_failed(r)]
    if not failed:
        print(f"  no failed cases in {path}")
        return 0
    if dry_run:
        print(f"  [dry-run] {len(failed)} failed / {len(d)} total in {path}")
        return len(failed)
    shutil.copy(path, path + ".preretry.bak")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"  removed {len(failed)} failed → {len(kept)} kept; backup: {path}.preretry.bak")
    return len(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="result JSON files (glob patterns OK)")
    parser.add_argument("--dry-run", action="store_true", help="report counts without modifying files")
    args = parser.parse_args()

    expanded = []
    for p in args.paths:
        m = sorted(glob.glob(p)) or [p]
        expanded.extend(m)

    total = 0
    for p in expanded:
        print(f"== {p} ==")
        try:
            total += patch(p, dry_run=args.dry_run)
        except FileNotFoundError:
            print(f"  ERROR: file not found")
        except json.JSONDecodeError as e:
            print(f"  ERROR: invalid JSON ({e})")

    label = "would remove" if args.dry_run else "removed"
    print(f"\nTotal failed cases {label}: {total}")


if __name__ == "__main__":
    main()
