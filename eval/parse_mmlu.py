#!/usr/bin/env python3
"""Compare MMLU-Pro accuracy across two (or more) lm-eval result dirs.

lm-eval writes a results JSON (``results_*.json``) under the ``--output_path``
directory. This script finds the newest one in each given dir, extracts the
MMLU-Pro accuracy, and prints a comparison table + a "Pass?" verdict (the
candidate passes if it does not regress more than --threshold accuracy points
below the baseline = the first dir).

Usage:
    python3 parse_mmlu.py results/baseline/mmlu_pro results/final/mmlu_pro \
        --names baseline final --out results/quality_check.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def find_results_json(d: Path):
    cands = sorted(d.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        cands = sorted(d.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def extract_acc(results_json: Path):
    """Return (accuracy_percent, metric_key) for the mmlu_pro task."""
    with open(results_json, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", {})
    # Prefer an exact 'mmlu_pro' key; else any key containing it (group or subtask).
    keys = list(results.keys())
    task_key = next((k for k in keys if k == "mmlu_pro"), None) \
        or next((k for k in keys if "mmlu_pro" in k), None) \
        or (keys[0] if keys else None)
    if task_key is None:
        return None, None
    metrics = results[task_key]
    # Pick the primary accuracy metric.
    for pref in ("acc,none", "exact_match,none", "acc_norm,none"):
        if pref in metrics:
            return round(float(metrics[pref]) * 100, 2), f"{task_key}:{pref}"
    for k, v in metrics.items():
        if ("acc" in k or "exact_match" in k) and not k.endswith("_stderr") \
                and isinstance(v, (int, float)):
            return round(float(v) * 100, 2), f"{task_key}:{k}"
    return None, task_key


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="lm-eval result dirs (first = baseline)")
    ap.add_argument("--names", nargs="*", help="labels for each dir")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="max allowed accuracy-point regression vs baseline (default 1.0)")
    ap.add_argument("--out", help="write the comparison CSV here")
    args = ap.parse_args()

    names = args.names or [Path(d).parent.name or d for d in args.dirs]
    rows = []
    for name, d in zip(names, args.dirs):
        path = Path(d)
        rj = find_results_json(path) if path.exists() else None
        if rj is None:
            print(f"WARN: no results_*.json under {d}", file=sys.stderr)
            rows.append({"Config": name, "MMLU-Pro (%)": "", "Metric": "", "_acc": None})
            continue
        acc, metric = extract_acc(rj)
        rows.append({"Config": name, "MMLU-Pro (%)": acc if acc is not None else "",
                     "Metric": metric or "", "_acc": acc})

    base_acc = rows[0]["_acc"]
    for r in rows:
        if r is rows[0] or base_acc is None or r["_acc"] is None:
            r["Delta"] = ""
            r["Pass?"] = "" if r is rows[0] else "N/A"
        else:
            delta = round(r["_acc"] - base_acc, 2)
            r["Delta"] = f"{delta:+.2f}"
            r["Pass?"] = "PASS" if delta >= -args.threshold else "FAIL"

    cols = ["Config", "MMLU-Pro (%)", "Metric", "Delta", "Pass?"]
    # Console table.
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([{c: r.get(c, "") for c in cols} for r in rows])
        print(f"\nWrote {args.out}", file=sys.stderr)

    # Non-zero exit if any candidate FAILed, so CI can gate on it.
    if any(r.get("Pass?") == "FAIL" for r in rows):
        sys.exit(2)


if __name__ == "__main__":
    main()
