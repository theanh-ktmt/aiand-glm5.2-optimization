#!/usr/bin/env python3
"""Sync one config's benchmark results to Weights & Biases.

Reads results/<config>.csv (produced by aggregate.py) and logs ONE W&B run per
config:
  - per-concurrency curves for each scenario (throughput / TTFT / TPOT) with
    concurrency on the x-axis, so configs overlay cleanly in the dashboard
  - a summary table of every row
  - the raw result JSONs + logs (server.log, bench.log, gpu.csv) as an artifact

Durability: results reach W&B as soon as each config finishes, so an instance
dying later loses nothing.

Auth: WANDB_API_KEY (or `wandb login`). Project via WANDB_PROJECT
(default aiand-glm5.2-fp8), entity via WANDB_ENTITY. Best-effort: missing
key/wandb/CSV -> loud skip (exit 3), upload error -> exit 1, success -> exit 0.

Usage:
    python3 wandb_sync.py --config baseline
    python3 wandb_sync.py --config baseline --dry-run    # parse + print, no upload
"""
import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Exit codes so callers (run.sh) can tell outcomes apart:
#   0 = synced, 3 = skipped (nothing uploaded), 1 = error during upload.
EXIT_OK, EXIT_ERROR, EXIT_SKIP = 0, 1, 3

# CSV column -> short metric name.
METRIC_MAP = {
    "Output tok/s": "output_tput",
    "Out tok/s/GPU": "out_tput_per_gpu",
    "Total tok/s": "total_tput",
    "TTFT Mean": "ttft_mean",
    "TTFT Median": "ttft_median",
    "TTFT P90": "ttft_p90",
    "TPOT Mean": "tpot_mean",
    "TPOT Median": "tpot_median",
    "TPOT P90": "tpot_p90",
}


def skip(reason):
    """Print a loud, un-missable banner and return the SKIP exit code."""
    bar = "!" * 70
    print(f"\n{bar}\n!! W&B SYNC SKIPPED - {reason}\n{bar}\n", file=sys.stderr)
    return EXIT_SKIP


def scen_label(isl, osl):
    def k(n):
        n = int(n)
        return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)
    return f"{k(isl)}_{k(osl)}"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config name (e.g. baseline)")
    ap.add_argument("--csv", help="CSV path (default results/<config>.csv)")
    ap.add_argument("--results-dir", help="dir with JSON/logs (default results/<config>)")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "aiand-glm5.2-fp8"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--group", default=os.environ.get("WANDB_GROUP"))
    ap.add_argument("--dry-run", action="store_true", help="parse + print, no W&B upload")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else REPO_ROOT / "results" / f"{args.config}.csv"
    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results" / args.config

    if not csv_path.is_file():
        return skip(f"CSV not found: {csv_path} - did this config finish (run.sh)?")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return skip(f"CSV is empty: {csv_path}")

    header = list(rows[0].keys())
    mode = rows[0].get("Mode", "")

    # --- best-effort guards (make the reason LOUD so a skip isn't missed) ----
    if not args.dry_run:
        have_key = bool((os.environ.get("WANDB_API_KEY") or "").strip())
        have_netrc = (Path.home() / ".netrc").exists()
        if not (have_key or have_netrc):
            return skip("WANDB_API_KEY not in environment and no ~/.netrc login. "
                        "It lives in .env, which only run.sh/run_all.sh load - "
                        "run via those, or `export WANDB_API_KEY=...`.")
        try:
            import wandb  # noqa: F401
        except ImportError:
            return skip("wandb not installed. `pip install wandb` "
                        "(container may have no internet).")

    # per-concurrency points for each scenario.
    concs = sorted({int(r["Conc"]) for r in rows if r.get("Conc")})
    per_conc = {c: {} for c in concs}
    for r in rows:
        if not r.get("Conc"):
            continue
        c = int(r["Conc"])
        s = scen_label(r["ISL"], r["OSL"])
        for col, short in METRIC_MAP.items():
            v = num(r.get(col))
            if v is not None:
                per_conc[c][f"{s}/{short}"] = v

    if args.dry_run:
        print(f"[dry-run] config={args.config} mode={mode} project={args.project}")
        print(f"[dry-run] rows={len(rows)} concs={concs}")
        for c in concs:
            print(f"[dry-run] step conc={c}: {per_conc[c]}")
        arts = [p.name for p in results_dir.glob("*")] if results_dir.is_dir() else []
        print(f"[dry-run] artifact files from {results_dir}: {arts}")
        return EXIT_OK

    import wandb

    # Strip stray whitespace/CR that a CRLF-edited .env can leave on values.
    project = (args.project or "").strip()
    entity = (args.entity or "").strip() or None
    key = (os.environ.get("WANDB_API_KEY") or "").strip()

    try:
        if key:
            wandb.login(key=key)   # explicit -> fails loudly on a bad/corrupt key
        run = wandb.init(
            project=project,
            entity=entity,
            group=(args.group or mode),
            name=args.config,
            job_type="benchmark",
            config={
                "config": args.config,
                "mode": mode,
                "model": os.environ.get("MODEL", "zai-org/GLM-5.2-FP8"),
                "tp": os.environ.get("TP", "8"),
                "num_gpus": os.environ.get("NUM_GPUS", "8"),
            },
        )

        # Concurrency as the x-axis for every metric.
        wandb.define_metric("conc")
        wandb.define_metric("*", step_metric="conc")
        for c in concs:
            wandb.log({"conc": c, **per_conc[c]})

        # Full table for at-a-glance inspection.
        table = wandb.Table(columns=header, data=[[r.get(h, "") for h in header] for r in rows])
        wandb.log({"results_table": table})

        # Raw JSONs + logs as an artifact (so nothing is lost if the box dies).
        art = wandb.Artifact(f"{args.config}-artifacts", type="benchmark")
        art.add_file(str(csv_path))
        if results_dir.is_dir():
            for p in results_dir.iterdir():
                if p.is_file():
                    art.add_file(str(p))
        run.log_artifact(art)
        run.finish()
    except Exception as exc:  # noqa: BLE001 - make the real cause obvious
        mode_env = os.environ.get("WANDB_MODE", "<unset>")
        print(f"ERROR: W&B sync failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"       project={project!r} entity={entity!r} "
              f"WANDB_MODE={mode_env} key_len={len(key)}", file=sys.stderr)
        print("       Hints: check the API key (no trailing chars), network to "
              "api.wandb.ai, and that WANDB_MODE isn't 'offline'.", file=sys.stderr)
        return EXIT_ERROR

    print(f"W&B sync done: project={project} run={args.config}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
