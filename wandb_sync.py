#!/usr/bin/env python3
"""Sync GLM-5.2 benchmark results to Weights & Biases.

Two modes, both logging into ONE W&B run per config (resumed across processes
via the WANDB_RUN_ID env var, so per-cell updates and the final table land in
the same run):

  --log-cell <result.json>   Log a single (scenario, concurrency) cell as soon
                             as bench.sh finishes it — live progress.
  --config <name>            Final push at end of the config: the full results
                             table + raw JSONs/logs as an artifact. With
                             --table-only it does NOT re-log the per-conc points
                             (they were already streamed per-cell).

Without WANDB_RUN_ID (per-cell disabled), --config logs everything in one fresh
run (per-conc curves + table + artifact) — the original behavior.

Durability: results reach W&B as each cell/config finishes, so an instance dying
later loses nothing.

Auth: WANDB_API_KEY (or `wandb login`). Project via WANDB_PROJECT
(default aiand-glm5.2-fp8), entity via WANDB_ENTITY. Best-effort: missing
key/wandb/CSV -> loud skip (exit 3), upload error -> exit 1, success -> exit 0.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Exit codes so callers (run.sh/bench.sh) can tell outcomes apart:
#   0 = synced, 3 = skipped (nothing uploaded), 1 = error during upload.
EXIT_OK, EXIT_ERROR, EXIT_SKIP = 0, 1, 3

# result filename convention from bench.sh:
#   <config>__<mode>_isl<ISL>_osl<OSL>_conc<CONC>.json
FNAME_RE = re.compile(
    r"^(?P<config>.+?)__(?P<mode>mtp|nonmtp)_isl(?P<isl>\d+)_osl(?P<osl>\d+)_conc(?P<conc>\d+)$"
)

# CSV column -> short metric name (final table path).
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

# raw benchmark_serving JSON key -> short metric name (per-cell path).
RAW_MAP = {
    "output_throughput": "output_tput",
    "total_token_throughput": "total_tput",
    "mean_ttft_ms": "ttft_mean",
    "median_ttft_ms": "ttft_median",
    "p90_ttft_ms": "ttft_p90",
    "mean_tpot_ms": "tpot_mean",
    "median_tpot_ms": "tpot_median",
    "p90_tpot_ms": "tpot_p90",
}


def skip(reason, quiet=False):
    """Print a loud banner (unless quiet) and return the SKIP exit code."""
    if not quiet:
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


def _num_gpus():
    try:
        return int(os.environ.get("NUM_GPUS", "8"))
    except ValueError:
        return 8


def _have_auth():
    return bool((os.environ.get("WANDB_API_KEY") or "").strip()) \
        or (Path.home() / ".netrc").exists()


def _init_run(wandb, name, mode, project, entity):
    """Init (or resume, if WANDB_RUN_ID is set) the single run for this config."""
    key = (os.environ.get("WANDB_API_KEY") or "").strip()
    if key:
        wandb.login(key=key)
    rid = (os.environ.get("WANDB_RUN_ID") or "").strip() or None
    return wandb.init(
        id=rid,
        resume=("allow" if rid else None),
        project=project,
        entity=entity,
        group=(os.environ.get("WANDB_GROUP") or mode),
        name=name,
        job_type="benchmark",
        config={
            "config": name,
            "mode": mode,
            "model": os.environ.get("MODEL", "zai-org/GLM-5.2-FP8"),
            "tp": os.environ.get("TP", "8"),
            "num_gpus": os.environ.get("NUM_GPUS", "8"),
        },
    )


# ---------------------------------------------------------------------------
# per-cell mode
# ---------------------------------------------------------------------------
def log_cell(json_path, project, entity):
    p = Path(json_path)
    stem = p.name[:-5] if p.name.endswith(".json") else p.stem
    m = FNAME_RE.match(stem)
    if not m:
        return skip(f"cannot parse cell filename: {p.name}", quiet=True)
    config, mode = m["config"], m["mode"]
    isl, osl, conc = int(m["isl"]), int(m["osl"]), int(m["conc"])

    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: per-cell read failed for {p.name}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    s = scen_label(isl, osl)
    metrics = {"conc": conc}
    for raw, short in RAW_MAP.items():
        v = num(d.get(raw))
        if v is not None:
            metrics[f"{s}/{short}"] = v
    ot = d.get("output_throughput")
    if isinstance(ot, (int, float)):
        metrics[f"{s}/out_tput_per_gpu"] = round(float(ot) / _num_gpus(), 2)

    # Per-cell runs quietly during the sweep (no loud banner spam).
    if not _have_auth():
        return skip("no WANDB_API_KEY for per-cell log", quiet=True)
    try:
        import wandb
    except ImportError:
        return skip("wandb not installed for per-cell log", quiet=True)

    try:
        run = _init_run(wandb, config, mode, project, entity)
        wandb.define_metric("conc")
        wandb.define_metric("*", step_metric="conc")
        wandb.log(metrics)
        run.finish()
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: per-cell W&B log failed ({p.name}): {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


# ---------------------------------------------------------------------------
# final per-config mode
# ---------------------------------------------------------------------------
def log_config(config, csv_path, results_dir, project, entity, table_only):
    if not csv_path.is_file():
        return skip(f"CSV not found: {csv_path} - did this config finish (run.sh)?")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return skip(f"CSV is empty: {csv_path}")

    header = list(rows[0].keys())
    mode = rows[0].get("Mode", "")

    if not _have_auth():
        return skip("WANDB_API_KEY not in environment and no ~/.netrc login. "
                    "It lives in .env, which only run.sh/run_all.sh load - "
                    "run via those, or `export WANDB_API_KEY=...`.")
    try:
        import wandb
    except ImportError:
        return skip("wandb not installed. `pip install wandb` "
                    "(container may have no internet).")

    # per-conc points (skipped when they were already streamed per-cell).
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

    try:
        run = _init_run(wandb, config, mode, project, entity)
        if not table_only:
            wandb.define_metric("conc")
            wandb.define_metric("*", step_metric="conc")
            for c in concs:
                wandb.log({"conc": c, **per_conc[c]})

        table = wandb.Table(columns=header, data=[[r.get(h, "") for h in header] for r in rows])
        wandb.log({"results_table": table})

        art = wandb.Artifact(f"{config}-artifacts", type="benchmark")
        art.add_file(str(csv_path))
        if results_dir.is_dir():
            for p in results_dir.iterdir():
                if p.is_file():
                    art.add_file(str(p))
        run.log_artifact(art)
        run.finish()
    except Exception as exc:  # noqa: BLE001 - make the real cause obvious
        mode_env = os.environ.get("WANDB_MODE", "<unset>")
        key = (os.environ.get("WANDB_API_KEY") or "").strip()
        print(f"ERROR: W&B sync failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"       project={project!r} entity={entity!r} "
              f"WANDB_MODE={mode_env} key_len={len(key)}", file=sys.stderr)
        print("       Hints: check the API key (no trailing chars), network to "
              "api.wandb.ai, and that WANDB_MODE isn't 'offline'.", file=sys.stderr)
        return EXIT_ERROR

    what = "table+artifact" if table_only else "curves+table+artifact"
    print(f"W&B sync done ({what}): project={project} run={config}")
    return EXIT_OK


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config name (final push). Not needed with --log-cell.")
    ap.add_argument("--log-cell", dest="log_cell", metavar="RESULT_JSON",
                    help="log a single result JSON (per-cell live mode)")
    ap.add_argument("--table-only", action="store_true",
                    help="final push logs only the table + artifact "
                         "(per-conc points already streamed per-cell)")
    ap.add_argument("--csv", help="CSV path (default results/<config>.csv)")
    ap.add_argument("--results-dir", help="dir with JSON/logs (default results/<config>)")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "aiand-glm5.2-fp8"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--dry-run", action="store_true", help="parse + print, no W&B upload")
    args = ap.parse_args()

    project = (args.project or "").strip()
    entity = (args.entity or "").strip() or None

    if args.log_cell:
        return log_cell(args.log_cell, project, entity)

    if not args.config:
        print("ERROR: --config is required (or use --log-cell)", file=sys.stderr)
        return EXIT_ERROR

    csv_path = Path(args.csv) if args.csv else REPO_ROOT / "results" / f"{args.config}.csv"
    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results" / args.config

    if args.dry_run:
        if not csv_path.is_file():
            print(f"[dry-run] CSV not found: {csv_path}")
            return EXIT_OK
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"[dry-run] config={args.config} rows={len(rows)} "
              f"table_only={args.table_only} run_id={os.environ.get('WANDB_RUN_ID', '<none>')}")
        return EXIT_OK

    return log_config(args.config, csv_path, results_dir, project, entity, args.table_only)


if __name__ == "__main__":
    sys.exit(main())
