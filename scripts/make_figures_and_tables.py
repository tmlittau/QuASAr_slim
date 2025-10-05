"""Orchestrate suite runs and figures for the QuASAr slim paper."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from plots.bar_disjoint import make_plot as make_disjoint_bars
from plots.bar_hybrid import make_plot as make_hybrid_bars

ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = ROOT / "suites"


def _run_command(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("[make_figures] $", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def _load_index(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _best_baseline(entry: Dict[str, Any]) -> Tuple[str, float] | None:
    baselines = entry.get("baselines", [])
    if isinstance(baselines, dict):
        items = baselines.get("entries", []) or []
    else:
        items = baselines
    best: Tuple[str, float] | None = None
    for candidate in items:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("ok") is False:
            continue
        scope = candidate.get("scope")
        if scope not in (None, "whole", "global", "circuit") and candidate.get("per_partition") is not False:
            continue
        elapsed = candidate.get("wall_s_measured")
        if elapsed is None:
            elapsed = candidate.get("wall_s_estimated")
        if elapsed is None:
            continue
        method = (candidate.get("method") or candidate.get("name") or "sv").lower()
        seconds = float(elapsed)
        if best is None or seconds < best[1]:
            best = (method, seconds)
    return best


def _summarize_suite(index_path: Path) -> List[Dict[str, Any]]:
    entries = _load_index(index_path)
    summary: List[Dict[str, Any]] = []
    for entry in entries:
        record: Dict[str, Any] = {
            "kind": entry.get("kind"),
            "params": entry.get("params", {}),
            "quasar_wall_s": entry.get("quasar_wall_s"),
        }
        best = _best_baseline(entry)
        if best is not None:
            record["baseline_method"] = best[0]
            record["baseline_wall_s"] = best[1]
        summary.append(record)
    return summary


def _write_markdown_table(path: Path, tables: Dict[str, Iterable[Dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# QuASAr slim: suite summary\n\n")
        for name, rows in tables.items():
            rows = list(rows)
            if not rows:
                continue
            fh.write(f"## {name}\n\n")
            fh.write("| Kind | Parameters | QuASAr wall (s) | Best baseline | Baseline wall (s) |\n")
            fh.write("| --- | --- | --- | --- | --- |\n")
            for row in rows:
                params = ", ".join(f"{k}={v}" for k, v in sorted(row.get("params", {}).items()))
                baseline = row.get("baseline_method", "—")
                baseline_time = row.get("baseline_wall_s")
                if baseline_time is None:
                    baseline_time_str = "—"
                else:
                    baseline_time_str = f"{baseline_time:.3g}"
                wall = row.get("quasar_wall_s")
                wall_str = "—" if wall is None else f"{float(wall):.3g}"
                fh.write(
                    f"| {row.get('kind','?')} | {params} | {wall_str} | {baseline} | {baseline_time_str} |\n"
                )
            fh.write("\n")


def _suite_command(script: str, out_dir: Path, extra: Sequence[str]) -> List[str]:
    return [
        sys.executable,
        str(SUITES_DIR / script),
        "--out-dir",
        str(out_dir),
        *extra,
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run suites and build plots for the paper")
    parser.add_argument("--workspace", type=Path, default=Path("paper_artifacts"), help="Root output directory")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip the hybrid suite run")
    parser.add_argument("--skip-dd-friendly", action="store_true", help="Skip the DD-friendly suite run")
    parser.add_argument("--skip-disjoint", action="store_true", help="Skip the disjoint suite run")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--hybrid-extra",
        type=str,
        default="",
        help="Additional arguments for the hybrid suite (quote the string, e.g. \"--num-qubits 64\")",
    )
    parser.add_argument(
        "--dd-extra",
        type=str,
        default="",
        help="Additional arguments for the DD-friendly suite",
    )
    parser.add_argument(
        "--disjoint-extra",
        type=str,
        default="",
        help="Additional arguments for the disjoint suite",
    )

    args = parser.parse_args(argv)

    workspace = (ROOT / args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    tables: Dict[str, List[Dict[str, Any]]] = {}

    if not args.skip_hybrid:
        hybrid_dir = workspace / "hybrid_suite"
        hybrid_dir.mkdir(parents=True, exist_ok=True)
        extra = shlex.split(args.hybrid_extra) if args.hybrid_extra else []
        cmd = _suite_command("run_hybrid_suite.py", hybrid_dir, extra)
        _run_command(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            make_hybrid_bars(
                str(hybrid_dir),
                out=str(workspace / "fig_hybrid_bars.png"),
                title="Hybrid circuits: QuASAr vs baseline",
            )
            tables["Hybrid suite"] = _summarize_suite(hybrid_dir / "index.json")

    if not args.skip_dd_friendly:
        dd_dir = workspace / "dd_friendly_suite"
        dd_dir.mkdir(parents=True, exist_ok=True)
        extra = shlex.split(args.dd_extra) if args.dd_extra else []
        cmd = _suite_command("run_dd_friendly_suite.py", dd_dir, extra)
        _run_command(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            make_hybrid_bars(
                str(dd_dir),
                out=str(workspace / "fig_dd_friendly_bars.png"),
                title="DD-friendly circuits: QuASAr vs baseline",
            )
            tables["DD-friendly suite"] = _summarize_suite(dd_dir / "index.json")

    if not args.skip_disjoint:
        dis_dir = workspace / "disjoint_suite"
        dis_dir.mkdir(parents=True, exist_ok=True)
        extra = shlex.split(args.disjoint_extra) if args.disjoint_extra else []
        cmd = _suite_command("run_disjoint_suite.py", dis_dir, extra)
        _run_command(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            make_disjoint_bars(
                str(dis_dir),
                out=str(workspace / "fig_disjoint_bars.png"),
                title="Disjoint circuits: QuASAr vs baseline",
            )
            tables["Disjoint suite"] = _summarize_suite(dis_dir / "index.json")

    if tables and not args.dry_run:
        _write_markdown_table(workspace / "summary.md", tables)
        print(f"[make_figures] Wrote {workspace / 'summary.md'}")


if __name__ == "__main__":
    main()
