"""Command-line helper to build QuASAr figures and tables for the paper.

This script exposes sub-commands that orchestrate running the benchmark
suites (if necessary) and invoking the corresponding plotting utilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = ROOT / "suites"


# ---------------------------------------------------------------------------
# Utilities


def _run_command(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    """Run *cmd*, printing it beforehand."""

    print("[make_figures] $", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _has_suite_results(suite_dir: Path) -> bool:
    """Return True if *suite_dir* looks like it already contains JSON results."""

    if not suite_dir.exists():
        return False
    for child in suite_dir.iterdir():
        if child.is_file() and child.suffix == ".json" and child.name != "index.json":
            return True
    return False


def _ensure_suite(
    script_name: str,
    *,
    suite_dir: Path,
    runner_args: Sequence[str],
    force: bool,
    dry_run: bool,
) -> None:
    """Run the suite if results are missing or ``force`` is True."""

    if force or not _has_suite_results(suite_dir):
        suite_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(SUITES_DIR / script_name), *runner_args]
        _run_command(cmd, dry_run=dry_run)
    else:
        print(f"[make_figures] Reusing existing results in {suite_dir}")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN check
        return None
    return result


def _extract_quasar_wall(entry: MutableMapping[str, Any]) -> Optional[float]:
    quasar = entry.get("quasar")
    if isinstance(quasar, MutableMapping):
        wall = quasar.get("wall_elapsed_s")
        if wall is not None:
            return _safe_float(wall)
        execution = quasar.get("execution")
        if isinstance(execution, MutableMapping):
            meta = execution.get("meta")
            if isinstance(meta, MutableMapping):
                meta_wall = _safe_float(meta.get("wall_elapsed_s"))
                if meta_wall is not None:
                    return meta_wall
            results = execution.get("results")
            if isinstance(results, Iterable):
                best: Optional[float] = None
                for res in results:
                    if not isinstance(res, MutableMapping):
                        continue
                    elapsed = _safe_float(res.get("elapsed_s"))
                    if elapsed is None:
                        continue
                    if best is None or elapsed > best:
                        best = elapsed
                if best is not None:
                    return best
    execution = entry.get("execution")
    if isinstance(execution, MutableMapping):
        meta = execution.get("meta")
        if isinstance(meta, MutableMapping):
            meta_wall = _safe_float(meta.get("wall_elapsed_s"))
            if meta_wall is not None:
                return meta_wall
    return None


def _baseline_entries(data: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    baselines = data.get("baselines")
    if isinstance(baselines, MutableMapping):
        entries = baselines.get("entries")
        if isinstance(entries, list):
            return [e for e in entries if isinstance(e, MutableMapping)]
    if isinstance(baselines, list):
        return [e for e in baselines if isinstance(e, MutableMapping)]
    return []


def _best_baseline(entries: Iterable[MutableMapping[str, Any]]) -> Optional[Tuple[str, float]]:
    best: Optional[Tuple[str, float]] = None
    for candidate in entries:
        if candidate.get("ok") is False:
            continue
        scope = candidate.get("scope")
        if scope not in (None, "whole", "global", "circuit") and candidate.get("per_partition") is not False:
            if "partition_id" in candidate or "chain_id" in candidate:
                continue
        elapsed = candidate.get("wall_s_measured")
        if elapsed is None:
            elapsed = candidate.get("wall_s_estimated")
        elapsed_f = _safe_float(elapsed)
        if elapsed_f is None:
            continue
        method = str(candidate.get("method") or candidate.get("name") or "sv").lower()
        if best is None or elapsed_f < best[1]:
            best = (method, elapsed_f)
    return best


def _load_suite_records(suite_dir: Path) -> List[Dict[str, Any]]:
    index = suite_dir / "index.json"
    if index.exists():
        try:
            with index.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                records: List[Dict[str, Any]] = []
                for entry in data:
                    if not isinstance(entry, MutableMapping):
                        continue
                    rec = {
                        "kind": entry.get("kind"),
                        "params": entry.get("params", {}),
                        "quasar_wall_s": entry.get("quasar_wall_s"),
                        "baselines": entry.get("baselines", []),
                    }
                    records.append(rec)
                return records
        except Exception:
            pass

    records: List[Dict[str, Any]] = []
    for path in sorted(suite_dir.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if not isinstance(data, MutableMapping):
            continue
        case = data.get("case") if isinstance(data.get("case"), MutableMapping) else {}
        record = {
            "kind": case.get("kind"),
            "params": case.get("params", {}),
            "quasar_wall_s": _extract_quasar_wall(data),
            "baselines": _baseline_entries(data),
        }
        records.append(record)
    return records


def _serialise_params(params: Dict[str, Any]) -> str:
    if not isinstance(params, MutableMapping):
        return ""
    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Sub-commands


def cmd_hybrid(args: argparse.Namespace) -> None:
    from plots.bar_hybrid import make_plot as make_hybrid_bars

    suite_dir = Path(args.results_dir or "suite_hybrid").resolve()
    out_path = Path(args.out).resolve()

    runner_args: List[str] = ["--out-dir", str(suite_dir)]
    if args.n:
        runner_args.extend(["--num-qubits", *map(str, args.n)])
    if args.block_size:
        runner_args.extend(["--block-size", *map(str, args.block_size)])
    if args.non_disjoint_qubits is not None:
        runner_args.extend(["--non-disjoint-qubits", str(args.non_disjoint_qubits)])
    runner_args.extend(["--max-ram-gb", str(args.max_ram_gb)])
    runner_args.extend(["--conv-factor", str(args.conv_factor)])
    runner_args.extend(["--twoq-factor", str(args.twoq_factor)])
    if args.sv_ampops_per_sec is not None:
        runner_args.extend(["--sv-ampops-per-sec", str(args.sv_ampops_per_sec)])

    _ensure_suite(
        "run_hybrid_suite.py",
        suite_dir=suite_dir,
        runner_args=runner_args,
        force=args.force,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return

    print(f"[make_figures] Building hybrid bar chart -> {out_path}")
    make_hybrid_bars(str(suite_dir), out=str(out_path), title=args.title)


def cmd_disjoint(args: argparse.Namespace) -> None:
    from plots.bar_disjoint import make_plot as make_disjoint_bars

    suite_dir = Path(args.results_dir or "suite_disjoint").resolve()
    out_path = Path(args.out).resolve()

    runner_args: List[str] = [
        "--out-dir",
        str(suite_dir),
        "--n",
        *map(str, args.n),
        "--blocks",
        *map(str, args.blocks),
        "--max-ram-gb",
        str(args.max_ram_gb),
        "--conv-factor",
        str(args.conv_factor),
        "--twoq-factor",
        str(args.twoq_factor),
    ]

    if args.tail_kind is not None:
        runner_args.extend(["--tail-kind", args.tail_kind])
    if args.tail_depth is not None:
        runner_args.extend(["--tail-depth", str(args.tail_depth)])
    if args.angle_scale is not None:
        runner_args.extend(["--angle-scale", str(args.angle_scale)])
    if args.sparsity is not None:
        runner_args.extend(["--sparsity", str(args.sparsity)])
    if args.bandwidth is not None:
        runner_args.extend(["--bandwidth", str(args.bandwidth)])
    if args.prep is not None:
        runner_args.extend(["--prep", args.prep])
    if args.sv_ampops_per_sec is not None:
        runner_args.extend(["--sv-ampops-per-sec", str(args.sv_ampops_per_sec)])
    if args.seed is not None:
        runner_args.extend(["--seed", str(args.seed)])

    _ensure_suite(
        "run_disjoint_suite.py",
        suite_dir=suite_dir,
        runner_args=runner_args,
        force=args.force,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return

    print(f"[make_figures] Building disjoint bar chart -> {out_path}")
    make_disjoint_bars(str(suite_dir), out=str(out_path), title=args.title)


@dataclass
class TableRow:
    suite: str
    kind: str
    params: str
    quasar_wall_s: Optional[float]
    baseline_method: Optional[str]
    baseline_wall_s: Optional[float]


def _build_table_rows(suite_dir: Path) -> List[TableRow]:
    records = _load_suite_records(suite_dir)
    rows: List[TableRow] = []
    for record in records:
        raw_baselines = record.get("baselines", [])
        if isinstance(raw_baselines, MutableMapping):
            baseline_candidates = raw_baselines.get("entries", []) or []
        else:
            baseline_candidates = raw_baselines
        baselines = _best_baseline(baseline_candidates)
        baseline_method: Optional[str]
        baseline_time: Optional[float]
        if baselines is None:
            baseline_method = None
            baseline_time = None
        else:
            baseline_method, baseline_time = baselines
        rows.append(
            TableRow(
                suite=suite_dir.name,
                kind=str(record.get("kind")),
                params=_serialise_params(record.get("params", {})),
                quasar_wall_s=_safe_float(record.get("quasar_wall_s")),
                baseline_method=baseline_method,
                baseline_wall_s=baseline_time,
            )
        )
    return rows


def _write_csv(path: Path, rows: Iterable[TableRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "suite",
                "kind",
                "params",
                "quasar_wall_s",
                "baseline_method",
                "baseline_wall_s",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.suite,
                    row.kind,
                    row.params,
                    "" if row.quasar_wall_s is None else f"{row.quasar_wall_s:.6g}",
                    row.baseline_method or "",
                    "" if row.baseline_wall_s is None else f"{row.baseline_wall_s:.6g}",
                ]
            )


def _write_markdown(path: Path, rows: Iterable[TableRow]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("| Suite | Kind | Parameters | QuASAr wall (s) | Baseline | Baseline wall (s) |\n")
        fh.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            q = "—" if row.quasar_wall_s is None else f"{row.quasar_wall_s:.3g}"
            b = row.baseline_method or "—"
            bt = "—" if row.baseline_wall_s is None else f"{row.baseline_wall_s:.3g}"
            fh.write(
                f"| {row.suite} | {row.kind} | {row.params} | {q} | {b} | {bt} |\n"
            )


def cmd_table(args: argparse.Namespace) -> None:
    suite_dirs = [Path(p).resolve() for p in args.suite_dir]
    for suite_dir in suite_dirs:
        if not suite_dir.exists():
            raise SystemExit(f"Suite directory not found: {suite_dir}")

    rows: List[TableRow] = []
    for suite_dir in suite_dirs:
        suite_rows = _build_table_rows(suite_dir)
        if not suite_rows:
            print(f"[make_figures] Warning: no entries found in {suite_dir}")
        rows.extend(suite_rows)

    if not rows:
        raise SystemExit("No rows produced; nothing to write")

    out_path = Path(args.out).resolve()
    suffix = out_path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        _write_markdown(out_path, rows)
    else:
        _write_csv(out_path, rows)
    print(f"[make_figures] Wrote table to {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--max-ram-gb", type=float, default=64.0, help="Planner/executor RAM budget")
    parent.add_argument("--conv-factor", type=float, default=64.0, help="Conversion amortisation factor")
    parent.add_argument("--twoq-factor", type=float, default=4.0, help="Statevector two-qubit factor")
    parent.add_argument("--sv-ampops-per-sec", type=float, default=None, help="Override SV amp-ops/sec speed")
    parent.add_argument(
        "--out-dir",
        "--suite-dir",
        dest="results_dir",
        type=str,
        default=None,
        help="Directory for suite JSON results",
    )
    parent.add_argument("--force", action="store_true", help="Re-run suites even if results already exist")
    parent.add_argument("--dry-run", action="store_true", help="Print commands without executing them")

    parser = argparse.ArgumentParser(
        description="Generate QuASAr paper figures and tables",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Hybrid -----------------------------------------------------------------
    hybrid = subparsers.add_parser(
        "hybrid",
        parents=[parent],
        help="Run the hybrid suite and create the stacked bar chart",
    )
    hybrid.add_argument("--n", type=int, nargs="+", default=[64, 96], help="Number of qubits to sweep")
    hybrid.add_argument("--block-size", type=int, nargs="+", default=[8], help="Hybrid block sizes")
    hybrid.add_argument("--non-disjoint-qubits", type=int, default=None, help="Limit non-disjoint qubits when planning")
    hybrid.add_argument("--out", type=str, required=True, help="Path to the output bar chart file")
    hybrid.add_argument("--title", type=str, default=None, help="Optional figure title override")
    hybrid.set_defaults(func=cmd_hybrid)

    # Disjoint ---------------------------------------------------------------
    disjoint = subparsers.add_parser(
        "disjoint",
        parents=[parent],
        help="Run the disjoint suite and create the QuASAr vs baseline bar chart",
    )
    disjoint.add_argument("--n", type=int, nargs="+", required=True, help="Numbers of qubits to sweep")
    disjoint.add_argument("--blocks", type=int, nargs="+", required=True, help="Block counts to evaluate")
    disjoint.add_argument(
        "--prep",
        type=str,
        default="mixed",
        help="Preparation routine kind override (default mixes GHZ and W blocks)",
    )
    disjoint.add_argument(
        "--tail-kind",
        type=str,
        default="mixed",
        help=(
            "Tail circuit kind override; use 'mixed' for alternating Clifford and "
            "random-rotation (diagonal) tails"
        ),
    )
    disjoint.add_argument(
        "--tail-depth", type=int, default=20, help="Tail depth override (layers per block)"
    )
    disjoint.add_argument(
        "--angle-scale",
        type=float,
        default=0.1,
        help="Tail rotation angle scale for diagonal tails",
    )
    disjoint.add_argument(
        "--sparsity",
        type=float,
        default=0.05,
        help="Tail sparsity for diagonal layers",
    )
    disjoint.add_argument(
        "--bandwidth",
        type=int,
        default=2,
        help="Tail bandwidth for diagonal layers",
    )
    disjoint.add_argument("--seed", type=int, default=None, help="Random seed for circuit construction")
    disjoint.add_argument("--out", type=str, required=True, help="Path to the output bar chart file")
    disjoint.add_argument("--title", type=str, default=None, help="Optional figure title override")
    disjoint.set_defaults(func=cmd_disjoint)

    # Table ------------------------------------------------------------------
    table = subparsers.add_parser(
        "table",
        help="Summarise suite results into a CSV or Markdown table",
    )
    table.add_argument("--suite-dir", type=str, nargs="+", required=True, help="Suite directory/directories to summarise")
    table.add_argument("--out", type=str, required=True, help="Output CSV/Markdown path")
    table.set_defaults(func=cmd_table)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

