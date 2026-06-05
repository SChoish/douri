#!/usr/bin/env python3
"""Parse train.csv / run logs for puzzle FBR displacement grid sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ACTOR_KEY = 'eval/success_rate_mean'
IDM_KEY = 'eval_idm/success_rate_mean'
EPOCH_KEY = 'train/epoch'

ZERO_TOL = 1e-6


def _decode_num_token(s: str) -> float:
    return float(s.replace('p', '.').replace('m', '-'))


def parse_config_path(cfg_path: Path) -> dict[str, object]:
    name = cfg_path.name
    m = re.match(
        r'^puzzle_(3x3|4x4|4x6)_a(?P<a>[^_]+)_gap(?P<g>[^_]+)_k(?P<k>[^.]+)\.yaml$',
        name,
    )
    if not m:
        raise ValueError(f'Unexpected config filename: {name}')
    tag = m.group(1)
    env_map = {'3x3': 'puzzle-3x3-play-v0', '4x4': 'puzzle-4x4-play-v0', '4x6': 'puzzle-4x6-play-v0'}
    return {
        'alpha': _decode_num_token(m.group('a')),
        'gap': _decode_num_token(m.group('g')),
        'kappa': _decode_num_token(m.group('k')),
        'file_tag': tag,
        'env_from_name': env_map[tag],
    }


def parse_run_dir_from_log(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    last = None
    pat = re.compile(r'run_dir=([^\s]+)')
    with open(log_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = pat.search(line)
            if m:
                last = m.group(1).strip()
    return last


def _float_cell(raw: str) -> float | None:
    raw = (raw or '').strip()
    if raw == '':
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_train_rows(train_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not train_csv.is_file():
        return [], []
    with open(train_csv, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = [dict(row) for row in r]
    return fieldnames, rows


def _csv_eval_points(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    upto_epoch: float,
) -> list[tuple[float, float | None, float | None]]:
    has_actor = ACTOR_KEY in fieldnames
    has_idm = IDM_KEY in fieldnames
    if not has_actor and not has_idm:
        return []
    out: list[tuple[float, float | None, float | None]] = []
    for row in rows:
        ep = _float_cell(row.get(EPOCH_KEY, ''))
        if ep is None or ep > float(upto_epoch) + 1e-6:
            continue
        av = _float_cell(row.get(ACTOR_KEY, '')) if has_actor else None
        iv = _float_cell(row.get(IDM_KEY, '')) if has_idm else None
        if av is None and iv is None:
            continue
        out.append((ep, av, iv))
    return out


def _iter_run_log_files(run_dir: Path) -> list[Path]:
    rd = Path(run_dir)
    paths: list[Path] = []
    main = rd / 'run.log'
    if main.is_file():
        paths.append(main)
    paths.extend(sorted(rd.glob('run_resume_from*.log')))
    return paths


def _log_eval_points(run_dir: Path, upto_epoch: float) -> list[tuple[float, float | None, float | None]]:
    """Parse idm/actor env_success_rate_mean from run.log (+ resume logs)."""
    start_pat = re.compile(r'===\s*EVAL\s+START\s+epoch=(\d+)')
    idm_pat = re.compile(r'idm\s+env_success_rate_mean=([\d.+\-eE]+)')
    actor_pat = re.compile(r'actor\s+env_success_rate_mean=([\d.+\-eE]+)')
    pts: list[tuple[float, float | None, float | None]] = []
    for logf in _iter_run_log_files(run_dir):
        cur_ep: float | None = None
        pending_idm: float | None = None
        with open(logf, encoding='utf-8', errors='replace') as f:
            for line in f:
                ms = start_pat.search(line)
                if ms:
                    cur_ep = float(ms.group(1))
                    pending_idm = None
                    continue
                if cur_ep is None or cur_ep > float(upto_epoch) + 1e-6:
                    continue
                mi = idm_pat.search(line)
                if mi:
                    pending_idm = float(mi.group(1))
                ma = actor_pat.search(line)
                if ma and pending_idm is not None:
                    pts.append((cur_ep, float(ma.group(1)), pending_idm))
                    cur_ep = None
                    pending_idm = None
    return pts


def _aggregate_eval_points(
    eval_points: list[tuple[float, float | None, float | None]],
    upto_epoch: float,
) -> dict[str, object]:
    eval_points = [(e, a, i) for e, a, i in eval_points if e <= float(upto_epoch) + 1e-6]

    if not eval_points:
        return {
            'ok': False,
            'reason': 'no_eval_rows',
            'metric_name': '',
            'best': float('nan'),
            'at_cutoff': float('nan'),
            'actor_best': float('nan'),
            'idm_best': float('nan'),
            'actor_at_cutoff': float('nan'),
            'idm_at_cutoff': float('nan'),
            'use_idm_fallback': False,
        }

    actor_vals = [a for _, a, _ in eval_points if a is not None]
    idm_vals = [i for _, _, i in eval_points if i is not None]

    use_idm_fallback = False
    if not actor_vals:
        use_idm_fallback = True
        metric_name = IDM_KEY
    elif all(abs(v) <= ZERO_TOL for v in actor_vals):
        use_idm_fallback = True
        metric_name = IDM_KEY
    else:
        metric_name = ACTOR_KEY

    def primary(a: float | None, i: float | None) -> float | None:
        if use_idm_fallback:
            return i
        return a if a is not None else i

    primaries = [primary(a, i) for _, a, i in eval_points if primary(a, i) is not None]
    if not primaries:
        return {
            'ok': False,
            'reason': 'primary_missing',
            'metric_name': metric_name,
            'best': float('nan'),
            'at_cutoff': float('nan'),
            'actor_best': max(actor_vals) if actor_vals else float('nan'),
            'idm_best': max(idm_vals) if idm_vals else float('nan'),
            'actor_at_cutoff': float('nan'),
            'idm_at_cutoff': float('nan'),
            'use_idm_fallback': use_idm_fallback,
        }

    best = max(primaries)
    actor_best = max(actor_vals) if actor_vals else float('nan')
    idm_best = max(idm_vals) if idm_vals else float('nan')

    at_actor = float('nan')
    at_idm = float('nan')
    at_primary = float('nan')
    for ep, a, i in sorted(eval_points, key=lambda t: t[0]):
        if abs(ep - float(upto_epoch)) < 0.25:
            at_actor = a if a is not None else float('nan')
            at_idm = i if i is not None else float('nan')
            p = primary(a, i)
            if p is not None:
                at_primary = float(p)

    return {
        'ok': not math.isnan(at_primary),
        'reason': 'ok' if not math.isnan(at_primary) else f'no_row_at_epoch_{int(upto_epoch)}',
        'metric_name': metric_name,
        'best': float(best),
        'at_cutoff': float(at_primary),
        'actor_best': float(actor_best),
        'idm_best': float(idm_best),
        'actor_at_cutoff': float(at_actor),
        'idm_at_cutoff': float(at_idm),
        'use_idm_fallback': use_idm_fallback,
    }


def analyze_train_csv(train_csv: Path, upto_epoch: int) -> dict[str, object]:
    """CSV-only metrics (may be empty if CsvLogger header omitted eval columns)."""
    fieldnames, rows = _read_train_rows(train_csv)
    pts = _csv_eval_points(fieldnames, rows, float(upto_epoch))
    return _aggregate_eval_points(pts, float(upto_epoch))


def analyze_run_dir_metrics(run_dir: Path, upto_epoch: int) -> dict[str, object]:
    """Prefer train.csv eval columns when populated; else parse run.log / resume logs."""
    rd = Path(run_dir)
    train_csv = rd / 'train.csv'
    fieldnames, rows = _read_train_rows(train_csv)
    csv_pts = _csv_eval_points(fieldnames, rows, float(upto_epoch))
    log_pts = _log_eval_points(rd, float(upto_epoch))

    if csv_pts:
        out = _aggregate_eval_points(csv_pts, float(upto_epoch))
        out['source'] = 'train.csv'
        return out
    if log_pts:
        out = _aggregate_eval_points(log_pts, float(upto_epoch))
        out['source'] = 'run.log'
        return out
    out = _aggregate_eval_points([], float(upto_epoch))
    out['reason'] = 'no_eval_train_csv_or_logs'
    out['source'] = 'none'
    return out


def cmd_parse_train_csv(args: argparse.Namespace) -> int:
    m = analyze_train_csv(Path(args.train_csv), int(args.upto_epoch))
    return _print_metrics_json(m)


def cmd_parse_metrics(args: argparse.Namespace) -> int:
    m = analyze_run_dir_metrics(Path(args.run_dir), int(args.upto_epoch))
    return _print_metrics_json(m)


def _print_metrics_json(m: dict[str, object]) -> int:

    def _json_safe(obj: object) -> object:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        return obj

    print(json.dumps(_json_safe(m), indent=2))
    return 0


def cmd_parse_run_dir(args: argparse.Namespace) -> int:
    p = parse_run_dir_from_log(Path(args.log))
    out = {'run_dir': p}
    print(json.dumps(out, indent=2))
    return 0 if p else 1


CSV_COLUMNS = [
    'env_name',
    'config_path',
    'alpha',
    'gap',
    'kappa',
    'discount',
    'batch_size',
    'seed',
    'run_dir',
    'stage1_completed',
    'metric_name',
    'best_metric_upto_200',
    'metric_at_200',
    'continued_to_400',
    'best_metric_upto_400',
    'final_metric_400',
    'idm_best_upto_200',
    'actor_best_upto_200',
    'idm_final_400',
    'actor_final_400',
    'status',
    'notes',
]


def idm_at_epoch(run_dir: Path, epoch: int) -> float:
    m = analyze_run_dir_metrics(run_dir, int(epoch))
    v = m.get('idm_at_cutoff')
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float('nan')
    return float(v)


def cmd_rank_top_idm(args: argparse.Namespace) -> int:
    """Rank completed runs in a phase CSV by IDM success at a fixed eval epoch."""
    csv_path = Path(args.csv)
    epoch = int(args.epoch)
    top_n = int(args.top)
    rows_out: list[dict[str, object]] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('status') != 'ok' or row.get('stage1_completed') != 'true':
                continue
            rd = row.get('run_dir', '').strip()
            if not rd:
                continue
            run_path = Path(rd)
            if not run_path.is_absolute():
                run_path = REPO_ROOT / rd
            idm = idm_at_epoch(run_path, epoch)
            if math.isnan(idm):
                continue
            rows_out.append({
                'rank_key': idm,
                'idm_at_epoch': idm,
                'config_path': row.get('config_path', ''),
                'run_dir': str(run_path),
                'alpha': row.get('alpha', ''),
                'gap': row.get('gap', ''),
                'kappa': row.get('kappa', ''),
            })
    rows_out.sort(key=lambda r: float(r['rank_key']), reverse=True)
    picked = rows_out[:top_n]
    print(json.dumps({'epoch': epoch, 'top_n': top_n, 'candidates': len(rows_out), 'top': picked}, indent=2))
    return 0


def cmd_append_row(args: argparse.Namespace) -> int:
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, encoding='utf-8') as f:
        row = json.load(f)
    missing = [k for k in CSV_COLUMNS if k not in row]
    if missing:
        print(f'Missing CSV keys: {missing}', file=sys.stderr)
        return 2
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in CSV_COLUMNS})
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('parse-train')
    p1.add_argument('--train-csv', required=True)
    p1.add_argument('--upto-epoch', type=int, required=True)
    p1.set_defaults(func=cmd_parse_train_csv)

    p1b = sub.add_parser('parse-metrics')
    p1b.add_argument('--run-dir', required=True)
    p1b.add_argument('--upto-epoch', type=int, required=True)
    p1b.set_defaults(func=cmd_parse_metrics)

    p2 = sub.add_parser('parse-run-dir')
    p2.add_argument('--log', required=True)
    p2.set_defaults(func=cmd_parse_run_dir)

    p3 = sub.add_parser('append-row')
    p3.add_argument('--csv', required=True)
    p3.add_argument('--json', required=True, help='Path to JSON object with CSV columns')
    p3.set_defaults(func=cmd_append_row)

    p4 = sub.add_parser('rank-top-idm')
    p4.add_argument('--csv', required=True, help='Phase sweep CSV with run_dir column')
    p4.add_argument('--epoch', type=int, default=200)
    p4.add_argument('--top', type=int, default=3)
    p4.set_defaults(func=cmd_rank_top_idm)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
