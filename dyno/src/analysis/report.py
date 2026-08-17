"""Render results.json into a human-readable analysis_report.txt.

results.json is the source of truth; this is a view of it. Findings come first
because the diagnostics are the product -- a reader who stops after the first
screen should already know whether to trust the numbers.
"""

import json
from datetime import datetime

_LEVEL_ORDER = {'error': 0, 'warn': 1, 'info': 2}
_LEVEL_LABEL = {'error': 'ERROR', 'warn': 'WARNING', 'info': 'note'}

_SKIP_METRIC_KEYS = ('pooled', 'branches')


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.6g}'
    if isinstance(v, list):
        return '[' + ', '.join(_fmt(x) for x in v) + ']'
    if v is None:
        return '-'
    return str(v)


def _wrap(text, width, indent):
    words, lines, cur = text.split(), [], ''
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f'{cur} {w}' if cur else w
    if cur:
        lines.append(cur)
    return ('\n' + ' ' * indent).join(lines)


def render(results, log_path, width=78):
    out = []
    add = out.append

    add('=' * width)
    add('POST-PROCESSING ANALYSIS REPORT')
    add('=' * width)
    add(f'Log:       {log_path}')
    add(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    add('')

    if not results:
        add('No processor was applicable to any segment in this log.')
        return '\n'.join(out) + '\n'

    for entry in results:
        add('-' * width)
        add(f"{entry['title']}  [{entry['processor']}]")
        add(f"Segments:  {entry['label']}  ({entry['n_segments']} span(s), "
            f"{entry['n_samples']} samples)")
        add(f"Selected:  {entry['selection_reason']}")
        add('-' * width)
        add('')

        if entry.get('summary'):
            add(_wrap(entry['summary'], width, 0))
            add('')

        findings = sorted(entry.get('findings', []),
                          key=lambda f: _LEVEL_ORDER.get(f['level'], 3))
        if findings:
            add('FINDINGS')
            for f in findings:
                tag = f'  [{_LEVEL_LABEL.get(f["level"], f["level"])}] '
                add(tag + _wrap(f['message'], width - len(tag), len(tag)))
            add('')

        metrics = {k: v for k, v in entry.get('metrics', {}).items()
                   if k not in _SKIP_METRIC_KEYS}
        if metrics:
            add('METRICS')
            klen = max(len(k) for k in metrics)
            for k, v in metrics.items():
                add(f'  {k:<{klen}}  {_fmt(v)}')
            add('')

        pooled = entry.get('metrics', {}).get('pooled')
        if pooled:
            add('POOLED FIT')
            for model in ('linear', 'tanh'):
                if model in pooled:
                    add(f'  {model}:')
                    for k, v in pooled[model].items():
                        add(f'    {k:<28}  {_fmt(v)}')
            add('')

        branches = entry.get('metrics', {}).get('branches')
        if branches:
            add('PER-BRANCH LINEAR FITS')
            add(f'  {"park angle":>12}  {"dir":<5} {"n":>8}  {"kt (Nm/A)":>11}  '
                f'{"offset (Nm)":>12}  {"R2":>9}')
            for b in branches:
                pos = b.get('level_position_rad')
                pos_s = f'{pos * 57.29578:8.3f} deg' if pos is not None else '     n/a'
                lin = b['linear']
                add(f'  {pos_s:>12}  {b["direction"]:<5} {b["n"]:>8}  '
                    f'{lin["kt_Nm_per_A"]:>11.4f}  {lin["offset_Nm"]:>12.4f}  '
                    f'{lin["r_squared"]:>9.5f}')
            add('')

        if entry.get('figures'):
            add('FIGURES')
            for name in entry['figures']:
                add(f'  {name}')
            add('')
        if entry.get('tables'):
            add('TABLES')
            for name in entry['tables']:
                add(f'  {name}')
            add('')

    return '\n'.join(out) + '\n'


def write(results, log_path, out_path):
    with open(out_path, 'w') as f:
        f.write(render(results, log_path))
    return out_path


def dump_json(results, out_path):
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return out_path
