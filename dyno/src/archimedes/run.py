"""Run the whole Archimedes pack for one unit, from one unit file.

The point of this module: between gearboxes, NOTHING here should need editing.
A new unit is a new YAML under units/ naming its log root, its rating and its
torque cap; the analyzers find the tests by segment name and read the direction
off the data. Splitting a test across a different number of files, running the
speeds in a different order, or getting the back-drive pass in a week late all
change the unit file and nothing else.

    PYTHONPATH=. .venv/bin/python -m dyno.src.archimedes units/gbx_1p2p0.yaml
    ./dyno/utilities/archimedes.sh gbx_1p2p0            # same thing

Output lands in one folder per unit: every figure as a PNG, every number as a
CSV beside it (the customer asked for the raw data), and a report.txt holding
the findings. The folder is wiped each run -- a stale figure from a previous
configuration is worse than no figure, and this is the same rule the main
analysis runner follows.
"""

import argparse
import json
import os
import shutil
import sys

import yaml

from . import dataset, naming, physics, report as texreport
from .analyzers import efficiency, slip, stiffness, velocity_ramp

ANALYZERS = {
    'velocity_ramp': velocity_ramp,
    'efficiency': efficiency,
    'slip': slip,
    'stiffness': stiffness,
}

# Order the customer's test list uses, so the report reads down their table.
ORDER = ['velocity_ramp', 'efficiency', 'slip', 'stiffness']

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS_DIR = os.path.join(HERE, 'units')

# Defaults for anything a unit file does not say. Every one of these is a
# property of the CUSTOMER'S request or of the bench, not of a gearbox, which
# is why they are not repeated in each unit file.
DEFAULTS = {
    'exclude': ['initial_setup', 'troubleshooting_initial', 'calibration'],
    'ratio': 43.88,
    'torque_step_nm': 5.0,
    'velocity_steps_rpm': (list(range(30, 301, 30))
                           + list(range(600, 3601, 300))),
    'efficiency_speeds_rpm': [20, 300, 1500, 3000],
    'correct_zero': True,
    'position_half_window_rad': None,
    'duplicates': 'longest',
}


def load_unit(path):
    """Read a unit file, resolving it against units/ when it is a bare name."""
    for candidate in (path, os.path.join(UNITS_DIR, path),
                      os.path.join(UNITS_DIR, f'{path}.yaml')):
        if os.path.isfile(candidate):
            with open(candidate) as fh:
                cfg = yaml.safe_load(fh) or {}
            cfg['_path'] = os.path.abspath(candidate)
            return _fill(cfg)
    known = sorted(f[:-5] for f in os.listdir(UNITS_DIR)
                   if f.endswith('.yaml'))
    raise FileNotFoundError(
        f'no unit file {path!r}; known units: {", ".join(known) or "(none)"}')


def _fill(cfg):
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, list(v) if isinstance(v, list) else v)
    cfg.setdefault('unit_label', cfg.get('unit', 'Archimedes drive'))
    root = cfg.get('log_root')
    if not root:
        raise ValueError(f'{cfg["_path"]}: no log_root')
    if not os.path.isabs(root):
        # Relative to the repo root, which is where every other path in this
        # tree is relative to.
        cfg['log_root'] = os.path.abspath(
            os.path.join(HERE, '..', '..', '..', root))
    return cfg


def run(cfg, out_dir=None, only=None, make_report=False):
    out_dir = out_dir or cfg.get('out_dir') or os.path.join(
        cfg['log_root'], 'analysis_pack')
    print(f'Unit:     {cfg["unit_label"]}')
    print(f'Logs:     {cfg["log_root"]}')
    print(f'Excluding: {", ".join(cfg["exclude"]) or "(nothing)"}')

    with dataset.load(cfg['log_root'], exclude=cfg['exclude'],
                      duplicates=cfg['duplicates']) as ds:
        if not ds.spans:
            print('\nNo recognised test segments found. Check log_root and '
                  'exclude.', file=sys.stderr)
            return 1
        _print_inventory(ds)

        ratio, measured = ds.ratio(cfg['ratio'])
        cfg['ratio'] = ratio
        print(f'\nRatio:    {ratio:.4f} '
              f'({"measured on the no-load sweep" if measured else "from the unit file"})')

        offsets, detail = physics.cell_offsets(ds.spans, ratio)
        cfg['cell_offsets'] = (offsets, detail)
        if offsets:
            print('Cell zero: ' + ', '.join(
                f'{k} {v:+.3f} Nm' for k, v in offsets.items()))
        else:
            print('Cell zero: no no-load traverses -- not corrected')

        names = [n for n in ORDER if not only or n in only]
        # A full run wipes the folder: a stale figure from a previous
        # configuration is worse than no figure, and this is the rule the main
        # analysis runner follows. A --only run must NOT wipe it -- rerunning
        # one analyzer would otherwise silently delete the other three, which
        # is exactly what someone iterating on one plot does not want. It
        # clears only what it is about to rewrite.
        if only:
            os.makedirs(out_dir, exist_ok=True)
            _clear(out_dir, names)
        else:
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir)
        results = []
        for name in names:
            print(f'\n--- {name} ---')
            try:
                res = ANALYZERS[name].analyze(ds, cfg)
            except Exception as exc:
                import traceback
                print(f'  ! FAILED: {type(exc).__name__}: {exc}')
                traceback.print_exc()
                continue
            _write(res, out_dir)
            results.append(res)

        # A partial run's report would describe only the analyzers it ran, so
        # it is written only when the pack is complete.
        if only:
            print('\n(partial run: report.txt and results.json left as they '
                  'were -- rerun without --only to refresh them)')
        else:
            _write_report(results, ds, cfg, out_dir)

    if make_report:
        _render_report(out_dir, cfg)
    print(f'\nWrote {out_dir}/')
    return 0


def _render_report(pack_dir, cfg):
    """Render the customer-facing LaTeX report from the pack just written.

    Kept separate from the analysis so a wording change costs a render and not
    a re-analysis: it reads results.json and the CSVs, never the HDF5.
    """
    print('\n--- report ---')
    try:
        driver, sections, pics = texreport.build(pack_dir, cfg)
    except Exception as exc:
        import traceback
        print(f'  ! report render failed: {type(exc).__name__}: {exc}')
        traceback.print_exc()
        print('  The analysis above is written and intact.')
        return
    print(f'  sections: {", ".join(sections)}')
    print(f'  figures:  {len(pics)} copied into report/pics/')
    print(f'  wrote:    {driver}')
    if cfg.get('build_pdf'):
        _build_pdf(driver)
    else:
        print(f'  build it: cd {os.path.dirname(driver)} && '
              f'pdflatex {os.path.basename(driver)}')


def _build_pdf(driver):
    """Run pdflatex twice over the rendered report.

    Twice because the document carries a table of contents and cross
    references: the first pass writes the .aux and .toc and the second
    resolves them. A missing TeX installation is reported as the one-line
    fact it is -- the .tex is written and intact either way.
    """
    import shutil as _sh
    import subprocess
    exe = _sh.which('pdflatex')
    if not exe:
        print('  ! pdflatex is not on PATH; the .tex above is written and '
              'intact.')
        print('    Install TeX with:  sudo apt install -y '
              'texlive-latex-recommended texlive-latex-extra texlive-science')
        return
    work = os.path.dirname(driver)
    name = os.path.basename(driver)
    for i in (1, 2):
        proc = subprocess.run(
            [exe, '-interaction=nonstopmode', '-halt-on-error',
             '-file-line-error', name],
            cwd=work, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f'  ! pdflatex failed on pass {i}:')
            # The error lines are what matters; the rest is banner noise.
            for line in proc.stdout.splitlines():
                if ':' in line and ('Error' in line or '! ' in line):
                    print(f'      {line}')
            print(f'    full log: {os.path.join(work, name[:-4] + ".log")}')
            return
    pdf = os.path.join(work, name[:-4] + '.pdf')
    size = os.path.getsize(pdf) / 1024 if os.path.isfile(pdf) else 0
    print(f'  built:    {pdf}  ({size:.0f} kB)')


def _clear(out_dir, names):
    """Remove the outputs of just these analyzers, leaving the rest alone."""
    prefixes = tuple(f'{n}__' for n in names)
    for f in os.listdir(out_dir):
        if f.startswith(prefixes):
            os.remove(os.path.join(out_dir, f))


def _print_inventory(ds):
    print(f'\nFound {len(ds.spans)} measured span(s) in {len(ds.logs)} log(s):')
    for (kind, direction), n in sorted(ds.kinds().items()):
        print(f'  {naming.KIND_TITLES.get(kind, kind):<24} '
              f'{direction:<10} {n:>4} span(s)')
    if ds.duplicates:
        pooled = any(d.get('pooled') for d in ds.duplicates)
        print(f'  ({len(ds.duplicates)} span(s) appear in more than one file, '
              + ('pooled' if pooled else 'collapsed to the longest copy')
              + ' -- see report.txt)')
    for note in ds.notes:
        print(f'  ! {note}')


def _write(res, out_dir):
    import matplotlib.pyplot as plt
    for slug, fig in res.figures:
        path = os.path.join(out_dir, f'{res.name}__{slug}.png')
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f'  saved: {os.path.basename(path)}')
    for slug, text in res.tables:
        if not text:
            continue
        path = os.path.join(out_dir, f'{slug}.csv')
        with open(path, 'w') as fh:
            fh.write(text)
        print(f'  saved: {os.path.basename(path)}')
    for f in res.findings:
        print(f'  [{f.level}] {f.message}')


def _write_report(results, ds, cfg, out_dir):
    lines = [
        f'Archimedes analysis pack -- {cfg["unit_label"]}',
        '=' * 72, '',
        f'Log root:   {cfg["log_root"]}',
        f'Unit file:  {cfg.get("_path", "(inline)")}',
        f'Ratio used: {cfg["ratio"]:.4f} (input rad per output rad)',
    ]
    offsets = (cfg.get('cell_offsets') or ({}, {}))[0]
    if offsets:
        lines.append('Cell zero:  ' + ', '.join(
            f'{k} {v:+.4f} Nm' for k, v in offsets.items())
            + ('  [subtracted]' if cfg['correct_zero'] else '  [NOT subtracted]'))
    lines += ['', 'Inventory', '-' * 72]
    for (kind, direction), n in sorted(ds.kinds().items()):
        lines.append(f'  {naming.KIND_TITLES.get(kind, kind):<24} '
                     f'{direction:<10} {n:>4} span(s)')
    if ds.duplicates:
        pooled = any(d.get('pooled') for d in ds.duplicates)
        lines += ['', 'Spans found in more than one file', '-' * 72]
        lines += (['  duplicates: all -- every copy is kept and pooled.']
                  if pooled else
                  ['  duplicates: longest -- a resumed run repeats the span it '
                   'died on, so the longer',
                   '  copy is kept; equal-length copies mean the sweep was '
                   're-run and the later file wins.',
                   '  Pass --duplicates all to pool them instead.'])
        for d in ds.duplicates:
            lines.append(f'  {d["segment"]} [{d["direction"]}] kept '
                         f'{d["kept"]} ({d["kept_n"]} samples, {d["why"]})')
            for drop in d['dropped']:
                lines.append(f'      dropped {drop["source"]} '
                             f'({drop["n"]} samples)')
            for pooled in d.get('pooled', []):
                lines.append(f'      pooled  {pooled["source"]} '
                             f'({pooled["n"]} samples)')
    if ds.skipped:
        lines += ['', 'Skipped', '-' * 72]
        for what, why in ds.skipped:
            lines.append(f'  {what}: {why}')
    if ds.notes:
        lines += ['', 'Notes', '-' * 72]
        lines += [f'  {n}' for n in ds.notes]

    for res in results:
        lines += ['', res.title, '=' * 72]
        if res.summary:
            lines += ['  ' + res.summary, '']
        for f in res.findings:
            lines.append(f'  [{f.level:>5}] {f.message}')
        if res.figures:
            lines.append('  figures: ' + ', '.join(
                f'{res.name}__{s}.png' for s, _ in res.figures))
        if res.tables:
            lines.append('  tables:  ' + ', '.join(
                f'{s}.csv' for s, _ in res.tables))

    text = '\n'.join(lines) + '\n'
    with open(os.path.join(out_dir, 'report.txt'), 'w') as fh:
        fh.write(text)

    payload = {
        'unit': cfg['unit_label'], 'log_root': cfg['log_root'],
        'ratio': cfg['ratio'],
        'cell_offsets': offsets,
        'inventory': {f'{k}/{d}': n for (k, d), n in ds.kinds().items()},
        'duplicates': ds.duplicates,
        'results': [{
            'name': r.name, 'title': r.title, 'summary': r.summary,
            'metrics': r.metrics,
            'findings': [vars(f) for f in r.findings],
        } for r in results],
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='archimedes',
        description='Run the Archimedes customer analysis pack for one unit.')
    ap.add_argument('unit', help='unit file, or its bare name under units/')
    ap.add_argument('--out', help='output folder '
                                  '(default: <log_root>/analysis_pack)')
    ap.add_argument('--only', action='append', metavar='NAME',
                    choices=sorted(ANALYZERS),
                    help='run one analyzer (repeatable): '
                         + ', '.join(sorted(ANALYZERS)))
    ap.add_argument('--list', action='store_true',
                    help='show what the logs hold and run nothing')
    ap.add_argument('--report', action='store_true',
                    help='also render the customer-facing LaTeX report into '
                         '<out>/report/')
    ap.add_argument('--report-only', action='store_true',
                    help='re-render that report from the analysis already on '
                         'disk, without re-analysing')
    ap.add_argument('--pdf', action='store_true',
                    help='also run pdflatex on the rendered report (implies '
                         '--report; needs a TeX installation on PATH)')
    ap.add_argument('--no-zero-correction', action='store_true',
                    help='do not subtract the measured torque-cell zero')
    ap.add_argument('--duplicates', choices=('longest', 'all'),
                    help='a span found in more than one file: keep the longest '
                         'copy (a resumed run) or pool them all (a deliberately '
                         'repeated sweep). Default from the unit file, '
                         'else longest')
    args = ap.parse_args(argv)

    try:
        cfg = load_unit(args.unit)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    if args.no_zero_correction:
        cfg['correct_zero'] = False
    if args.duplicates:
        cfg['duplicates'] = args.duplicates
    if args.pdf:
        args.report = True
    cfg['build_pdf'] = args.pdf
    if args.report_only:
        pack = args.out or cfg.get('out_dir') or os.path.join(
            cfg['log_root'], 'analysis_pack')
        if not os.path.isfile(os.path.join(pack, 'results.json')):
            print(f'error: no analysis pack at {pack}; run without '
                  f'--report-only first', file=sys.stderr)
            return 2
        with open(os.path.join(pack, 'results.json')) as fh:
            saved = json.load(fh)
        # The ratio and the cell zeros are properties of the analysis that
        # produced the pack, not of the unit file, so they come from the pack.
        cfg['ratio'] = saved.get('ratio', cfg['ratio'])
        _render_report(pack, cfg)
        return 0

    if args.list:
        with dataset.load(cfg['log_root'], exclude=cfg['exclude'],
                          duplicates=cfg['duplicates']) as ds:
            _print_inventory(ds)
            for span in sorted(ds.spans, key=lambda s: (s.kind, s.direction,
                                                        s.point.raw)):
                print(f'  {span.kind:<12} {span.direction:<10} '
                      f'{span.point.raw:<24} {span.n:>8} '
                      f'{span.source}')
        return 0
    return run(cfg, out_dir=args.out,
               only=set(args.only) if args.only else None,
               make_report=args.report)


if __name__ == '__main__':
    sys.exit(main())
