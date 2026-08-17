"""Plan construction, execution, and every file write.

Processors return numbers and Figures; this module decides what they are called
and where they land. Output naming is therefore uniform across the catalog and
matplotlib's backend is chosen in exactly one place.
"""

import argparse
import json
import os
import shutil
import sys

import matplotlib

from . import report
from .registry import all_processors, get_processor
from .segment import open_log

ANALYSIS_DIRNAME = 'analysis'
PLAN_FILENAME = 'analysis_plan.json'


def build_plan(log, only=None, segment=None, overrides=None):
    """Ask every processor which segment groups it can handle.

    Returns a list of plan entries. Applicability is recorded even when the
    verdict is 'no', so --list can explain why something was skipped.
    """
    procs = [get_processor(only)] if only else all_processors()
    plan = []
    for proc in procs:
        for label, segs in log.grouped(proc.granularity):
            if segment and not any(s.raw_id == segment or s.behavior == segment
                                   for s in segs):
                continue
            try:
                verdict = proc.applies_to(segs)
            except Exception as exc:               # a broken predicate must not
                verdict = None                     # take down the whole run
                print(f'  ! {proc.name} predicate failed on {label}: '
                      f'{type(exc).__name__}: {exc}')
            if verdict is None:
                continue
            params = proc.defaults()
            params.update(verdict.params)
            params.update(overrides or {})
            plan.append({
                'processor': proc.name,
                'title': proc.title,
                'label': label,
                'segments': [s.raw_id for s in segs],
                'verdict': verdict.verdict,
                'reason': verdict.reason,
                'params': params,
                '_proc': proc,
                '_segs': segs,
            })
    return plan


def _coerce(proc, key, raw):
    spec = proc.params.get(key)
    if not spec:
        raise KeyError(f'{proc.name} has no parameter {key!r}')
    caster = spec[1]
    if caster is bool:
        return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')
    return caster(raw)


def execute(plan, out_dir, interactive=False):
    results = []
    for entry in plan:
        if entry['verdict'] == 'no':
            continue
        proc, segs = entry['_proc'], entry['_segs']
        print(f"\n{proc.title}  [{entry['label']}]")
        print(f"  selected: {entry['reason']}")

        try:
            res = proc.run(segs, entry['params'])
        except Exception as exc:
            print(f'  ! FAILED: {type(exc).__name__}: {exc}')
            results.append({
                'processor': proc.name, 'title': proc.title,
                'label': entry['label'], 'n_segments': len(segs),
                'n_samples': sum(len(s) for s in segs),
                'selection_reason': entry['reason'],
                'params': entry['params'],
                'findings': [{'level': 'error', 'code': 'processor_failed',
                              'message': f'{type(exc).__name__}: {exc}'}],
                'metrics': {}, 'figures': [], 'tables': [],
            })
            continue

        fig_names, table_names = [], []
        for slug, fig in res.figures:
            fname = f"{entry['label']}__{proc.name}__{slug}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=200)
            if not interactive:
                matplotlib.pyplot.close(fig)
            fig_names.append(fname)
            print(f'  saved: {fname}')
        for slug, text in res.tables:
            fname = f"{entry['label']}__{proc.name}__{slug}.csv"
            with open(os.path.join(out_dir, fname), 'w') as fh:
                fh.write(text)
            table_names.append(fname)
            print(f'  saved: {fname}')

        for f in res.findings:
            print(f'  [{f.level}] {f.message}')

        results.append({
            'processor': proc.name, 'title': proc.title,
            'label': entry['label'], 'n_segments': len(segs),
            'n_samples': sum(len(s) for s in segs),
            'selection_reason': entry['reason'],
            'params': entry['params'],
            'summary': res.summary,
            'findings': [vars(f) for f in res.findings],
            'metrics': res.metrics,
            'figures': fig_names, 'tables': table_names,
        })
    return results


def save_plan(plan, path):
    with open(path, 'w') as f:
        json.dump([{k: v for k, v in e.items() if not k.startswith('_')}
                   for e in plan], f, indent=2, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='analyze',
        description='Run post-processors against a dyno log directory.')
    ap.add_argument('log_dir', help='Log folder containing the .hdf5')
    ap.add_argument('--list', action='store_true',
                    help='Show segments and applicable processors; run nothing')
    ap.add_argument('--only', metavar='NAME', help='Restrict to one processor')
    ap.add_argument('--segment', metavar='ID', help='Restrict to one segment')
    ap.add_argument('--set', action='append', default=[], metavar='K=V',
                    help='Override a processor parameter (repeatable)')
    ap.add_argument('--interactive', action='store_true',
                    help='Display figures as well as saving them')
    args = ap.parse_args(argv)

    matplotlib.use('TkAgg' if args.interactive else 'Agg')
    global plt
    import matplotlib.pyplot as plt
    matplotlib.pyplot = plt

    with open_log(args.log_dir) as log:
        print(f'Log:      {log.path}')
        print(f'Segments: {len(log.segments)}')
        for s in log.segments:
            print(f'  {s.raw_id:<32} n={len(s):>8}  {s.duration:7.1f}s')

        raw_overrides = dict(kv.split('=', 1) for kv in args.set)
        plan = build_plan(log, only=args.only, segment=args.segment)

        # Coerce late: casting needs the proc. A --set key only has to be known
        # to the processors it is aimed at -- a plan routinely mixes processors
        # with disjoint parameters, so a key that is meaningless to one of them
        # is normal and must not abort the others. Only a key that no processor
        # in the plan recognizes is a typo worth stopping for.
        for k, v in raw_overrides.items():
            takers = [e for e in plan if k in e['_proc'].params]
            if not takers:
                print(f'\nerror: no processor in this plan has a parameter '
                      f'{k!r}', file=sys.stderr)
                for entry in plan:
                    proc = entry['_proc']
                    print(f'\n{proc.name} accepts:', file=sys.stderr)
                    for pk, (default, _t, help_) in proc.params.items():
                        print(f'  {pk:<22} (default {default})  {help_}',
                              file=sys.stderr)
                return 2
            for entry in takers:
                try:
                    entry['params'][k] = _coerce(entry['_proc'], k, v)
                except ValueError as exc:
                    print(f'\nerror: {entry["_proc"].name} parameter {k!r} '
                          f'rejected {v!r}: {exc}', file=sys.stderr)
                    return 2

        if args.list:
            print('\nApplicability:')
            if not plan:
                print('  (no processors registered)')
            for e in plan:
                print(f"  [{e['verdict']:>5}] {e['processor']:<20} {e['label']}")
                print(f"          {e['reason']}")
            return 0

        runnable = [e for e in plan if e['verdict'] != 'no']
        if not runnable:
            print('\nNo processor was applicable. Run with --list to see why.')
            return 1

        out_dir = os.path.join(log.log_dir, ANALYSIS_DIRNAME)
        # Wiped each run: stale figures from a previous selection are worse
        # than no figures.
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        results = execute(plan, out_dir, interactive=args.interactive)

        report.dump_json(results, os.path.join(out_dir, 'results.json'))
        report.write(results, log.path,
                     os.path.join(out_dir, 'analysis_report.txt'))
        save_plan(plan, os.path.join(log.log_dir, PLAN_FILENAME))

        print(f'\nWrote {out_dir}/')
        print(f'  results.json, analysis_report.txt')
        if args.interactive:
            plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
