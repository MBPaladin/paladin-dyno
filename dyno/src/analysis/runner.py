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

    `overrides` is the raw {key: str} from --set. It is coerced and installed on
    each processor *before* its predicate runs, because predicates gate on
    defaults(): an override applied only to the plan entry afterwards would
    reach run() but not the decision about whether run() ever happens.
    Raises ValueError if a key is recognized by no processor, or if a value
    does not cast.
    """
    procs = [get_processor(only)] if only else all_processors()
    for k, raw in (overrides or {}).items():
        takers = [p for p in procs if k in p.params]
        if not takers:
            raise ValueError(_unknown_param_message(k, procs))
        for proc in takers:
            try:
                value = _coerce(proc, k, raw)
            except ValueError as exc:
                raise ValueError(f'{proc.name} parameter {k!r} rejected '
                                 f'{raw!r}: {exc}') from exc
            proc.overrides = {**proc.overrides, k: value}

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
            # defaults() already carries this processor's overrides, and they
            # win: an operator forcing a value outranks the predicate's guess.
            params = proc.defaults()
            params.update(verdict.params)
            params.update(proc.overrides)
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


def _output_name(out_dir, label, proc_name, slug, ext):
    """Build an output filename, honouring a subfolder in the slug.

    Flat `{label}__{proc}__{slug}.{ext}` inside analysis/ is the default. A slug
    may carry a leading path -- 'runs/sweep_03' -- which becomes a subfolder, so
    a processor emitting one figure per repetition does not bury its headline
    figures under fifty siblings. The leaf keeps the full prefix so a file stays
    identifiable once it is dragged out of the folder.
    """
    subdir, _, leaf = slug.rpartition('/')
    fname = os.path.join(subdir, f'{label}__{proc_name}__{leaf}.{ext}')
    if subdir:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)
    return fname


def _announce(names, limit=6):
    """Print what was written, collapsing long per-repetition runs."""
    for name in names[:limit]:
        print(f'  saved: {name}')
    if len(names) > limit:
        print(f'  saved: ... and {len(names) - limit} more')


def _unknown_param_message(key, procs):
    """A --set typo should print the whole menu, not just the rejection."""
    lines = [f'no processor in this plan has a parameter {key!r}']
    for proc in procs:
        lines.append(f'\n{proc.name} accepts:')
        for pk, (default, _t, help_) in proc.params.items():
            lines.append(f'  {pk:<22} (default {default})  {help_}')
    return '\n'.join(lines)


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
            fname = _output_name(out_dir, entry['label'], proc.name, slug, 'png')
            fig.savefig(os.path.join(out_dir, fname), dpi=200)
            if not interactive:
                matplotlib.pyplot.close(fig)
            fig_names.append(fname)
        _announce(fig_names)
        for slug, text in res.tables:
            fname = _output_name(out_dir, entry['label'], proc.name, slug, 'csv')
            with open(os.path.join(out_dir, fname), 'w') as fh:
                fh.write(text)
            table_names.append(fname)
        _announce(table_names)

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
    # A processor builds every figure before returning -- that is the Result
    # contract -- so a per-repetition plotter legitimately holds dozens open
    # until execute() saves and closes them. The warning is aimed at leaks.
    plt.rcParams['figure.max_open_warning'] = 0

    with open_log(args.log_dir) as log:
        print(f'Log:      {log.path}')
        print(f'Segments: {len(log.segments)}')
        for s in log.segments:
            print(f'  {s.raw_id:<32} n={len(s):>8}  {s.duration:7.1f}s')

        # A --set key only has to be known to the processors it is aimed at --
        # a plan routinely mixes processors with disjoint parameters, so a key
        # that is meaningless to one of them is normal and must not abort the
        # others. Only a key that no processor recognizes is a typo worth
        # stopping for.
        raw_overrides = dict(kv.split('=', 1) for kv in args.set)
        try:
            plan = build_plan(log, only=args.only, segment=args.segment,
                              overrides=raw_overrides)
        except ValueError as exc:
            print(f'\nerror: {exc}', file=sys.stderr)
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
