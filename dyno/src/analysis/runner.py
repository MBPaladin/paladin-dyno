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
from .processor import Finding
from .registry import all_processors, get_processor
from .segment import open_log

ANALYSIS_DIRNAME = 'analysis'
PLAN_FILENAME = 'analysis_plan.json'
PLAN_VERSION = 2


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
                # Keep a structured entry rather than vanishing: a UI (or
                # --list) must be able to show that this predicate crashed.
                plan.append({
                    'processor': proc.name,
                    'title': proc.title,
                    'label': label,
                    'segments': [s.raw_id for s in segs],
                    'verdict': 'error',
                    'reason': f'predicate crashed: '
                              f'{type(exc).__name__}: {exc}',
                    'params': proc.defaults(),
                    '_proc': proc,
                    '_segs': segs,
                    '_baseline': proc.defaults(),
                })
                continue
            # defaults() already carries this processor's overrides, and they
            # win: an operator forcing a value outranks the predicate's guess.
            params = proc.defaults()
            params.update(verdict.params)
            params.update(proc.overrides)
            # What this entry's params would be with no operator input at all;
            # save_plan writes only the deltas from this, so a saved plan does
            # not pin today's defaults against future improvements.
            baseline = {k: v[0] for k, v in proc.params.items()}
            baseline.update(verdict.params)
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
                '_baseline': baseline,
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
        if entry['verdict'] == 'error':
            continue
        if entry['verdict'] == 'no' and not entry.get('_forced'):
            continue
        proc, segs = entry['_proc'], entry['_segs']
        print(f"\n{proc.title}  [{entry['label']}]")
        print(f"  selected: {entry['reason']}")
        if entry.get('_forced'):
            print(f'  ! forced: predicate said no -- {entry["reason"]}')

        try:
            res = proc.run(segs, entry['params'])
        except Exception as exc:
            print(f'  ! FAILED: {type(exc).__name__}: {exc}')
            failure_findings = [{'level': 'error', 'code': 'processor_failed',
                                 'message': f'{type(exc).__name__}: {exc}'}]
            if entry.get('_forced'):
                failure_findings.insert(0, {
                    'level': 'warn', 'code': 'predicate_objected',
                    'message': entry['reason']})
            results.append({
                'processor': proc.name, 'title': proc.title,
                'label': entry['label'], 'n_segments': len(segs),
                'n_samples': sum(len(s) for s in segs),
                'selection_reason': entry['reason'],
                'params': entry['params'],
                'findings': failure_findings,
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

        if entry.get('_forced'):
            # The operator overrode the predicate; the report must say so.
            res.findings.insert(0, Finding(
                'warn', 'predicate_objected', entry['reason']))

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


def save_plan(plan, path, log_dir=''):
    """Write the v2 plan file: an object with version, log, and entries.

    `params` in the file holds only deltas from what build_plan would produce
    on its own (defaults + predicate detection), so re-running a saved plan
    picks up improved processor defaults instead of pinning today's.
    `selected` records what actually ran (or, absent an explicit selection,
    which entries qualified).
    """
    entries = []
    for e in plan:
        out = {k: v for k, v in e.items() if not k.startswith('_')}
        baseline = e.get('_baseline', {})
        out['params'] = {k: v for k, v in e['params'].items()
                         if k not in baseline or baseline[k] != v}
        out['selected'] = e.get(
            '_selected', e['verdict'] in ('yes', 'maybe'))
        entries.append(out)
    with open(path, 'w') as f:
        json.dump({'version': PLAN_VERSION, 'log': log_dir,
                   'entries': entries}, f, indent=2, default=str)


def load_plan(path):
    """Read a plan file; returns its entry list.

    Accepts the v1 format (a bare top-level list, full param snapshots, no
    `selected` field) as well as v2. For v1, every entry whose verdict was not
    'no' is treated as selected, and its full param snapshot is applied
    verbatim -- the pinned-defaults cost is accepted for old files.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):                                  # v1
        for e in data:
            e.setdefault('selected', e.get('verdict') != 'no')
        return data
    return data['entries']


def apply_plan_file(plan, file_entries):
    """Filter a freshly rebuilt plan down to a plan file's selection.

    Matches on (processor, label). A selected file entry with no rebuilt
    counterpart is a hard error -- silently running a subset of what was asked
    for is the worst available outcome. A selected entry whose rebuilt verdict
    is 'no' is forced: it runs, flagged so execute() stamps a
    `predicate_objected` warning into its results. Unknown param keys are
    rejected by name.
    """
    selected = [fe for fe in file_entries if fe.get('selected')]
    if not selected:
        raise ValueError('plan file selects no entries')
    for e in plan:            # an explicit selection covers the whole plan --
        e['_selected'] = False  # rewriting it must not resurrect the default
    index = {(e['processor'], e['label']): e for e in plan}
    to_run = []
    for fe in selected:
        key = (fe.get('processor'), fe.get('label'))
        entry = index.get(key)
        if entry is None:
            raise ValueError(
                f'plan entry {key[0]!r} on {key[1]!r} matches nothing in the '
                f'rebuilt plan -- the log or the processor catalog has '
                f'changed; rebuild the plan')
        if entry['verdict'] == 'error':
            raise ValueError(
                f"{key[0]!r} on {key[1]!r}: {entry['reason']}")
        unknown = [k for k in fe.get('params', {})
                   if k not in entry['_proc'].params]
        if unknown:
            raise ValueError(
                f'{key[0]!r} has no parameter(s) {unknown} '
                f'(from plan entry on {key[1]!r})')
        entry['params'].update(fe.get('params', {}))
        entry['_selected'] = True
        if entry['verdict'] == 'no':
            entry['_forced'] = True
        to_run.append(entry)
    return to_run


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
    ap.add_argument('--plan', metavar='FILE',
                    help='Run exactly the entries selected in a plan file '
                         '(see analysis_plan.json); excludes '
                         '--only/--segment/--set')
    ap.add_argument('--interactive', action='store_true',
                    help='Display figures as well as saving them')
    args = ap.parse_args(argv)

    # --plan is a complete statement of what to run; combining it with the
    # ad-hoc selectors would need a precedence nobody will remember.
    if args.plan and (args.only or args.segment or args.set):
        ap.error('--plan cannot be combined with --only/--segment/--set')

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

        if args.plan:
            try:
                to_run = apply_plan_file(plan, load_plan(args.plan))
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f'\nerror: {exc}', file=sys.stderr)
                return 2
        else:
            to_run = [e for e in plan if e['verdict'] in ('yes', 'maybe')]
            for e in to_run:
                e['_selected'] = True
            if not to_run:
                print('\nNo processor was applicable. '
                      'Run with --list to see why.')
                return 1

        out_dir = os.path.join(log.log_dir, ANALYSIS_DIRNAME)
        # Wiped each run: stale figures from a previous selection are worse
        # than no figures.
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        results = execute(to_run, out_dir, interactive=args.interactive)

        report.dump_json(results, os.path.join(out_dir, 'results.json'))
        report.write(results, log.path,
                     os.path.join(out_dir, 'analysis_report.txt'))
        save_plan(plan, os.path.join(log.log_dir, PLAN_FILENAME),
                  log_dir=log.log_dir)

        print(f'\nWrote {out_dir}/')
        print(f'  results.json, analysis_report.txt')
        if args.interactive:
            plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
