r"""Render analysis results into LaTeX report sections.

This reads results.json and nothing else. No HDF5 is opened, no processor runs,
no figure is replotted -- which is the point: report prose gets iterated on far
more often than the maths behind it, and a wording change should cost a second,
not a re-analysis.

    python -m dyno.src.analysis.tex <report-root>

The report root is the directory holding `report.yaml`. That file is the anchor
for everything, and it exists because log trees are ragged: one test lives at
`small_motor/blocked_rotor`, another at `small_motor/inertia/full_system`, and
some sections need two logs at once (inertia wants the full chain and the
decoupled structure; torque ripple and running torque each want drive-on and
drive-off). Discovering that by walking the tree would mean guessing which
folder is which. Declaring it costs twenty lines, once per device under test.

Everything the manifest names is resolved relative to the root, so nesting depth
never enters. Output lands in two places:

    <root>/sections/*.tex      generated, safe to delete, never hand-edited
    <root>/pics/<alias>.png    figures copied under stable semantic names

Figures get semantic aliases rather than their analysis filenames on purpose.
`SEG1-RUN0__current_torque__pooled_fit.png` carries a segment label that changes
whenever segmentation changes, and a .tex full of those breaks silently the next
time a behavior is renamed. `pics/current_torque_fit.png` does not.

Nothing in `pics/` is ever deleted except files this tool previously wrote --
recorded in `pics/.generated.json`. The directory also holds hand-placed assets
(paladinLogo.png in the mac_motors report), and wiping it to keep the generated
set tidy would take those with it.
"""

import argparse
import json
import os
import shutil
import sys

from . import tex_sections
from .runner import ANALYSIS_DIRNAME

MANIFEST_NAME = 'report.yaml'
SECTIONS_DIRNAME = 'sections'
PICS_DIRNAME = 'pics'
OWNED_INDEX = '.generated.json'

# How far up to look for a manifest before giving up. A log folder that is not
# part of any report is a normal thing -- a one-off run -- so this returns None
# rather than raising, and the caller decides whether that is worth mentioning.
_MAX_WALK_UP = 8


# ---------------------------------------------------------------- manifest

def find_root(start):
    """Nearest ancestor of `start` holding a report.yaml, or None.

    Same discovery shape as git or package.json, for the same reason: the tool
    is most useful when it can be pointed at any folder in the tree.
    """
    here = os.path.abspath(start)
    if os.path.isfile(here):
        here = os.path.dirname(here)
    for _ in range(_MAX_WALK_UP):
        if os.path.isfile(os.path.join(here, MANIFEST_NAME)):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def load_manifest(root):
    """Read and normalize report.yaml.

    `sections` accepts either a bare path (one log, role 'main') or a mapping of
    role -> path. Normalizing here means every section renderer sees the same
    shape and none of them has to handle the string case.
    """
    import yaml

    path = os.path.join(root, MANIFEST_NAME)
    with open(path, encoding='utf-8') as f:
        man = yaml.safe_load(f) or {}

    sections = {}
    for key, value in (man.get('sections') or {}).items():
        if isinstance(value, str):
            sections[key] = {'main': value}
        elif isinstance(value, dict):
            sections[key] = {str(r): str(p) for r, p in value.items()}
        else:
            raise ValueError(
                f'{MANIFEST_NAME}: section {key!r} must be a log path or a '
                f'mapping of role -> log path, got {type(value).__name__}')
    man['sections'] = sections
    man.setdefault('dut', {})
    man.setdefault('summary', {})
    man.setdefault('current_units', {})
    return man


# ------------------------------------------------------------------ results

def load_entries(log_dir):
    """The results.json entries for one log folder, or None when absent.

    Absent is the common case while a report is being assembled -- a test that
    has not been analysed yet -- so it is a return value, not an exception.
    """
    path = os.path.join(log_dir, ANALYSIS_DIRNAME, 'results.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def entry_for(entries, processor):
    """The one entry produced by `processor`, or None.

    A log can carry several entries for one processor when the plan grouped it
    per behavior. The first is taken: every section this module renders uses
    processors whose granularity yields one entry per log, and picking silently
    is better than the alternative of rendering several copies of a section.
    """
    for e in entries or ():
        if e.get('processor') == processor:
            return e
    return None


# ------------------------------------------------------------------ figures

class FigureCollector:
    """Records (source figure -> alias) copy jobs while sections render.

    Sections declare figures as they write them; the copies happen afterwards,
    in one pass, so a section that fails to render leaves no half-copied pics
    directory behind.
    """

    def __init__(self, root):
        self.root = root
        self.jobs = {}          # alias.png -> absolute source path
        self.missing = []       # (alias, log_dir, slug) for reporting

    def declare(self, log_dir, slug, alias):
        """Resolve one figure slug in one log's analysis folder.

        Returns the path a .tex should reference, or None when the processor
        did not emit that figure -- the caller then emits a placeholder rather
        than an \\includegraphics that fails to compile.
        """
        entries = load_entries(log_dir) or []
        wanted = f'__{slug}.png'
        for e in entries:
            for name in e.get('figures', []):
                if name.endswith(wanted):
                    src = os.path.join(log_dir, ANALYSIS_DIRNAME, name)
                    if os.path.isfile(src):
                        self.jobs[f'{alias}.png'] = src
                        return f'{PICS_DIRNAME}/{alias}.png'
        self.missing.append((alias, log_dir, slug))
        return None

    def commit(self):
        """Copy every declared figure, then retire aliases we no longer make.

        Only files listed in the previous run's index are removed. Anything
        else in pics/ was put there by a person and stays.
        """
        pics = os.path.join(self.root, PICS_DIRNAME)
        os.makedirs(pics, exist_ok=True)
        index_path = os.path.join(pics, OWNED_INDEX)
        previously = set()
        if os.path.isfile(index_path):
            try:
                with open(index_path, encoding='utf-8') as f:
                    previously = set(json.load(f).get('files', []))
            except (ValueError, OSError):
                previously = set()

        for alias, src in sorted(self.jobs.items()):
            shutil.copyfile(src, os.path.join(pics, alias))

        retired = sorted(previously - set(self.jobs))
        for alias in retired:
            target = os.path.join(pics, alias)
            if os.path.isfile(target):
                os.remove(target)

        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump({'files': sorted(self.jobs)}, f, indent=2)
        return retired


# ------------------------------------------------------------------- render

def render(root, verbose=True):
    """Render every section the manifest names. Returns a summary dict."""
    man = load_manifest(root)
    figures = FigureCollector(root)
    out_dir = os.path.join(root, SECTIONS_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    written, skipped, notes = [], [], []

    # Load each section's logs once, so a section renderer never touches the
    # filesystem and stays a pure function of its inputs.
    loaded = {}
    for key, roles in man['sections'].items():
        section = tex_sections.SECTIONS.get(key)
        if section is None:
            skipped.append((key, f'no renderer for section {key!r}; known: '
                                 + ', '.join(sorted(tex_sections.SECTIONS))))
            continue
        by_role = {}
        for role, rel in roles.items():
            log_dir = os.path.normpath(os.path.join(root, rel))
            if not os.path.isdir(log_dir):
                notes.append(f'{key}/{role}: {rel} is not a folder')
                continue
            entries = load_entries(log_dir)
            if entries is None:
                notes.append(f'{key}/{role}: {rel} has no '
                             f'{ANALYSIS_DIRNAME}/results.json -- run the '
                             f'analysis on it first')
                continue
            entry = entry_for(entries, section.processor)
            if entry is None:
                notes.append(f'{key}/{role}: {rel} carries no '
                             f'{section.processor} result')
                continue
            by_role[role] = {'entry': entry, 'log_dir': log_dir, 'rel': rel}
        loaded[key] = by_role

        ctx = tex_sections.Context(key, by_role, man, figures)
        body = section.render(ctx)
        path = os.path.join(out_dir, f'{key}.tex')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(_header(key) + body)
        written.append(path)
        notes.extend(ctx.notes)

    # The summary table spans every section, so it is rendered last, off the
    # same loaded entries rather than by reopening anything.
    ctx = tex_sections.Context('summary', {}, man, figures)
    summary_body = tex_sections.render_summary(ctx, loaded)
    summary_path = os.path.join(out_dir, 'summary_table.tex')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(_header('summary_table') + summary_body)
    written.append(summary_path)
    notes.extend(ctx.notes)

    driver_path = os.path.join(root, 'report_generated.tex')
    with open(driver_path, 'w', encoding='utf-8') as f:
        f.write(tex_sections.render_driver(man, list(man['sections']), root))
    written.append(driver_path)

    retired = figures.commit()

    if verbose:
        print(f'Report root: {root}')
        for p in written:
            print(f'  wrote: {os.path.relpath(p, root)}')
        for alias in sorted(figures.jobs):
            print(f'  figure: {PICS_DIRNAME}/{alias}')
        for alias in retired:
            print(f'  retired: {PICS_DIRNAME}/{alias}')
        for alias, log_dir, slug in figures.missing:
            print(f'  ! no {slug} figure in {os.path.relpath(log_dir, root)} '
                  f'-- {alias} left as a placeholder')
        for note in notes:
            print(f'  ! {note}')
        for key, why in skipped:
            print(f'  ! skipped {key}: {why}')

    return {'written': written, 'figures': sorted(figures.jobs),
            'missing': figures.missing, 'notes': notes, 'skipped': skipped}


def _header(key):
    return (f'% {key}.tex -- generated by dyno.src.analysis.tex\n'
            f'% Do not edit: this file is overwritten on every render.\n'
            f'% Commentary belongs in the driver, beside the \\input line.\n\n')


# ---------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='report',
        description='Render analysis results into LaTeX report sections.')
    ap.add_argument('path', nargs='?', default='.',
                    help='Report root, or any folder beneath one '
                         f'(searched upward for {MANIFEST_NAME})')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    root = find_root(args.path)
    if root is None:
        print(f'error: no {MANIFEST_NAME} in {os.path.abspath(args.path)} or '
              f'its parents. Create one at the report root -- see '
              f'dyno/src/analysis/tex.py for the schema.', file=sys.stderr)
        return 2
    render(root, verbose=not args.quiet)
    return 0


if __name__ == '__main__':
    sys.exit(main())
