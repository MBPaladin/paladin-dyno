"""Plan handling for the analysis UI. Deliberately Qt-free.

The window is a thin view over this module: grouping plan entries into tree
rows, resolving a chosen folder to its one log file, and restoring a previous
session's selection from an existing analysis_plan.json. Keeping that logic
here is what makes it testable without a display.
"""

import os

from dyno.src.analysis import runner


def resolve_log_dir(folder):
    """A chosen folder must hold exactly one .hdf5 -- return its path.

    The CLI's find_log_file silently resolves a parent folder to the most
    recent run beneath it; the UI does not inherit that silence. Zero or
    multiple logs is an error the operator fixes by choosing again.
    """
    logs = sorted(f for f in os.listdir(folder) if f.endswith('.hdf5'))
    if not logs:
        raise ValueError(f'{folder} contains no .hdf5 log file. '
                         'Choose the folder of a single run.')
    if len(logs) > 1:
        raise ValueError(f'{folder} contains {len(logs)} .hdf5 files '
                         f'({", ".join(logs)}). Choose the folder of a '
                         'single run.')
    return os.path.join(folder, logs[0])


def display_entries(plan):
    """Strip a built plan down to what the window needs: no live Processor or
    Segment objects (the log is closed once the plan is built), but keep
    `_baseline` so save_plan can still write deltas-only params, and the
    processor's param spec so the dialog can build its form. Underscore
    prefixes keep both out of the written plan file."""
    keep = ('processor', 'title', 'label', 'segments', 'verdict', 'reason',
            'params', '_baseline')
    out = []
    for e in plan:
        d = {k: e[k] for k in keep if k in e}
        d['_param_spec'] = dict(e['_proc'].params)
        out.append(d)
    return out


def deltas(entry):
    """The operator's edits: params differing from the entry's baseline
    (defaults + predicate detection). This is exactly what save_plan writes."""
    baseline = entry.get('_baseline', {})
    return {k: v for k, v in entry['params'].items()
            if k not in baseline or baseline[k] != v}


def group_by_label(entries):
    """[(label, [entry])] in order of first appearance -- the operator thinks
    "what did that test produce", so the tree groups segment first."""
    order, buckets = [], {}
    for e in entries:
        if e['label'] not in buckets:
            buckets[e['label']] = []
            order.append(e['label'])
        buckets[e['label']].append(e)
    return [(label, buckets[label]) for label in order]


def restore_selection(entries, plan_path):
    """Best-effort merge of a previous analysis_plan.json onto fresh entries.

    Unlike --plan execution this must not hard-error: the file may predate a
    catalog change, and reopening a log should never refuse to show it. File
    entries that no longer match anything are simply dropped.
    """
    if not os.path.isfile(plan_path):
        return
    try:
        file_entries = runner.load_plan(plan_path)
    except Exception:
        return                       # a corrupt old file must not block the UI
    index = {(e['processor'], e['label']): e for e in entries}
    for fe in file_entries:
        entry = index.get((fe.get('processor'), fe.get('label')))
        if entry is None:
            continue
        entry['_selected'] = bool(fe.get('selected'))
        params = {k: v for k, v in fe.get('params', {}).items()
                  if k in entry['params']}
        entry['params'].update(params)


def write_plan(entries, selected_keys, log_dir):
    """Stamp the checkbox state onto the entries and write the plan file.

    Returns the plan file path. `selected_keys` is the set of
    (processor, label) pairs the operator ticked.
    """
    for e in entries:
        e['_selected'] = (e['processor'], e['label']) in selected_keys
    path = os.path.join(log_dir, runner.PLAN_FILENAME)
    runner.save_plan(entries, path, log_dir=log_dir)
    return path
