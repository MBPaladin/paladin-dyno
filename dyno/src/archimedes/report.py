r"""Render an analysis pack into a LaTeX report.

    ./dyno/utilities/archimedes.sh gbx_1p2p0 --report

Reads `results.json` and the CSVs the pack already wrote, and nothing else. No
HDF5 is opened and no figure is replotted, for the same reason
`dyno.src.analysis.tex` works that way: report wording gets iterated on far
more often than the maths behind it, and a rewording should cost a second.

TONE. This is a customer deliverable and it is deliberately flat. It reports
what was measured and under what conditions, and it stops there:

  * measured values, the conditions they were measured under, and the stop
    reason a run actually recorded -- all stated plainly;
  * no compliance verdicts, no pass/fail, no comparison against a rating;
  * no attribution of cause, and no speculation about why a number came out
    where it did;
  * method notes limited to what a reader needs to reproduce the number.

Measurement conditions that bear on how a number should be read are NOT
omitted -- leaving those out would be worse than editorialising -- but they are
written as conditions rather than as conclusions. "The output shaft was held by
the absorber's position loop rather than a fixed ground" is a condition. "The
stiffness is therefore understated" is a conclusion, and it belongs to whoever
reads the report.

Output, beside the analysis pack:

    <pack>/report/report_generated.tex   the driver; regenerated every run
    <pack>/report/sections/*.tex         generated, never hand-edited
    <pack>/report/pics/*.png             figures under stable semantic names

Figures are copied under semantic aliases rather than their analysis filenames
so hand-written commentary can \includegraphics them without breaking the next
time an analyzer renames an output.
"""

import csv
import json
import os
import re
import shutil

import numpy as np

from .. import analysis
from ..analysis import texfmt as T
from . import naming
from .analyzers.efficiency import CREEP_MIN_LEG_S

SECTIONS_DIRNAME = 'sections'
PICS_DIRNAME = 'pics'
DRIVER_NAME = 'report_generated.tex'

NM = r'\newton\meter'
RPM = r'\rpm'
NM_RAD = r'\newton\meter\per\radian'
MRAD = r'\milli\radian'

# analysis-pack filename -> the name the report refers to it by.
FIGURES = {
    'velocity_ramp__ripple_vs_speed.png': 'torque_ripple.png',
    'velocity_ramp__speed_tracking.png': 'speed_tracking.png',
    'velocity_ramp__no_load_drag_forward.png': 'no_load_drag_forward.png',
    'velocity_ramp__no_load_drag_backdrive.png':
        'no_load_drag_backdrive.png',
    'velocity_ramp__output_travel.png': 'output_travel.png',
    'efficiency__efficiency_map_forward.png': 'efficiency_map_forward.png',
    'efficiency__efficiency_map_backdrive.png':
        'efficiency_map_backdrive.png',
    'efficiency__efficiency_map_backdriven_in_forward_test.png':
        'efficiency_map_backdriven_forward_test.png',
    'efficiency__efficiency_map_forward_flow_in_backdrive_test.png':
        'efficiency_map_forward_flow_backdrive_test.png',
    'efficiency__efficiency_vs_torque_backdriven_in_forward_test.png':
        'efficiency_curve_backdriven_forward_test.png',
    'efficiency__efficiency_vs_torque_forward_flow_in_backdrive_test.png':
        'efficiency_curve_forward_flow_backdrive_test.png',
    'efficiency__efficiency_vs_torque_forward.png': 'efficiency_forward.png',
    'efficiency__efficiency_vs_torque_backdrive.png': 'efficiency_backdrive.png',
    'efficiency__loss_vs_torque_forward.png': 'loss_forward.png',
    'efficiency__loss_vs_torque_backdrive.png': 'loss_backdrive.png',
    'efficiency__creep_vs_torque.png': 'creep.png',
    'efficiency__coverage.png': 'efficiency_coverage.png',
    'efficiency__cell_zero_and_drag.png': 'cell_zero.png',
    'slip__ramp_traces_forward.png': 'slip_ramps_forward.png',
    'slip__ramp_traces_backdrive.png': 'slip_ramps_backdrive.png',
    'slip__slip_detail_forward.png': 'slip_detail_forward.png',
    'slip__slip_detail_backdrive.png': 'slip_detail_backdrive.png',
    'stiffness__hysteresis.png': 'stiffness_hysteresis.png',
    'stiffness__stiffness_vs_torque.png': 'stiffness_vs_torque.png',
    'stiffness__tracking.png': 'stiffness_tracking.png',
}

DIRECTION_WORD = {'forward': 'forward driving', 'backdrive': 'back-driving'}

# The slip section is split by which shaft was locked, because that is how the
# customer's test list asks for it ('forward driving (low speed side lock)',
# 'back-driving (high speed side lock)') and because the two halves are read in
# different frames: a forward ramp pushes the input and a back-drive ramp the
# output, 43:1 apart.
_SLIP_HEADING = {
    'forward': 'Forward driving, output (low-speed) shaft held',
    'backdrive': 'Back-driving, input (high-speed) shaft held',
}
_SLIP_INTRO = {
    'forward':
        'Torque was ramped on the input (high-speed) shaft with the output '
        '(low-speed) shaft held.',
    'backdrive':
        'Torque was ramped on the output (low-speed) shaft with the input '
        '(high-speed) shaft held. Torque values in this subsection are '
        'measured at the shaft each ramp was applied to, so the breakaway '
        'torques below are at the output cell and are not comparable with the '
        'input-cell values in the preceding subsection without referring one '
        'through the measured gear ratio.',
}


# ------------------------------------------------------------------ helpers

def _read_csv(pack, name):
    path = os.path.join(pack, name)
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    """A CSV cell as a float, or None when blank/unparseable."""
    v = (row or {}).get(key)
    if v in (None, ''):
        return None
    try:
        return float(v)
    except ValueError:
        return None


# Shrink a tabular to the text width, but only when it is actually too wide --
# \resizebox on its own would also SCALE UP a narrow table, which makes the
# two-column ones look like a different document. The \ifdim test is what keeps
# a table that already fits at its natural size.
_FITBOX_OPEN = (r'\resizebox{\ifdim\width>\linewidth\linewidth'
                r'\else\width\fi}{!}{%')


def _table(caption, header, rows, label=None, spec=None, size=r'\small'):
    """texfmt.table, set one size down and scaled to fit the text width.

    These tables are wider than the motor reports' -- an efficiency point
    carries a speed, two torques, a count and a spread, and the stiffness table
    carries eight columns -- and at full size several of them ran past the
    right margin. The size command is applied inside the float so it does not
    leak into the surrounding text.
    """
    out = T.table(caption, header, rows, label=label, spec=spec)
    out = out.replace('  \\centering', '  \\centering\n  %s' % size, 1)
    out = out.replace('  \\begin{tabular}',
                      '  %s\n  \\begin{tabular}' % _FITBOX_OPEN, 1)
    out = out.replace('  \\end{tabular}', '  \\end{tabular}}', 1)
    return out


def _fig(alias, caption, label, width=0.92):
    return (r'\resultfig{%s/%s}{%s}{%s}{fig:%s}'
            % (PICS_DIRNAME, alias, width, caption, label))


def _have(pack, analysis_name):
    return os.path.isfile(os.path.join(pack, analysis_name))


def _para(*lines):
    return '\n'.join(lines) + '\n'


def _n(v, sig=3):
    r"""\num{...}, dropping a pointless decimal tail on a whole number.

    Commanded speeds and torque levels are integers -- 20 rpm, 5 N.m -- and
    three significant figures renders them as 20.00 and 5.00, which reads as a
    measurement precision that was never claimed.
    """
    if v is not None and float(v).is_integer():
        return T.num(int(v))
    return T.num(v, sig)


def _rpm(v):
    return r'\SI{%s}{\rpm}' % (int(v) if float(v).is_integer()
                                else T.raw(v, 4))


def _and_list(items):
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    if len(items) <= 1:
        return ''.join(items)
    return '%s and %s' % (', '.join(items[:-1]), items[-1])


def _directions_phrase(directions):
    """'forward driving', 'back-driving', or 'forward and back-driving'.

    Fixed order rather than the order the inventory happened to be keyed in,
    and the shared word is carried by the last term only -- 'forward driving
    and back-driving' says 'driving' twice to no purpose.
    """
    words = [DIRECTION_WORD.get(d, d) for d in ('forward', 'backdrive')
             if d in set(directions)]
    # Anything the fixed order does not know about still gets listed.
    words += [DIRECTION_WORD.get(d, d) for d in directions
              if d not in ('forward', 'backdrive')]
    if len(words) <= 1:
        return ''.join(words)
    trimmed = [w[:-len(' driving')] if w.endswith(' driving') else w
               for w in words[:-1]]
    return _and_list(trimmed + words[-1:])


# ----------------------------------------------------------------- sections

def _conditions(results, cfg, pack):
    """Bench, instrumentation, and the corrections applied. Facts only."""
    ratio = results.get('ratio')
    out = [r'\section{Test conditions}', '']
    # One sentence per source line. The conditions paragraph is the one that
    # gets reworded most often between units, and a single 500-character line
    # makes every such edit a whole-paragraph diff.
    out.append(_para(
        'The drive under test was installed on the Paladin in-house '
        'dynamometer between two servo motors.',
        'The input (high-speed) shaft was coupled to a %s servo motor and the '
        'output (low-speed) shaft to an %s servo motor.'
        % (T.escape(cfg.get('input_motor', 'Kollmorgen AKM2G-431')),
           T.escape(cfg.get('output_motor', 'Akribis ADR220-A165'))),
        'Torque was measured by a load cell on each shaft.',
        'Position and velocity were taken from the drive encoders.',
        'All channels were logged at \\SI{1}{\\kilo\\hertz}.'))

    rows = [
        # _n, not T.num: a whole-number nameplate ratio is 43, and T.num's
        # four significant figures renders it 43.00, which reads as a measured
        # value rather than the number stamped on the drive.
        ['Nominal gear ratio', _n(cfg.get('ratio_nameplate', 43.88), 4)],
        ['Measured gear ratio, no load', T.num(ratio, 6)],
        ['Input torque cell, full scale', T.si(20, NM)],
        ['Output torque cell, full scale', T.si(500, NM)],
        ['Logging rate', T.si(1, r'\kilo\hertz')],
    ]
    half = cfg.get('position_half_window_rad')
    if half:
        rows.append(['Output position window, half width',
                     T.si(half, r'\radian')])
    cap = cfg.get('torque_cap_nm')
    if cap and cfg.get('report_torque_cap', True):
        rows.append(['Output torque limit applied during testing',
                     T.si(cap, NM)])
    out.append(_table('Test conditions.', ['Quantity', 'Value'], rows,
                       label='tab:conditions', spec='lr'))

    out.append(_para(
        'The gear ratio in this report is the value measured from the no-load '
        'velocity sweep, taken as the slope of input shaft speed against '
        'output shaft speed. It is used to refer input-side quantities to the '
        'output shaft throughout.'))

    # The torque-cell zero offsets and the train's Coulomb drag used to be
    # stated here, with the bar figure that separated them. The drag moved to
    # the velocity ramp section, where it is measured and where it is now read
    # against speed rather than as one pooled number; the zero offsets are no
    # longer reported at all. Both were dropped deliberately -- do not put
    # them back without asking.
    out.append(_para(
        'Each torque cell is tared prior to each test using a '
        '%s second still duration with no applied torque.'
        % _n(cfg.get('tare_still_s', 2.0))))
    return '\n'.join(out)


# Tests whose results the report does not present. Their data is still
# collected, still analysed and still shipped as CSV -- it is just not listed
# here, because an inventory row for a test the reader will not find a section
# for reads as a section that went missing.
_INVENTORY_SKIP = {'stiffness'}


def _inventory(results, cfg, pack):
    """What data exists, and what each run recorded as its stop reason."""
    out = [r'\section{Data collected}', '']
    inv = results.get('inventory') or {}
    # One row per test, not one per test and direction. A test run both ways is
    # one entry in the customer's test list, and splitting it across two rows
    # made the table twice as long without saying anything the Direction column
    # does not already say.
    by_test = {}
    skip = set(_INVENTORY_SKIP)
    if cfg.get('report_stiffness'):
        # The report has a stiffness section, so the inventory lists the test.
        skip.discard('stiffness')
    if not cfg.get('report_slip'):
        # No slip section, so no slip row: the breakaway runs ride with it.
        skip |= {naming.SLIP, naming.BREAKAWAY}
    for key in sorted(inv):
        kind, _, direction = key.partition('/')
        if kind in skip:
            continue
        title = naming.KIND_TITLES.get(
            kind, kind.replace('_', ' ').capitalize())
        if kind == naming.STIFFNESS:
            title = 'Torsional stiffness'
        by_test.setdefault(title, []).append(direction)
    rows = []
    for title in sorted(by_test):
        # Segment counts are dropped on purpose. They are a property of how the
        # campaign was split into files, not of the drive, and a reader who
        # sees 249 against one row and 3 against another reads a coverage
        # difference that is not there.
        rows.append([T.escape(title),
                     T.escape(_directions_phrase(by_test[title]))])
    if rows:
        out.append(_table(
            'Tests performed, and the directions each was run in.',
            ['Test', 'Direction'], rows,
            label='tab:inventory', spec='ll'))

    # The customer's test list asks for pk-pk torque ripple, but no segment was
    # run for it -- it comes off the velocity ramp's constant-speed legs. With
    # no row of its own in the table above, a reader looking for it finds
    # nothing, so the table is told where it went.
    if any(k.partition('/')[0] == naming.VELOCITY for k in inv):
        out.append(_para(
            'Peak-to-peak torque ripple was not run as a test of its own. It '
            'was inferred using the data from the velocity ramp up segments, '
            'from the constant-speed legs of each speed step; '
            'Section~\\ref{sec:velocity} gives the method and the results.'))

    dups = results.get('duplicates') or []
    if dups and _throws(cfg):
        out.append(_para(
            'Some tests were recorded across more than one file. '
            '%d operating point%s recorded in more than one file, and the '
            'longest recording was used in each case. The efficiency sweep was '
            'recorded in several runs; where a level was recorded in more than '
            'one, every leg is included and the reported value is the median '
            'across them.'
            % (len(dups), ' was' if len(dups) == 1 else 's were')))
    elif dups:
        # The affected segment names used to be listed out here, inside a
        # \sloppypar because a long comma-separated run of \texttt identifiers
        # gives TeX almost no break points. Both are gone: the names are the
        # bench's internal segment ids and mean nothing to the customer, and
        # with them gone the paragraph breaks like ordinary prose.
        out.append(_para(
            'Several tests were recorded across more than one file, where a '
            'run was stopped and resumed due to slip-induced position limit '
            'breaches.',
            '%d operating point%s recorded in more than one file.'
            % (len(dups), ' was' if len(dups) == 1 else 's were'),
            'In each case the longest recording was used and the others were '
            'set aside.'))
    return '\n'.join(out)


def _throws(cfg):
    """True when the unit's tests were run as velocity-mode THROWS (limit to
    limit at one constant speed, each steady leg logged separately) rather than
    as a bipolar position sawtooth. The unit file says so: `test_shape: throws`.
    It changes how the test is DESCRIBED, never how it is analysed."""
    return cfg.get('test_shape') == 'throws'


def _velocity(results, cfg, pack):
    rows_csv = _read_csv(pack, 'velocity_ramp__per_step.csv')
    if not rows_csv:
        return ''
    out = [r'\section{Velocity ramp and no-load torque ripple}'
           r'\label{sec:velocity}', '']
    out.append(_para(
        'The drive was commanded through a series of constant-speed steps at '
        'no load, with the shaft not being commanded held at '
        '\\SI{0}{\\newton\\meter}. Steps were \\SI{30}{\\rpm} apart up to '
        '\\SI{300}{\\rpm} and \\SI{300}{\\rpm} apart from there to '
        '\\SI{3600}{\\rpm}, referred to the input shaft. '
        + ('Each step is a series of throws: the commanded shaft is driven at '
           'the step speed to a turnaround angle on the output, stopped, and '
           'driven back the other way, so each step holds its speed over '
           'several separate constant-speed legs.' if _throws(cfg) else
           'Each step is a bipolar traverse, so it holds its speed over '
           'several separate constant-speed legs.')))
    if len({r['direction'] for r in rows_csv}) > 1:
        # Two tables follow and they were taken under different control
        # configurations. Saying so here is the difference between a reader
        # comparing two speed sweeps and a reader comparing two different
        # tests.
        out.append(_para(
            'The sweep was run twice. Forward driving, the input '
            '(high-speed) shaft was the shaft commanded through the speed '
            'steps and the output (low-speed) shaft was held at '
            '\\SI{0}{\\newton\\meter}. Back-driving, the output shaft was '
            'commanded and the input shaft was held at '
            '\\SI{0}{\\newton\\meter}. Commanded speeds are referred to the '
            'input shaft in both cases, so a back-drive step labelled '
            '\\SI{300}{\\rpm} commanded the output at the speed that '
            'corresponds to \\SI{300}{\\rpm} at the input through the '
            'measured gear ratio. Measured input speed is shown against the '
            'commanded value in Figure~\\ref{fig:tracking}.'))
    out.append(_para(
        'Torque ripple is reported as a robust peak-to-peak value: the span '
        'between the 0.5th and 99.5th percentiles of the input torque cell '
        'reading over one constant-speed leg. The percentile span is used in '
        'place of the full minimum-to-maximum range so that isolated single '
        'samples do not set the reported value. The value reported for a step '
        'is the median across its legs, and the bars in '
        'Figure~\\ref{fig:ripple} span them.'))

    # The per-step ripple tables that used to stand here -- one per direction,
    # 21 rows each -- are gone. Every column they carried is in the two figures
    # below, read more easily: the ripple and its leg spread in
    # Figure~\ref{fig:ripple} and the speed holding in
    # Figure~\ref{fig:tracking}. The numbers themselves are still shipped, in
    # `velocity_ramp__per_step.csv`.

    if _have(pack, 'velocity_ramp__ripple_vs_speed.png'):
        out.append(_fig('torque_ripple.png',
                        'No-load input torque ripple against commanded input '
                        'speed. Bars span the constant-speed legs of each '
                        'step.', 'ripple'))
    if _have(pack, 'velocity_ramp__speed_tracking.png'):
        out.append(_fig('speed_tracking.png',
                        'Measured input speed against commanded input speed.',
                        'tracking'))
    # `report_output_travel: false` in the unit file leaves the end-stop travel
    # figure and its paragraph out; the figure is still written to the pack.
    if (cfg.get('report_output_travel', True)
            and _have(pack, 'velocity_ramp__output_travel.png')):
        out.append('')
        out.append(_para(
            'The Archimedes drive has an internal end stop on the output '
            'shaft. Output travel was monitored throughout and a software '
            'position window was applied; Figure~\\ref{fig:travel} shows the '
            'travel recorded during each step against that window.'))
        out.append(_fig('output_travel.png',
                        'Output shaft travel during each velocity step, '
                        'referenced to the start of the step. Dashed lines '
                        'mark the position window in force.', 'travel'))

    # Last, and not before the end-stop paragraph: it is the only subsection
    # this section has, and anything emitted after its heading would be filed
    # under it.
    out.append(_drag_block(cfg, pack))
    return '\n'.join(out)


def _drag_block(cfg, pack):
    """No-load drag, split into a Coulomb and a viscous term against speed.

    This used to sit in Test conditions as one pooled Coulomb number beside the
    torque-cell zeros. It belongs here: the number is measured from these
    constant-speed legs, and read against speed it separates into a term that
    does not change with speed and a term that grows with it, which one pooled
    value cannot show.

    Still a condition and not a conclusion. The split is stated as what the
    cell readings do -- one part reverses with travel direction, one part grows
    with speed -- and no claim is made about which part belongs to the bearings,
    the seals or the lubricant.
    """
    fits = {r['direction']: r for r in
            _read_csv(pack, 'velocity_ramp__drag_fit.csv')}
    have_figs = [d for d in ('forward', 'backdrive')
                 if _have(pack, 'velocity_ramp__no_load_drag_%s.png' % d)]
    if not fits and not have_figs:
        return ''

    out = ['', r'\subsection{No-load drag}', '']
    out.append(_para(
        'The cell reading on a no-load constant-speed leg carries the train\'s '
        'own drag. It was separated into two terms by reading each step in '
        'both travel directions: a term that reverses when the direction of '
        'travel reverses, quoted below as the Coulomb term, and a term '
        'proportional to shaft speed, quoted as the viscous term. Each '
        'direction is read on the torque cell of the shaft that was driven and '
        'against that shaft\'s own speed, so neither column below is referred '
        'through the gear ratio and the two are not directly comparable.'))

    if fits:
        body = []
        for direction in ('forward', 'backdrive'):
            r = fits.get(direction)
            if not r:
                continue
            body.append([
                T.escape(DIRECTION_WORD[direction].capitalize()),
                T.escape(r['shaft']),
                T.num(_f(r, 'coulomb_nm'), 3),
                # Signed: the input-side drag FALLS with speed over this
                # sweep, and a magnitude here would hide that.
                T.num(_f(r, 'viscous_nm_per_krpm'), 3),
                _rpm(_f(r, 'rpm_max')),
                T.num(_f(r, 'drag_at_rpm_max_nm'), 3),
                T.num(_f(r, 'r2'), 2),
            ])
        out.append(_table(
            'No-load drag on the driven shaft, split into a speed-independent '
            'Coulomb term and a speed-proportional viscous term. Values are at '
            'the torque cell of the shaft named, not referred through the gear '
            'ratio.',
            ['Direction', 'Shaft measured',
             r'Coulomb (\si{\newton\meter})',
             r'Viscous (\si{\newton\meter}/\si{\kilo\rpm})',
             'Top speed reached',
             r'Total at top speed (\si{\newton\meter})',
             r'$R^2$'],
            body, label='tab:drag_split', spec='llrrrrr'))
        # The viscous slope is quoted per 1000 rpm because that is the readable
        # unit for the input shaft, which ran to 3600 rpm. The output shaft
        # never passed 85 rpm, so its slope in those units is a unit
        # conversion and not a reading taken at 1000 rpm -- hence the last
        # column, which is the drag actually seen at the top of each sweep.
        note = ['The viscous term is quoted as a signed slope per '
                '\\SI{1000}{\\rpm} of the shaft it was measured on.']
        # Only say it where it applies. The output shaft turns at 1/43 of the
        # input and never comes near 1000 rpm; the input shaft passes it on
        # every sweep, so on a unit with no back-drive ramp this caveat has
        # nothing to caveat.
        if 'backdrive' in fits:
            note.append(
                'The output shaft did not reach that speed during these '
                'tests.')
        note.append(
            'The table also gives the total drag at the highest speed each '
            'sweep actually reached, which is a measured figure rather than '
            'an extrapolated one. $R^2$ is that of the two-term fit against '
            'the per-step drag.')
        out.append(_para(' '.join(note)))

        # Called out because a negative viscous term reads as a typo otherwise:
        # drag that FALLS as the shaft speeds up is not what the words 'viscous
        # drag' lead a reader to expect. Written only when the input-side term
        # actually comes out negative -- it does on 1.2.0 and does not on
        # 1.2.1, and a sentence hardcoded here would describe the wrong unit.
        #
        # Stated and left there. A draft of this carried a second sentence
        # attributing it to a static-to-kinetic friction transition; that was
        # cut, because why the number came out negative is the reader's to
        # judge and this report does not attribute cause anywhere else.
        fwd = _f(fits.get('forward') or {}, 'viscous_nm_per_krpm')
        if fwd is not None and fwd < 0:
            out.append(_para(
                'Notably, the input shaft carries a negative viscous term: '
                'the drag measured at the input cell falls as speed rises, '
                'rather than growing with it.'))

    for direction in have_figs:
        out.append(_fig(
            'no_load_drag_%s.png' % direction,
            'No-load drag, %s. Mean cell reading on each constant-speed leg '
            'against the driven shaft\'s speed, signed by travel direction. '
            'The fitted line on each flank gives the Coulomb term as half the '
            'difference of the two intercepts and the viscous term as their '
            'slope.' % DIRECTION_WORD[direction],
            'drag_%s' % direction))
    return '\n'.join(out)


def _efficiency_quadrants(cfg, pack):
    """The efficiency section laid out as the four combinations of which shaft
    was commanded (the test) and which way power flowed (the legs).

    Every throw is classified by the direction power actually flowed, so each
    test yields both: the legs where the held torque opposed the motion and
    the legs where it assisted. The plain section reports only the legs whose
    flow matches their test; this one reports all four, each under its own
    heading with its own statement of what the shafts were doing. Switched by
    `efficiency_quadrants: true` in the unit file.
    """
    allpts = _read_csv(pack, 'efficiency__per_point.csv')
    speeds = sorted({_f(r, 'rpm') for r in allpts if _f(r, 'rpm') is not None})
    step = cfg.get('torque_step_nm', 5.0)
    out = [r'\section{Efficiency}', '']
    out.append(_para(
        'Efficiency was measured at input speeds of %s, in output torque steps '
        'of \\SI{%s}{\\newton\\meter}. Two tests were run. In the forward-driving '
        'test the input (high-speed) shaft was commanded through constant-speed '
        'throws to a turnaround angle on the output, reversing direction between '
        'throws, while the output (low-speed) shaft held a constant torque. In '
        'the back-driving test the output shaft was commanded through the same '
        'kind of throws, at the output speed that corresponds to the stated '
        'input speed through the measured gear ratio, while the input shaft held '
        'a constant torque.'
        % (_and_list([_rpm(s) for s in speeds]),
           T.raw(int(step) if float(step).is_integer() else step, 2))))
    out.append(_para(
        'Because the motion reverses and the held torque keeps one sign through '
        'a segment, the held torque opposes the motion during the throws in one '
        'direction and assists it during the throws in the other. Power '
        'therefore flows from input to output during some throws and from '
        'output to input during others. Each throw was measured separately and '
        'classified by the direction in which power flowed. The results are '
        'presented in four subsections: each of the two tests, split by the '
        'direction of power flow.'))
    out.append(_para(
        'Efficiency is the ratio of the two measured shaft powers, output over '
        'input where power flowed from input to output and input over output '
        'where it flowed from output to input. Shaft power is the product of '
        'the measured torque and the measured speed on that shaft. Values are '
        'the median over the repeat legs at each point. Legs where the two '
        'shaft powers did not share a sign were excluded, as no power was being '
        'transmitted through the drive on those legs. Every map uses the same '
        'grid of commanded input speed and commanded output torque and a '
        'colour scale fixed at \\SIrange{0}{100}{\\percent}; grey cells were not '
        'measured.'))

    def have(name):
        return _have(pack, 'efficiency__' + name + '.png')

    def sub(title, label, text, map_name, map_alias, curve=None, loss=None):
        if not have(map_name):
            return
        out.append(r'\subsection{%s}' % title)
        out.append(_para(text))
        lc = title[0].lower() + title[1:]
        out.append(_fig(map_alias,
                        'Efficiency (\\si{\\percent}), %s.' % lc,
                        'eff_map_' + label, width=0.78))
        if curve and have(curve[0]):
            out.append(_fig(curve[1],
                            'Efficiency against measured output torque, %s. '
                            'Bands span the repeat legs at each point.' % lc,
                            'eff_curve_' + label))
        if loss and have(loss[0]) and cfg.get('report_loss_plot', True):
            out.append(_fig(loss[1],
                            'Power loss, %s: measured input shaft power minus '
                            'measured output shaft power.' % lc,
                            'eff_loss_' + label))

    sub('Forward driving, power from input to output', 'ff',
        'The input shaft was commanded and the held output torque opposed the '
        'motion: the input shaft supplied the power and the output shaft '
        'absorbed it.',
        'efficiency_map_forward', 'efficiency_map_forward.png',
        curve=('efficiency_vs_torque_forward', 'efficiency_forward.png'),
        loss=('loss_vs_torque_forward', 'loss_forward.png'))
    sub('Forward driving, power from output to input', 'fb',
        'The input shaft was commanded and the held output torque assisted the '
        'motion, so power flowed from the output shaft to the input shaft. The '
        'input shaft was still the shaft commanded at the constant speed, so '
        'this is a different arrangement from the back-driving test.',
        'efficiency_map_backdriven_in_forward_test',
        'efficiency_map_backdriven_forward_test.png',
        curve=('efficiency_vs_torque_backdriven_in_forward_test',
               'efficiency_curve_backdriven_forward_test.png'))
    sub('Back-driving, power from output to input', 'bb',
        'The output shaft was commanded and the held input torque opposed the '
        'motion: the output shaft supplied the power and the input shaft '
        'absorbed it.',
        'efficiency_map_backdrive', 'efficiency_map_backdrive.png',
        curve=('efficiency_vs_torque_backdrive', 'efficiency_backdrive.png'))
    sub('Back-driving, power from input to output', 'bf',
        'The output shaft was commanded and the held input torque assisted the '
        'motion, so power flowed from the input shaft to the output shaft. The '
        'output shaft was still the shaft commanded at the constant speed, so '
        'this is a different arrangement from the forward-driving test.',
        'efficiency_map_forward_flow_in_backdrive_test',
        'efficiency_map_forward_flow_backdrive_test.png',
        curve=('efficiency_vs_torque_forward_flow_in_backdrive_test',
               'efficiency_curve_forward_flow_backdrive_test.png'))

    if _have(pack, 'efficiency__coverage.png'):
        out.append(_fig('efficiency_coverage.png',
                        'Operating points covered by the efficiency sweep.',
                        'eff_cov'))
    out.append(_creep(cfg, pack, speeds=speeds))
    return '\n'.join(x for x in out if x)


def _efficiency(results, cfg, pack):
    pts = [r for r in _read_csv(pack, 'efficiency__per_point.csv')
           if r.get('flow') and r['flow'] == r.get('span_direction')]
    if not pts:
        return ''
    if cfg.get('efficiency_quadrants'):
        return _efficiency_quadrants(cfg, pack)
    out = [r'\section{Efficiency}', '']
    speeds = sorted({_f(r, 'rpm') for r in pts if _f(r, 'rpm') is not None})
    step = cfg.get('torque_step_nm', 5.0)
    out.append(_para(
        'Efficiency was measured at input speeds of %s, in output torque steps '
        'of \\SI{%s}{\\newton\\meter}. At each operating point the input shaft '
        'was commanded through %s while the output shaft held a constant '
        'torque.'
        % (_and_list([_rpm(s) for s in speeds]),
           T.raw(int(step) if float(step).is_integer() else step, 2),
           'constant-speed throws to a turnaround angle on the output, '
           'reversing direction between throws, ' if _throws(cfg) else
           'a bipolar constant-speed traverse')))
    out.append(_para(
        'Because the motion reverses, the held output torque opposes the '
        'motion on one half of each traverse and assists it on the other. '
        'Power therefore flows from input to output on one half and from '
        'output to input on the other. Each half-traverse was measured '
        'separately and classified by the direction in which power flowed, '
        'and only the half-traverses whose power flow matches the test they '
        'came from are reported: the forward-driving halves of the '
        'forward-driving test give the forward-driving efficiency, and the '
        'back-driven halves of the back-driving test give the back-driving '
        'efficiency. Each figure below is therefore a single, isolated '
        'direction of power flow measured under the drive arrangement the '
        'test was set up for.'))
    if _throws(cfg):
        # Same statement, in the vocabulary of a throw test.
        for old_, new_ in (
                ('on one half of each traverse and assists it on the other',
                 'during the throws in one direction and assists it during the '
                 'throws in the other'),
                ('Each half-traverse was measured', 'Each throw was measured'),
                ('only the half-traverses whose', 'only the throws whose')):
            out[-1] = out[-1].replace(old_, new_)
    out.append(_para(
        'Efficiency is the ratio of the two measured shaft powers, output '
        'over input where the input was driving and input over output where '
        'the output was driving. Shaft power is the product of the measured '
        'torque and the measured speed on that shaft. Values are the median '
        'over the repeat legs at each point. Legs where the two shaft powers '
        'did not share a sign were excluded, as no power was being '
        'transmitted through the drive on those legs.'))

    directions = [d for d in ('forward', 'backdrive')
                  if any(r['flow'] == d for r in pts)]
    for direction in directions:
        alias = 'efficiency_map_%s.png' % direction
        if not _have(pack, 'efficiency__' + alias):
            continue
        out.append(_fig(alias,
                        'Efficiency (\\si{\\percent}), %s, over the measured '
                        'grid of commanded input speed and commanded output '
                        'torque. The colour scale is fixed at \\SIrange{0}{100}{\\percent} '
                        'in both directions and across units. Grey cells were '
                        'not measured.' % DIRECTION_WORD[direction],
                        'eff_map_%s' % direction, width=0.78))

    # `report_cross_term_map: true` adds the forward-driving test's own
    # output-to-input legs as a map. They are kept out of the reported results
    # above on purpose (see the paragraph before the figures), so the text says
    # exactly what they are.
    if (cfg.get('report_cross_term_map')
            and _have(pack, 'efficiency__efficiency_map_backdriven_in_forward_test.png')):
        out.append(_para(
            'The forward-driving test also contains legs on which the held '
            'output torque assisted the motion, so that power flowed from the '
            'output shaft to the input shaft. Figure~\\ref{fig:eff_map_cross} '
            'shows the efficiency over those legs, on the same grid and colour '
            'scale. On these legs the input shaft was still the shaft '
            'commanded at the constant speed, so this is a different '
            'arrangement from the back-driving test, in which the output shaft '
            'was commanded.'))
        out.append(_fig('efficiency_map_backdriven_forward_test.png',
                        'Efficiency (\\si{\\percent}) over the legs of the '
                        'forward-driving test on which power flowed from the '
                        'output shaft to the input shaft, over the measured '
                        'grid of commanded input speed and commanded output '
                        'torque. The colour scale is fixed at '
                        '\\SIrange{0}{100}{\\percent}. Grey cells were not '
                        'measured.', 'eff_map_cross', width=0.78))

    if _have(pack, 'efficiency__efficiency_vs_torque_forward.png'):
        out.append(_fig('efficiency_forward.png',
                        'Efficiency against measured output torque, forward '
                        'driving. Bands span the repeat legs at each point.',
                        'eff_fwd'))
    # The back-driving line plot is written into the pack but not placed here.
    # Its x axis is MEASURED output torque, and in the back-drive segments that
    # runs 18-55 Nm for a commanded 5-40 Nm, so the curves shear against each
    # other and the collapse above ~40 Nm reads as noise. The back-driving map
    # carries the same numbers on the commanded grid. Add it back with a
    # _fig('efficiency_backdrive.png', ...) here if it is wanted.
    if (cfg.get('report_loss_plot', True)
            and _have(pack, 'efficiency__loss_vs_torque_forward.png')):
        out.append(_fig('loss_forward.png',
                        'Power loss, forward driving: measured input shaft '
                        'power minus measured output shaft power.', 'loss'))
    if _have(pack, 'efficiency__coverage.png'):
        out.append(_fig('efficiency_coverage.png',
                        'Operating points covered by the efficiency sweep.',
                        'eff_cov'))

    top = max((_f(r, 't_cmd_nm') or 0) for r in pts)
    cap = cfg.get('torque_cap_nm')
    if cap and top < cap:
        out.append(_para(
            'The sweep covered output torque up to \\SI{%s}{\\newton\\meter}. '
            'The test request specifies a range to \\SI{%s}{\\newton\\meter}; '
            'points above \\SI{%s}{\\newton\\meter} were not recorded.'
            % (T.raw(top, 3), T.raw(cap, 3), T.raw(top, 3))))

    # A 'Speed holding' subsection used to sit between here and the creep
    # ratio, tabulating how much of each segment was spent at the commanded
    # speed. It is gone: it reported on the shape of the bench's own test
    # plan -- the lead-in, settle and turnaround dwells that do not shorten as
    # the commanded speed rises -- rather than on the drive, and the customer
    # asked for the drive. The underlying numbers are unchanged and still
    # written to efficiency__speed_coverage.csv, and the analyzer still raises
    # its low_speed_coverage finding into report.txt.
    out.append(_creep(cfg, pack, speeds=speeds))
    return '\n'.join(x for x in out if x)


def _creep(cfg, pack, speeds=()):
    rows = _read_csv(pack, 'efficiency__creep.csv')
    if not rows:
        return ''
    out = [r'\subsection{Creep ratio}', '']
    out.append(_para(
        'Creep ratio is reported as $1 - (r \\cdot \\omega_{\\mathrm{out}}) / '
        '\\omega_{\\mathrm{in}}$, where $r$ is the gear ratio measured from '
        'the no-load sweep. It is evaluated over the constant-speed legs of '
        'each operating point.'
        + (' The test request specifies that creep ratio should stay below '
           '\\SI{%s}{\\percent}.' % T.raw(cfg.get('creep_limit_pct', 5.0), 2)
           if cfg.get('creep_limit_line', True) else '')))
    by_speed = {}
    for r in rows:
        by_speed.setdefault(_f(r, 'rpm_cmd'), []).append(r)
    body = []
    for sp in sorted(by_speed):
        rs = by_speed[sp]
        vals = [(_f(r, 'creep_pct'), abs(_f(r, 't_cmd_nm') or 0)) for r in rs
                if _f(r, 'creep_pct') is not None]
        if not vals:
            continue
        worst, at = max(vals)
        over = sum(1 for v, _ in vals if v > 5.0)
        body.append([_n(sp), str(len(vals)),
                     T.num(min(v for v, _ in vals), 3),
                     T.num(worst, 3), T.num(at, 3), str(over)])
    # A speed the efficiency sweep ran but creep could not be evaluated at
    # must be SAID, not just left out of the table. The section opens by
    # naming every speed measured, so a speed missing from the table below
    # reads as an omission rather than as a result -- which is how 1.2.1's
    # 3000 rpm points, none of which held the commanded speed long enough to
    # evaluate the ratio over, would otherwise have left the report.
    missing = [sp for sp in (speeds or ()) if sp not in by_speed]
    # `report_creep_table: false` leaves the table out and the figure in; the
    # numbers are still in efficiency__creep.csv.
    if cfg.get('report_creep_table', True):
        out.append(_table(
            'Creep ratio by input speed, across the output torque levels '
            'measured at that speed.',
            [r'Input speed (\si{\rpm})', 'Levels',
             r'Min (\si{\percent})', r'Max (\si{\percent})',
             r'At torque (\si{\newton\meter})',
             r'Above \SI{5}{\percent}'],
            body, label='tab:creep', spec='rrrrrr'))
    if missing:
        out.append(_para(
            'Creep ratio is not reported at %s. The creep ratio is evaluated '
            'over the constant-speed legs of each operating point, and at '
            '%s no leg held the commanded speed for the minimum leg length '
            'of \\SI{%s}{\\second}.'
            % (_and_list([_rpm(s) for s in missing]),
               'that speed' if len(missing) == 1 else 'those speeds',
               T.raw(CREEP_MIN_LEG_S, 2))))
    if _have(pack, 'efficiency__creep_vs_torque.png'):
        out.append(_fig('creep.png',
                        'Creep ratio against commanded output torque, by '
                        'input speed.'
                        + (' The dashed line marks \\SI{%s}{\\percent}.'
                           % T.raw(cfg.get('creep_limit_pct', 5.0), 2)
                           if cfg.get('creep_limit_line', True) else ''),
                        'creep'))
    return '\n'.join(out)


def _slip_short(results, cfg, pack):
    """A short static-slip section: how many ramps, how high, and which slipped.

    Written from slip__ramp_summary.csv, which reads every ramp out of the raw
    record (see analyzers.slip._ramp_summary). No breakaway, no per-ramp tables.
    Switched by `slip_summary: short` in the unit file.
    """
    rows = _read_csv(pack, 'slip__ramp_summary.csv')
    if not rows:
        return ''
    ratio = results.get('ratio') or 1.0
    out = [r'\section{Static slip torque}', '']
    out.append(_para(
        'Torque was ramped on one shaft while the other was held under '
        'position control, which grounds the drive train on the held side. '
        'Successive ramps alternated between positive and negative torque. '
        'A ramp is recorded as slip where the two shafts stopped turning at '
        'the gear ratio and the angle between them, referred to the output, '
        'began to change.'))
    table, text = [], []
    nominal = cfg.get('ratio_nameplate') or ratio
    for direction in ('forward', 'backdrive'):
        sel = [r for r in rows if r['direction'] == direction]
        if not sel:
            continue
        slips = [r for r in sel if r['outcome'] == 'slip']
        held = len(sel) - len(slips)
        peak = max(_f(r, 'cmd_peak_nm') for r in sel)
        # The ramps are commanded to a target, and the target is what is
        # reported: 3.5 N.m at the input is the 150 N.m asked for at the
        # output (through the nameplate ratio), whatever the cell then read.
        target = peak * nominal if direction == 'forward' else peak
        target = 5 * round(target / 5)
        table.append([
            T.escape(DIRECTION_WORD[direction].capitalize()),
            T.escape('input' if direction == 'forward' else 'output'),
            T.escape('output' if direction == 'forward' else 'input'),
            str(len(sel)), str(len(slips)), T.num(target, 3)])
        if direction == 'forward':
            how = ('The input shaft torque was ramped to \\SI{%s}{\\newton\\meter} '
                   'commanded, which is \\SI{%s}{\\newton\\meter} at the output, with '
                   'the output shaft held.' % (T.raw(peak, 3), T.raw(target, 3)))
        else:
            how = ('The output shaft torque was ramped to \\SI{%s}{\\newton\\meter} '
                   'commanded, with the input shaft held.' % T.raw(target, 3))
        if not slips:
            result = 'All %d ramps reached the commanded torque without slip.' % len(sel)
        else:
            parts = []
            for r in slips:
                at_cell = _f(r, 'slip_out_nm')
                cmd_out = _f(r, 'cmd_peak_nm') * (nominal if direction == 'forward' else 1)
                parts.append(
                    'slip occurred at \\SI{%s}{\\newton\\meter} commanded '
                    '(\\SI{%s}{\\newton\\meter} measured at the output torque cell)'
                    % (T.raw(cmd_out, 3), T.raw(abs(at_cell), 3)) if at_cell is not None
                    else 'slip occurred')
            result = ('%d of the %d ramps reached the commanded torque without slip. '
                      'On the remaining %s, %s.'
                      % (held, len(sel),
                         'ramp' if len(slips) == 1 else '%d ramps' % len(slips),
                         ' and '.join(parts)))
        text.append(_para('%s: %s %s' % (DIRECTION_WORD[direction].capitalize(),
                                          how, result)))
    out.append(_table(
        'Static slip ramps. The torque is the commanded output torque the '
        'ramps were run to.',
        ['Test', 'Ramped shaft', 'Held shaft', 'Ramps', 'Slipped',
         r'Commanded torque (\si{\newton\meter})'],
        table, label='tab:slip', spec='lllrrr'))
    out.extend(text)
    return '\n'.join(out)


def _slip(results, cfg, pack):
    rows = _read_csv(pack, 'slip__events.csv')
    if not rows:
        return ''
    out = [r'\section{Static slip and breakaway torque}', '']
    out.append(_para(
        'Torque was ramped on one shaft while the other was either held under '
        'position control, which grounds the drive train, or commanded to '
        '\\SI{0}{\\newton\\meter}, which does not. Ramps were run in both '
        'directions: with the output (low-speed) shaft held, and with the '
        'input (high-speed) shaft held. The configuration of the far shaft on '
        'each ramp is given in the tables below.'))
    out.append(_para(
        'Two outcomes are distinguished. Where the two shafts stopped turning '
        'at the gear ratio, the event is recorded as slip; this was '
        'identified as the point at which the relative angle between the '
        'shafts, referred to the output, began to change at more than '
        '\\SI{0.2}{\\radian\\per\\second}. Where the shafts remained '
        'coupled at the gear ratio and the train began to turn as a whole, '
        'the event is recorded as breakaway, identified at the point the '
        'ramped shaft began to turn. Where a ramp reached the torque it was '
        'configured to stop at with neither event identified, no event is '
        'recorded and the torque reached is given instead.'))
    out.append(_para(
        'Torque values are the largest sustained reading before the '
        'identified event, taken from a \\SI{0.1}{\\second} median-filtered '
        'trace. Ramps are numbered within each direction in the order they '
        'were recorded; the segment identifier of each is given in the CSV '
        'data accompanying this report.'))

    ratio = results.get('ratio') or 1.0
    # `ramp` is the readable per-direction label the analyzer assigns. A pack
    # written before it existed has only the segment ID, and --report-only over
    # an old pack must still render rather than die on a missing column.
    for r in rows:
        r.setdefault('ramp', r.get('segment', ''))
        if not r['ramp']:
            r['ramp'] = r.get('segment', '')
    for direction in ('forward', 'backdrive'):
        sel = [r for r in rows if r['direction'] == direction]
        if not sel:
            continue
        out.append('')
        out.append(r'\subsection{%s}' % _SLIP_HEADING[direction])
        out.append('')
        out.append(_para(_SLIP_INTRO[direction]))

        body = []
        # By the ordinal, not the label: 'Slip ramp 10' sorts before
        # 'Slip ramp 2' as a string.
        for r in sorted(sel, key=lambda r: (r['kind'],
                                            _f(r, 'ramp_no') or 0)):
            t_out, t_in = _f(r, 't_out_at_event_nm'), _f(r, 't_in_at_event_nm')
            # Magnitudes in the table, with the rotation sense in its own
            # column: a signed torque column mixes 'how much' with 'which way'
            # and reads as an inconsistency next to the magnitudes quoted in
            # the text. The sense is taken from the OUTPUT cell, which is
            # signed the same way on every ramp here; the input cell reverses
            # with the drive's own sign convention.
            sense = ('---' if t_out is None else
                     'positive' if t_out > 0 else 'negative')
            # A ramp with no event has no torque 'at the event'. The number
            # that carries the result is what it reached, so that is what goes
            # in the row, marked as such by the event column.
            if t_out is None:
                t_out, t_in = _f(r, 't_out_peak_nm'), _f(r, 't_in_peak_nm')
            body.append([
                T.escape(r['ramp']),
                T.escape({'slip': 'Slip', 'breakaway': 'Breakaway',
                          'none': 'No event'}.get(r['event'], r['event'])),
                T.escape('%s, %s' % (r['lock_side'].split(' (')[0],
                                     r.get('far_shaft_mode', ''))).rstrip(', '),
                sense,
                T.num(abs(t_out) if t_out is not None else None, 3),
                T.num(abs(t_in) if t_in is not None else None, 3),
            ])
        out.append(_table(
            'Ramp tests, %s. Torque magnitudes are those recorded at the '
            'identified event, or the largest reached where no event was '
            'identified; the sense column gives the direction of the ramp.'
            % DIRECTION_WORD[direction],
            ['Ramp', 'Event', 'Far shaft', 'Sense',
             r'Output cell (\si{\newton\meter})',
             r'Input cell (\si{\newton\meter})'],
            body, label='tab:slip_%s' % direction, spec='llllrr'))

        slips = [r for r in sel if r['event'] == 'slip']
        if slips:
            vals = [abs(_f(r, 't_out_at_event_nm')) for r in slips]
            if len(slips) == 1:
                r = slips[0]
                t_in = _f(r, 't_in_at_event_nm')
                out.append(_para(
                    'Slip occurred at \\SI{%s}{\\newton\\meter} measured at '
                    'the output torque cell. The input torque cell read '
                    '\\SI{%s}{\\newton\\meter} at the same point, which is '
                    '\\SI{%s}{\\newton\\meter} referred to the output through '
                    'the measured ratio.'
                    % (T.raw(vals[0], 3), T.raw(abs(t_in) if t_in else None, 3),
                       T.raw(abs(t_in) * ratio if t_in else None, 3))))
            else:
                n_slip_ramps = len([r for r in sel if r['kind'] == 'slip'])
                out.append(_para(
                    'Slip occurred on %d of the %d slip ramps. Measured at '
                    'the output torque cell the values were %s '
                    '\\si{\\newton\\meter}, with a median of '
                    '\\SI{%s}{\\newton\\meter} and a spread of '
                    '\\SI{%s}{\\newton\\meter} across the ramps.'
                    % (len(slips), n_slip_ramps,
                       _and_list([T.raw(v, 3) for v in sorted(vals)]),
                       T.raw(float(np.median(vals)), 3),
                       T.raw(float(np.ptp(vals)), 3))))
                # Both senses were run, and they did not give the same number.
                # Reporting one median over the pair would hide that, so the
                # two are given separately -- as measurements, with no account
                # of why they differ.
                by_sense = {}
                for r in slips:
                    v = _f(r, 't_out_at_event_nm')
                    by_sense.setdefault('positive' if v > 0 else 'negative',
                                        []).append(abs(v))
                if len(by_sense) > 1:
                    out.append(_para(
                        'Taken by the sense of the ramp, the medians were %s.'
                        % _and_list([
                            '\\SI{%s}{\\newton\\meter} over %d %s ramp%s'
                            % (T.raw(float(np.median(v)), 3), len(v), k,
                               '' if len(v) == 1 else 's')
                            for k, v in sorted(by_sense.items())])))

        brks = [r for r in sel if r['event'] == 'breakaway']
        if brks:
            # The frame follows the ramped shaft: the bench's own detector logs
            # the command value on the motor it was ramping, so a forward ramp
            # yields an input-side torque and a back-drive ramp an output-side
            # one. Quoting either in the other's frame would be wrong by the
            # gear ratio.
            cell = ('t_in_at_event_nm' if direction == 'forward'
                    else 't_out_at_event_nm')
            shaft = 'input' if direction == 'forward' else 'output'
            vals = [abs(_f(r, 'controller_breakaway_nm'))
                    if _f(r, 'controller_breakaway_nm') is not None
                    else abs(_f(r, cell)) for r in brks]
            vals = [v for v in vals if v is not None]
            if vals:
                out.append(_para(
                    'Breakaway was identified on %d ramp%s, at %s '
                    '\\si{\\newton\\meter} measured at the %s torque cell. On '
                    '%s the two shafts continued to turn at the gear ratio and '
                    'no slip was identified.'
                    % (len(vals), '' if len(vals) == 1 else 's',
                       _and_list([T.raw(v, 3) for v in vals]), shaft,
                       'this ramp' if len(vals) == 1 else 'each of these ramps')))

        none = [r for r in sel if r['event'] == 'none']
        if none:
            cell = ('t_in_peak_nm' if direction == 'forward'
                    else 't_out_peak_nm')
            shaft = 'input' if direction == 'forward' else 'output'
            vals = [abs(_f(r, cell)) for r in none if _f(r, cell) is not None]
            if vals:
                out.append(_para(
                    'On %d ramp%s neither event was identified: %s reached %s '
                    '\\si{\\newton\\meter} at the %s torque cell and ended '
                    'there, at the torque the ramp was configured to stop at, '
                    'with the two shafts still turning at the gear ratio.'
                    % (len(vals), '' if len(vals) == 1 else 's',
                       'that ramp' if len(vals) == 1 else 'those ramps',
                       _and_list([T.raw(v, 3) for v in vals]), shaft)))

        alias = 'slip_ramps_%s.png' % direction
        if _have(pack, 'slip__ramp_traces_%s.png' % direction):
            out.append(_fig(alias,
                            'Torque against time for each ramp, %s. The input '
                            'cell is shown referred to the output through the '
                            'measured ratio. The marker shows the identified '
                            'event.' % DIRECTION_WORD[direction],
                            'slip_ramps_%s' % direction, width=1.0))
        if _have(pack, 'slip__slip_detail_%s.png' % direction):
            out.append(_fig('slip_detail_%s.png' % direction,
                            'Output torque and relative shaft angle for the '
                            'ramps in which slip was identified, %s.'
                            % DIRECTION_WORD[direction],
                            'slip_detail_%s' % direction, width=1.0))
    return '\n'.join(out)


_DIRECTION_WORDS = {'forward': 'Forward driving', 'backdrive': 'Back-driving'}


def _stiffness(results, cfg, pack):
    """Torsional stiffness. NOT WIRED INTO THE REPORT -- see `build`.

    Left whole rather than deleted: the analyzer still produces everything
    this reads, so restoring the section is a matter of putting its entry back
    in the builders list.
    """
    rows = _read_csv(pack, 'stiffness__fits.csv')
    if not rows:
        return ''
    out = [r'\section{Torsional stiffness}', '']
    # `report_stiffness_detail: false` drops the lost-motion and shaft-tracking
    # columns, the tracking paragraph and the shaft-angle figure. All of it is
    # still in the pack (stiffness__fits.csv, stiffness__tracking.png).
    detail = cfg.get('report_stiffness_detail', True)
    out.append(_para(
        'Torque was ramped to a target, held, and reversed. Wind-up is '
        'reported at the output shaft as the difference between the output '
        'shaft angle and the input shaft angle divided by the measured no-load '
        'gear ratio. Stiffness is the slope of output torque against that '
        'wind-up, fitted over the middle of the torque range and excluding the '
        'reversal.' + (' Lost motion is the width of the loop at zero torque.'
                       if detail else '')))
    both = len({r.get('direction') for r in rows}) > 1
    if both:
        out.append(_para(
            'The ramps were run in both directions. Forward driving, torque '
            'was applied by the input (high-speed) shaft motor and the output '
            '(low-speed) shaft was held by the absorber motor under position '
            'control. Back-driving, torque was applied by the output motor and '
            'the input shaft was held by its motor under position control. In '
            'neither case was the held shaft fixed to a mechanical ground, so '
            'the values below are measured across the drive and its holding '
            'servo in series.'))
    else:
        out.append(_para(
            'During these tests the output shaft was held by the absorber '
            'motor under position control rather than by a fixed mechanical '
            'ground. The values below are measured across the drive and its '
            'holding servo in series.'))

    # Compliance of the test chain outside the drive. Each motor's encoder sits
    # behind its coupler, torque cell and second coupler, all of which are in
    # series with the drive in every K below. The output-side chain was measured
    # against a mechanical lockout; the input-side one is still to be measured.
    chain_k = cfg.get('chain_k_output_nm_per_rad')
    if chain_k:
        out.append(_para(
            'The measured wind-up also includes the compliance of the test '
            'chain between each motor encoder and the gearbox shaft it drives: '
            'coupler, torque cell and second coupler. The output-side chain '
            '(output motor, coupler, torque cell, coupler, blocked against a '
            'mechanical lockout plate) was measured separately at '
            '\\SI{%s}{\\newton\\meter\\per\\radian}, from lockout runs to '
            '\\SI{100}{\\newton\\meter}.' % T.raw(chain_k, 3)
            + (' The input-side chain (lockout plate, input coupler, input '
               'torque cell, input coupler, input motor) has not yet been '
               'measured, so the values below are not corrected for either '
               'chain.' if cfg.get('chain_input_pending') else '')))

    body = []
    for r in rows:
        k = _f(r, 'k_nm_per_rad')
        body.append(([T.escape(_DIRECTION_WORDS.get(r.get('direction'),
                                                   r.get('direction') or ''))]
                     if both else []) + [
            T.tt(r['segment']),
            T.num(_f(r, 'target_pct'), 2),
            T.num(_f(r, 'torque_peak_nm'), 3),
            T.num(_f(r, 'windup_pk_pk_rad') * 1e3
                  if _f(r, 'windup_pk_pk_rad') is not None else None, 3),
            # Rounded to whole Nm/rad: the fit does not support a fraction of
            # one, and four significant figures pushes it into exponent form.
            T.num(round(k) if k is not None else None),
            T.num(_f(r, 'k_r2'), 3),
        ] + ([T.num(_f(r, 'hysteresis_rad') * 1e3
                    if _f(r, 'hysteresis_rad') is not None else None, 3),
              T.num(_f(r, 'ratio_tracking_r'), 5)] if detail else []))
    out.append(_table(
        'Stiffness ramps. The nominal target is the fraction of static slip '
        'torque the ramp was configured for, using the slip value available '
        'when the test was generated; the peak torque column gives what was '
        'recorded.' + (' Shaft tracking is the correlation between the two '
                       'shaft angles over the ramp.' if detail else ''),
        (['Direction'] if both else []) + ['Segment', r'Target (\si{\percent})',
         r'Peak torque (\si{\newton\meter})',
         r'Wind-up (\si{\milli\radian})',
         r'$K$ (\si{\newton\meter\per\radian})', '$R^2$']
        + ([r'Lost motion (\si{\milli\radian})', 'Tracking'] if detail else []),
        body, label='tab:stiff',
        spec=('l' if both else '') + ('lrrrrrrr' if detail else 'lrrrrr')))

    loose = [r for r in rows if (_f(r, 'ratio_tracking_r') or 1.0) < 0.99]
    if loose and detail:
        out.append(_para(
            'In %s the two shaft angles tracked the gear ratio with a '
            'correlation below 0.99 over the ramp, meaning the shafts did not '
            'remain coupled at a fixed ratio throughout. The wind-up recorded '
            'for %s therefore includes relative motion that is not elastic '
            'deflection.'
            % (', '.join(T.tt(r['segment']) for r in loose),
               'these ramps' if len(loose) > 1 else 'this ramp')))

    if _have(pack, 'stiffness__hysteresis.png'):
        out.append(_fig('stiffness_hysteresis.png',
                        'Output torque against wind-up. The dashed line is '
                        'the fitted slope.', 'stiff_hyst', width=1.0))
    if detail and _have(pack, 'stiffness__tracking.png'):
        out.append(_fig('stiffness_tracking.png',
                        'Shaft angles during each ramp, with the input '
                        'referred to the output through the measured ratio. '
                        'Wind-up is the difference between them, shown on the '
                        'right-hand axis.', 'stiff_track', width=1.0))
    if (cfg.get('report_stiffness_vs_torque', True)
            and _have(pack, 'stiffness__stiffness_vs_torque.png')):
        out.append(_fig('stiffness_vs_torque.png',
                        'Local slope of the wind-up curve, in torque bins.',
                        'stiff_vs_t'))
    return '\n'.join(out)


def _methods(results, cfg, pack):
    out = [r'\section{Measurement notes}', '']
    out.append(_para(
        'The following apply to the values in this report.'))
    items = [
        # The frames are named by their MOTOR and by their handedness, not
        # just as 'input'/'output'. The two machines are mounted facing each
        # other, so their positive senses are opposite; a reader referring a
        # torque from one cell to the other without that gets the sign wrong.
        # Hand-written onto the 1.2.0 report and folded back in here.
        ('Reference frames',
         'Input shaft quantities are in the input motor (right-handed) frame '
         'and output shaft quantities are in the output motor (left-handed) '
         'frame. Input-side quantities '
         'referred to the output are divided by the measured no-load gear '
         'ratio, and are identified as such wherever they appear.'),
        ('Constant-speed selection',
         ('Measurements taken during a throw use only the constant-speed part '
          'of the throw, which was logged separately from its ramp-up and '
          'stop. Within it, samples are taken at the commanded constant '
          'speed, which is used as the reference rather than a value derived '
          'from the data.' if _throws(cfg) else
          'Measurements taken during a traverse use only the samples at the '
          'commanded constant speed. The commanded speed is used as the '
          'reference rather than a value derived from the data, because the '
          'traverse overshoots at each reversal.')),
        ('Averaging',
         'Where a point was measured over more than one leg, the reported '
         'value is the median across legs and the quoted range is the span '
         'across them.'),
        ('Train drag',
         'The measured torques include the running friction of the complete '
         'test train, which was \\SI{%s}{\\newton\\meter} referred to the '
         'output at no load. This has not been subtracted from any value in '
         'this report.'
         % T.raw((results.get('train_drag_nm')), 3)),
    ]
    # A 'Torque cell zero' note used to sit here, citing the offsets that Test
    # conditions listed. Those offsets were dropped from the report, so the
    # note went with them -- it cross-referenced a table row that no longer
    # exists. The correction itself is unchanged: `correct_zero` still governs
    # whether the offsets are subtracted, the report simply no longer says so.
    out.append(r'\begin{description}')
    for name, text in items:
        if T.MISSING in text:
            continue
        out.append(r'  \item[%s] %s' % (T.escape(name), text))
    out.append(r'\end{description}')
    # A closing 'the complete per-leg and per-point data is provided as CSV'
    # line used to sit here. Scope already opens the report with 'The complete
    # raw data for every test is provided alongside this report', so this
    # repeated it as the last words of the document. Struck from the 1.2.0
    # report by hand and dropped here so the next unit does not carry it back.
    return '\n'.join(out)


# -------------------------------------------------------------------- driver

# LaTeX is full of both '%' (comments) and '{}' (groups), so the template is
# filled by plain token replacement rather than %-formatting or str.format --
# either of those would need half the preamble escaped.
PREAMBLE = r"""% report_generated.tex -- generated by dyno.src.archimedes.report
% Regenerated on every render. To keep commentary, copy this driver to a
% name of your own and edit that, or put the commentary in its own file
% and \input it beside the section it belongs to.

\documentclass[11pt]{article}

\usepackage[letterpaper,margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{siunitx}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}
\hypersetup{linkcolor=blue!50!black}

\DeclareSIUnit\rpm{rpm}
\sisetup{per-mode=symbol,detect-weight=true,detect-family=true,inter-unit-product={}}

\newcommand{\resultfig}[4]{%
  \begin{figure}[H]
    \centering
    \includegraphics[width=#2\linewidth]{#1}
    \caption{#3}
    \label{#4}
  \end{figure}}

\title{\textbf{Dynamometer Test Report: @TITLE@}}
\author{Paladin Engineering@LOGO@}
\date{\today}

\begin{document}
\maketitle

\section{Scope}
This report presents dynamometer measurements made on @TITLE@. It states the
measurements taken, the conditions under which they were taken, and the
methods used. It does not assess the results against any
specification.

The complete raw data for every test is provided alongside this report.

\tableofcontents

"""


def build(pack_dir, cfg, results=None, out_dir=None):
    """Render the report for one analysis pack. Returns the driver's path."""
    pack_dir = os.path.abspath(pack_dir)
    if results is None:
        with open(os.path.join(pack_dir, 'results.json')) as fh:
            results = json.load(fh)
    out_dir = out_dir or os.path.join(pack_dir, 'report')
    sections_dir = os.path.join(out_dir, SECTIONS_DIRNAME)
    pics_dir = os.path.join(out_dir, PICS_DIRNAME)
    os.makedirs(sections_dir, exist_ok=True)
    os.makedirs(pics_dir, exist_ok=True)

    # Drag is a measurement note, and it lives in the efficiency findings
    # rather than in the metrics, so it is lifted out here once.
    results.setdefault('train_drag_nm', _drag_from(results))

    logo_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'utilities', 'assets',
                            'paladinLogo.png')
    logo = ''
    if os.path.isfile(logo_src):
        shutil.copy2(logo_src, os.path.join(pics_dir, 'paladinLogo.png'))
        logo = (r'\hspace{0.6em}%' '\n'
                r'  \raisebox{-0.4\height}{\includegraphics[height=2.2em]'
                r'{pics/paladinLogo.png}}')

    # Static slip and torsional stiffness are OPTIONAL sections, switched by
    # the unit file rather than by editing this list. The 1.2.2 Paladin Test
    # Request asks for both on 1.2.2 and for neither on 1.2.0 or 1.2.1 ('Slip
    # torque: No', 'Stiffness: No' in its test list), so which of them a report
    # carries is a property of the UNIT, the same as its torque cap.
    #
    # Both analyzers run regardless: the pack still carries their figures,
    # CSVs and findings for the analyst, and only the customer-facing section
    # is withheld. Note that `_slip` also carries the breakaway results, so
    # switching it on brings those back with it.
    #
    # On stiffness there is a standing caveat to repeat wherever it is
    # reported: the absorber holds the output with a position loop rather than
    # a ground, so its servo stiffness is in series with the drive's and every
    # K is a LOWER BOUND.
    builders = [
        ('conditions', _conditions),
        ('data_collected', _inventory),
        ('velocity_ramp', _velocity),
        ('efficiency', _efficiency),
    ]
    if cfg.get('report_slip'):
        builders.append(('slip', _slip_short if cfg.get('slip_summary') == 'short'
                         else _slip))
    if cfg.get('report_stiffness'):
        builders.append(('stiffness', _stiffness))
    builders.append(('measurement_notes', _methods))
    written, body_text = [], []
    for name, fn in builders:
        text = fn(results, cfg, pack_dir)
        if not text or not text.strip():
            continue
        with open(os.path.join(sections_dir, f'{name}.tex'), 'w') as fh:
            # Bind the section to ITS driver. Several report roots exist at
            # once -- one per gearbox, plus the mac_motors report -- and an
            # editor that finds the root by scanning for \documentclass will
            # otherwise build a section against another unit's document.
            fh.write('%% !TEX root = ../%s\n\n' % DRIVER_NAME)
            fh.write(text.rstrip() + '\n')
        written.append(name)
        body_text.append(text)

    # A section that is no longer built must not leave its .tex behind. The
    # driver stops \input-ing it, so it renders nothing -- but it sits in
    # sections/ looking current, and the next person to read the folder has no
    # way to tell it from a live one.
    for stale in set(os.listdir(sections_dir)) - {f'{n}.tex' for n in written}:
        if stale.endswith('.tex'):
            os.remove(os.path.join(sections_dir, stale))

    # Sections are built first so only the figures they reference get copied.
    referenced = set(re.findall(r'%s/([^}]+)' % PICS_DIRNAME,
                                '\n'.join(body_text)))
    copied = []
    for src, alias in FIGURES.items():
        if alias not in referenced:
            continue
        s_path = os.path.join(pack_dir, src)
        if os.path.isfile(s_path):
            shutil.copy2(s_path, os.path.join(pics_dir, alias))
            copied.append(alias)
    # A rerun that drops a section must not leave its figure behind.
    for stale in set(os.listdir(pics_dir)) - set(copied) - {'paladinLogo.png'}:
        os.remove(os.path.join(pics_dir, stale))

    title = T.escape(cfg.get('unit_label', 'Archimedes drive'))
    body = [PREAMBLE.replace('@TITLE@', title).replace('@LOGO@', logo)]
    for name in written:
        body.append('%% ---- %s %s' % (name, '-' * max(0, 56 - len(name))))
        body.append(r'\input{%s/%s}' % (SECTIONS_DIRNAME, name))
        body.append('')
    body.append(r'\end{document}')

    driver = os.path.join(out_dir, DRIVER_NAME)
    with open(driver, 'w') as fh:
        fh.write('\n'.join(body))
    return driver, written, copied


def _drag_from(results):
    """Pull the output-referred train drag out of the efficiency metrics."""
    for res in results.get('results') or []:
        if res.get('name') != 'efficiency':
            continue
        for f in res.get('findings') or []:
            if f.get('code') == 'train_drag':
                import re
                m = re.search(r'is\s+([\d.]+)\s*Nm', f.get('message', ''))
                if m:
                    return float(m.group(1))
    return None
