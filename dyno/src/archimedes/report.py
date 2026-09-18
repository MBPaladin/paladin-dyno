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
    'velocity_ramp__no_load_drag.png': 'no_load_torque.png',
    'velocity_ramp__output_travel.png': 'output_travel.png',
    'efficiency__efficiency_vs_torque_forward.png': 'efficiency_forward.png',
    'efficiency__efficiency_vs_torque_backdrive.png': 'efficiency_backdrive.png',
    'efficiency__efficiency_both_directions.png': 'efficiency_both.png',
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


# ----------------------------------------------------------------- sections

def _conditions(results, cfg, pack):
    """Bench, instrumentation, and the corrections applied. Facts only."""
    ratio = results.get('ratio')
    offsets = results.get('cell_offsets') or {}
    out = [r'\section{Test conditions}', '']
    out.append(_para(
        'The drive under test was installed on the Paladin in-house '
        'dynamometer between two servo motors. The input (high-speed) shaft '
        'was coupled to an %s servo motor and the output (low-speed) shaft to '
        'an %s servo motor. Torque was measured by a load cell on each shaft. '
        'Position and velocity were taken from the drive encoders. All '
        'channels were logged at \\SI{1}{\\kilo\\hertz}.'
        % (T.escape(cfg.get('input_motor', 'RTC-0020')),
           T.escape(cfg.get('output_motor', 'RTC-0200')))))

    rows = [
        ['Nominal gear ratio', T.num(cfg.get('ratio_nameplate', 43.88), 4)],
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
    if cap:
        rows.append(['Output torque limit applied during testing',
                     T.si(cap, NM)])
    out.append(_table('Test conditions.', ['Quantity', 'Value'], rows,
                       label='tab:conditions', spec='lr'))

    out.append(_para(
        'The gear ratio in this report is the value measured from the no-load '
        'velocity sweep, taken as the slope of input shaft speed against '
        'output shaft speed. It is used to refer input-side quantities to the '
        'output shaft throughout.'))

    if offsets:
        lines = [
            'A zero offset was measured on each torque cell from the no-load '
            'traverses, as the component of the cell reading that does not '
            'reverse when the direction of travel reverses:']
        vals = ', '.join(
            '%s %s' % (T.tt(k), T.si(v, NM, 3)) for k, v in offsets.items())
        lines.append(vals + '.')
        lines.append(
            'These offsets %s subtracted from the torque values reported in '
            'this document.'
            % ('were' if cfg.get('correct_zero', True) else 'were not'))
        out.append(_para(*lines))
        if _have(pack, 'efficiency__cell_zero_and_drag.png'):
            out.append(_fig('cell_zero.png',
                            'Torque cell zero offset and train running '
                            'friction, separated by their behaviour under a '
                            'reversal of travel direction.', 'cell_zero',
                            width=0.72))
    return '\n'.join(out)


def _inventory(results, cfg, pack):
    """What data exists, and what each run recorded as its stop reason."""
    out = [r'\section{Data collected}', '']
    inv = results.get('inventory') or {}
    rows = []
    for key in sorted(inv):
        kind, _, direction = key.partition('/')
        rows.append([T.escape(naming.KIND_TITLES.get(
                         kind, kind.replace('_', ' ').capitalize())),
                     T.escape(DIRECTION_WORD.get(direction, direction)),
                     str(inv[key])])
    if rows:
        out.append(_table(
            'Measured segments by test and direction. A segment is one '
            'commanded operating point.',
            ['Test', 'Direction', 'Segments'], rows,
            label='tab:inventory', spec='llr'))

    dups = results.get('duplicates') or []
    if dups:
        # Listed here rather than pointed at. This used to refer the reader to
        # `report.txt`, which is the analyst's working file: it carries
        # absolute paths on the bench machine and the internal folder name for
        # the unit, neither of which belongs in a customer deliverable.
        # \sloppypar because a long comma-separated run of \texttt
        # identifiers gives TeX almost no break points and it sets the line
        # 35pt past the margin. Loosening the interword glue for this one
        # paragraph is the standard remedy and affects nothing else.
        out.append(_para(
            r'\begin{sloppypar}',
            'Several tests were recorded across more than one file, where a '
            'run was stopped and resumed. %d operating point%s recorded in '
            'more than one file. In each case the longest recording was used '
            'and the others were set aside. The operating points affected '
            'were %s.'
            % (len(dups), ' was' if len(dups) == 1 else 's were',
               _and_list([T.tt(d['segment']) for d in dups])),
            r'\end{sloppypar}'))
    return '\n'.join(out)


def _velocity(results, cfg, pack):
    rows_csv = _read_csv(pack, 'velocity_ramp__per_step.csv')
    if not rows_csv:
        return ''
    out = [r'\section{Velocity ramp and no-load torque ripple}', '']
    out.append(_para(
        'The drive was commanded through a series of constant-speed steps at '
        'no load, with the shaft not being commanded held at '
        '\\SI{0}{\\newton\\meter}. Steps were \\SI{30}{\\rpm} apart up to '
        '\\SI{300}{\\rpm} and \\SI{300}{\\rpm} apart from there to '
        '\\SI{3600}{\\rpm}, referred to the input shaft. Each step is a '
        'bipolar traverse, so it holds its speed over several separate '
        'constant-speed legs.'))
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
            'measured gear ratio. Measured input speed is given beside the '
            'commanded value in each table.'))
    out.append(_para(
        'Torque ripple is reported as a robust peak-to-peak value: the span '
        'between the 0.5th and 99.5th percentiles of the input torque cell '
        'reading over one constant-speed leg. The percentile span is used in '
        'place of the full minimum-to-maximum range so that isolated single '
        'samples do not set the reported value. Values below are the median '
        'across the legs of each step.'))

    for direction in ('forward', 'backdrive'):
        sel = [r for r in rows_csv if r['direction'] == direction]
        if not sel:
            continue
        body = []
        for r in sorted(sel, key=lambda r: _f(r, 'rpm_cmd') or 0):
            body.append([
                _n(_f(r, 'rpm_cmd')),
                T.num(_f(r, 'rpm_in_meas'), 4),
                T.num(_f(r, 'ripple_nm'), 3),
                T.num(_f(r, 'ripple_lo_nm'), 3) + '--'
                + T.num(_f(r, 'ripple_hi_nm'), 3),
                str(int(_f(r, 'n_legs') or 0)),
                T.num(_f(r, 't_in_nm'), 3),
            ])
        out.append(_table(
            'No-load results, %s. Speeds are at the input shaft. Ripple and '
            'running torque are at the input torque cell.'
            % DIRECTION_WORD[direction],
            [r'Commanded (\si{\rpm})', r'Measured (\si{\rpm})',
             r'Ripple pk--pk (\si{\newton\meter})',
             r'Leg range (\si{\newton\meter})', 'Legs',
             r'Running torque (\si{\newton\meter})'],
            body, label='tab:vel_%s' % direction,
            spec='rrrrrr'))

    if _have(pack, 'velocity_ramp__ripple_vs_speed.png'):
        out.append(_fig('torque_ripple.png',
                        'No-load input torque ripple against commanded input '
                        'speed. Bars span the constant-speed legs of each '
                        'step.', 'ripple'))
    if _have(pack, 'velocity_ramp__speed_tracking.png'):
        out.append(_fig('speed_tracking.png',
                        'Measured input speed against commanded input speed.',
                        'tracking'))
    if _have(pack, 'velocity_ramp__no_load_drag.png'):
        out.append(_fig('no_load_torque.png',
                        'Running torque at no load against speed, from both '
                        'torque cells.', 'noload'))
    if _have(pack, 'velocity_ramp__output_travel.png'):
        out.append(_para(
            'The Archimedes drive has an internal end stop on the output '
            'shaft. Output travel was monitored throughout and a software '
            'position window was applied; Figure~\\ref{fig:travel} shows the '
            'travel recorded during each step against that window.'))
        out.append(_fig('output_travel.png',
                        'Output shaft travel during each velocity step, '
                        'referenced to the start of the step. Dashed lines '
                        'mark the position window in force.', 'travel'))
    return '\n'.join(out)


def _efficiency(results, cfg, pack):
    pts = _read_csv(pack, 'efficiency__per_point.csv')
    if not pts:
        return ''
    out = [r'\section{Efficiency}', '']
    speeds = sorted({_f(r, 'rpm') for r in pts if _f(r, 'rpm') is not None})
    step = cfg.get('torque_step_nm', 5.0)
    out.append(_para(
        'Efficiency was measured at input speeds of %s, in output torque steps '
        'of \\SI{%s}{\\newton\\meter}. At each operating point the input shaft '
        'was commanded through a bipolar constant-speed traverse while the '
        'output shaft held a constant torque.'
        % (_and_list([_rpm(s) for s in speeds]),
           T.raw(int(step) if float(step).is_integer() else step, 2))))
    out.append(_para(
        'Because the traverse reverses, the held output torque opposes the '
        'motion on one half of each traverse and assists it on the other. '
        'Power therefore flows from input to output on one half and from '
        'output to input on the other. Each half-traverse was measured '
        'separately and classified by the direction in which power flowed. '
        'Efficiency is the ratio of the two measured shaft powers, output '
        'over input where the input was driving and input over output where '
        'the output was driving. Shaft power is the product of the measured '
        'torque and the measured speed on that shaft.'))
    out.append(_para(
        'Values are the median over the repeat legs at each point. Points '
        'where the two shaft powers did not share a sign were excluded, as no '
        'power was being transmitted through the drive on those legs.'))

    combos = []
    for flow in ('forward', 'backdrive'):
        for sweep in ('forward', 'backdrive'):
            if any(r['flow'] == flow and r.get('span_direction') == sweep
                   for r in pts):
                combos.append((flow, sweep))
    for direction, sweep in combos:
        sel = [r for r in pts
               if r['flow'] == direction and r.get('span_direction') == sweep]
        tag = direction if direction == sweep else f'{direction}_from_{sweep}'
        if direction != sweep:
            out.append(_para(
                'The values in Table~\\ref{tab:eff_%s} were measured on the '
                'half-traverses of the %s sweep during which the %s shaft was '
                'driving. The control configuration was unchanged throughout '
                'that sweep: the %s shaft held the traverse under position '
                'control and the %s shaft held a constant torque.'
                % (tag, sweep,
                   'output' if direction == 'backdrive' else 'input',
                   'input' if sweep == 'forward' else 'output',
                   'output' if sweep == 'forward' else 'input')))
        levels = sorted({_f(r, 't_cmd_nm') for r in sel})
        by = {(_f(r, 'rpm'), _f(r, 't_cmd_nm')): r for r in sel}
        body = []
        for lv in levels:
            cells = [_n(lv)]
            for sp in speeds:
                r = by.get((sp, lv))
                cells.append(T.num(_f(r, 'eta') * 100, 3)
                             if r and _f(r, 'eta') is not None else T.MISSING)
            body.append(cells)
        out.append(_table(
            'Efficiency (\\si{\\percent}), %s, by commanded output torque and '
            'input speed.%s'
            % (DIRECTION_WORD[direction],
               '' if direction == sweep else
               ' Measured on the half-traverses of the %s sweep during which '
               'the %s shaft was driving.'
               % (sweep, 'output' if direction == 'backdrive' else 'input')),
            [r'Output torque (\si{\newton\meter})']
            + [_rpm(s) for s in speeds],
            body, label='tab:eff_%s' % tag,
            spec='r' * (len(speeds) + 1)))

    if _have(pack, 'efficiency__efficiency_vs_torque_forward.png'):
        out.append(_fig('efficiency_forward.png',
                        'Efficiency against measured output torque, forward '
                        'driving. Bars span the repeat legs at each point.',
                        'eff_fwd'))
    if _have(pack, 'efficiency__efficiency_both_directions.png'):
        out.append(_para(
            'Figure~\\ref{fig:eff_both} shows the two power-flow directions '
            'measured from the same traverses, at each speed.'))
        out.append(_fig('efficiency_both.png',
                        'Efficiency against measured output torque for both '
                        'directions of power flow, by input speed.',
                        'eff_both', width=1.0))
    if _have(pack, 'efficiency__loss_vs_torque_forward.png'):
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

    out.append(_speed_holding(cfg, pack))
    out.append(_creep(cfg, pack))
    return '\n'.join(x for x in out if x)


def _speed_holding(cfg, pack):
    """How much of each segment was actually spent at the commanded speed.

    Stated because it changes what the efficiency table means at the low-speed
    points: there the shaft does not turn steadily, it stands still and catches
    up in bursts, and the measurements come from the bursts. The numbers are
    given plainly and the reader draws their own conclusion.
    """
    rows = _read_csv(pack, 'efficiency__speed_coverage.csv')
    if not rows:
        return ''
    by = {}
    for r in rows:
        by.setdefault((r.get('direction', ''), _f(r, 'rpm_cmd')), []).append(r)
    body = []
    for (d, rpm), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        med = lambda k: float(np.median(  # noqa: E731
            [_f(r, k) for r in rs if _f(r, k) is not None]))
        body.append([
            T.escape(DIRECTION_WORD.get(d, d)), _n(rpm), str(len(rs)),
            T.num(med('coverage_frac') * 100, 2),
            T.num(med('frac_at_rest') * 100, 2),
            # Whole rpm: these are medians of measured speeds, and a third
            # significant figure renders 1430 rpm as 1.43e+03 in the table.
            T.num(round(med('rpm_in_during_legs'))),
            T.num(round(med('rpm_in_span_average'))),
        ])
    out = [r'\subsection{Speed holding}', '']
    out.append(_para(
        'The efficiency values above are taken from the samples at the '
        'commanded speed. The table below gives how much of each segment that '
        'was, how much of it the shaft spent at rest, the input speed measured '
        'over the on-speed samples, and the input speed averaged over the '
        'whole segment. The span average is computed from shaft travel divided '
        'by segment duration, so it includes the stationary intervals.'))
    out.append(_table(
        'Speed holding by test direction and commanded speed. Median across '
        'the torque levels measured at each combination.',
        ['Direction', r'Commanded (\si{\rpm})', 'Points',
         r'On speed (\si{\percent} of span)',
         r'At rest (\si{\percent} of span)',
         r'Input speed on speed (\si{\rpm})',
         r'Input speed, span average (\si{\rpm})'],
        body, label='tab:holding', spec='llrrrrr'))
    out.append(_para(
        'Each segment carries a fixed lead-in, a settling interval at the '
        'commanded torque, and a dwell at each end of the traverse, during '
        'which the shaft is stationary. Those intervals do not shorten as the '
        'commanded speed rises, while the traverse itself does, so the '
        'proportion of a segment spent at rest increases with speed. Where the '
        'shaft was moving, the measured input speed is close to the commanded '
        'value, with the exception noted below.'))
    out.append(_para(
        'In the back-driven segments at \\SI{20}{\rpm} the input shaft turned '
        'at a small fraction of the speed the gear ratio implies for the '
        'output speed being commanded. The efficiency and creep values for '
        'those points are reported as measured.'))
    return '\n'.join(out)


def _creep(cfg, pack):
    rows = _read_csv(pack, 'efficiency__creep.csv')
    if not rows:
        return ''
    out = [r'\subsection{Creep ratio}', '']
    out.append(_para(
        'Creep ratio is reported as $1 - (r \\cdot \\omega_{\\mathrm{out}}) / '
        '\\omega_{\\mathrm{in}}$, where $r$ is the gear ratio measured from '
        'the no-load sweep. It is evaluated over the constant-speed legs of '
        'each operating point. The test request specifies that creep ratio '
        'should stay below \\SI{5}{\\percent}.'))
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
    out.append(_table(
        'Creep ratio by input speed, across the output torque levels measured '
        'at that speed.',
        [r'Input speed (\si{\rpm})', 'Levels',
         r'Min (\si{\percent})', r'Max (\si{\percent})',
         r'At torque (\si{\newton\meter})',
         r'Above \SI{5}{\percent}'],
        body, label='tab:creep', spec='rrrrrr'))
    if _have(pack, 'efficiency__creep_vs_torque.png'):
        out.append(_fig('creep.png',
                        'Creep ratio against commanded output torque, by '
                        'input speed. The dashed line marks the '
                        '\\SI{5}{\\percent} value given in the test request.',
                        'creep'))
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


def _stiffness(results, cfg, pack):
    rows = _read_csv(pack, 'stiffness__fits.csv')
    if not rows:
        return ''
    out = [r'\section{Torsional stiffness}', '']
    out.append(_para(
        'Torque was ramped to a target, held, and reversed. Wind-up is '
        'reported at the output shaft as the difference between the output '
        'shaft angle and the input shaft angle divided by the measured no-load '
        'gear ratio. Stiffness is the slope of output torque against that '
        'wind-up, fitted over the middle of the torque range and excluding the '
        'reversal. Lost motion is the width of the loop at zero torque.'))
    out.append(_para(
        'During these tests the output shaft was held by the absorber motor '
        'under position control rather than by a fixed mechanical ground. The '
        'values below are measured across the drive and its holding servo in '
        'series.'))

    body = []
    for r in rows:
        k = _f(r, 'k_nm_per_rad')
        body.append([
            T.tt(r['segment']),
            T.num(_f(r, 'target_pct'), 2),
            T.num(_f(r, 'torque_peak_nm'), 3),
            T.num(_f(r, 'windup_pk_pk_rad') * 1e3
                  if _f(r, 'windup_pk_pk_rad') is not None else None, 3),
            # Rounded to whole Nm/rad: the fit does not support a fraction of
            # one, and four significant figures pushes it into exponent form.
            T.num(round(k) if k is not None else None),
            T.num(_f(r, 'k_r2'), 3),
            T.num(_f(r, 'hysteresis_rad') * 1e3
                  if _f(r, 'hysteresis_rad') is not None else None, 3),
            T.num(_f(r, 'ratio_tracking_r'), 5),
        ])
    out.append(_table(
        'Stiffness ramps. The nominal target is the fraction of static slip '
        'torque the ramp was configured for, using the slip value available '
        'when the test was generated; the peak torque column gives what was '
        'recorded. Shaft tracking is the correlation between the two shaft '
        'angles over the ramp.',
        ['Segment', r'Target (\si{\percent})',
         r'Peak torque (\si{\newton\meter})',
         r'Wind-up (\si{\milli\radian})',
         r'$K$ (\si{\newton\meter\per\radian})', '$R^2$',
         r'Lost motion (\si{\milli\radian})', 'Tracking'],
        body, label='tab:stiff', spec='lrrrrrrr'))

    loose = [r for r in rows if (_f(r, 'ratio_tracking_r') or 1.0) < 0.99]
    if loose:
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
    if _have(pack, 'stiffness__tracking.png'):
        out.append(_fig('stiffness_tracking.png',
                        'Shaft angles during each ramp, with the input '
                        'referred to the output through the measured ratio. '
                        'Wind-up is the difference between them, shown on the '
                        'right-hand axis.', 'stiff_track', width=1.0))
    if _have(pack, 'stiffness__stiffness_vs_torque.png'):
        out.append(_fig('stiffness_vs_torque.png',
                        'Local slope of the wind-up curve, in torque bins.',
                        'stiff_vs_t'))
    return '\n'.join(out)


def _methods(results, cfg, pack):
    out = [r'\section{Measurement notes}', '']
    out.append(_para(
        'The following apply to the values in this report.'))
    items = [
        ('Reference frames',
         'Input shaft quantities are in the input (motor) frame and output '
         'shaft quantities are in the output frame. Input-side quantities '
         'referred to the output are divided by the measured no-load gear '
         'ratio, and are identified as such wherever they appear.'),
        ('Constant-speed selection',
         'Measurements taken during a traverse use only the samples at the '
         'commanded constant speed. The commanded speed is used as the '
         'reference rather than a value derived from the data, because the '
         'traverse overshoots at each reversal.'),
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
    offsets = results.get('cell_offsets') or {}
    if offsets and cfg.get('correct_zero', True):
        items.insert(3, (
            'Torque cell zero',
            'The torque cell zero offsets given in Table~\\ref{tab:conditions} '
            'have been subtracted from the torque values in this report. They '
            'were measured as the component of each cell reading that does not '
            'reverse with the direction of travel.'))
    out.append(r'\begin{description}')
    for name, text in items:
        if T.MISSING in text:
            continue
        out.append(r'  \item[%s] %s' % (T.escape(name), text))
    out.append(r'\end{description}')
    out.append('')
    out.append(_para(
        'The complete per-leg and per-point data behind every table and figure '
        'is provided as CSV alongside this report.'))
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
methods used to reduce them. It does not assess the results against any
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

    builders = [
        ('conditions', _conditions),
        ('data_collected', _inventory),
        ('velocity_ramp', _velocity),
        ('efficiency', _efficiency),
        ('slip', _slip),
        ('stiffness', _stiffness),
        ('measurement_notes', _methods),
    ]
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
