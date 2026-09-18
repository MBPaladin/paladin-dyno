"""Static slip torque, and the breakaway (stiction) ramps run beside it.

Customer ask (1.2.2 Paladin Test Request, 'Static slip test'):

    Gradually increase torque until the drive starts to slip
    Forward driving (low speed side lock)
    Back-driving (high speed side lock)

'Which side is locked' is what the customer means by direction here, and it is
the same thing dataset.direction_of answers: the ramped shaft is the driver and
the held one is the lock. Forward locks the output (low-speed side); back-drive
locks the input.

Two different events are measured by two different plans, and they must not be
confused -- the generator split them apart for exactly this reason:

    BREAKAWAY (STICT, RAMP)  the far shaft is at 0 Nm, so nothing grounds the
                             train. The ramp fights friction only, and the
                             whole train breaks free and turns TOGETHER at the
                             ratio. This is drivetrain stiction.
    SLIP (SLIP)              the far shaft is position-held, so the train is
                             grounded and the ramp winds the traction contact
                             up until it lets go. The input runs away while the
                             output stays put.

The discriminator between them is therefore not the torque -- it is whether the
two shafts stayed coupled. That is what `_ratio_break` measures, and it is why
a breakaway ramp is never reported as a slip torque.
"""

import numpy as np

from .. import dataset, naming, physics, plotting
from ..result import Result

# Gross slip is detected on the RATE of decoupling, not on how much has
# accumulated. Accumulated decoupling cannot separate the two things this ramp
# does: the contact creeps steadily under load all the way up -- the
# 2026-09-17 hunt had drifted 0.067 output rad before anything let go -- and
# then lets go all at once. A displacement threshold low enough to catch the
# release fires on the creep long before it, and reports a slip torque taken
# from halfway up the ramp.
#
# 0.2 output rad/s sits an order of magnitude above the creep rate of a static
# ramp and an order below the release (3.5 output rad/s on 2026-09-17).
SLIP_RATE_RAD_S = 0.2
# Fallback for a ramp that walks away without a clean release: far above any
# creep a static ramp can legitimately accumulate.
SLIP_BREAK_RAD = 0.5
# Speed on the RAMPED shaft that counts as 'moving' for a breakaway detection.
# The stiction ramps break free and immediately stall against the far shaft's
# drag, so they never reach a large speed -- the 2026-09-17 ramps peak under
# 0.5 rad/s and one of them under 0.3.
#
# It has to be the ramped shaft, in the ramped shaft's own frame, and there is
# one threshold per direction because the two shafts are 43:1 apart. Reading
# the INPUT either way looks tempting -- it is the same motion amplified, so it
# is the more sensitive detector -- and that is exactly the problem
# back-driving: 0.1 rad/s at the input is 0.002 rad/s at the output, which is
# wind-up, not a train that has broken free. On the 2026-09-18 back-drive
# ramps it called BRK_P2 and BRK_P3 breakaways at ~20 Nm when the output had
# moved 0.005 rad in total and the bench's own detector never fired.
MOVE_RAD_S = 0.1
# Back-driving, on the output. The bench's own RampBreak detector for these
# plans uses this figure (BWD_BREAKAWAY_V_THRESH_RAD_S in the generator), so
# our reading and the controller's `breakaway_torque` marker agree about what
# counts as motion.
MOVE_OUT_RAD_S = 0.05

# Window for the median filter applied before any torque is read off a ramp.
# The output cell swings +/-50 Nm sample to sample during a slip hunt; its
# trend is real and its excursions are not, and an unfiltered peak reports the
# worst noise sample as the drive's slip torque.
SMOOTH_S = 0.1


def analyze(ds, cfg):
    res = Result('slip', 'Static slip torque and breakaway')
    spans = ds.select(kind=(naming.SLIP, naming.BREAKAWAY))
    if not spans:
        res.add('warn', 'no_data', 'no slip or breakaway ramps in this campaign')
        return res

    ratio = cfg['ratio']
    events = [_ramp_event(s, ratio, cfg) for s in spans]
    _number(events)
    res.tables.append(('slip__events', plotting.csv(
        [[e.get(k) for k in _COLS] for e in events], _COLS)))

    # One figure per direction rather than one for the campaign. With the
    # back-drive ramps in, a single grid is twelve panels and comes out three
    # times taller than it is wide, which at \linewidth in the report is a
    # page-and-a-half of unreadable thumbnails. Split, each figure is a page.
    for direction in (dataset.FORWARD, dataset.BACKDRIVE):
        sel = sorted(((sp, ev) for sp, ev in zip(spans, events)
                      if sp.direction == direction),
                     key=lambda p: (p[1]['kind'], p[1]['ramp_no']))
        if not sel:
            continue
        res.figures.append((f'ramp_traces_{direction}',
                            _fig_traces(sel, ratio, cfg, direction)))
        if any(ev['event'] == 'slip' for _, ev in sel):
            res.figures.append((f'slip_detail_{direction}',
                                _fig_slip_detail(sel, ratio, cfg, direction)))
    _findings(res, events, cfg)
    res.metrics['events'] = {f'{e["direction"]} {e["ramp"]}': {
        'event': e['event'], 'slip_torque_out_nm': e['t_out_at_event_nm'],
        'breakaway_in_nm': e['t_in_at_event_nm']} for e in events}
    return res


_COLS = ['ramp', 'ramp_no', 'segment', 'kind', 'lock_side', 'far_shaft_mode', 'direction',
         'source', 'event',
         't_event_s', 't_out_at_event_nm', 't_in_at_event_nm',
         't_out_peak_nm', 't_in_peak_nm', 'controller_breakaway_nm',
         'decouple_rad', 'driven_travel_out_rad', 'held_travel_rad']


def _number(events):
    """Give every ramp a label that is unique within its test and direction.

    The segment ID is not unique across a campaign and cannot be made so: every
    fresh launch of the back-drive slip plan starts its attempts at ramp 1, so
    the six runs on 2026-09-18 logged `SLIP_P1` twice and `SLIP_N1` four times.
    Those are six separate measurements, and a table that labels four rows
    identically is unreadable -- so the customer-facing label is an ordinal
    within the (test, direction) group and the raw ID and source file stay
    beside it in the CSV.

    Numbered in the order the ramps were recorded, which is `source` order: the
    per-run folders sort by the time they were written.
    """
    groups = {}
    for ev in sorted(events, key=lambda e: (e['source'], e['segment'])):
        key = (ev['kind'], ev['direction'])
        groups[key] = groups.get(key, 0) + 1
        word = 'Slip' if ev['kind'] == naming.SLIP else 'Breakaway'
        ev['ramp_no'] = groups[key]
        ev['ramp'] = f'{word} ramp {groups[key]}'
    # A lone ramp needs no ordinal -- 'Slip ramp 1' of one reads as a promise
    # of a ramp 2 that does not exist.
    for ev in events:
        if groups[(ev['kind'], ev['direction'])] == 1:
            ev['ramp'] = ev['ramp'].rsplit(' ', 1)[0]


def _ramp_event(span, ratio, cfg):
    """Find where this ramp let go, and what the cells read at that moment."""
    seg = span.seg
    t = seg['time'] - seg['time'][0]
    n_smooth = physics.samples_for(seg, SMOOTH_S)
    ti = physics.median_filter(seg['input_torque'], n_smooth)
    to = physics.median_filter(seg['load_torque'], n_smooth)
    # The ramped shaft, in its own frame: input forward, output back-driving.
    driven_w, move_floor = ((seg['dut_velocity'], MOVE_RAD_S)
                            if span.direction == 'forward'
                            else (seg['load_velocity'], MOVE_OUT_RAD_S))
    dp = seg['dut_output_position'] / ratio        # output-referred input angle
    lp = seg['load_position']
    walk = (dp - dp[0]) - (lp - lp[0])
    decouple = np.abs(walk)
    # Decoupling rate, output rad/s, smoothed over the same window. This is the
    # controller's own `ratio_slip` quantity, recomputed here so a log that
    # predates that channel still gets a slip detection.
    rate = np.abs(physics.median_filter(np.gradient(walk, t), n_smooth))

    far_position = seg.is_active('load_position_command'
                                 if span.direction == 'forward'
                                 else 'dut_position_command')
    row = {
        'segment': span.point.raw, 'kind': span.kind,
        'lock_side': ('output (low speed)' if span.direction == 'forward'
                      else 'input (high speed)'),
        # Position on the far shaft grounds the train; torque at 0 Nm grounds
        # nothing. Stating which is what separates a slip ramp from a
        # breakaway ramp, and the table must not imply a lock that was not
        # there.
        'far_shaft_mode': ('position-held' if far_position
                           else 'free at 0 Nm'),
        'direction': span.direction, 'source': span.source,
        'event': 'none', 't_event_s': None,
        't_out_at_event_nm': None, 't_in_at_event_nm': None,
        't_out_peak_nm': float(np.nanmax(np.abs(to))),
        't_in_peak_nm': float(np.nanmax(np.abs(ti))),
        'controller_breakaway_nm': _controller_value(seg),
        'decouple_rad': float(np.nanmax(decouple)),
        'driven_travel_out_rad': float(np.ptp(dp)),
        'held_travel_rad': float(np.ptp(lp)),
    }

    # Slip first: it is the stronger claim and the one that invalidates a
    # breakaway reading taken from the same ramp.
    idx = _first_true(rate > SLIP_RATE_RAD_S)
    if idx is None:
        idx = _first_true(decouple > SLIP_BREAK_RAD)
    if idx is not None:
        row.update(event='slip', t_event_s=float(t[idx]),
                   t_out_at_event_nm=float(_peak_before(to, idx)),
                   t_in_at_event_nm=float(_peak_before(ti, idx)))
        return row

    # Otherwise: did the train turn together? That is breakaway.
    idx = _first_true(np.abs(driven_w) > move_floor)
    if idx is not None:
        row.update(event='breakaway', t_event_s=float(t[idx]),
                   t_out_at_event_nm=float(_peak_before(to, idx)),
                   t_in_at_event_nm=float(_peak_before(ti, idx)))
    return row


def _first_true(mask, debounce=50):
    """First index where `mask` holds for `debounce` consecutive samples.

    A bare first-True lands on a single noise sample, which on a 1 kHz cell is
    guaranteed to happen before the real event and would report a slip torque
    far below the truth.
    """
    if not mask.any():
        return None
    run = np.convolve(mask.astype(int), np.ones(debounce, int), mode='valid')
    hits = np.flatnonzero(run == debounce)
    return int(hits[0]) if hits.size else None


def _peak_before(x, idx):
    """Largest magnitude the cell sustained up to the event, signed.

    Reads from an already median-filtered trace, so 'peak' means the highest
    the cell actually held rather than its worst single excursion. The torque
    AT the event sample would understate the drive: the contact lets go at the
    peak and the cell has begun to fall by the time the debounce is satisfied.
    """
    seg = x[:max(idx, 1)]
    seg = seg[np.isfinite(seg)]
    if not seg.size:
        return float('nan')
    return seg[np.argmax(np.abs(seg))]


def _controller_value(seg):
    """The controller's own breakaway detection, if it fired on this ramp.

    Logged as a channel that is NaN everywhere except the single sample where
    RampBreak released, so it is one number, not a trace.
    """
    if not seg.has('breakaway_torque'):
        return None
    v = seg['breakaway_torque']
    v = v[np.isfinite(v)]
    return float(v[0]) if v.size else None


def _grid_shape(n):
    """rows, cols for `n` panels, kept near a printable page's proportions.

    Two columns up to six panels, three beyond -- a twelve-panel grid two
    across is 3:1 tall and unreadable once the report scales it to the text
    width.
    """
    cols = 1 if n == 1 else (2 if n <= 6 else 3)
    return int(np.ceil(n / cols)), cols


def _fig_traces(sel, ratio, cfg, direction):
    """Every ramp, torque against time, with its detected event marked."""
    n = len(sel)
    rows, cols = _grid_shape(n)
    fig, axes = plotting.grid_figure(
        rows, cols, f'{cfg["unit_label"]}  --  slip and breakaway ramps, '
        f'{plotting.DIR_LABEL[direction]}',
        'output cell (left axis) and input cell referred to the output '
        '(right); the marker is where the shafts decoupled')
    for ax, (span, ev) in zip(axes.ravel(), sel):
        seg = span.seg
        t = seg['time'] - seg['time'][0]
        ax.plot(t, seg['load_torque'], lw=0.9, color=plotting.FORWARD_C,
                label='output cell')
        ax.plot(t, seg['input_torque'] * ratio, lw=0.9, color='#7f7f7f',
                label='input cell x ratio')
        if ev['t_event_s'] is not None:
            ax.axvline(ev['t_event_s'], ls='--', lw=1.2,
                       color=plotting.BACKDRIVE_C)
            ax.annotate(f'{ev["event"]}\n{ev["t_out_at_event_nm"]:+.1f} Nm out',
                        xy=(ev['t_event_s'], 0), fontsize=8,
                        color=plotting.BACKDRIVE_C,
                        xytext=(4, 4), textcoords='offset points')
        # Two short lines, not one long one. At three columns a title
        # carrying the segment ID and the full far-shaft description runs into
        # its neighbours and every panel in the grid becomes unreadable; the
        # ID is in the CSV and the lock side is constant across the figure, so
        # what is left is the ramp and how the far shaft was held.
        ax.set_title(f'{ev["ramp"]}\nfar shaft {ev["far_shaft_mode"]}',
                     fontsize=9)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('output-referred torque (Nm)')
        ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return plotting.finish_grid(fig)


def _fig_slip_detail(sel, ratio, cfg, direction):
    """For the ramps that slipped: torque and decoupling on one time base."""
    sel = [(s, e) for s, e in sel if e['event'] == 'slip']
    rows, cols = _grid_shape(len(sel))
    fig, axes = plotting.grid_figure(
        rows, cols, f'{cfg["unit_label"]}  --  static slip detail, '
        f'{plotting.DIR_LABEL[direction]}',
        'the contact lets go where the two shafts stop tracking at the ratio')
    for ax, (span, ev) in zip(axes.ravel(), sel):
        seg = span.seg
        t = seg['time'] - seg['time'][0]
        dp = seg['dut_output_position'] / ratio
        lp = seg['load_position']
        decouple = (dp - dp[0]) - (lp - lp[0])
        ax.plot(t, seg['load_torque'], lw=1.0, color=plotting.FORWARD_C,
                label='output torque (Nm)')
        ax.set_ylabel('output torque (Nm)')
        ax.set_xlabel('time (s)')
        twin = ax.twinx()
        twin.plot(t, decouple, lw=1.0, color=plotting.BACKDRIVE_C,
                  label='decoupling (rad, output-referred)')
        twin.set_ylabel('input-minus-output angle (rad)')
        twin.axhline(SLIP_BREAK_RAD, ls=':', lw=1, color=plotting.BACKDRIVE_C)
        if ev['t_event_s'] is not None:
            ax.axvline(ev['t_event_s'], ls='--', lw=1.2, color='#444444')
        ax.set_title(
            f'{ev["ramp"]}: slip at {ev["t_out_at_event_nm"]:+.1f} Nm output\n'
            f'({ev["t_in_at_event_nm"] * ratio:+.1f} Nm output-referred '
            'on the input cell)', fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
        twin.legend(loc='lower right', fontsize=8)
    for ax in axes.ravel()[len(sel):]:
        ax.set_visible(False)
    return plotting.finish_grid(fig)


def _findings(res, events, cfg):
    slips = [e for e in events if e['event'] == 'slip']
    brks = [e for e in events if e['event'] == 'breakaway']
    none = [e for e in events if e['event'] == 'none']

    for direction in ('forward', 'backdrive'):
        sel = [e for e in slips if e['direction'] == direction]
        if not sel:
            continue
        vals = [abs(e['t_out_at_event_nm']) for e in sel]
        res.add('info', f'slip_{direction}',
                f'{direction} static slip ({sel[0]["lock_side"]} locked): '
                f'{np.median(vals):.1f} Nm at the output over {len(vals)} '
                f'ramp(s)' + (f', spread {np.ptp(vals):.1f} Nm'
                              if len(vals) > 1 else ''))
        rated = cfg.get('rated_torque_nm')
        if rated and np.median(vals) < 0.8 * rated:
            res.add('warn', 'slip_below_rating',
                    f'{direction} slip torque {np.median(vals):.1f} Nm is '
                    f'{np.median(vals) / rated * 100:.0f}% of the '
                    f'{rated:g} Nm rating for this unit')

    # Breakaway is reported per direction and in the RAMPED shaft's own frame.
    # The two are not interchangeable and pooling them is nonsense: the forward
    # ramps push the input and read tenths of a Nm, the back-drive ramps push
    # the output and read tens, so one median over both came out at 0.152 Nm
    # "worst 21.554" -- two different quantities in one sentence.
    for direction in ('forward', 'backdrive'):
        sel = [e for e in brks if e['direction'] == direction]
        if not sel:
            continue
        # `controller_breakaway_nm` is the command value on the RAMPED motor at
        # the sample the bench's own detector fired, so it is already in the
        # right frame for its own ramp -- input Nm forward, output Nm
        # back-driving. That is the number to quote where it exists; ours is
        # the cross-check.
        cell = 't_in_at_event_nm' if direction == 'forward' else 't_out_at_event_nm'
        where = ('at the input' if direction == 'forward'
                 else 'at the output')
        vals = [abs(e['controller_breakaway_nm'])
                if e['controller_breakaway_nm'] is not None
                else abs(e[cell]) for e in sel]
        res.add('info', f'breakaway_{direction}',
                f'{direction} drivetrain breakaway (stiction) {where}: '
                f'{np.median(vals):.3f} Nm over {len(vals)} ramp(s), worst '
                f'{np.max(vals):.3f} Nm -- the train turned together, the '
                'contact never let go')
        ours = [abs(e[cell]) for e in sel
                if e[cell] is not None and np.isfinite(e[cell])]
        if ours:
            res.add('info', f'breakaway_crosscheck_{direction}',
                    f'same ramps re-detected from the logged velocity ({where}): '
                    + ', '.join(f'{c:.3f}' for c in ours) + ' Nm')
    if none:
        # A ramp that reached its ceiling without an event is a measurement,
        # not a gap: it says the train did not let go below the torque the ramp
        # got to. Quoting that ceiling is the whole content of the result, so
        # it goes in the finding rather than only in the CSV.
        for direction in ('forward', 'backdrive'):
            sel = [e for e in none if e['direction'] == direction]
            if not sel:
                continue
            cell = ('t_in_peak_nm' if direction == 'forward'
                    else 't_out_peak_nm')
            where = 'input' if direction == 'forward' else 'output'
            res.add('warn', f'no_event_{direction}',
                    f'{len(sel)} {direction} ramp(s) reached their ceiling '
                    'without the shafts decoupling or the train turning: '
                    + ', '.join(
                        f'{e["ramp"]} (to {abs(e[cell]):.1f} Nm at the '
                        f'{where} cell)' for e in sel))

    missing = {'forward', 'backdrive'} - {e['direction'] for e in slips}
    if missing:
        res.add('warn', 'direction_missing',
                'no static slip measured for: ' + ', '.join(sorted(missing)))

    res.summary = (f'{len(events)} ramp(s): {len(slips)} slipped, '
                   f'{len(brks)} broke away without slipping, '
                   f'{len(none)} did neither')
