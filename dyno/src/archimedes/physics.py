"""Shared mechanics: plateaus, traverses, power, and the cell zero.

Three facts about this bench drive everything in here.

1. FRAMES. `dut_velocity` and `dut_output_position` are INPUT (motor) frame;
   `load_*` are OUTPUT frame. Measured on the 2026-09-17 no-load sweep: the
   position spans divide at 43.84 and the velocity slope agrees. Despite its
   name `dut_output_position` is not output-referred.

2. EVERY SHUTTLE CONTAINS BOTH DIRECTIONS OF POWER FLOW. The traverse is a
   bipolar sawtooth against a held torque, so on one half-traverse the held
   torque OPPOSES the motion (the input drives the output: forward) and on the
   other it ASSISTS (the output drives the input: back-drive). Averaging a
   segment whole mixes the two and produces a number that is neither. Legs are
   therefore split by travel direction and classified by where the power
   actually flows, not by which plan the file came from.

3. THE OUTPUT CELL CARRIES A ZERO OFFSET. On the 2026-09-17 gravity map the
   direction-independent part of `load_torque` is flat against output angle at
   +1.4 Nm (slow) to +1.7 Nm (fast). A real gravity term would go as sin(angle)
   and would not be flat, so this is residual zero, not the flange -- a tare
   taken while the absorber was holding something (see the bench note on
   standstill tares). It is 28% of the 5 Nm efficiency step, which is the
   difference between a believable low-torque efficiency point and one over
   100%. `cell_offsets` estimates it; the analyzers subtract it when asked and
   say so either way.
"""

import numpy as np

# Fraction of the plateau speed a sample must reach to count as steady. The
# sawtooth spends real time accelerating at both ends of every traverse, and
# those samples are the ones that carry the inertia torque rather than the
# gearbox's.
PLATEAU_FRAC = 0.85

# A leg must hold the plateau this long to be measured. Below it the estimate
# is dominated by whatever the turnaround transient left behind.
MIN_LEG_S = 0.05

# Trimmed off the FRONT of every leg. The controller lands the shaft on speed
# with an impulse -- on the 2026-09-18 back-drive ramps the output cell reaches
# 45 Nm at motion onset and settles to 20 Nm, and the same catch-up repeats at
# every reversal. The speed mask does not exclude it: the impulse is what puts
# the shaft on speed, so it lands inside the band, at the leg's leading edge.
# 0.1 s covers the sharp part while leaving the short legs at the top speeds
# usable; what it costs is reported rather than assumed.
LEG_LEAD_IN_S = 0.1

FORWARD = 'forward'
BACKDRIVE = 'backdrive'


def _smooth(x, n):
    """Centred moving average, edge-padded. Used only to stop mask flicker."""
    n = max(1, int(n) | 1)
    if n < 3 or x.size < n:
        return x
    pad = n // 2
    padded = np.concatenate((np.full(pad, x[0]), x, np.full(pad, x[-1])))
    kern = np.ones(n) / n
    return np.convolve(padded, kern, mode='valid')


def median_filter(x, n):
    """Rolling median, edge-padded.

    Both torque cells on this bench chatter hard while the absorber holds a
    load -- the output cell swings tens of Nm sample to sample during a slip
    hunt and during the stiffness dwells. The trend is real and the excursions
    are not, so anything that reads a torque VALUE off a ramp (a slip torque, a
    stiffness slope) filters first. Anything measuring the chatter itself
    (ripple) must not.
    """
    x = np.asarray(x, dtype=float)
    n = max(1, int(n) | 1)
    if n < 3 or x.size < n:
        return x
    pad = n // 2
    padded = np.concatenate((np.full(pad, x[0]), x, np.full(pad, x[-1])))
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, n),
                     axis=-1)


def samples_for(seg, seconds):
    """How many samples `seconds` is on this segment's clock."""
    dt = seg.dt
    if not (np.isfinite(dt) and dt > 0):
        dt = 1e-3
    return max(1, int(round(seconds / dt)))


def _close_gaps(mask, max_gap):
    """Fill runs of False shorter than `max_gap` that sit between two True runs.

    Velocity noise at the low steps drops the odd sample below the band and
    would otherwise cut one traverse into dozens of fragments -- 322 of them on
    the 30 rpm forward step before this was added. A short gap in the middle of
    a plateau is noise; a gap at either end is the turnaround, and filling that
    would merge two opposing traverses, so only interior gaps are closed.
    """
    mask = np.asarray(mask, dtype=bool)
    if max_gap < 1 or mask.all() or not mask.any():
        return mask
    out = mask.copy()
    # Boundaries of every run of False, as [start, stop) pairs.
    padded = np.concatenate(([True], mask, [True]))
    off = np.flatnonzero(~padded[1:] & padded[:-1])      # True -> False
    on = np.flatnonzero(padded[1:] & ~padded[:-1])       # False -> True
    for a, b in zip(off, on):
        # A gap touching either end is the lead-in or the turnaround that
        # closes the segment, not a dropout inside a plateau.
        if a == 0 or b >= mask.size:
            continue
        if b - a <= max_gap:
            out[a:b] = True
    return out


def plateau_mask(v, frac=PLATEAU_FRAC, target=None, band=0.20, smooth_n=1,
                 gap=0):
    """Samples travelling at the intended constant speed, either sign.

    `target` is the commanded shaft speed in rad/s and is strongly preferred to
    the data-derived reference. The sawtooth overshoots hard at its turnarounds
    -- the 30 rpm forward step peaks at twice its plateau, and the back-driven
    steps reach twenty times theirs -- so a percentile of |v| sits ABOVE the
    speed the segment actually held, and the plateau falls outside its own
    mask. The commanded speed has no such problem: it is what the plateau is.

    Without a target this falls back to the 90th percentile of |v|, which is
    right for a span whose commanded speed is unknown.
    """
    v = np.asarray(v, dtype=float)
    speed = _smooth(np.abs(v), smooth_n)
    finite = np.isfinite(v)
    if target and target > 0:
        m = finite & (speed > (1 - band) * target) & (speed < (1 + band) * target)
    else:
        ref = (np.percentile(speed[finite], 90) if finite.any() else 0.0)
        if not ref:
            return np.zeros_like(speed, dtype=bool)
        m = finite & (speed > frac * ref)
    return _close_gaps(m, gap)


def legs(seg, drive_channel, target=None, band=0.20, min_s=MIN_LEG_S,
         frac=PLATEAU_FRAC, smooth_s=0.02, gap_s=0.05,
         lead_in_s=LEG_LEAD_IN_S):
    """Contiguous constant-speed traverses, as (slice, sign) pairs.

    One leg is one half of the sawtooth: the drive shaft holds speed in one
    direction until the turnaround. Legs are split on travel-direction changes
    as well as on the mask, so a quick turnaround that leaves few sub-threshold
    samples cannot merge two opposing traverses into one -- which would cancel
    the very drag asymmetry the campaign is measuring.
    """
    v = seg[drive_channel]
    dt = seg.dt
    if not (np.isfinite(dt) and dt > 0):
        dt = 1e-3
    n_smooth = int(round(smooth_s / dt))
    m = plateau_mask(v, frac=frac, target=target, band=band,
                     smooth_n=n_smooth, gap=int(round(gap_s / dt)))
    if not m.any():
        return []

    sign = np.sign(_smooth(v, n_smooth))
    cuts = np.flatnonzero((np.diff(m.astype(np.int8)) != 0)
                          | (np.diff(sign) != 0)) + 1
    bounds = np.unique(np.concatenate(([0], cuts, [len(v)])))

    min_n = max(10, int(min_s / dt))
    lead_n = int(round(lead_in_s / dt)) if lead_in_s else 0
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < min_n or not m[a]:
            continue
        # Drop the controller's catch-up impulse at the leading edge, but only
        # where the leg can spare it -- a leg shorter than the trim plus the
        # minimum is kept whole and carries the transient, which is visible in
        # the onset diagnostic rather than silently averaged away.
        if lead_n and (b - a) > (lead_n + min_n):
            a = a + lead_n
        s = np.sign(np.median(v[a:b]))
        if s == 0:
            continue
        # The mask runs on |v| smoothed, so a span whose RAW velocity reverses
        # faster than the smoothing window can pass it while its signed mean is
        # near zero -- the drive judders hard at the low-speed points. Such a
        # span is not a constant-speed leg: anything dividing by its mean speed
        # (creep is speed_out / speed_in) then divides by almost nothing, which
        # is where reported creep maxima of 1e6 percent came from.
        if target and target > 0:
            mean_v = float(np.nanmean(v[a:b]))
            if not np.isfinite(mean_v) or abs(mean_v) < (1 - band) * target:
                continue
        out.append((slice(int(a), int(b)), int(s)))
    return out


def drive_target(rpm_in, direction, ratio):
    """Commanded speed of the DRIVING shaft, rad/s, from the plan's step name.

    Both plans name their steps by INPUT-referred rpm -- the customer asked for
    the back-drive steps to use input-side speeds so the two sweeps line up --
    so the back-drive target is that speed divided by the ratio, because there
    the output is the shaft being commanded.
    """
    if not rpm_in:
        return None
    w_in = rpm_in * 2.0 * np.pi / 60.0
    return w_in if direction == FORWARD else w_in / ratio


def leg_power(seg, sl, t_out_offset=0.0, t_in_offset=0.0):
    """Mean shaft powers and their terms over one leg.

    Torques are averaged before multiplying by the mean speed rather than
    averaging the instantaneous product. On a plateau the speed is constant by
    construction, so the two agree to the noise; the separated form is what
    lets the report quote the torque that produced a given efficiency.
    """
    wi = float(np.nanmean(seg['dut_velocity'][sl]))
    wo = float(np.nanmean(seg['load_velocity'][sl]))
    ti = float(np.nanmean(seg['input_torque'][sl])) - t_in_offset
    to = float(np.nanmean(seg['load_torque'][sl])) - t_out_offset
    return {
        'w_in': wi, 'w_out': wo, 't_in': ti, 't_out': to,
        'p_in': ti * wi, 'p_out': to * wo,
        't_in_sd': float(np.nanstd(seg['input_torque'][sl])),
        't_out_sd': float(np.nanstd(seg['load_torque'][sl])),
        'n': int(sl.stop - sl.start),
    }


def classify_leg(p, ratio):
    """Which way power flowed on this leg, and the efficiency that follows.

    The discriminator is |T_out| against ratio x |T_in|: below it the input is
    the source and the output cell sees what survived the gearbox (forward);
    above it the output is the source and the input cell sees what survived
    coming back (back-drive). Equivalent to comparing the two shaft powers,
    because the speeds are locked at the ratio -- but stated on torques, which
    is the form that says WHY a leg was called one way or the other.

    Efficiency is the ratio of the two mechanical powers, output over input
    when driving forward and input over output when back-driven. It is NOT
    clamped: a value over 1 is a measurement fault (a cell zero, a cell scale,
    a leg that was really coasting) and hiding it would hide the fault.
    """
    p_in, p_out = p['p_in'], p['p_out']
    driven = abs(p['t_out']) < ratio * abs(p['t_in'])
    direction = FORWARD if driven else BACKDRIVE
    src, dst = (p_in, p_out) if driven else (p_out, p_in)
    eta = abs(dst) / abs(src) if abs(src) > 1e-12 else float('nan')
    # Power must flow one way through a passive train. Opposite signs mean both
    # shafts are sourcing or both absorbing, which no gearbox does -- it is a
    # coasting leg or a bad zero, and its efficiency is meaningless.
    coherent = (p_in > 0) == (p_out > 0)
    return {**p, 'direction': direction, 'eta': eta, 'coherent': bool(coherent),
            'loss_w': abs(src) - abs(dst)}


def cell_offsets(spans, ratio):
    """Estimate the zero on each torque cell from the no-load traverses.

    A no-load velocity-ramp leg carries only the train's own drag, and drag
    REVERSES with travel while a zero offset does not. So over a matched pair
    of opposing legs the mean is the offset and the half-difference is the
    Coulomb drag. Both come out of the same arithmetic, and neither needs a
    model of the flange.

    Returns (offsets, detail). `offsets` is {channel: Nm} ready to subtract.
    Missing when there are no no-load traverses to read it from -- the caller
    then runs uncorrected and says so.
    """
    from . import naming

    pos, neg = {'input_torque': [], 'load_torque': []}, {'input_torque': [], 'load_torque': []}
    n_legs = 0
    for span in spans:
        if span.point.kind not in (naming.VELOCITY, naming.GRAVITY):
            continue
        seg = span.seg
        drive = ('dut_velocity' if span.direction == 'forward'
                 else 'load_velocity')
        for sl, sign in legs(seg, drive):
            n_legs += 1
            for ch in ('input_torque', 'load_torque'):
                if not seg.has(ch):
                    continue
                val = float(np.nanmean(seg[ch][sl]))
                (pos if sign > 0 else neg)[ch].append(val)

    offsets, detail = {}, {}
    for ch in ('input_torque', 'load_torque'):
        if not pos[ch] or not neg[ch]:
            continue
        p, n = float(np.median(pos[ch])), float(np.median(neg[ch]))
        offsets[ch] = (p + n) / 2.0          # survives a direction flip: zero
        detail[ch] = {
            'offset_nm': (p + n) / 2.0,
            'drag_nm': abs(p - n) / 2.0,     # reverses with travel: friction
            'n_legs': n_legs,
            'mean_pos': p, 'mean_neg': n,
        }
    return offsets, detail


def robust_pk_pk(x, lo=0.5, hi=99.5):
    """Peak-to-peak with the tails trimmed.

    The customer asks for 'robust pk-pk torque ripple'. A true max-minus-min on
    a 1 kHz cell picks up single-sample spikes from the drive's current loop
    and from the occasional late EtherCAT cycle, neither of which is ripple.
    The 0.5/99.5 percentile span keeps a real ripple peak while dropping those.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float('nan')
    return float(np.percentile(x, hi) - np.percentile(x, lo))


def rpm_of(rad_s):
    return rad_s * 60.0 / (2.0 * np.pi)
