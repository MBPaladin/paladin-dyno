"""Test-side inertia from constant-acceleration ramp pairs.

Written for the sawtooth inertia test shape: a handful of behaviors, each
sweeping the shaft up and back down at one constant acceleration, repeated a
few times.

    SEG2   4x [ramp up at +alpha_1] [plateau] [ramp down at -alpha_1]
    SEG3   4x  ...                  at +-alpha_2
    ...

Nothing below keys off those names. Every logged span is scanned for
constant-alpha ramp branches, so a re-ordered or re-named test still analyzes
correctly, and the accelerations are recovered by clustering what was measured
rather than by reading the recipe.

The one estimator
-----------------
Inside one repeat, difference the up and down branches at matched speed:

    C_up(w) - C_dn(w) = (alpha_up - alpha_dn) * J

The loss curve -- Coulomb, viscous, windage, whatever shape it really has --
appears identically in both branches and cancels exactly. So this needs no loss
model, no loss-estimation ramps, and no tare: a cell zero offset is common to
both branches and subtracts out with everything else. That is the whole reason
this analysis is short. An earlier version of it fitted loss(w) from a set of
constant-speed dwells and used it as a second estimator; the dwells cost most of
the test's duration and the cross-check never moved the answer, so both are gone.

What the test shape has to deliver for that to hold:

1. The two branches of a pair are ~1 s apart, so there is no room for thermal
   drift or cell-zero movement to enter between them. This is what makes the
   tare unnecessary rather than merely convenient.
2. Alpha is 100-660 rad/s^2 rather than single digits, so the inertial torque
   J*alpha is ~1 Nm against a ~30 mN.m zero uncertainty. The slow-ramp failure
   mode -- where zero drift over the up/down time gap swamps J*alpha -- is gone
   by construction.
3. The ramps really are constant-acceleration. A jerk-limited or curved ramp
   has no single alpha to divide by, and the per-branch polynomial fit would
   quietly absorb the curvature into C(w). `const_acc_tol` checks this per
   branch instead of assuming it, and drops the branches that fail.
4. Each acceleration is repeated, and several accelerations are run. Repeats
   give the run-to-run spread; the disagreement BETWEEN accelerations is the
   systematic floor, and it is quoted as-is -- not divided by sqrt(n), because
   a systematic does not average down. See the pooling note in `run`.
   The repeats are not only a spread to report, they are the sanity check on
   whether the pooled mean means anything at all: two layers of averaging can
   turn wildly scattered pairs into group means that agree by construction, so
   `run` compares the within-acceleration scatter against the reported J and
   warns when the pairs disagree at the level of the answer.

Measured per-branch alpha is used throughout, because the drive does not always
deliver symmetric up/down rates at the aggressive end.

Everything measured here is the whole test-side string. Set `structure_inertia`
to the dyno structure's contribution in kg.m^2 (couplers, adapters, cell rotor
-- whatever spins that is not the motor under test) to have it subtracted, so
the reported J and the plot are the motor shaft alone.
"""

import numpy as np

from ..processor import Applicability, Processor, Result
from ..registry import register
from .running_torque import moving_average, sample_rate

PALETTE = dict(blue='#2a78d6', aqua='#1baf7a', yellow='#eda100', green='#008300',
               violet='#4a3aa7', red='#e34948', magenta='#e87ba4', orange='#eb6834')
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8a85'

_RC = {
    'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.color': '#e3e3e0',
    'grid.linewidth': 0.7, 'axes.axisbelow': True, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
}

# Relative gap at which two measured |alpha| values are called different
# accelerations. The recipe steps them by 100 rad/s^2 and the drive holds each
# to well under 1%, so anything in the middle of this range partitions them
# identically -- it is not a tuned number.
_ALPHA_CLUSTER_TOL = 0.05

# Within-acceleration scatter, as a fraction of J, above which the per-pair
# estimates are called too noisy to average. Set from the logs on hand: a
# healthy string repeats to 0.5-2.3% of J, while a string small enough to sit
# near the cell's noise floor scattered to 59%. Anything an order of magnitude
# above the healthy case and well below the broken one separates them; 10% is
# in the middle of that gap rather than tuned to either.
_SCATTER_WARN_FRAC = 0.10


def _runs_of(mask, min_len):
    """Contiguous True runs of `mask` at least `min_len` long -> [(start, stop)]."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    stops = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(len(mask))
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


def _cluster(values, rel_tol=_ALPHA_CLUSTER_TOL):
    """Partition `values` into ascending clusters, splitting at any relative
    gap wider than `rel_tol`. Returns a list of index lists."""
    order = list(np.argsort(values))
    groups, cur = [], [order[0]]
    for prev, i in zip(order, order[1:]):
        if values[i] - values[prev] > rel_tol * max(abs(values[i]), 1e-9):
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    return groups


@register
class Inertia(Processor):
    name = 'inertia'
    title = 'Test-Side Inertia (constant-acceleration ramp pairs)'
    description = (
        'Rotational inertia of the test-side string from paired up/down velocity '
        'ramps run at constant acceleration. Differencing the two branches of a '
        'pair at matched speed cancels the loss curve and the cell zero exactly, '
        'so no loss model, no loss-estimation dwells and no tare are needed. '
        'Repeats give the run-to-run spread; the disagreement between different '
        'accelerations is reported as the systematic floor.'
    )
    granularity = 'log'

    params = {
        # -- what to read (auto-detected; override to force) ------------------
        'velocity_channel': (None,  str,   'Shaft velocity channel (auto-detected)'),
        'torque_channel':   (None,  str,   'Torque cell channel (auto-detected '
                                           'from the input port, else from the data)'),
        'torque_sign':      (0.0,   float, 'Sign applied to the torque cell so that '
                                           'driving the shaft forward reads positive. '
                                           '0 = detect from the data; +1 or -1 to force'),
        # -- rig description ---------------------------------------------------
        'structure_inertia': (0.0,  float, 'Inertia of the dyno structure on the test '
                                           'side -- couplers, adapters, cell rotor, '
                                           'anything spinning that is not the motor '
                                           'under test -- in kg.m^2 (e.g. 1.2e-3). '
                                           'Subtracted from the measured total so the '
                                           'plot and the reported J are the motor '
                                           'shaft alone'),
        # -- ramp detection ----------------------------------------------------
        'smooth_s':         (0.05,  float, 'Rolling-average width for velocity and '
                                           'acceleration [s]'),
        'ramp_min_acc':     (20.0,  float, 'Sustained |alpha| that counts as a ramp '
                                           '[rad/s^2]'),
        'ramp_min_s':       (0.10,  float, 'Shortest usable ramp branch [s]'),
        'ramp_edge_s':      (0.06,  float, 'Trimmed from each end of a branch to drop '
                                           'the jerk-limited entry and exit corners [s]'),
        'fit_min_vel':      (15.0,  float, 'Ignore |w| below this on a branch [rad/s]'),
        'fit_max_vel':      (0.0,   float, 'Ignore |w| above this on a branch [rad/s]; '
                                           '0 = no upper limit'),
        'min_branch_samples': (40,  int,   'Samples a branch needs inside the shared '
                                           'speed window before it is fitted'),
        'const_acc_tol':    (0.10,  float, 'A branch is constant-acceleration if the '
                                           'alpha fitted to its first half and to its '
                                           'second half agree to this fraction. '
                                           'Branches that fail are dropped'),
        'poly_deg':         (3,     int,   'Order of the per-branch C(w) fit'),
    }

    # -- channel discovery ---------------------------------------------------

    def _pick_velocity(self, seg, segs):
        """Pick the channel that reports the shaft this test actually spun.

        Choosing by port prefix is wrong whenever a port is unattached: a
        disconnected servo still streams its encoder, so `dut_velocity` exists
        and reads +-0.1 rad/s of noise. Downstream that is indistinguishable
        from a shaft that never moved -- the whole log scans as one long
        standstill and no ramp is found. Rank by how much each channel actually
        moved instead; the real shaft wins by orders of magnitude. Prefer the
        input port only to break a genuine tie, which is the rigidly-coupled
        case where both encoders see the same sweep.
        """
        in_prefix = (seg.port_by_role('input') or {}).get('prefix', '')
        scored = []
        for ch in seg.channels():
            if not ch.endswith('_velocity'):
                continue
            v = np.concatenate([np.nan_to_num(s[ch], nan=0.0) for s in segs])
            if not np.isfinite(v).any():
                continue
            scored.append((float(np.max(np.abs(v))),
                           ch.startswith(in_prefix) if in_prefix else False, ch))
        if not scored:
            return None, 'no *_velocity channel in this log'
        peak = max(scored)[0]
        live = [s for s in scored if s[0] > 0.5 * peak]
        pick = max(live, key=lambda s: (s[1], s[0]))
        return pick[2], f'{pick[2]} peaks {pick[0]:.1f} rad/s'

    def _pick_cell(self, seg, segs, w, acc):
        """Trust the configured cell unless the log shows it is dead.

        The tempting gate here -- "require the cell to correlate with speed" --
        is backwards for an unloaded string. With nothing on the far side the
        cell reads the coupler's inertial torque, which tracks ACCELERATION and
        is near-orthogonal to speed, while the absorber's own `*_torque`
        estimate tracks speed strongly. Scoring on speed alone therefore throws
        away the real cell and picks the motor's internal estimate -- precisely
        inverted. Config knows the wiring; only override it when the named
        channel is absent or dead.
        """
        named = (seg.port_by_role('input') or {}).get('cell')
        if named and seg.has(named) and seg.is_active(named):
            return named, f'{named}, named by the input port'

        cands = [c for c in seg.channels()
                 if c.endswith('_torque') and seg.is_active(c)]
        if not cands:
            return None, 'no live torque channel in this log'
        # Among live channels the shaft cell is the one that responds to what
        # the shaft is doing -- either term, since which dominates depends on
        # whether anything is attached on the far side.
        scored = []
        for ch in cands:
            c = np.concatenate([np.nan_to_num(s[ch], nan=0.0) for s in segs])
            scored.append((max(abs(_pearson(c, np.abs(w))), abs(_pearson(c, acc))), ch))
        best = max(scored)
        why = (f'{best[1]}, picked from the data (|corr| with the shaft '
               f'{best[0]:.2f})')
        if named:
            why += (f'; the input port names {named!r} but it is '
                    f'{"missing from" if not seg.has(named) else "dead in"} this log')
        return best[1], why

    # -- ramp extraction ------------------------------------------------------

    def _prep(self, seg, vch, tch, params, fs):
        """Smoothed velocity, acceleration and raw torque for one span."""
        n = max(3, int(round(params['smooth_s'] * fs)))
        w = moving_average(np.nan_to_num(seg[vch], nan=0.0), n)
        acc = moving_average(np.gradient(w) * fs, n)
        C = np.nan_to_num(seg[tch], nan=0.0)
        return w, acc, C

    def _branches(self, segs, vch, tch, params, fs, sign=1.0):
        """Every constant-alpha ramp branch in the log, tagged with its span.

        Returned per span rather than pooled, because pairing an up branch with
        a down branch is only meaningful inside one span -- that is what keeps a
        pair's two branches adjacent in time, and therefore at the same thermal
        state and the same cell zero.
        """
        out, rejected = {}, 0
        edge = int(round(params['ramp_edge_s'] * fs))
        vmax = params['fit_max_vel'] or np.inf
        for seg in segs:
            w, acc, C = self._prep(seg, vch, tch, params, fs)
            t = seg['time']
            for direction in (+1, -1):
                for x, y in _runs_of(direction * acc > params['ramp_min_acc'],
                                     params['ramp_min_s'] * fs):
                    core = np.arange(x + edge, y - edge)
                    if len(core) < 0.05 * fs:
                        continue
                    aw = np.abs(w[core])
                    keep = core[(aw > params['fit_min_vel']) & (aw < vmax)]
                    if len(keep) < max(params['min_branch_samples'], 0.05 * fs):
                        continue
                    alpha = float(np.polyfit(t[keep], w[keep], 1)[0])
                    # Constant-acceleration test. Fitting alpha to each half of
                    # the branch and comparing is immune to the velocity noise
                    # that differentiating amplifies -- each half is a
                    # least-squares slope over hundreds of samples -- and it is
                    # a direct test of the thing that matters: whether one alpha
                    # describes the whole branch, or whether it is curving.
                    h = len(keep) // 2
                    a1 = float(np.polyfit(t[keep[:h]], w[keep[:h]], 1)[0])
                    a2 = float(np.polyfit(t[keep[h:]], w[keep[h:]], 1)[0])
                    curvature = abs(a2 - a1) / max(abs(alpha), 1e-9)
                    if curvature > params['const_acc_tol']:
                        rejected += 1
                        continue
                    out.setdefault(seg.raw_id, []).append(dict(
                        dirn='up' if direction > 0 else 'dn', alpha=alpha,
                        t=float(t[keep].mean()), curvature=curvature,
                        w=np.abs(w[keep]), C=sign * C[keep]))
        return out, rejected

    def _pairs(self, by_span, params):
        """One J per up/down ramp pair."""
        out = []
        for span, brs in by_span.items():
            # Pair each up branch with the next down branch in time. A repeat
            # has exactly one of each, so sequential pairing keeps a pair's two
            # branches adjacent -- which is the assumption the whole estimator
            # rests on.
            pairs, pending = [], None
            for br in sorted(brs, key=lambda b: b['t']):
                if br['dirn'] == 'up':
                    pending = br
                elif pending is not None:
                    pairs.append((pending, br))
                    pending = None

            for up, dn in pairs:
                dalpha = up['alpha'] - dn['alpha']
                if abs(dalpha) < 1e-6:
                    continue
                lo = max(up['w'].min(), dn['w'].min())
                hi = min(up['w'].max(), dn['w'].max())
                mu = (up['w'] >= lo) & (up['w'] <= hi)
                md = (dn['w'] >= lo) & (dn['w'] <= hi)
                nmin = params['min_branch_samples']
                if hi - lo < 1.0 or mu.sum() < nmin or md.sum() < nmin:
                    continue
                # Speed-binning the two branches wastes the aggressive runs: at
                # 500 rad/s^2 a 5 rad/s bin holds ~10 samples, so any fixed
                # per-bin count either discards the run or forces bins so wide
                # that only two survive. Fitting each branch's C(w) with a low
                # order polynomial instead uses every sample, is well
                # conditioned on a short branch, and lets the two be differenced
                # on a common grid regardless of how fast either was swept.
                deg = min(params['poly_deg'],
                          max(1, min(mu.sum(), md.sum()) // 20 - 1))
                pu = np.polyfit(up['w'][mu], up['C'][mu], deg)
                pd = np.polyfit(dn['w'][md], dn['C'][md], deg)
                grid = np.linspace(lo, hi, 40)
                curve = (np.polyval(pu, grid) - np.polyval(pd, grid)) / dalpha
                # Residual scatter of the branch fits -> the uncertainty this
                # pair actually earns, rather than the spread of a smooth curve.
                resid = np.hypot(
                    np.std(up['C'][mu] - np.polyval(pu, up['w'][mu])) / np.sqrt(mu.sum()),
                    np.std(dn['C'][md] - np.polyval(pd, dn['w'][md])) / np.sqrt(md.sum()))
                out.append(dict(
                    span=span, alpha_up=up['alpha'], alpha_dn=dn['alpha'],
                    alpha_mag=0.5 * (abs(up['alpha']) + abs(dn['alpha'])),
                    curvature=max(up['curvature'], dn['curvature']), deg=int(deg),
                    n_up=int(mu.sum()), n_dn=int(md.sum()), w_lo=float(lo),
                    w_hi=float(hi), J=float(curve.mean()),
                    sd=float(curve.std(ddof=1)), se=float(resid / abs(dalpha))))
        out.sort(key=lambda d: d['alpha_mag'])
        return out

    # -- applicability --------------------------------------------------------

    def applies_to(self, segs):
        p = self.defaults()
        seg = segs[0]

        vch, why_v = self._pick_velocity(seg, segs)
        if vch is None:
            return Applicability('no', why_v)

        fs = _sample_rate(segs)
        w = np.concatenate([np.nan_to_num(s[vch], nan=0.0) for s in segs])
        acc = np.gradient(moving_average(w, max(3, int(round(p['smooth_s'] * fs))))) * fs
        if float(np.max(np.abs(acc))) < p['ramp_min_acc']:
            return Applicability(
                'no', f'peak |acceleration| on {vch} is {float(np.max(np.abs(acc))):.1f} '
                      f'rad/s^2, under the {p["ramp_min_acc"]:g} that counts as a ramp; '
                      f'this log holds no ramps to pair')

        tch, why_c = self._pick_cell(seg, segs, w, acc)
        if tch is None:
            return Applicability('no', why_c)

        by_span, rejected = self._branches(segs, vch, tch, p, fs)
        pairs = self._pairs(by_span, p)
        if not pairs:
            n_br = sum(len(v) for v in by_span.values())
            return Applicability(
                'no', f'no span contained a usable up-ramp/down-ramp pair '
                      f'({n_br} constant-alpha branch(es) found, {rejected} more '
                      f'rejected as not constant-acceleration). This analysis wants '
                      f'each repeat to ramp up and back down within one span, both '
                      f'at a steady alpha')

        alphas = np.array([d['alpha_mag'] for d in pairs])
        n_groups = len(_cluster(alphas))
        return Applicability(
            'yes',
            f'{len(pairs)} constant-acceleration ramp pair(s) across {n_groups} '
            f'acceleration(s) from {alphas.min():.0f} to {alphas.max():.0f} '
            f'rad/s^2; shaft speed from {why_v}; torque cell {why_c}',
            {'velocity_channel': vch, 'torque_channel': tch})

    # -- run ------------------------------------------------------------------

    def run(self, segs, params):
        res = Result()
        vch, tch = params['velocity_channel'], params['torque_channel']
        structure = float(params['structure_inertia'])

        fs = _sample_rate(segs)
        w = np.concatenate([np.nan_to_num(s[vch], nan=0.0) for s in segs])
        C = np.concatenate([np.nan_to_num(s[tch], nan=0.0) for s in segs])
        acc = np.gradient(moving_average(
            w, max(3, int(round(params['smooth_s'] * fs))))) * fs

        sign, why_sign = self._sign(C, w, acc, params)
        res.add('info', 'channels',
                f'Shaft speed from {vch}, torque from {tch} with sign {sign:+.0f} '
                f'({why_sign}). The sign only has to make forward driving read '
                f'positive; the estimator below is a difference, so a cell offset '
                f'does not enter at all.')

        by_span, rejected = self._branches(segs, vch, tch, params, fs, sign)
        pairs = self._pairs(by_span, params)
        if not pairs:
            res.add('error', 'no_ramp_pairs',
                    'No usable up-ramp/down-ramp pair survived. Nothing to report.')
            return res
        if rejected:
            res.add('warn', 'non_constant_ramps',
                    f'{rejected} ramp branch(es) were dropped because their '
                    f'acceleration was not constant to within '
                    f'{params["const_acc_tol"]:.0%} (alpha fitted to the first half '
                    f'of the branch disagreed with the second half by more than '
                    f'that). A curving ramp has no single alpha to divide by, so it '
                    f'cannot contribute. If most of the log went this way, the drive '
                    f'is jerk-limiting the commanded ramp rate -- lower the '
                    f'sawtooth rate until it can follow, or raise const_acc_tol if '
                    f'you accept the bias.')

        # -- pooling -----------------------------------------------------------
        alphas = np.array([d['alpha_mag'] for d in pairs])
        groups = _cluster(alphas)
        rows = []
        for g in groups:
            v = np.array([pairs[i]['J'] for i in g])
            rows.append(dict(alpha=float(alphas[list(g)].mean()),
                             mean=float(v.mean()),
                             sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                             n=len(v)))
            for i in g:
                pairs[i]['group_alpha'] = rows[-1]['alpha']

        # Within one acceleration the repeats agree to well under 1%, so a
        # pooled standard error over all the pairs would claim far more
        # precision than the disagreement BETWEEN accelerations supports.
        # Weight each acceleration equally and take the uncertainty from their
        # spread: that is what survives repeatability, i.e. the systematic.
        gmeans = np.array([r['mean'] for r in rows])
        J = float(gmeans.mean())
        Jsys = float(gmeans.std(ddof=1)) if len(gmeans) > 1 else float('nan')
        # Reported for completeness, NOT quoted -- see the note below.
        Jse = float(Jsys / np.sqrt(len(gmeans))) if len(gmeans) > 1 else float('nan')

        if len(gmeans) > 1:
            res.add('info', 'systematic_floor',
                    f'The {len(gmeans)} accelerations disagree by '
                    f'{Jsys * 1e3:.4f}e-3 kg.m^2 ({100 * Jsys / abs(J):.1f}% of J), '
                    f'and that standard deviation IS the quoted error bar. A real '
                    f'inertia does not depend on how fast it was accelerated, so the '
                    f'spread between accelerations is a systematic, and a systematic '
                    f'does not shrink by averaging more accelerations -- dividing it '
                    f'by sqrt({len(gmeans)}) would quote '
                    f'+/-{Jse * 1e3:.4f}e-3 and claim {np.sqrt(len(gmeans)):.1f}x more '
                    f'accuracy than this measurement has earned. That standard error '
                    f'is still in the metrics as J_test_se_kgm2 for anyone who wants '
                    f'it; it is not what this report stands behind.')
        else:
            res.add('warn', 'single_acceleration',
                    f'Every ramp pair ran at the same acceleration '
                    f'({alphas.mean():.0f} rad/s^2), so there is no '
                    f'between-acceleration spread and no systematic error bar. The '
                    f'repeats agree to {np.std([d["J"] for d in pairs], ddof=1) * 1e3:.4f}'
                    f'e-3 kg.m^2, but that is repeatability only -- it cannot see a '
                    f'bias that is common to every ramp. Run at least two '
                    f'accelerations to get a real uncertainty.')

        # -- is the scatter small enough for the mean to mean anything? --------
        # The pooling above averages 4 pairs into a group mean and then 7 group
        # means into J. Two layers of averaging pull badly scattered pairs into
        # group means that agree with each other by construction, so a tight
        # between-acceleration spread can sit on top of per-pair estimates that
        # disagree by 30x. Jsys cannot see that -- it only ever sees the means.
        # This check looks underneath them, at the scatter the pairs actually
        # have, and is the one number that separates a real measurement from
        # noise that happens to average well.
        repeat = float(np.mean([r['sd'] for r in rows]))
        n_neg = sum(1 for d in pairs if d['J'] - structure <= 0)
        # Relative to the value actually REPORTED, not to the test-side total.
        # The pairs scatter around J_test, but a fixed structure subtraction
        # shrinks the number being quoted without shrinking the noise on it, so
        # a 59%-of-J_test scatter is 86% of a J_motor half that size. The
        # fraction only means anything against the figure the reader takes away.
        J_report = J - structure
        scatter_frac = repeat / abs(J_report) if J_report else float('inf')
        if scatter_frac > _SCATTER_WARN_FRAC:
            jv = np.array([d['J'] for d in pairs]) - structure
            res.add('warn', 'pairs_disagree',
                    f'The individual ramp pairs scatter by {repeat * 1e3:.4f}e-3 '
                    f'kg.m^2 within a single acceleration -- '
                    f'{100 * scatter_frac:.0f}% of the reported '
                    f'{"J_motor" if structure else "J_test"}, spanning '
                    f'{jv.min() * 1e3:.4f}e-3 to {jv.max() * 1e3:.4f}e-3 across the '
                    f'{len(pairs)} pairs. Repeats of one acceleration are the same '
                    f'measurement made twice, so they should agree to a few percent; '
                    f'at this level the pooled J is an average of noise, and the '
                    f'quoted error bar -- built from the group MEANS, which average '
                    f'that noise away -- does not reflect it. Treat the headline '
                    f'number as an order of magnitude, not a spec. The usual cause '
                    f'is a string whose inertial torque J*alpha is down near the '
                    f'cell noise: raise the ramp acceleration, or measure a larger '
                    f'assembly and subtract.')
        if n_neg:
            res.add('warn', 'negative_pairs',
                    f'{n_neg} of {len(pairs)} ramp pair(s) returned a non-physical '
                    f'J <= 0 and were still averaged in. A single pair cannot have '
                    f'negative inertia, so those are noise excursions and their '
                    f'presence means the per-pair uncertainty is at least as large '
                    f'as J itself.')

        if J <= 0:
            res.add('error', 'negative_inertia',
                    f'The pooled J came out NEGATIVE ({J * 1e3:.4f}e-3 kg.m^2), which '
                    f'is not physical. The usual cause is the torque cell sign: '
                    f'{tch} is being read with sign {sign:+.0f}, and the estimator '
                    f'inverts with it. Force the other one with '
                    f'torque_sign={-sign:+.0f} and check that the result is positive.')

        # -- structure subtraction ---------------------------------------------
        # Every number above sees the whole test-side string. Removing the known
        # structure inertia is the only step that makes this a MOTOR spec.
        # Subtracting a constant leaves the error bar alone, so the uncertainty
        # reported here is still the measurement's -- it does not include however
        # well the structure figure itself is known.
        J_motor = J - structure
        if structure:
            res.add('info', 'structure_removed',
                    f'structure_inertia={structure * 1e3:.4f}e-3 kg.m^2 has been '
                    f'subtracted from the {J * 1e3:.4f}e-3 measured for the whole '
                    f'test-side string, leaving J_motor = {J_motor * 1e3:.4f}e-3 '
                    f'kg.m^2. It is treated as an exact constant: it shifts every '
                    f'point by the same amount and widens no uncertainty, so the '
                    f'error bar quoted here does not include how well the structure '
                    f'figure itself is known.')
            if J_motor <= 0:
                res.add('error', 'structure_too_large',
                        f'The structure inertia is at or above the measured '
                        f'test-side total, so J_motor is non-physical. Check its '
                        f'units -- this parameter wants kg.m^2, e.g. 1.2e-3.')

        # -- metrics ------------------------------------------------------------
        # The commanded speed the ramps were run to, for the report's
        # experimental description. Taken from the command channel rather than
        # from the measured peak: the measured peak includes overshoot, and
        # "driven up to X" is a statement about the command.
        # The commanding shaft is routinely not the measured one -- this test
        # measures the DUT's shaft while the absorber holds the velocity ramp --
        # so the channel is looked up rather than derived from vch.
        cmd_v, cmd_ch = None, None
        for s in segs:
            channel = s.command_channel('velocity', prefer=vch.rsplit('_', 1)[0])
            span = s.cmd_span(channel) if channel else None
            if span:
                cmd_ch = channel
                cmd_v = max(cmd_v or 0.0, span['amplitude'])
        res.metrics.update({
            'velocity_channel': vch,
            'torque_channel': tch,
            'cmd_velocity_channel': cmd_ch,
            'cmd_peak_velocity_rad_s': cmd_v,
            'torque_cell_sign': sign,
            'sample_rate_hz': fs,
            'n_ramp_pairs': len(pairs),
            'n_accelerations': len(rows),
            'alpha_min_rad_s2': float(alphas.min()),
            'alpha_max_rad_s2': float(alphas.max()),
            'speed_window_rad_s': [float(min(d['w_lo'] for d in pairs)),
                                   float(max(d['w_hi'] for d in pairs))],
            'worst_ramp_curvature': float(max(d['curvature'] for d in pairs)),
            'J_test_kgm2': J,
            # The quoted bar is the between-acceleration standard deviation.
            # *_se_kgm2 is that divided by sqrt(n_accelerations); it is kept
            # because it is a real number some readers want, but a systematic
            # does not average down, so nothing here quotes it.
            'J_quoted_uncertainty_kgm2': Jsys,
            'J_test_se_kgm2': Jse,
            'J_test_systematic_kgm2': Jsys,
            'J_structure_kgm2': structure,
            'J_motor_kgm2': J_motor,
            'J_motor_se_kgm2': Jse,
            'repeatability_kgm2': repeat,
            'repeatability_frac_of_reported_J': scatter_frac,
            'J_pair_min_kgm2': float(min(d['J'] for d in pairs)),
            'J_pair_max_kgm2': float(max(d['J'] for d in pairs)),
            'n_pairs_nonphysical': n_neg,
        })
        sym = 'J_motor' if structure else 'J_test'
        # With one acceleration there is no systematic estimate, and printing
        # "+/- nan" as the headline number is worse than saying so: a reader
        # skimming the report would take the nan for a formatting glitch rather
        # than for the missing uncertainty it actually is.
        bar = (f'+/- {Jsys * 1e3:.4f} e-3 kg.m^2' if np.isfinite(Jsys)
               else 'e-3 kg.m^2, uncertainty UNKNOWN (only one acceleration)')
        tail = (f'The bar is the standard deviation across the {len(rows)} '
                f'accelerations ({100 * Jsys / abs(J):.1f}% of J), not that spread '
                f'divided by sqrt(n).' if np.isfinite(Jsys)
                else f'Repeats agree to {repeat * 1e3:.4f}'
                     f'e-3, which is repeatability, not accuracy.')
        if scatter_frac > _SCATTER_WARN_FRAC:
            tail += (f' PAIRS DISAGREE by {100 * scatter_frac:.0f}% within one '
                     f'acceleration -- see the pairs_disagree warning before '
                     f'quoting this.')
        res.summary = (
            f'{sym} = {(J_motor if structure else J) * 1e3:.4f} {bar} from '
            f'{len(pairs)} constant-acceleration ramp pairs across {len(rows)} '
            f'acceleration(s), {alphas.min():.0f}-{alphas.max():.0f} rad/s^2. {tail}'
        )

        res.figures.append(('inertia',
                            self._fig_inertia(pairs, rows, J, Jsys, structure)))
        res.tables.append(('ramp_pairs', self._csv_pairs(pairs)))
        return res

    # -- sign -----------------------------------------------------------------

    def _sign(self, C, w, acc, params):
        """Sign convention: C is torque delivered INTO the test side, so driving
        the shaft forward must read positive. Detect it from the data rather
        than trusting the config's flip_torque_sign, which describes the motor's
        wiring and not the cell's.

        Which "forward" to use depends on what the cell is transmitting. With a
        load attached the cell carries loss torque and is signed by speed; with
        the far side unloaded it carries only J*alpha, so speed says nothing and
        acceleration is the only handle. Pick whichever coupling the log
        actually shows.

        Correlate rather than average a product: on a test that only ever spins
        forward sign(w) is identically +1, so mean(C*sign(w)) collapses to the
        cell's DC zero offset and "detects" a sign from the tare rather than
        from the shaft. Correlation is immune to that offset.
        """
        forced = float(params['torque_sign'])
        if forced:
            return (1.0 if forced > 0 else -1.0), 'forced by torque_sign'
        r_speed, r_acc = _pearson(C, w), _pearson(C, acc)
        speed_wins = abs(r_speed) >= abs(r_acc)
        ref = r_speed if speed_wins else r_acc
        basis = 'speed' if speed_wins else 'acceleration'
        return (-1.0 if ref < 0 else 1.0), f'from {basis}, r={ref:+.2f}'

    # -- figure ---------------------------------------------------------------

    def _fig_inertia(self, pairs, rows, J, Jsys, structure):
        """J vs alpha. Everything measured here is the whole test-side string, so
        `structure` (couplers, adapter, cell rotor -- whatever is bolted between
        the absorber and the motor) is subtracted to leave the motor shaft alone.
        It is treated as an exact constant: it shifts every point by the same
        amount and widens no uncertainty.

        Two spreads are drawn, because they answer different questions and the
        plot is misleading without both. The shaded band is the standard
        deviation ACROSS accelerations -- the systematic, and the quoted bar.
        The whisker on each acceleration is the standard deviation of the
        repeats WITHIN it. A band much narrower than the whiskers is the
        signature of noise averaging away into agreeable group means, and is
        exactly the case the pairs_disagree warning fires on; drawing only the
        band would hide it."""
        import matplotlib.pyplot as plt

        P = PALETTE
        sym = 'J_{motor}' if structure else 'J_{test}'
        Jm = J - structure

        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(6.4, 4.6))

            # No band when there is only one acceleration to compare: a
            # zero-width band would draw "+-0.000" and read as a perfectly known
            # answer, when in fact nothing here can estimate the uncertainty.
            band = Jsys if np.isfinite(Jsys) else 0.0
            if band:
                ax.axhspan((Jm - band) * 1e3, (Jm + band) * 1e3,
                           color=P['blue'], alpha=.13, lw=0)
                label = ('pooled  $%s=%.3f\\pm%.3f\\times10^{-3}$  (s.d. across '
                         '$\\alpha$)' % (sym, Jm * 1e3, band * 1e3))
            else:
                label = ('pooled  $%s=%.3f\\times10^{-3}$  (one $\\alpha$, no '
                         'error bar)' % (sym, Jm * 1e3))
            ax.axhline(Jm * 1e3, color=P['blue'], lw=2, label=label)

            # Show every ramp pair rather than a bar summarising them. Each pair
            # is an independent estimate, so the raw scatter IS the
            # repeatability -- a drawn error bar is one number derived from
            # these points, and at n=4 it hides whether the spread is a clean
            # cluster or one outlier. Each pair also carries its own measured
            # alpha, which separates the repeats horizontally on its own; no
            # jitter is invented to spread them.
            allv = []
            for k, r in enumerate(rows):
                sel = [d for d in pairs if d.get('group_alpha') == r['alpha']]
                vals = (np.array([d['J'] for d in sel]) - structure) * 1e3
                amags = np.array([d['alpha_mag'] for d in sel])
                # The whisker is the within-alpha s.d. It is drawn UNDER the
                # dots, not instead of them: at n=4 one number cannot say
                # whether the spread is a clean cluster or one outlier, and the
                # raw points still answer that. What the whisker adds is a
                # readable comparison against the shaded band -- eyeballing the
                # dot cloud against a faint axhspan does not work once the two
                # differ by 30x.
                if r['n'] > 1:
                    ax.errorbar(amags.mean(), vals.mean(),
                                yerr=r['sd'] * 1e3, fmt='none', ecolor=INK2,
                                elinewidth=1.3, capsize=5, capthick=1.3, zorder=3,
                                label='repeats within one $\\alpha$ (s.d.)'
                                      if k == 0 else None)
                ax.plot(amags, vals, 'o', ms=5, color=P['violet'], alpha=.75, zorder=4,
                        label='individual ramp pairs (n=%d)' % len(pairs) if k == 0 else None)
                # A dash, not a bigger dot: within one group the repeats agree
                # to well under a marker width, so any filled symbol at the mean
                # simply covers the points this plot exists to show. A wide flat
                # tick marks the level without hiding the cluster under it.
                ax.plot(amags.mean(), vals.mean(), '_', ms=17, mew=2.2, color=INK,
                        zorder=5, label='per $\\alpha$ mean' if k == 0 else None)
                allv.extend(vals)
                allv.extend([vals.mean() - r['sd'] * 1e3, vals.mean() + r['sd'] * 1e3])
            if structure:
                ax.plot([], [], ' ',
                        label='$J_{test}=%.3f$, structure$=%.3f\\times10^{-3}$'
                              % (J * 1e3, structure * 1e3))

            ax.set_xlabel('ramp acceleration  $|\\alpha|$  (rad/s²)')
            ax.set_ylabel('$%s$  ($10^{-3}$ kg·m²)' % sym)
            ax.set_title('Calculated %s Inertia vs Acceleration'
                         % ('Motor' if structure else 'Test-Side'),
                         loc='left', fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            # Scale to the scatter, not to zero. The run-to-run spread here is a
            # couple of percent of J, so a zero-based axis flattens every
            # individual estimate into one line and throws away the only thing
            # this plot is for. The axis is read off the ticks, and both the
            # pooled value and its band stay on screen.
            lo = min(min(allv), (Jm - band) * 1e3)
            hi = max(max(allv), (Jm + band) * 1e3)
            # Asymmetric on purpose: the headroom is there to clear the legend,
            # and the footroom is not. One symmetric pad big enough for the
            # legend leaves a third of the axes empty below the data -- which
            # was tolerable when the points were a tight cluster and is not now
            # that the whiskers set the range.
            span = hi - lo
            pad_lo = max(span * 0.10, abs(Jm * 1e3) * 0.01, 5e-4)
            pad_hi = max(span * 0.55, abs(Jm * 1e3) * 0.04, 2e-3)
            ax.set_ylim(lo - pad_lo, hi + pad_hi)
            fig.tight_layout()
        return fig

    # -- table ----------------------------------------------------------------

    def _csv_pairs(self, pairs):
        lines = ['span,alpha_up_rad_s2,alpha_dn_rad_s2,alpha_mag_rad_s2,'
                 'speed_lo_rad_s,speed_hi_rad_s,n_up,n_dn,poly_deg,curvature,'
                 'J_kgm2,J_grid_sd_kgm2,J_se_kgm2']
        for d in pairs:
            lines.append(
                f'{d["span"]},{d["alpha_up"]:.2f},{d["alpha_dn"]:.2f},'
                f'{d["alpha_mag"]:.2f},{d["w_lo"]:.2f},{d["w_hi"]:.2f},'
                f'{d["n_up"]},{d["n_dn"]},{d["deg"]},{d["curvature"]:.4f},'
                f'{d["J"]:.6e},{d["sd"]:.6e},{d["se"]:.6e}')
        return '\n'.join(lines) + '\n'


def _sample_rate(segs):
    """Median per-span sample rate.

    Not the rate of the concatenation: spans are not contiguous in time, and on
    this test the gaps between them are seconds long, so spanning the whole
    record under-reads fs by ~1.5%. Nothing physical rides on it -- alpha is
    fitted against logged time, not counted in samples -- but every window
    length below is in seconds and is converted with it.
    """
    rates = [sample_rate(s['time']) for s in segs if len(s) > 1]
    return float(np.median(rates)) if rates else float('nan')


def _pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return 0.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])
