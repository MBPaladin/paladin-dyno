"""Test-side inertia and loss from an aggressive-ramp inertia log.

Written for the `inertia_test_better` test shape:

    SEG1        long warm-up dwell at speed
    SEG6        step-through of loss setpoints, each hold bracketed by a
                zero-speed window
    SEG2..SEG5  groups of RUNs, each RUN = [stationary tare dwell]
                [ramp up at +alpha] [plateau] [ramp down at -alpha]

Nothing below keys off those names. The log is scanned for three kinds of
span -- stationary tare windows, constant-speed dwells, and constant-alpha
ramp branches -- so a re-ordered or re-named test still analyzes correctly.

What the test shape buys, and how it is used here
-------------------------------------------------
1. Every ramp RUN opens with its own stationary dwell and every loss hold is
   bracketed by two, so the cell zero is measured ~28 times instead of being
   modelled. `make_tare` then spends those windows the way the data warrants:
   pooled when the zero is stable (each window carries ~33 mN.m of noise, so
   a local correction would add more error than it removes), bracketing only
   when the zero is actually drifting. On this log pooling takes the loss-fit
   residual from 24 mN.m to 1.5 mN.m.
2. Alpha is 100-660 rad/s^2 rather than single digits, so the inertial torque
   J*alpha is ~1 Nm against a ~30 mNm zero uncertainty. The slow-ramp failure
   mode -- where zero drift over the up/down time gap swamps J*alpha -- is
   gone by construction, and `drift_check` verifies that rather than assuming
   it.
3. Each alpha is repeated 4x, so the error bar is the observed run-to-run
   spread, not a propagated guess.
4. Up and down branches inside one RUN are ~1 s apart, so the primary
   estimator differences them and the loss curve cancels exactly:

       C_up(w) - C_dn(w) = (alpha_up - alpha_dn) * J

   This needs no loss model and no tare at all. Measured per-branch alpha is
   used because the drive does not deliver symmetric up/down rates at the
   aggressive end (SEG5 ran +665 / -640).
5. The SEG6 holds and the in-run plateaus give alpha=0 references, so the
   loss curve doubles as a cross-check estimator: (C_ramp - loss(w)) / alpha.

Usage:
    python dyno/src/post_processors/mac_motor_inertia.py <log.hdf5|log_dir>
        [-o OUTDIR] [-s STRUCTURE_INERTIA]

The positional argument is either the .hdf5 log itself, the run folder holding
it, or any parent folder -- a parent resolves to its most recently modified
subfolder that contains a log. Outputs land next to the log unless -o says
otherwise.

Everything measured here is the whole test-side string. Pass `-s` with the dyno
structure's inertia in kg.m^2 (couplers, adapters, cell rotor -- whatever spins
that is not the motor under test) to have it subtracted, so the reported J and
the plot are the motor shaft alone.
"""

import argparse
import json
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Tunables ---------------------------------------------------------------
STILL_VEL = 0.10          # rad/s; |w| below this is "stationary"
STILL_MIN_S = 0.40        # s; shortest usable tare window
STILL_EDGE_S = 0.15       # s trimmed from each end of a tare window

DWELL_MIN_VEL = 2.0       # rad/s; a loss dwell must actually be turning
DWELL_MAX_ACC = 3.0       # rad/s^2; and be at constant speed
DWELL_MIN_S = 0.50
DWELL_EDGE_FRAC = 0.30    # drop this fraction off the front (settling)
DWELL_ACC_HALF_S = 0.15   # half-width of the central difference used to test
                          # "is this a dwell". Differentiating a 1 kHz velocity
                          # sample-to-sample amplifies its noise enough to shred
                          # a genuine low-speed hold (a clean 10 rad/s dwell
                          # measures +-3 rad/s^2 that way); over +-0.15 s the
                          # same hold reads well under 1.

RAMP_MIN_ACC = 20.0       # rad/s^2; sustained |alpha| that counts as a ramp
RAMP_MIN_S = 0.10
RAMP_EDGE_S = 0.06        # trim ramp corners (jerk-limited entry/exit)
RAMP_VEL_LO = 15.0        # speed window kept from each branch
RAMP_VEL_HI = 108.0

BIN_W = 5.0               # rad/s, speed granularity the overlap must exceed
BIN_MIN_N = 25            # samples needed in a bin, per branch
BRANCH_MIN_N = 40         # samples a branch needs before it is fitted
BRANCH_POLY_DEG = 3       # order of the per-branch C(w) fit

SMOOTH_S = 0.05           # s, smoothing window for velocity/acceleration

PALETTE = dict(blue='#2a78d6', aqua='#1baf7a', yellow='#eda100', green='#008300',
               violet='#4a3aa7', red='#e34948', magenta='#e87ba4', orange='#eb6834')
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8a85'


# --- small helpers ----------------------------------------------------------
def _smooth(x, n):
    n = max(int(n) | 1, 1)
    if n <= 1:
        return x.copy()
    return np.convolve(x, np.ones(n) / n, mode='same')


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


def _group(diffs):
    """Bucket ramp pairs by their behavior (SEG2, SEG3, ...). The test shape
    already groups one acceleration per behavior with its repeats, which is a
    truer grouping than rounding a measured alpha."""
    groups = {}
    for d in diffs:
        groups.setdefault(d['span'].split('-RUN')[0], []).append(d)
    return groups


def _wls(X, y):
    """Ordinary least squares returning (beta, stderr, resid)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), resid


# --- log loading ------------------------------------------------------------
def find_log(log_dir):
    log_dir = os.path.abspath(log_dir.rstrip('/'))
    if os.path.isfile(log_dir):
        return os.path.dirname(log_dir), log_dir
    here = sorted(f for f in os.listdir(log_dir) if f.endswith('.hdf5'))
    if len(here) == 1:
        return log_dir, os.path.join(log_dir, here[0])
    if len(here) > 1:
        raise ValueError(f'{log_dir} holds {len(here)} .hdf5 files; pick one run folder')
    subs = [os.path.join(log_dir, d) for d in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, d))]
    subs = [s for s in subs if any(x.endswith('.hdf5') for x in os.listdir(s))]
    if not subs:
        raise FileNotFoundError(f'no .hdf5 under {log_dir}')
    return find_log(max(subs, key=os.path.getmtime))


class Bench:
    """Channel resolution + sign convention for one inertia log."""

    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.config = json.loads(self.f.attrs.get('resolved_config', '{}'))
        self.time = self.f['time'][:].astype(float)
        self.fs = 1.0 / float(np.median(np.diff(self.time)))

        ports = self.config.get('ports') or []
        inp = next((p for p in ports if p.get('role') == 'input'), {})
        out = next((p for p in ports if p.get('role') == 'output'), {})
        self.in_prefix = inp.get('prefix', 'dut')
        self.out_prefix = out.get('prefix', 'load')

        self.vel_ch = self._resolve_velocity()
        self.raw_w = self.f[self.vel_ch][:].astype(float)
        self.raw_acc = _smooth(np.gradient(self.raw_w) * self.fs, SMOOTH_S * self.fs)
        self.cell = self._resolve_cell(inp)

        dev = out.get('device', 'LOAD')
        params = self.config.get('devices', {}).get(dev, {}).get('params', {})
        self.kt = float(params.get('motor_params', {}).get('kt', np.nan))

        raw_w = self.raw_w
        raw_c = self.f[self.cell][:].astype(float)
        # Sign convention: C is torque delivered INTO the test side, so driving
        # the shaft forward must read positive. Detect the sign from the data
        # rather than trusting flip_torque_sign, which describes the motor's
        # wiring and not the cell's.
        #
        # Which "forward" to use depends on what the cell is transmitting. With
        # a load attached the cell carries loss torque and is signed by speed;
        # with the far side unloaded it carries only J*alpha, so speed says
        # nothing (0.08 correlation on the disconnected log) and acceleration
        # is the only handle. Pick whichever coupling the log actually shows.
        #
        # Correlate rather than average a product: on a test that only ever
        # spins forward sign(w) is identically +1, so mean(C*sign(w)) collapses
        # to the cell's DC zero offset and "detects" a sign from the tare
        # rather than from the shaft. Correlation is immune to that offset.
        r_speed = self._pearson(raw_c, raw_w)
        r_acc = self._pearson(raw_c, self.raw_acc)
        speed_wins = abs(r_speed) >= abs(r_acc)
        ref = r_speed if speed_wins else r_acc
        self.sign_basis = f'{"speed" if speed_wins else "acceleration"} (r={ref:+.2f})'
        self.cell_sign = -1.0 if ref < 0 else 1.0

        self.w = _smooth(raw_w, SMOOTH_S * self.fs)
        self.C = self.cell_sign * raw_c
        # Two acceleration estimates, because dwell detection and ramp
        # detection want opposite things. `acc` is narrow enough to resolve a
        # 0.14 s ramp at 660 rad/s^2; `acc_slow` is wide enough that velocity
        # noise does not fake acceleration during a slow hold.
        self.acc = _smooth(np.gradient(self.w) * self.fs, SMOOTH_S * self.fs)
        k = max(int(DWELL_ACC_HALF_S * self.fs), 1)
        self.acc_slow = np.empty_like(self.w)
        self.acc_slow[k:-k] = (self.w[2 * k:] - self.w[:-2 * k]) * self.fs / (2 * k)
        self.acc_slow[:k] = self.acc_slow[k]
        self.acc_slow[-k:] = self.acc_slow[-k - 1]
        cur = self.f[f'{self.out_prefix}_current'][:].astype(float)
        self.A = cur * self.kt if np.isfinite(self.kt) else np.full_like(cur, np.nan)

        ids = self.f['behavior_ids'][:]
        idx = self.f['behavior_indices'][:]
        self.spans = []
        for i, raw in enumerate(ids):
            name = raw.decode() if isinstance(raw, bytes) else str(raw)
            a, b = int(idx[i, 0]), int(idx[i, 1])
            if name and b > a:
                self.spans.append((name, a, b))

    def _resolve_velocity(self):
        """Pick the channel that reports the shaft this test actually spun.

        Choosing by port prefix is wrong whenever a port is unattached: a
        disconnected servo still streams its encoder, so `dut_velocity` exists
        and reads +-0.1 rad/s of noise. Downstream that is indistinguishable
        from a shaft that never moved -- the whole log scans as one long
        standstill, no dwell or ramp is found, and the loss fit dies on an
        empty vector. Rank by how much each channel actually moved instead;
        the real shaft wins by three orders of magnitude, so the test is not
        close. Prefer the input prefix only to break a genuine tie (both ports
        rigidly coupled, which is the connected case).
        """
        cands = [k for k in self.f if k.endswith('_velocity')
                 and self.f[k].ndim == 1 and not k.endswith('_velocity_command')]
        scored = []
        for k in cands:
            v = self.f[k][:].astype(float)
            if not np.isfinite(v).any():
                continue
            scored.append((float(np.nanmax(np.abs(v))), k.startswith(self.in_prefix), k))
        if not scored:
            raise RuntimeError('no usable *_velocity channel in this log')
        scored.sort(reverse=True)
        best = scored[0]
        live = [s for s in scored if s[0] > 0.5 * best[0]]
        pick = max(live, key=lambda s: (s[1], s[0]))[2]
        if len(scored) > 1 and pick != f'{self.in_prefix}_velocity':
            others = ', '.join(f'{k} peaks {p:.2f}' for p, _, k in scored if k != pick)
            print(f'  shaft speed from {pick} (peaks {best[0]:.1f} rad/s); {others}')
        return pick

    def _resolve_cell(self, inp):
        """Trust the configured cell unless the log shows it is dead.

        The tempting gate here -- "require the cell to correlate with speed" --
        is backwards for an unloaded string. With nothing on the far side the
        cell reads the coupler's inertial torque, which tracks ACCELERATION and
        is near-orthogonal to speed (0.08 on this log), while the absorber's own
        `load_torque` tracks speed strongly (0.59). Scoring on speed therefore
        throws away the real cell and picks the motor's internal estimate --
        precisely inverted. Config knows the wiring; only override it when the
        named channel is absent, all-NaN, or a flat line.
        """
        cell = inp.get('cell')
        if cell and cell in self.f and self._is_live(cell):
            return cell
        if cell:
            print(f'  WARNING: ports names {cell!r} as the input cell but it is '
                  f'{"missing from the log" if cell not in self.f else "dead (flat or all-NaN)"}; '
                  f'searching for a live torque channel')
        cands = [k for k in self.f if k.endswith('_torque') and self.f[k].ndim == 1
                 and not k.endswith('_torque_command') and self._is_live(k)]
        # Among live channels, the shaft cell is the one that responds to what
        # the shaft is doing -- either term, since which one dominates depends
        # on whether anything is attached.
        scored = sorted(((max(self._corr(c, np.abs(self.raw_w)),
                              self._corr(c, self.raw_acc)), c) for c in cands), reverse=True)
        if not scored:
            raise RuntimeError('no live torque channel in this log')
        print(f'  torque cell: {scored[0][1]} (|corr| with shaft = {scored[0][0]:.3f})')
        return scored[0][1]

    @staticmethod
    def _pearson(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            return 0.0
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    def _is_live(self, ch):
        v = self.f[ch][:].astype(float)
        return bool(np.isfinite(v).any() and np.nanstd(v) > 0)

    def _corr(self, ch, ref):
        c = self.f[ch][:].astype(float)
        k = max(int(len(ref) / 4000), 1)
        r, c = ref[::k], c[::k]
        ok = np.isfinite(r) & np.isfinite(c)
        if ok.sum() < 10 or np.std(r[ok]) == 0 or np.std(c[ok]) == 0:
            return -np.inf
        return abs(float(np.corrcoef(r[ok], c[ok])[0, 1]))

    def close(self):
        self.f.close()


# --- span extraction --------------------------------------------------------
def extract(bench):
    """Scan every logged span for tare windows, constant-speed dwells and
    constant-alpha ramp branches."""
    fs = bench.fs
    w, C, A, acc, t = bench.w, bench.C, bench.A, bench.acc, bench.time
    tares, dwells, branches = [], [], []

    for name, a, b in bench.spans:
        sl = np.arange(a, b)
        aw, aacc = np.abs(w[sl]), np.abs(bench.acc_slow[sl])

        for x, y in _runs_of(aw < STILL_VEL, STILL_MIN_S * fs):
            e = int(STILL_EDGE_S * fs)
            core = sl[x + e:y - e]
            if len(core) < 0.2 * fs:
                continue
            tares.append(dict(span=name, t=float(t[core].mean()), n=len(core),
                              C=float(C[core].mean()), A=float(A[core].mean()),
                              sd=float(C[core].std())))

        for x, y in _runs_of((aw > DWELL_MIN_VEL) & (aacc < DWELL_MAX_ACC),
                             DWELL_MIN_S * fs):
            core = sl[x + int(DWELL_EDGE_FRAC * (y - x)):y]
            if len(core) < 0.2 * fs or np.std(w[core]) > 1.0:
                continue
            dwells.append(dict(span=name, t=float(t[core].mean()), n=len(core),
                               w=float(w[core].mean()), C=float(C[core].mean()),
                               A=float(A[core].mean()),
                               se=float(C[core].std() / np.sqrt(len(core) / fs))))

        for sign in (+1, -1):
            for x, y in _runs_of(sign * acc[sl] > RAMP_MIN_ACC, RAMP_MIN_S * fs):
                e = int(RAMP_EDGE_S * fs)
                core = sl[x + e:y - e]
                if len(core) < 0.05 * fs:
                    continue
                keep = core[(np.abs(w[core]) > RAMP_VEL_LO) & (np.abs(w[core]) < RAMP_VEL_HI)]
                if len(keep) < 0.05 * fs:
                    continue
                alpha = float(np.polyfit(t[keep], w[keep], 1)[0])
                branches.append(dict(span=name, dirn='up' if sign > 0 else 'dn',
                                     alpha=alpha, idx=keep, t=float(t[keep].mean())))
    branches.sort(key=lambda x: x['t'])
    return tares, dwells, branches


def make_tare(tares, drift):
    """Choose how to spend the standstill windows, and return f(t, field).

    The test parks at zero between every hold and before every ramp RUN so the
    cell can be re-zeroed. There are two ways to use that, and which one is
    right depends on whether the zero actually moves:

      * moving zero  -> average the windows bracketing each measurement, so
                        the correction tracks it.
      * static zero  -> pool every window. Each one carries ~33 mN.m of noise,
                        so subtracting a local pair would ADD ~24 mN.m to an
                        otherwise smooth loss curve; pooling 28 windows
                        instead pins the offset to ~6 mN.m.

    Picking the local correction unconditionally is the tempting mistake: it
    looks more careful and is measurably worse when the zero is stable.
    """
    if not tares:
        return (lambda t, field='C': 0.0), 'none (no standstill windows found)'

    significant = bool(drift and abs(drift['drift']) > 2 * drift['drift_se'])
    if significant:
        def f(t_ref, field='C'):
            before = [x for x in tares if x['t'] <= t_ref]
            after = [x for x in tares if x['t'] > t_ref]
            picks = []
            if before:
                picks.append(max(before, key=lambda x: x['t'])[field])
            if after:
                picks.append(min(after, key=lambda x: x['t'])[field])
            return float(np.mean(picks)) if picks else 0.0
        return f, 'bracketing windows (zero drifts significantly)'

    pooled = {k: float(np.mean([x[k] for x in tares])) for k in ('C', 'A')}
    return ((lambda t_ref, field='C': pooled.get(field, 0.0)),
            f'pooled over {len(tares)} windows (zero is stable): '
            f'C {pooled["C"] * 1e3:+.1f} mN·m')


# --- estimators -------------------------------------------------------------
def loss_curve(bench, dwells, tare_fn):
    """Tare-corrected loss torque vs speed from every constant-speed dwell."""
    raw = []
    for d in dwells:
        raw.append(dict(w=abs(d['w']),
                        C=d['C'] - tare_fn(d['t'], 'C'),
                        A=d['A'] - tare_fn(d['t'], 'A'),
                        se=d['se'], t=d['t'], span=d['span'], n=d['n']))
    # Every ramp RUN parks at the same top speed, so that one setpoint arrives
    # ~16 times while the rest arrive once. Collapse repeats to one point per
    # setpoint, otherwise the fit is effectively anchored to a single speed.
    merged = {}
    for r in raw:
        merged.setdefault(round(r['w'] * 2) / 2, []).append(r)
    rows = []
    for speed, grp in merged.items():
        wts = np.array([g['n'] for g in grp], dtype=float)
        vals = np.array([g['C'] for g in grp])
        rows.append(dict(
            w=speed, C=float(np.average(vals, weights=wts)),
            A=float(np.average([g['A'] for g in grp], weights=wts)),
            n=int(wts.sum()), repeats=len(grp),
            spread=float(vals.std(ddof=1)) if len(grp) > 1 else 0.0,
            spans=','.join(sorted({g['span'] for g in grp}))))
    rows.sort(key=lambda r: r['w'])
    return rows


def inertia_by_difference(bench, branches):
    """Primary estimator. Inside one RUN, difference the up and down branches
    at matched speed: the loss term cancels identically, so

        C_up(w) - C_dn(w) = (alpha_up - alpha_dn) * J

    No loss model, no tare, and -- because the two branches are ~1 s apart --
    no room for thermal hysteresis or zero drift to enter.
    """
    by_span = {}
    for br in branches:
        by_span.setdefault(br['span'], []).append(br)

    out = []
    for span, brs in by_span.items():
        # Pair each up-ramp with the next down-ramp in time. A ramp RUN has
        # exactly one of each; a step-through span has one pair per hold, and
        # pairing sequentially keeps each pair's two branches adjacent in time
        # (and therefore at the same thermal state and the same cell zero).
        pairs, pending = [], None
        for br in sorted(brs, key=lambda x: x['t']):
            if br['dirn'] == 'up':
                pending = br
            elif pending is not None:
                pairs.append((pending, br))
                pending = None

        for up, dn in pairs:
            dalpha = up['alpha'] - dn['alpha']
            wu, wd = np.abs(bench.w[up['idx']]), np.abs(bench.w[dn['idx']])
            cu, cd = bench.C[up['idx']], bench.C[dn['idx']]
            lo, hi = max(wu.min(), wd.min()), min(wu.max(), wd.max())
            if hi - lo < 4 * BIN_W or len(wu) < BRANCH_MIN_N or len(wd) < BRANCH_MIN_N:
                continue
            # Speed-binning the two branches wastes the aggressive runs: at
            # 660 rad/s^2 a 5 rad/s bin holds ~8 samples, so any fixed
            # per-bin count either discards the run or forces bins so wide
            # that only two survive. Fitting each branch's C(w) with a low
            # order polynomial instead uses every sample, is well conditioned
            # on a short branch, and lets the two be differenced on a common
            # grid regardless of how fast either was swept.
            mu = (wu >= lo) & (wu <= hi)
            md = (wd >= lo) & (wd <= hi)
            if mu.sum() < BRANCH_MIN_N or md.sum() < BRANCH_MIN_N:
                continue
            deg = min(BRANCH_POLY_DEG, max(1, min(mu.sum(), md.sum()) // 20 - 1))
            pu = np.polyfit(wu[mu], cu[mu], deg)
            pd = np.polyfit(wd[md], cd[md], deg)
            grid = np.linspace(lo, hi, 40)
            per_bin = np.column_stack(
                [grid, (np.polyval(pu, grid) - np.polyval(pd, grid)) / dalpha])
            amag = (abs(up['alpha']) + abs(dn['alpha'])) / 2
            # Residual scatter of the branch fits -> the uncertainty this pair
            # actually earns, rather than the spread of a smooth curve.
            resid = np.hypot(np.std(cu[mu] - np.polyval(pu, wu[mu])) / np.sqrt(mu.sum()),
                             np.std(cd[md] - np.polyval(pd, wd[md])) / np.sqrt(md.sum()))
            out.append(dict(span=span, alpha_up=up['alpha'], alpha_dn=dn['alpha'],
                            alpha_mag=amag, deg=deg, n_up=int(mu.sum()),
                            n_dn=int(md.sum()), w_lo=lo, w_hi=hi,
                            J=float(per_bin[:, 1].mean()),
                            sd=float(per_bin[:, 1].std(ddof=1)),
                            se=float(resid / abs(dalpha)), bins=per_bin))
    return out


def inertia_vs_loss_curve(bench, branches, tare_fn, loss_fit):
    """Cross-check. Referenced to the alpha=0 loss curve, every branch gives
    J = (C - loss(w)) / alpha independently -- including its sign."""
    out = []
    for br in branches:
        i = br['idx']
        wv = np.abs(bench.w[i])
        c = bench.C[i] - tare_fn(br['t'], 'C')
        j = (c - np.polyval(loss_fit, wv)) / br['alpha']
        out.append(dict(span=br['span'], dirn=br['dirn'], alpha=br['alpha'],
                        J=float(np.mean(j)), sd=float(np.std(j))))
    return out


def drift_check(bench, tares):
    """The slow-ramp killer was cell-zero movement between the up and down
    passes. Quantify it here so the assumption that it is now negligible is
    checked rather than asserted."""
    if len(tares) < 3:
        return None
    t = np.array([x['t'] for x in tares])
    c = np.array([x['C'] for x in tares])
    a = np.array([x['A'] for x in tares])
    beta, se, _ = _wls(np.column_stack([np.ones_like(t), t - t.mean()]), c)
    # A cell reading at standstill is only a zero offset if nothing is pushing
    # on the shaft. If the absorber is drawing current at the same time, the
    # cell is reporting real held torque -- wind-up against a cogging detent,
    # typically -- and subtracting it corrupts the loss curve by that amount.
    # Distinguishing the two costs one comparison and was worth a wrong answer
    # on an earlier log, so it is checked rather than assumed.
    held = bool(abs(a.mean()) > 0.3 * abs(c.mean()) and abs(c.mean()) > 0.02)
    return dict(zero_mean=float(c.mean()), zero_sd=float(c.std(ddof=1)),
                absorber_mean=float(a.mean()), looks_like_held_torque=held,
                drift=float(beta[1]), drift_se=float(se[1]),
                n=len(tares), span_s=float(t.max() - t.min()))


# --- figures ----------------------------------------------------------------
def _style():
    plt.rcParams.update({
        'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
        'figure.facecolor': 'white', 'axes.grid': True, 'grid.color': '#e3e3e0',
        'grid.linewidth': 0.7, 'axes.axisbelow': True, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False})


def figure_losses(rows, fit, drift, tares, out_path):
    _style()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={'width_ratios': [1.45, 1]})
    P = PALETTE
    wv = np.array([r['w'] for r in rows])
    cv = np.array([r['C'] for r in rows])
    av = np.array([r['A'] for r in rows])
    xx = np.linspace(0, max(wv.max() * 1.08, 1), 200)

    ax[0].plot(xx, np.polyval(fit, xx), color=P['blue'], lw=2, zorder=3,
               label='fit  $\\tau=%.3f%+.5f\\,\\omega%+.2e\\,\\omega^2$' % (fit[2], fit[1], fit[0]))
    ax[0].plot(wv, cv, 'o', ms=5, mfc='white', mec=P['blue'], mew=1.6, zorder=4,
               label='torque cell (tared)')
    if np.isfinite(av).any():
        ax[0].plot(wv, av, 's', ms=4.5, color=P['aqua'], zorder=4,
                   label='absorber current $\\times k_t$')
    ax[0].set_xlabel('shaft speed  $\\omega$  (rad/s)')
    ax[0].set_ylabel('loss torque (N·m)')
    ax[0].set_title('Test-side loss vs speed', loc='left', fontweight='bold')
    ax[0].legend(loc='upper left', fontsize=8)
    ax[0].set_xlim(0, xx[-1])

    tt = np.array([x['t'] for x in tares])
    cc = np.array([x['C'] for x in tares])
    ax[1].axhline(0, color=MUTED, lw=1)
    ax[1].plot(tt - tt.min(), cc * 1e3, 'o', ms=5, color=P['red'])
    if drift:
        ax[1].axhline(drift['zero_mean'] * 1e3, color=P['red'], ls='--', lw=1.2)
        ax[1].set_title('Cell zero at standstill — mean %+.1f, sd %.1f mN·m'
                        % (drift['zero_mean'] * 1e3, drift['zero_sd'] * 1e3),
                        loc='left', fontweight='bold')
    ax[1].set_xlabel('time into run (s)')
    ax[1].set_ylabel('cell zero (mN·m)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def figure_inertia(diffs, J, Jse, out_path, structure=0.0):
    """J vs alpha. Everything measured here is the whole test-side string, so
    `structure` (couplers, adapter, cell rotor -- whatever is bolted between the
    absorber and the motor) is subtracted to leave the motor shaft alone. It is
    treated as an exact constant: it shifts every point by the same amount and
    widens no uncertainty."""
    _style()
    P = PALETTE
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    groups = _group(diffs)
    order = sorted(groups, key=lambda k: np.mean([d['alpha_mag'] for d in groups[k]]))
    sym = 'J_{motor}' if structure else 'J_{test}'

    # Show every ramp pair rather than a bar summarising them. Each pair is an
    # independent estimate, so the raw scatter IS the repeatability -- a drawn
    # error bar is one number derived from these points, and at n=4 it hides
    # whether the spread is a clean cluster or one outlier. Each pair also
    # carries its own measured alpha, which separates the repeats horizontally
    # on its own; no jitter is invented to spread them.
    Jm = J - structure
    ax.axhspan((Jm - Jse) * 1e3, (Jm + Jse) * 1e3, color=P['blue'], alpha=.13, lw=0)
    ax.axhline(Jm * 1e3, color=P['blue'], lw=2,
               label='pooled  $%s=%.3f\\pm%.3f\\times10^{-3}$' % (sym, Jm * 1e3, Jse * 1e3))

    xs, ys, allv = [], [], []
    for k, g in enumerate(order):
        vals = (np.array([d['J'] for d in groups[g]]) - structure) * 1e3
        amags = np.array([d['alpha_mag'] for d in groups[g]])
        ax.plot(amags, vals, 'o', ms=5, color=P['violet'], alpha=.75, zorder=4,
                label='individual ramp pairs (n=%d)' % len(diffs) if k == 0 else None)
        # A dash, not a bigger dot: within one group the repeats agree to well
        # under a marker width, so any filled symbol at the mean simply covers
        # the points this plot exists to show. A wide flat tick marks the level
        # without hiding the cluster under it.
        ax.plot(amags.mean(), vals.mean(), '_', ms=17, mew=2.2, color=INK,
                zorder=5, label='per $\\alpha$ mean' if k == 0 else None)
        xs.append(amags.mean())
        ys.append(vals.mean())
        allv.extend(vals)
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
    # Scale to the scatter, not to zero. The run-to-run spread here is ~2% of J,
    # so a zero-based axis flattens every individual estimate into one line and
    # throws away the only thing this plot is for. The axis is read off the
    # ticks, and both the pooled value and its band stay on screen.
    lo = min(min(allv), (Jm - Jse) * 1e3)
    hi = max(max(allv), (Jm + Jse) * 1e3)
    pad = max((hi - lo) * 0.35, abs(Jm * 1e3) * 0.02, 1e-3)
    ax.set_ylim(lo - pad, hi + pad * 1.9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --- main -------------------------------------------------------------------
def analyze(log_dir, out_dir=None, structure_inertia=0.0):
    resolved, path = find_log(log_dir)
    out_dir = out_dir or resolved
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    print(f'Log: {path}')

    b = Bench(path)
    print(f'  cell={b.cell} (sign {b.cell_sign:+.0f} from {b.sign_basis})  vel={b.vel_ch}  '
          f'kt={b.kt:g}  fs={b.fs:.0f} Hz')

    tares, dwells, branches = extract(b)
    print(f'  found {len(tares)} tare windows, {len(dwells)} constant-speed '
          f'dwells, {len(branches)} ramp branches')

    drift = drift_check(b, tares)
    if drift:
        print(f'  cell zero: {drift["zero_mean"]*1e3:+.1f} mN·m, sd {drift["zero_sd"]*1e3:.1f}, '
              f'drift {drift["drift"]*1e6:+.1f} ± {drift["drift_se"]*1e6:.1f} µN·m/s')
        if drift['looks_like_held_torque']:
            print(f'  WARNING: the absorber is holding {drift["absorber_mean"]*1e3:+.0f} mN·m '
                  f'while "stationary", so the {drift["zero_mean"]*1e3:+.0f} mN·m cell '
                  f'reading is real transmitted torque, not a zero offset.\n'
                  f'           Subtracting it would bias the loss curve by that much. '
                  f'Inertia is unaffected (it comes from slopes vs alpha).')

    tare_fn, tare_mode = make_tare(tares, drift)
    print(f'  tare strategy: {tare_mode}')
    rows = loss_curve(b, dwells, tare_fn)
    if len(rows) < 3:
        peak = float(np.abs(b.w).max())
        raise RuntimeError(
            f'only {len(rows)} constant-speed dwell(s) found -- not enough for a loss '
            f'curve.\n'
            f'  Peak |speed| on {b.vel_ch} is {peak:.2f} rad/s against a DWELL_MIN_VEL '
            f'of {DWELL_MIN_VEL:g}.\n'
            f'  If that peak looks far too low, the wrong velocity channel was picked '
            f'and every span scanned as standstill.')
    wv = np.array([r['w'] for r in rows])
    cv = np.array([r['C'] for r in rows])
    fit = np.polyfit(wv, cv, 2)
    lin = np.polyfit(wv, cv, 1)
    print(f'  loss fit (quadratic): tau = {fit[2]:+.4f} {fit[1]:+.5f} w {fit[0]:+.3e} w^2  '
          f'(rms {np.std(cv - np.polyval(fit, wv))*1e3:.2f} mN·m)')
    print(f'  loss fit (linear):    tau = {lin[1]:+.4f} {lin[0]:+.5f} w  '
          f'(rms {np.std(cv - np.polyval(lin, wv))*1e3:.2f} mN·m)')

    diffs = inertia_by_difference(b, branches)
    if not diffs:
        peak = float(np.abs(b.acc).max())
        raise RuntimeError(
            f'no logged span contained a usable up-ramp/down-ramp pair.\n'
            f'  Peak |acceleration| in this log is {peak:.1f} rad/s^2 against a '
            f'RAMP_MIN_ACC of {RAMP_MIN_ACC:g}, and {len(branches)} branches '
            f'were found.\n'
            f'  This analysis wants each RUN to ramp up and back down within one '
            f'span. A log of slow single triangles (a few rad/s^2) does not '
            f'qualify: at that acceleration the inertial torque is comparable to '
            f'cell zero drift, which is the reason for the aggressive-ramp shape.')
    allJ = np.array([d['J'] for d in diffs])
    J = float(allJ.mean())
    Jse = float(allJ.std(ddof=1) / np.sqrt(len(allJ))) if len(allJ) > 1 else float('nan')

    print(f'\n  {"run":12s} {"a_up":>8s} {"a_dn":>8s} {"J (1e-3)":>10s} {"bin sd":>8s}')
    for d in sorted(diffs, key=lambda x: x['alpha_mag']):
        print(f'  {d["span"]:12s} {d["alpha_up"]:8.1f} {d["alpha_dn"]:8.1f} '
              f'{d["J"]*1e3:10.3f} {d["sd"]*1e3:8.3f}')

    groups = _group(diffs)
    print()
    gmeans = []
    for g in sorted(groups, key=lambda k: np.mean([d['alpha_mag'] for d in groups[k]])):
        v = np.array([d['J'] for d in groups[g]])
        amag = np.mean([d['alpha_mag'] for d in groups[g]])
        gmeans.append(v.mean())
        print(f'  {g:11s} |alpha|~{amag:6.0f}: J = {v.mean() * 1e3:.3f} +/- '
              f'{(v.std(ddof=1) if len(v) > 1 else 0) * 1e3:.3f} e-3  '
              f'(n={len(v)}, run-to-run)')
    gmeans = np.array(gmeans)
    # Within one acceleration the repeats agree to <0.5%, so a pooled standard
    # error over all 23 pairs would claim far more precision than the
    # disagreement BETWEEN accelerations supports. Weight each acceleration
    # equally and take the uncertainty from their spread: that is what
    # survives repeatability, i.e. the systematic.
    J = float(gmeans.mean())
    Jsys = float(gmeans.std(ddof=1)) if len(gmeans) > 1 else float('nan')
    Jse = float(Jsys / np.sqrt(len(gmeans))) if len(gmeans) > 1 else Jse
    print(f'\n  POOLED  J_test = {J * 1e3:.3f} +/- {Jse * 1e3:.3f} e-3 kg.m^2 '
          f'({len(allJ)} ramp pairs in {len(gmeans)} acceleration groups)')
    print(f'          between-acceleration spread = {Jsys * 1e3:.3f} e-3 '
          f'({100 * Jsys / J:.1f}%) -- the systematic floor')

    # Every estimator above sees the whole test-side string. Removing the known
    # structure inertia is the only step that makes the number a MOTOR spec.
    # Subtracting a constant leaves the error bar alone, so the uncertainty
    # reported here is still the measurement's -- it does not include however
    # well the structure figure itself is known.
    J_motor = J - structure_inertia
    if structure_inertia:
        print(f'          minus dyno structure {structure_inertia * 1e3:.3f} e-3'
              f'  ->  J_motor = {J_motor * 1e3:.3f} +/- {Jse * 1e3:.3f} e-3 kg.m^2')
        if J_motor <= 0:
            print(f'  WARNING: the structure inertia is at or above the measured '
                  f'test-side total, so J_motor is non-physical. Check its units '
                  f'(this flag wants kg.m^2, e.g. 1.2e-3).')

    xchecks = inertia_vs_loss_curve(b, branches, tare_fn, fit)
    xv = np.array([x['J'] for x in xchecks])
    print(f'  cross-check vs loss curve: {xv.mean()*1e3:.3f} +/- '
          f'{xv.std(ddof=1)*1e3:.3f} e-3 (n={len(xv)} branches)')

    f1 = os.path.join(out_dir, f'{stem}_losses.png')
    f2 = os.path.join(out_dir, f'{stem}_inertia.png')
    figure_losses(rows, fit, drift, tares, f1)
    figure_inertia(diffs, J, Jse, f2, structure_inertia)

    csv = os.path.join(out_dir, f'{stem}_loss_curve.csv')
    with open(csv, 'w') as fh:
        fh.write('speed_rad_s,loss_cell_Nm,loss_absorber_Nm,samples,repeats,'
                 'repeat_sd_Nm,spans\n')
        for r in rows:
            fh.write(f'{r["w"]:.3f},{r["C"]:.5f},{r["A"]:.5f},{r["n"]},'
                     f'{r["repeats"]},{r["spread"]:.5f},"{r["spans"]}"\n')

    summary = dict(
        log=path, cell=b.cell, cell_sign=b.cell_sign, kt=b.kt, fs=b.fs,
        J_test_kgm2=J, J_test_se=Jse, J_test_systematic=Jsys, n_runs=len(allJ),
        J_structure_kgm2=float(structure_inertia),
        J_motor_kgm2=float(J_motor), J_motor_se=Jse,
        J_by_group={str(g): dict(
            alpha=float(np.mean([d['alpha_mag'] for d in v])),
            mean=float(np.mean([d['J'] for d in v])),
            sd=float(np.std([d['J'] for d in v], ddof=1)) if len(v) > 1 else 0.0,
            n=len(v)) for g, v in groups.items()},
        J_crosscheck=float(xv.mean()), J_crosscheck_sd=float(xv.std(ddof=1)),
        loss_fit_quadratic=list(map(float, fit)),
        loss_fit_linear=list(map(float, lin)), tare_mode=tare_mode,
        cell_zero=drift)
    with open(os.path.join(out_dir, f'{stem}_inertia.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f'\n  wrote {f1}\n        {f2}\n        {csv}')
    b.close()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('log_dir', metavar='LOG',
                    help='.hdf5 log file, its run folder, or a parent folder')
    ap.add_argument('-o', '--out-dir', default=None)
    ap.add_argument('-s', '--structure-inertia', type=float, default=0.0,
                    metavar='KGM2',
                    help='inertia of the dyno structure on the test side -- '
                         'couplers, adapters, cell rotor, anything spinning that '
                         'is not the motor under test -- in kg.m^2 (e.g. 1.2e-3). '
                         'Subtracted from the measured total so the plot and the '
                         'reported J are the motor shaft alone.')
    a = ap.parse_args()
    analyze(a.log_dir, a.out_dir, a.structure_inertia)


if __name__ == '__main__':
    main()
