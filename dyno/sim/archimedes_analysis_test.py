"""Logic-only exercise of the Archimedes analysis pack's decision rules.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/archimedes_analysis_test.py

No logs, no HDF5, no figures -- hand-built arrays and a stub Segment, so it
runs instantly and covers the four places this pack can go quietly wrong on a
future unit:

  * the segment-ID grammar, including the back-drive IDs that look identical to
    forward ones and the `_RC` recentres that are travel, not data
  * direction detection, which must call the RAMPED shaft the driver and not
    the position-HELD one (getting this backwards inverts every slip and
    stiffness test)
  * de-duplication across a restart, which must keep the whole copy and not the
    truncated one
  * leg finding against a commanded target, which must survive the turnaround
    overshoot that made a percentile reference unusable

dyno/logs/archimedes/*/analysis_pack is the end-to-end counterpart on real data.
"""
import os
import re
import shutil
import sys

import numpy as np

from dyno.src.archimedes import dataset, naming, physics  # noqa: E402

ok = True


def check(cond, msg):
    global ok
    if not cond:
        ok = False
        print('FAIL:', msg)


class FakeSegment:
    """Enough of analysis.segment.Segment for the rules under test."""

    def __init__(self, channels, dt=1e-3):
        self._c = {k: np.asarray(v, dtype=float) for k, v in channels.items()}
        self._dt = dt

    def has(self, ch):
        return ch in self._c

    def __getitem__(self, ch):
        return self._c[ch]

    def __len__(self):
        return len(next(iter(self._c.values())))

    def is_active(self, ch):
        return ch in self._c and not np.all(np.isnan(self._c[ch]))

    @property
    def dt(self):
        return self._dt


def nan(n):
    return np.full(n, np.nan)


# -- 1. the segment-ID grammar ------------------------------------------------
cases = [
    ('V0300-RUN0', naming.VELOCITY, dict(rpm=300.0)),
    ('V0030-RUN0', naming.VELOCITY, dict(rpm=30.0)),
    ('V3600-RUN0', naming.VELOCITY, dict(rpm=3600.0)),
    ('E1500_P40_C3-RUN0', naming.EFFICIENCY,
     dict(rpm=1500.0, torque_nm=40.0, cycle=3)),
    ('E0020_N05-RUN0', naming.EFFICIENCY, dict(rpm=20.0, torque_nm=-5.0)),
    ('K45-RUN0', naming.STIFFNESS, dict(pct=45)),
    ('K90-RUN0', naming.STIFFNESS, dict(pct=90)),
    ('SLIP-RUN0-SETPOINT1', naming.SLIP, {}),
    ('STICT-RUN0-SETPOINT5', naming.BREAKAWAY, {}),
    ('RAMP-RUN1-SETPOINT5', naming.BREAKAWAY, {}),
    # The back-drive ramp plans give every attempt its own segment, so their
    # IDs carry a sense and an attempt number the forward ones do not. Unparsed
    # these come back UNKNOWN and the whole back-drive slip and breakaway pass
    # vanishes from the report without a word.
    ('SLIP_P1-RUN0-SETPOINT0', naming.SLIP, dict(attempt=1, sense=1)),
    ('SLIP_N2-RUN0-SETPOINT0', naming.SLIP, dict(attempt=2, sense=-1)),
    ('BRK_P1-RUN0-SETPOINT0', naming.BREAKAWAY, dict(attempt=1, sense=1)),
    ('BRK_N3-RUN0-SETPOINT0', naming.BREAKAWAY, dict(attempt=3, sense=-1)),
    ('BRK_P1_RC-RUN0', naming.RECENTRE, {}),
    ('SLOW-RUN0', naming.GRAVITY, {}),
    ('V0300_RC-RUN0', naming.RECENTRE, {}),
    ('E1500_P40_C3_RC-RUN0', naming.RECENTRE, {}),
    ('SOMETHING_ELSE-RUN0', naming.UNKNOWN, {}),
]
for raw, kind, fields in cases:
    p = naming.parse(raw)
    check(p.kind == kind, f'{raw}: kind {p.kind!r}, expected {kind!r}')
    for k, v in fields.items():
        check(getattr(p, k) == v,
              f'{raw}: {k} = {getattr(p, k)!r}, expected {v!r}')

check(naming.parse('V0300_RC-RUN0').is_data is False,
      'a recentre must not count as data')
check(naming.parse('V0300-RUN0').is_data is True,
      'a velocity step must count as data')
check(naming.parse('RAMP-RUN1-SETPOINT5').run == 1,
      'run index must survive a SETPOINT suffix')
check(naming.parse('RAMP-RUN1-SETPOINT5').setpoint == 5,
      'setpoint index must parse')
check(naming.parse('BRK_P1_RC-RUN0').is_data is False,
      'a back-drive ramp recentre must not count as data')
check(naming.parse('SLIPPY-RUN0').kind == naming.UNKNOWN,
      'the ramp pattern must be anchored, not a prefix match')

# -- 2. direction detection ---------------------------------------------------
N = 2000
sweep = np.linspace(0, 50, N)                   # a real traverse
held = np.zeros(N)                              # a shaft being grounded

# Forward shuttle: input position-commanded, output torque-held at a level.
check(dataset.direction_of(FakeSegment({
    'dut_position_command': sweep, 'load_position_command': nan(N),
    'dut_torque_command': nan(N), 'load_torque_command': np.full(N, 20.0),
    'dut_velocity_command': nan(N), 'load_velocity_command': nan(N),
})) == dataset.FORWARD, 'forward shuttle must read as forward')

# The case that actually pins the tier ORDER: a real efficiency segment sweeps
# BOTH commands -- the held level ramps in over the first second, so
# load_torque_command spans its full 5 Nm -- while the input runs the traverse.
# Position must be tried before torque, or the shaft merely taking up its load
# is called the driver and every efficiency segment reads as back-drive.
ramp_in = np.concatenate([np.linspace(0, 5, N // 4), np.full(N - N // 4, 5.0)])
check(dataset.direction_of(FakeSegment({
    'dut_position_command': sweep, 'load_position_command': nan(N),
    'dut_torque_command': nan(N), 'load_torque_command': ramp_in,
    'dut_velocity_command': nan(N), 'load_velocity_command': nan(N),
})) == dataset.FORWARD,
    'position must outrank torque: a traverse beats a level ramping in')

# And the mirror, so the rule is not just "always forward".
check(dataset.direction_of(FakeSegment({
    'dut_position_command': nan(N), 'load_position_command': sweep / 43.0,
    'dut_torque_command': ramp_in / 43.0, 'load_torque_command': nan(N),
    'dut_velocity_command': nan(N), 'load_velocity_command': nan(N),
})) == dataset.BACKDRIVE,
    'position must outrank torque on the back-drive side too')

# Back-drive shuttle: SAME segment ID in the real logs, output commanded.
check(dataset.direction_of(FakeSegment({
    'dut_position_command': nan(N), 'load_position_command': sweep / 43.0,
    'dut_torque_command': held, 'load_torque_command': nan(N),
    'dut_velocity_command': nan(N), 'load_velocity_command': nan(N),
})) == dataset.BACKDRIVE, 'back-drive shuttle must read as back-drive')

# Stiffness / slip: input RAMPS torque, output is position-HELD. Reading
# 'position-commanded' as 'driving' would call this back-drive and invert the
# customer's forward/back-drive labelling on every ramp test.
check(dataset.direction_of(FakeSegment({
    'dut_position_command': nan(N), 'load_position_command': held,
    'dut_torque_command': np.linspace(0, -1.1, N),
    'load_torque_command': nan(N),
    'dut_velocity_command': nan(N), 'load_velocity_command': nan(N),
})) == dataset.FORWARD,
    'a torque ramp against a position-held output is FORWARD driving')

# Back-drive slip: output ramps torque, input position-held.
check(dataset.direction_of(FakeSegment({
    'dut_position_command': held, 'load_position_command': nan(N),
    'dut_torque_command': nan(N), 'load_torque_command': np.linspace(0, 60, N),
    'dut_velocity_command': nan(N), 'load_velocity_command': nan(N),
})) == dataset.BACKDRIVE,
    'a torque ramp against a position-held input is BACK-driving')

# Nothing sweeping names no driver.
check(dataset.direction_of(FakeSegment({
    'dut_position_command': held, 'load_position_command': nan(N),
    'dut_torque_command': nan(N), 'load_torque_command': held,
    'dut_velocity_command': nan(N), 'load_velocity_command': nan(N),
})) is None, 'a segment with nothing sweeping must name no driver')

# -- 3. de-duplication across a restart --------------------------------------
class FakeSpan:
    def __init__(self, raw, n, mtime, source, direction=dataset.FORWARD):
        self.point = naming.parse(raw)
        self.direction = direction
        self.n = n
        self.mtime = mtime
        self.source = source
        self.seg = None

    @property
    def kind(self):
        return self.point.kind

    key = dataset.Span.key


class FakeDS:
    duplicates = None
    skipped = None

    def __init__(self):
        self.duplicates = []
        self.skipped = []


# A resumed run: the earlier file holds the truncated copy.
ds = FakeDS()
truncated = FakeSpan('E1500_P40_C2-RUN0', 3436, 100.0, 'pt_2/log.hdf5')
whole = FakeSpan('E1500_P40_C2-RUN0', 5821, 200.0, 'pt_3/log.hdf5')
kept = dataset._resolve({truncated.key(): [truncated, whole]}, ds)
check(len(kept) == 1 and kept[0] is whole,
      'the whole copy must win over the truncated one')
check(ds.duplicates[0]['why'] == 'longer',
      f'reason should be "longer", got {ds.duplicates[0]["why"]!r}')

# A re-run sweep: identical lengths, so the later file supersedes.
ds = FakeDS()
early = FakeSpan('V0300-RUN0', 10430, 100.0, 'ramp_1/log.hdf5')
later = FakeSpan('V0300-RUN0', 10430, 500.0, 'ramp_4/log.hdf5')
kept = dataset._resolve({early.key(): [early, later]}, ds)
check(len(kept) == 1 and kept[0] is later,
      'on equal lengths the later file must win')
check(ds.duplicates[0]['why'] == 'later re-run',
      f'reason should be "later re-run", got {ds.duplicates[0]["why"]!r}')

# Three copies must resolve once, not pairwise -- a report that names an
# intermediate winner as "kept" contradicts itself.
ds = FakeDS()
a = FakeSpan('V0030-RUN0', 77301, 100.0, 'ramp_1/log.hdf5')
b = FakeSpan('V0030-RUN0', 77301, 200.0, 'ramp_4/log.hdf5')
c = FakeSpan('V0030-RUN0', 78418, 300.0, 'ramp_6/log.hdf5')
kept = dataset._resolve({a.key(): [a, b, c]}, ds)
check(len(kept) == 1 and kept[0] is c, 'the longest of three copies must win')
check(len(ds.duplicates) == 1 and len(ds.duplicates[0]['dropped']) == 2,
      'three copies must produce ONE record listing two drops')

# duplicates='all' keeps every copy for a deliberately repeated sweep.
ds = FakeDS()
kept = dataset._resolve({a.key(): [a, b, c]}, ds, mode='all')
check(len(kept) == 3, "duplicates='all' must keep every copy")
check(len(ds.duplicates[0]['pooled']) == 3,
      "duplicates='all' must record what was pooled")

# A RAMP is never collapsed, however identical its ID. Every launch of the
# back-drive slip plan starts its attempts at ramp 1 and logs `SLIP_P1-RUN0`,
# so six separate runs share two IDs between them -- and each one is a
# measurement of a stochastic release, not a repeat of the same number. Keyed
# like a velocity step they collapse to two and four measured slips disappear
# from the customer's table.
r1 = FakeSpan('SLIP_P1-RUN0-SETPOINT0', 32365, 100.0,
              'slip_1/log.hdf5', dataset.BACKDRIVE)
r2 = FakeSpan('SLIP_P1-RUN0-SETPOINT0', 33536, 200.0,
              'slip_2/log.hdf5', dataset.BACKDRIVE)
check(r1.key() != r2.key(),
      'two slip ramps in different files must not share a de-dup identity')
ds = FakeDS()
kept = dataset._resolve({r1.key(): [r1], r2.key(): [r2]}, ds)
check(len(kept) == 2, 'both slip ramps must survive de-duplication')
check(ds.duplicates == [],
      'independent ramps must not be reported as duplicates')

# The same ramp really recorded twice in ONE file would still collapse, which
# is the behaviour the source in the key preserves rather than discards.
dup = FakeSpan('SLIP_P1-RUN0-SETPOINT0', 100, 300.0,
               'slip_1/log.hdf5', dataset.BACKDRIVE)
check(dup.key() == r1.key(),
      'the same ramp from the same file must still share an identity')

# And an efficiency level is unaffected: it is one operating point, so a
# resumed copy must still collapse.
check(FakeSpan('E1500_P40_C2-RUN0', 1, 1.0, 'a.hdf5').key()
      == FakeSpan('E1500_P40_C2-RUN0', 1, 2.0, 'b.hdf5').key(),
      'an efficiency level must still de-duplicate across files')

# -- 4. leg finding against a commanded target -------------------------------
# A sawtooth that OVERSHOOTS at each turnaround. A percentile reference sits
# above the plateau the segment actually held, so the plateau falls outside its
# own mask and the step reports nothing -- which is what the 30 rpm forward
# step did before the target was used.
dt = 1e-3
plateau = 3.14                       # 30 rpm at the input
hold = int(0.30 / dt)                # time actually spent at the plateau
ring = int(0.22 / dt)                # time spent overshooting past it
over = np.concatenate([
    np.linspace(0, 2 * plateau, ring // 2),
    np.full(ring, 2 * plateau),                 # turnaround overshoot, 2x
    np.linspace(2 * plateau, plateau, ring // 2),
    np.full(hold, plateau),                     # the real plateau, forward
    np.linspace(plateau, -2 * plateau, ring),
    np.full(ring, -2 * plateau),                # overshoot the other way
    np.linspace(-2 * plateau, -plateau, ring // 2),
    np.full(hold, -plateau),                    # the real plateau, reverse
    np.linspace(-plateau, 0, ring // 2),
])
# Guard the guard: unless the overshoot really does pull the 90th percentile
# above the plateau, this case cannot tell a target-referenced mask from a
# percentile one, and mutation-testing it proves nothing.
check(np.percentile(np.abs(over), 90) > 1.2 * plateau,
      'test setup: the overshoot must dominate the 90th percentile')
seg = FakeSegment({'dut_velocity': over, 'time': np.arange(len(over)) * dt}, dt)

legs = physics.legs(seg, 'dut_velocity', target=plateau)
check(len(legs) == 2, f'target-referenced: expected 2 legs, got {len(legs)}')
# The same span WITHOUT the commanded target: the percentile reference lands on
# the overshoot, so the plateau falls outside its own mask. This is the failure
# the target was introduced to fix, and it is asserted so the fix cannot be
# quietly undone.
untargeted = physics.legs(seg, 'dut_velocity')
missed = all(abs(abs(np.mean(over[sl])) - plateau) > 0.2 * plateau
             for sl, _ in untargeted) if untargeted else True
check(missed,
      'without a target the percentile reference must latch onto the '
      'overshoot rather than the plateau -- if it finds the plateau anyway, '
      'this case no longer tests what it claims to')
check({s for _, s in legs} == {1, -1},
      'the two legs must have opposite travel signs')
for sl, sign in legs:
    got = np.mean(over[sl])
    check(abs(abs(got) - plateau) < 0.05 * plateau,
          f'leg mean {got:.3f} should be near +/-{plateau}')

# Noise that dips a few samples below the band must not fragment a leg.
noisy = over.copy()
noisy[np.argmax(noisy == plateau) + hold // 2] = 0.0   # one dropout mid-plateau
seg = FakeSegment({'dut_velocity': noisy,
                   'time': np.arange(len(noisy)) * dt}, dt)
check(len(physics.legs(seg, 'dut_velocity', target=plateau)) == 2,
      'a single-sample dropout must not split a leg')

check(physics._close_gaps(
    np.array([0, 1, 1, 0, 1, 1, 0], dtype=bool), 1).tolist()
    == [False, True, True, True, True, True, False],
    'an interior one-sample gap must close, the edges must not')

# -- 5. leg classification ----------------------------------------------------
R = 43.8
fwd = physics.classify_leg(
    {'w_in': -5.2, 'w_out': -0.12, 't_in': 0.67, 't_out': 24.7,
     'p_in': 0.67 * -5.2, 'p_out': 24.7 * -0.12, 'n': 100,
     't_in_sd': 0.0, 't_out_sd': 0.0}, R)
check(fwd['direction'] == physics.FORWARD,
      'output torque below ratio x input torque is forward driving')
check(0.5 < fwd['eta'] < 1.0, f'forward eta {fwd["eta"]:.3f} should be sane')
check(fwd['coherent'], 'same-sign shaft powers must read as coherent')

bwd = physics.classify_leg(
    {'w_in': 5.05, 'w_out': 0.123, 't_in': 0.234, 't_out': 21.7,
     'p_in': 0.234 * 5.05, 'p_out': 21.7 * 0.123, 'n': 100,
     't_in_sd': 0.0, 't_out_sd': 0.0}, R)
check(bwd['direction'] == physics.BACKDRIVE,
      'output torque above ratio x input torque is back-driving')
check(0.0 < bwd['eta'] < 1.0,
      f'back-drive eta {bwd["eta"]:.3f} should be sane')

coasting = physics.classify_leg(
    {'w_in': 5.0, 'w_out': 0.12, 't_in': -0.1, 't_out': 6.0,
     'p_in': -0.5, 'p_out': 0.72, 'n': 100,
     't_in_sd': 0.0, 't_out_sd': 0.0}, R)
check(not coasting['coherent'],
      'opposite-sign shaft powers must be flagged incoherent')

# -- 6. robust pk-pk ----------------------------------------------------------
clean = np.sin(np.linspace(0, 40 * np.pi, 20000))           # pk-pk 2.0
spiked = clean.copy()
spiked[7] = 50.0                                            # one bad sample
check(abs(physics.robust_pk_pk(clean) - 2.0) < 0.05,
      'robust pk-pk should recover a clean sine amplitude')
check(abs(physics.robust_pk_pk(spiked) - 2.0) < 0.05,
      'a single spike must not set the reported ripple')
check(np.ptp(spiked) > 45, 'the raw pk-pk really is ruined by that spike')

# -- 7. cell zero vs drag -----------------------------------------------------
# A cell reading a +1.3 Nm zero offset plus 0.75 Nm of Coulomb drag: the offset
# does not reverse with travel, the drag does.
class ZeroSpan:
    def __init__(self, sign, value):
        n = 4000
        v = np.full(n, sign * 3.14)
        self.point = naming.parse('V0300-RUN0')
        self.direction = 'forward'
        self.seg = FakeSegment({
            'dut_velocity': v, 'load_velocity': v / R,
            'load_torque': np.full(n, value),
            'input_torque': np.zeros(n),
            'time': np.arange(n) * dt,
        }, dt)

    @property
    def kind(self):
        return self.point.kind


offs, detail = physics.cell_offsets(
    [ZeroSpan(+1, 1.3 + 0.75), ZeroSpan(-1, 1.3 - 0.75)], R)
check(abs(offs['load_torque'] - 1.3) < 1e-6,
      f'zero offset {offs["load_torque"]:.4f} should be 1.3')
check(abs(detail['load_torque']['drag_nm'] - 0.75) < 1e-6,
      f'drag {detail["load_torque"]["drag_nm"]:.4f} should be 0.75')

# -- 8. the customer report ---------------------------------------------------
# Built from a synthetic pack rather than from real logs, so this stays a
# logic test: no HDF5 is opened and nothing is re-analysed.
import tempfile  # noqa: E402

from dyno.src.archimedes import report as texreport  # noqa: E402

PACK = tempfile.mkdtemp(prefix='archimedes_report_')
with open(os.path.join(PACK, 'results.json'), 'w') as fh:
    __import__('json').dump({
        'unit': 'Test unit', 'ratio': 43.4342,
        'cell_offsets': {'load_torque': 1.3, 'input_torque': 0.0027},
        'inventory': {'efficiency/forward': 8, 'slip/forward': 1,
                      'stiffness/forward': 1, 'velocity/forward': 2},
        'duplicates': [],
        'results': [{'name': 'efficiency', 'title': 'Efficiency',
                     'summary': '', 'metrics': {},
                     'findings': [{'level': 'info', 'code': 'train_drag',
                                   'message': 'drag of the whole train is '
                                              '0.54 Nm; it is'}]}],
    }, fh)


def _write(name, header, rows):
    with open(os.path.join(PACK, name), 'w') as fh:
        fh.write(header + '\n')
        for r in rows:
            fh.write(r + '\n')


_write('velocity_ramp__per_step.csv',
       'direction,rpm_cmd,n_legs,rpm_in_meas,speed_err_pct,ripple_nm,'
       'ripple_spread_nm,ripple_lo_nm,ripple_hi_nm,ripple_out_nm,t_in_nm,'
       't_out_nm',
       ['forward,30,3,29.97,-0.09,0.366,0.086,0.32,0.41,3.07,0.282,1.37',
        'forward,600,5,597.2,-0.47,0.947,0.34,0.78,1.12,6.92,0.262,0.44'])
_write('efficiency__per_point.csv',
       'flow,span_direction,rpm,t_out_nm,t_cmd_nm,n_legs,eta,eta_spread,'
       'eta_lo,eta_hi,loss_w,t_in_nm,p_in_w,p_out_w',
       ['forward,forward,20,6.69,5,2,0.428,0.021,0.42,0.44,0.43,0.24,1.0,0.43',
        'forward,forward,20,23.1,20,3,0.722,0.017,0.71,0.73,0.42,0.75,1.5,1.1',
        'backdrive,forward,20,10.8,10,2,0.0986,0.01,0.09,0.11,0.9,0.1,1.0,0.1'])
_write('efficiency__creep.csv',
       'segment,rpm_cmd,t_cmd_nm,creep_pct,creep_pct_p95,source_channel',
       ['E0020_P05-RUN0,20,5,0.42,0.7,velocities',
        'E0020_P20-RUN0,20,20,6.1,8.0,velocities'])
# Both directions, because the two halves of this section are read in
# different frames: a forward ramp pushes the input and a back-drive ramp the
# output, and the back-drive rows also cover a ramp that reached its ceiling
# with no event -- which is a result, not a gap, and has no 'at the event'
# torque for the section to quote.
_write('slip__events.csv',
       'ramp,segment,kind,lock_side,far_shaft_mode,direction,source,event,'
       't_event_s,t_out_at_event_nm,t_in_at_event_nm,t_out_peak_nm,'
       't_in_peak_nm,controller_breakaway_nm,decouple_rad,'
       'driven_travel_out_rad,held_travel_rad',
       ['Slip ramp,SLIP-RUN0,slip,output (low speed),position-held,forward,'
        'a.hdf5,slip,26.9,-50.9,-1.26,51.7,1.3,,2.1,2.1,0.32',
        'Breakaway ramp,STICT-RUN0,breakaway,output (low speed),free at 0 Nm,'
        'forward,a.hdf5,breakaway,3.9,-3.31,-0.121,3.31,0.121,-0.179,0.00003,'
        '0.0017,0.0016',
        'Slip ramp 1,SLIP_P1-RUN0,slip,input (high speed),position-held,'
        'backdrive,b.hdf5,slip,28.0,59.9,0.947,59.9,0.947,,0.64,0.0006,0.64',
        'Slip ramp 2,SLIP_N1-RUN0,slip,input (high speed),position-held,'
        'backdrive,c.hdf5,slip,31.2,-66.3,-1.11,66.3,1.11,,0.59,0.0009,0.59',
        'Breakaway ramp 1,BRK_P1-RUN0,breakaway,input (high speed),'
        'free at 0 Nm,backdrive,d.hdf5,breakaway,21.4,23.8,0.106,24.9,0.289,'
        '21.554,0.0025,0.227,0.230',
        'Breakaway ramp 2,BRK_N1-RUN0,breakaway,input (high speed),'
        'free at 0 Nm,backdrive,d.hdf5,none,,,,37.8,0.0529,,0.0023,0.0001,'
        '0.0024'])
_write('stiffness__fits.csv',
       'segment,direction,target_pct,source,n,torque_pk_pk_nm,'
       'torque_pk_pk_raw_nm,torque_peak_nm,windup_pk_pk_rad,k_nm_per_rad,'
       'k_r2,hysteresis_rad,sign_used,ratio_tracking_r,'
       'residual_frac_of_travel',
       ['K45-RUN0,forward,45,a.hdf5,16039,43.3,91.1,22.3,0.00319,11032.4,'
        '0.934,0.00154,1,0.99995,0.0107'])

cfg = {'unit_label': 'Test unit 1.2.0', 'ratio': 43.4342,
       'torque_cap_nm': 100.0, 'torque_step_nm': 5.0, 'correct_zero': True,
       'position_half_window_rad': 1.5}
driver, sections, pics = texreport.build(PACK, cfg)
text = '\n'.join(open(os.path.join(PACK, 'report', 'sections', f'{n}.tex')).read()
                 for n in sections)

check(os.path.isfile(driver), 'the report driver must be written')
for want in ('conditions', 'velocity_ramp', 'efficiency', 'slip', 'stiffness',
             'measurement_notes'):
    check(want in sections, f'section {want!r} should have been written')

# Braces must balance or pdflatex will not build the document.
_t = re.sub(r'\\[{}]', '', open(driver).read() + text)
_d = 0
for _ch in _t:
    _d += (_ch == '{') - (_ch == '}')
check(_d == 0, f'generated LaTeX braces must balance (off by {_d})')

# Every figure the sections reference must have been copied.
_figs = re.findall(r'\\resultfig\{pics/([^}]+)\}', text)
check(_figs == [] or set(_figs) <= set(pics),
      f'referenced figures {sorted(set(_figs) - set(pics))} were not copied')
check(set(pics) <= set(_figs),
      f'figures {sorted(set(pics) - set(_figs))} were copied but unreferenced')

# TONE. This is a customer deliverable and it reports measurements, not
# verdicts. These are the words that turn a measurement into a claim, and
# none of them belongs in generated prose.
BANNED = ('fail', 'pass', 'compliant', 'non-compliant', 'does not meet',
          'below spec', 'out of spec', 'unacceptable', 'poor', 'excessive',
          'should have', 'expected to', 'suggests that', 'indicates that',
          'likely', 'appears to', 'unfortunately', 'only reached',
          'rating', 'specification')
# Matched on word boundaries: a substring test flags 'ope-rating point' and
# 'pass-es', and a guard that cries wolf gets deleted.
low = text.lower()
for word in BANNED:
    check(re.search(r'\b%s\b' % re.escape(word), low) is None,
          f'report prose must not contain {word!r}')

# Numbers must not reach the page as bare exponent strings.
check('e+0' not in text and 'e-0' not in text,
      'numbers must be formatted through siunitx, not left as 1.103e+04')

# A back-driven curve taken from the forward sweep must say so in BOTH places
# -- the table caption travels with the table, and the body paragraph is what a
# reader going through the section in order actually sees. Asserting only one
# lets the other be deleted silently.
check(text.count('half-traverses of the forward sweep') >= 2,
      'a back-driven result derived from the forward sweep must be labelled '
      'in its caption AND in the body text')
check('control configuration was unchanged' in text,
      'the body text must state that the control configuration did not change '
      'between the forward and back-driven halves')

# The slip torque must be stated as a measurement.
check('slip occurred at' in low, 'the slip torque must be stated plainly')

# Both directions of the ramp tests must reach the page. The customer asks for
# slip and breakaway in both, and back-drive ramps arrive weeks after the
# forward ones -- so the failure mode is a section that silently renders only
# the half that was there when the wording was last touched.
check('tab:slip_forward' in text and 'tab:slip_backdrive' in text,
      'both directions of the ramp tests must get a table')

# Breakaway torque is measured on the RAMPED shaft, which is the input forward
# and the output back-driving -- 43:1 apart. One sentence quoting both in one
# frame is wrong by the gear ratio, and it reads as if the drive changed rather
# than the reference.
_brk = re.findall(r'Breakaway was identified on .*?no slip was identified\.',
                  text, re.S)
check(len(_brk) == 2, f'expected a breakaway sentence per direction, got {_brk}')
check(any('input torque cell' in b for b in _brk)
      and any('output torque cell' in b for b in _brk),
      f'each breakaway sentence must name the cell it was measured at: {_brk}')

# A ramp that reached its ceiling without letting go is a measurement of the
# torque it held to, and dropping it would leave the reader counting rows to
# find out what happened to it.
check('neither event was identified' in low,
      'a ramp that reached its ceiling without an event must be reported')
check('37.8' in text,
      'the torque a no-event ramp reached must be quoted')

shutil.rmtree(PACK, ignore_errors=True)

print('PASS' if ok else 'FAILURES ABOVE')
sys.exit(0 if ok else 1)
