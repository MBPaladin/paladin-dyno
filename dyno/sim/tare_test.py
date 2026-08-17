"""Validation of the torque-cell tare: the channel maths, the controller's
averaging against the fake bus, and what ends up in the log.

Run from repo root:  PYTHONPATH=. .venv/bin/python dyno/sim/tare_test.py [mode]
(mode defaults to actuator_production)
"""
import json
import multiprocessing
import os
import sys
import tempfile
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else 'actuator_production'
os.environ['DYNO_SIM'] = MODE

import h5py  # noqa: E402

from dyno.src import setup_summary  # noqa: E402
from dyno.src.devices import ScaledChannels  # noqa: E402
from dyno.src.dyno_controller import Controller  # noqa: E402

ok = True


def fail(msg):
    global ok
    ok = False
    print(f'FAIL: {msg}')


# --- 1. The channel maths ----------------------------------------------------
print('--- 1. ScaledChannels: offset and tare stay separate ---')


class FakeModule(ScaledChannels):
    def __init__(self):
        self._init_channels()


m = FakeModule()
m._publish_channel('load_torque', 5.0)          # offset already folded in
if not (m.load_torque == 5.0 and m.untared['load_torque'] == 5.0):
    fail(f'untared publish wrong: {m.load_torque} / {m.untared}')

m.tare['load_torque'] = -0.4                     # a bias of -0.4
m._publish_channel('load_torque', 5.0)
if abs(m.load_torque - 4.6) > 1e-9:
    fail(f'tare not applied: expected 4.6, got {m.load_torque}')
if abs(m.untared['load_torque'] - 5.0) > 1e-9:
    fail(f'untared should stay pre-tare, got {m.untared["load_torque"]}')
print(f'  published={m.load_torque}  untared={m.untared["load_torque"]}  '
      f'tare={m.tare["load_torque"]}   (re-taring measures the cell, not the bias)')

# Two instances must not share state -- a class-level dict would be a real bug.
m2 = FakeModule()
if m2.tare:
    fail(f'tare dict leaked across instances: {m2.tare}')
print('  separate instances hold separate tare dicts')


# --- 2. The controller, against the fake bus ---------------------------------
print('\n--- 2. Controller: tare over the real control loop ---')
telemetry_q = multiprocessing.Queue()
command_q = multiprocessing.Queue()
proc = multiprocessing.Process(target=Controller,
                               args=[telemetry_q, command_q, MODE], name='ctrl')
proc.start()


def drain(seconds):
    """Collect control-state dicts (telemetry slot -1) for a while."""
    states, deadline = [], time.time() + seconds
    while time.time() < deadline:
        try:
            sample = telemetry_q.get(timeout=0.5)
        except Exception:
            if not proc.is_alive():
                fail('controller process died')
                return states
            continue
        if isinstance(sample, list) and isinstance(sample[-1], dict):
            states.append(sample[-1])
    return states


print('  waiting for bring-up...')
states = drain(10.0)
if not states:
    fail('no telemetry from the controller')
else:
    print(f'  telemetry flowing ({len(states)} samples), tare so far: '
          f'{states[-1].get("tare")}')

print('  sending tare...')
command_q.put_nowait(['tare', 0])
states = drain(8.0)

active_seen = any(s.get('tare_active') for s in states)
final = states[-1] if states else {}
tare = final.get('tare')
print(f'  saw tare_active during the window: {active_seen}')
print(f'  final tare_message: {final.get("tare_message")}')

if not tare:
    fail(f'no tare recorded; message was {final.get("tare_message")!r}')
else:
    for sensor, record in sorted(tare.items()):
        frac = record.get('frac_fs')
        print(f'    {sensor:<14} bias={record["bias"]:+.6g}  '
              f'{"%.3f%% FS" % (100 * frac) if frac is not None else "FS ?"}  '
              f'sd={record["stddev"]:.4g}  n={record["samples"]}  '
              f'fs={record.get("full_scale")}')
        # A 3 s window at the bus rate must gather far more than a handful.
        if record['samples'] < 100:
            fail(f'{sensor}: only {record["samples"]} samples in the window')
        if record['bias'] != -record['raw_mean']:
            fail(f'{sensor}: bias is not the negated mean')

    # Re-taring must measure the cell again, not ratchet the old bias to zero.
    print('  re-taring (bias must not collapse toward zero)...')
    first = {s: r['bias'] for s, r in tare.items()}
    command_q.put_nowait(['tare', 0])
    states = drain(8.0)
    second = {s: r['bias'] for s, r in (states[-1].get('tare') or {}).items()}
    print(f'    first : { {k: round(v, 6) for k, v in first.items()} }')
    print(f'    second: { {k: round(v, 6) for k, v in second.items()} }')
    for sensor, bias in second.items():
        if first.get(sensor) and abs(bias) < 0.1 * abs(first[sensor]):
            fail(f'{sensor}: re-tare collapsed the bias -- averaging tared data?')

    print('  clearing tare...')
    command_q.put_nowait(['clear_tare', 0])
    states = drain(4.0)
    if states and states[-1].get('tare'):
        fail(f'clear_tare left a tare behind: {states[-1].get("tare")}')
    else:
        print(f'    cleared; message: {states[-1].get("tare_message") if states else "?"}')

command_q.put_nowait(['shutdown', 0])
proc.join(timeout=15)
if proc.is_alive():
    proc.terminate()


# --- 3. What lands in the log ------------------------------------------------
print('\n--- 3. Log attribute and setup report ---')
sample_tare = {
    'load_torque': {'bias': -0.4213, 'raw_mean': 0.4213, 'stddev': 0.0121,
                    'samples': 3000, 'at': '2026-08-12 11:05:02',
                    'full_scale': 500.0, 'frac_fs': 0.0008426},
    'input_torque': {'bias': 0.0102, 'raw_mean': -0.0102, 'stddev': 0.0041,
                     'samples': 3000, 'at': '2026-08-12 11:05:02',
                     'full_scale': 20.0, 'frac_fs': 0.00051},
}
path = os.path.join(tempfile.mkdtemp(), 'tared.hdf5')
with h5py.File(path, 'w') as f:
    f.attrs['tare'] = json.dumps(sample_tare)

with h5py.File(path, 'r') as f:
    read_back = json.loads(f.attrs['tare'])
if read_back != sample_tare:
    fail('tare attribute did not round-trip through the log')
else:
    print('  tare attribute round-trips through HDF5')

resolved = {'mode': MODE, 'timestamp': '2026-08-12 11:04:00', 'devices': {}}
meta = {'test_name': 'efficiency', 'log_dir': '2026-08-12_110803',
        'tare': sample_tare}
print('\n  rendered into the setup report:')
for line in setup_summary.build_overview(resolved, meta, in_hdf5=True):
    print('    ' + line)

rendered = '\n'.join(setup_summary.build_overview(resolved, meta))
if 'Session tare' not in rendered or '0.08% of full scale' not in rendered:
    fail('tare block missing or malformed in the rendered overview')

# A log with no tare must render exactly as before.
if 'Session tare' in '\n'.join(setup_summary.build_overview(resolved, {'test_name': 'x'})):
    fail('untared log rendered a tare block')
else:
    print('\n  a log with no tare renders no tare block')

print('\nTARE TEST', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
