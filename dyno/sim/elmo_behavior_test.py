"""The fake Elmo (fake_pysoem.ElmoBehavior) against the real driver
(devices.ELMO), with no controller and no 1 kHz loop.

Each step is one EtherCAT cycle done by hand: the real ELMO writes its RxPDO,
the fake decodes it into shaft torque on the plant, the plant steps, and the
fake's TxPDO goes back through the real ELMO decode. So the two ends of every
unit conversion -- per-thousand of i_rated, counts/rev, counts/s, the DS402
modes 8/9/10, the 0x60B2 offset -- are checked against each other, not against
a copy of the arithmetic.

No bus, no real-time thread, no resolved_config: safe to run next to a live
rig session. Run from the repo root:

    PYTHONPATH=. .venv/bin/python dyno/sim/elmo_behavior_test.py [mode]

mode defaults to inhouse, the rig with an Elmo DUT.
"""
import math
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'inhouse'
os.environ.setdefault('DYNO_SIM', MODE)

import yaml  # noqa: E402

from deployment import dyno_paths  # noqa: E402
from dyno.sim.fake_pysoem import ElmoBehavior, CYCLE_S  # noqa: E402
from dyno.sim.plant import Plant, load_sim_params  # noqa: E402
from dyno.src.devices import ELMO  # noqa: E402


class _Slave:
    """Just enough of a pysoem slave for devices.ELMO, backed by the fake."""
    def __init__(self, behavior):
        self.b = behavior
        self.input = b'\x00' * behavior.input_size
        self.output = b'\x00' * behavior.output_size

    def sdo_read(self, index, subindex, size=None, ca=False):
        return self.b.sdo_read(index, subindex, size, ca)

    def sdo_write(self, index, subindex, data, ca=False):
        self.b.sdo_write(index, subindex, data, ca)


failures = []


def check(name, ok, detail):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failures.append(name)


def main():
    with open(f'{dyno_paths.dyno_config_directory}/{MODE}_dyno_config.yaml') as f:
        cfg = yaml.safe_load(f)
    entry = next((e for e in cfg['expected_slave_layout'] if e.get('model') == 'ELMO'), None)
    if entry is None:
        print(f'{MODE} has no ELMO in its slave layout; nothing to test')
        sys.exit(2)
    # The torque-sign check below compares against the device-side flip, so run
    # with the hardware convention (the AKD fake's sim_hardware_torque_sign).
    entry = dict(entry, params=dict(entry['params'], sim_hardware_torque_sign=True))
    params = dict(entry['params'])
    plant = Plant(load_sim_params())
    fake = ElmoBehavior(entry)
    fake.set_plant(plant)
    slave = _Slave(fake)
    dev = ELMO(slave, entry['name'], params)
    dev.setup()
    kt = dev.params['motor_params']['kt']
    cpr = dev.counts_per_rev
    G = dev.params['gear_ratio']
    print(f'--- fake Elmo vs devices.ELMO, {MODE} {entry["name"]}: kt {kt}, i_rated '
          f'{dev.i_rated}, {cpr} counts/rev, profile {dev.pdo_profile}, gear {G} ---')

    def cycle(command=None, torque_ff=0.0):
        dev.process_txpdo()
        dev.update_modes()
        if command is not None and not dev.switching_modes and dev.enabled:
            dev.send_command(command, torque_ff)
        dev.write_rxpdo()
        fake.pre_step(slave.output)
        plant.step(CYCLE_S)
        slave.input = fake.step(CYCLE_S, slave.output)

    def run(n, command=None, torque_ff=0.0):
        for _ in range(n):
            cycle(command, torque_ff)

    def enter(mode):
        dev.command_operating_mode(mode)
        dev.sw_enable = True
        for _ in range(3000):
            cycle()
            if not dev.switching_modes and dev.enabled:
                return True
        return False

    # identity, as the absorbers lookup sees it
    run(5)
    check('device name', fake.drive_name == 'SIM-' + entry['name'],
          f'0x1008 reads {fake.drive_name!r}')

    # --- torque mode (CST, 10) ---------------------------------------------
    ok = enter('torque')
    check('enters torque mode', ok and fake.op_mode == 10,
          f'drive op mode {fake.op_mode}, enabled {dev.enabled}')
    tc = 0.1 * dev.torque_limit
    run(20, tc)
    i_expect = tc / kt * (-1 if dev.flip_torque_sign else 1)   # drive frame (k_tanh ~ linear here)
    check('current command scaling', abs(dev.current - i_expect) < 0.05 * abs(i_expect),
          f'{tc:g} Nm -> logged {dev.current:+.3f} A, expected {i_expect:+.3f} A '
          f'(per-mille of i_rated {dev.i_rated:g} through the fake and back)')
    check('motor torque', abs(abs(fake.tau_motor) - tc) < 0.05 * tc,
          f'fake applies {fake.tau_motor:+.4f} Nm at the motor for {tc:g} Nm commanded')
    # the shaft accelerates the way the device-side polarity says it should
    w0 = dev.velocity
    run(300, tc)
    pol = (-1 if dev.flip_torque_sign else 1) * (-1 if dev.flip_direction_sign else 1)
    dw = dev.velocity - w0
    check('torque polarity', dw * pol > 0,
          f'+{tc:g} Nm changed logged velocity by {dw:+.3f} rad/s; flags predict sign {pol:+d}')
    # 0x60B2 offset adds in CST
    base = fake.tau_motor
    run(5, tc, torque_ff=tc)
    check('torque offset in CST', abs(abs(fake.tau_motor) - 2 * tc) < 0.1 * tc,
          f'command {tc:g} + offset {tc:g} Nm -> {fake.tau_motor:+.4f} Nm (was {base:+.4f})')
    run(400, 0.0)

    # --- velocity mode (CSV, 9) --------------------------------------------
    ok = enter('velocity')
    check('enters velocity mode', ok and fake.op_mode == 9, f'drive op mode {fake.op_mode}')
    v = 20.0                                   # rad/s at the motor
    p0 = dev.position
    run(3000, v)
    check('velocity tracking and counts/s scaling', abs(dev.velocity - v) < 0.05 * v,
          f'commanded {v:g} rad/s, logged {dev.velocity:+.3f} rad/s')
    run(1000, v)
    dp = dev.position - p0
    check('position integrates velocity (counts/rev)', dp * v > 0 and abs(dp / (4.0 * v) - 1) < 0.1,
          f'{dp:+.2f} rad over 4 s at {v:g} rad/s (about {4 * v:.0f} expected, less the 3 s ramp-in)')
    run(2000, 0.0)

    # --- position mode (CSP, 8) --------------------------------------------
    ok = enter('position')
    check('enters position mode', ok and fake.op_mode == 8, f'drive op mode {fake.op_mode}')
    hold = dev.position_command_frame
    target = hold + 3.0
    run(3000, target)
    err = dev.position_command_frame - target
    check('position target in counts', abs(err) < 0.05,
          f'commanded {target:+.3f} rad (command frame), reached {dev.position_command_frame:+.3f} '
          f'({err:+.4f} rad off)')

    print('\nELMO BEHAVIOR TEST', 'FAILED: ' + ', '.join(failures) if failures else 'PASSED')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
