"""Release the AKD drives from the fieldbus so Workbench can enable them.

After the GUI runs, each AKD is left parked in DS402 'Ready to Switch On':
devices.AKD.update_modes walks the controlword down to 0x0006 (Shutdown) when
sw_enable goes false and never sends 0x0000, and Master.run()'s teardown then
requests INIT without confirming the chain got there. The drive retains that
last controlword, keeps statusword bit 9 ('remote') set, and lets the latched
DS402 state override the service channel - so Workbench's Enable is accepted
for an instant and then pulled straight back down by the stale Shutdown. The
symptom is the Enable button flicking to 'Disable' and reverting, with
DRV.ACTIVE reading 0 and DRV.DISSOURCES still reporting the software disable.

This tool undoes that:

    1. controlword 0x0000 (Disable Voltage) over SDO to every AKD, moving the
       DS402 state machine to 'Switch on Disabled' - the state a drive should
       be handed back in.
    2. Confirm each drive actually got there (statusword bit 6).
    3. Take the chain to EtherCAT INIT and *verify* it, so CoE - and with it the
       DS402 state machine - is genuinely down before Workbench takes over.

Every write here is in the disabling direction. 0x0000 removes the power stage
enable; no command in this file can energize a drive or produce motion. It
refuses to run if any drive is in 'Operation enabled', because Disable Voltage
on a spinning machine drops the power stage and lets a loaded dyno coast rather
than ramping it down - stop the test in the GUI first.

Usage:  ./dyno/utilities/release_drives.sh                 # every AKD found
        ./dyno/utilities/release_drives.sh RTC-0200        # only named drives
        ./dyno/utilities/release_drives.sh --clear-faults  # also reset faults
"""
import os
import sys
import time

import yaml

if os.environ.get('DYNO_SIM'):
    from dyno.sim import fake_pysoem as pysoem
    print('#### SIMULATION MODE: fake EtherCAT bus, no hardware ####')
else:
    import pysoem

from deployment import dyno_paths
from dyno.src.drive_status import (AKD_PRODUCT_CODE, ds402_state,
                                   read_drive_name)

CONTROLWORD = 0x6040
STATUSWORD = 0x6041

# DS402 controlword commands used here. Both are disabling; nothing in this
# file commands an enable.
CW_DISABLE_VOLTAGE = 0x0000  # -> Switch on Disabled, legal from any state
CW_FAULT_RESET = 0x0080      # bit 7, edge triggered, only with --clear-faults

SW_FAULT = 0x0008                # statusword bit 3
SW_SWITCH_ON_DISABLED = 0x0040   # statusword bit 6

RELEASE_TIMEOUT_S = 1.0
POLL_INTERVAL_S = 0.02


def read_statusword(slave):
    return int.from_bytes(slave.sdo_read(STATUSWORD, 0, size=2), 'little')


def write_controlword(slave, value):
    # Legal over the mailbox because the chain is in PRE-OP: no PDOs are mapped
    # or exchanged, so nothing is racing this write for ownership of 0x6040.
    slave.sdo_write(index=CONTROLWORD, subindex=0,
                    data=value.to_bytes(2, 'little'))


def is_operation_enabled(statusword):
    return statusword & 0x6F == 0x27


def release_drive(slave, position, name, clear_faults=False):
    """Walk one AKD to Switch on Disabled. True if it got there."""
    statusword = read_statusword(slave)
    print(f"\nslave {position}: {name}")
    print(f"  before       0x{statusword:04X}  {ds402_state(statusword)}")

    if statusword & SW_SWITCH_ON_DISABLED:
        print('  already released, nothing to do')
        return True

    if statusword & SW_FAULT:
        if clear_faults:
            # Bit 7 is edge triggered, so it has to go 0 -> 1. The drive still
            # will not leave Fault until the underlying cause is gone.
            print('  fault bit set, sending fault reset (0x0080)')
            write_controlword(slave, CW_FAULT_RESET)
            time.sleep(POLL_INTERVAL_S)
        else:
            print('  fault bit set - Workbench will refuse to enable until it '
                  'is cleared. Run drive_status.sh for the emergency history, '
                  'then re-run with --clear-faults once the cause is handled.')

    write_controlword(slave, CW_DISABLE_VOLTAGE)

    deadline = time.monotonic() + RELEASE_TIMEOUT_S
    while time.monotonic() < deadline:
        statusword = read_statusword(slave)
        if statusword & SW_SWITCH_ON_DISABLED:
            print(f"  after        0x{statusword:04X}  {ds402_state(statusword)}")
            return True
        time.sleep(POLL_INTERVAL_S)

    print(f"  after        0x{statusword:04X}  {ds402_state(statusword)}")
    print(f"  FAILED to reach Switch on Disabled within {RELEASE_TIMEOUT_S}s")
    return False


def to_init(master):
    """INIT the chain and confirm it - the step Master.run()'s teardown skips.

    A bare write_state() is one broadcast datagram with nobody checking the
    result, which is exactly how the chain ends up half transitioned and the
    DS402 layer stays alive.
    """
    print('\nTaking the chain to EtherCAT INIT')
    master.state = pysoem.INIT_STATE
    master.write_state()
    reached = master.state_check(pysoem.INIT_STATE, timeout=500_000)

    master.read_state()
    stragglers = [(i, slave.state) for i, slave in enumerate(master.slaves)
                  if slave.state != pysoem.INIT_STATE]

    if reached == pysoem.INIT_STATE and not stragglers:
        print('  all slaves in INIT - CoE is down, the drives are Workbench\'s')
        return True

    for position, state in stragglers:
        print(f"  slave {position} did not reach INIT (state 0x{state:02X})")
    print('  the DS402 layer may still be live; Workbench can still be blocked')
    return False


def main(argv):
    args = [arg.strip() for arg in argv[1:] if arg.strip()]
    clear_faults = '--clear-faults' in args
    wanted = [arg for arg in args if not arg.startswith('--')]

    unknown = [arg for arg in args
               if arg.startswith('--') and arg != '--clear-faults']
    if unknown:
        print(f"unknown option(s): {', '.join(unknown)}")
        print(__doc__.split('Usage:')[-1].strip())
        return 2

    with open(f"{dyno_paths.dyno_config_directory}/master_config.yaml") as f:
        ifname = yaml.safe_load(f)['interface']

    master = pysoem.Master()
    try:
        master.open(ifname)
    except Exception as e:
        print(f"could not open {ifname}: {e}")
        print("  - link up?      ip -br link show " + ifname)
        print("  - bring it up?  sudo systemctl start ethercat-link@" + ifname)
        print("  - capabilities? getcap .venv/bin/python")
        return 1

    try:
        count = master.config_init()
        if count <= 0:
            print(f"{ifname} opened, but config_init() found no slaves "
                  f"(returned {count}). Check chain power and the IN/OUT port "
                  f"direction on the first coupler.")
            return 1

        # SDO access needs PRE-OP. config_init() leaves the chain there, but a
        # half transitioned bus from a killed GUI cannot be assumed.
        master.state = pysoem.PREOP_STATE
        master.write_state()
        master.state_check(pysoem.PREOP_STATE, timeout=500_000)

        print(f"\n{count} slave(s) on {ifname}; looking for AKD drives")

        targets = []
        for position, slave in enumerate(master.slaves):
            if slave.id != AKD_PRODUCT_CODE:
                continue
            try:
                name = read_drive_name(slave)
            except Exception as e:
                print(f"\nslave {position}: AKD present but DRV.NAME read "
                      f"failed: {e}")
                continue
            if wanted and name not in wanted:
                print(f"\nslave {position}: {name} (skipped, not requested)")
                continue
            targets.append((position, slave, name))

        if not targets:
            print('\nNo AKD drives to release. Run ./dyno/utilities/bus_scan.sh '
                  'to see what is actually on the chain.')
            return 1

        # Safety gate, before any write: Disable Voltage cuts the power stage
        # outright. On a drive that is still enabled and turning that means a
        # coupled dyno coasts down uncontrolled, so refuse and let the operator
        # stop the test properly instead.
        spinning = [(position, name) for position, slave, name in targets
                    if is_operation_enabled(read_statusword(slave))]
        if spinning:
            for position, name in spinning:
                print(f"\nslave {position}: {name} is in Operation Enabled")
            print('\nREFUSING to release an enabled drive: Disable Voltage '
                  'drops the power stage and a loaded machine would coast.\n'
                  'Stop the test and shut the GUI down cleanly first.')
            return 1

        released = [release_drive(slave, position, name, clear_faults)
                    for position, slave, name in targets]

        init_ok = to_init(master)

        if all(released) and init_ok:
            print('\nDone. Connect Workbench and enable the axis.')
            return 0
        print('\nRelease incomplete - see above. Workbench may still refuse to '
              'enable; a 24V logic power cycle clears any latched drive state.')
        return 1
    finally:
        master.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
