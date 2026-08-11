"""Read-only DS402 status word query for the AKD drives on the bus.

Opens the NIC named in master_config.yaml, runs config_init(), then walks the
chain and mailbox-reads each AKD's identity and status word over SDO:

    0x2031:00  DRV.NAME       drive name  (e.g. RTC-0200, 13030X)
    0x6041:00  Statusword     DS402 state machine + condition bits
    0x6061:00  Modes of op    display of the active control mode
    0x1001:00  Error register CANopen error class, only when the fault bit is set
    0x1003:nn  Error field    emergency codes, most recent first

Note the AKD does NOT implement the DS402 standard error code at 0x603F - it
raises "object does not exist" - so fault detail comes from the CANopen error
objects above instead.

Like bus_scan, this is deliberately inert: config_init() reads slave EEPROM and
leaves the chain in PRE-OP, where SDO mailbox traffic is legal but no PDOs are
mapped, no process data is exchanged and no drive is enabled. Nothing is
energized. It is safe to run on a live rig.

Usage:  ./dyno/drive_status.sh              # every AKD on the chain
        ./dyno/drive_status.sh RTC-0200     # only drives whose name matches
        ./dyno/drive_status.sh RTC-0200 13030X
"""
import os
import sys

import yaml

if os.environ.get('DYNO_SIM'):
    from dyno.sim import fake_pysoem as pysoem
    print('#### SIMULATION MODE: fake EtherCAT bus, no hardware ####')
else:
    import pysoem

from deployment import dyno_paths
from dyno.src.devices import DEVICE_CLASSES

AKD_PRODUCT_CODE = DEVICE_CLASSES['AKD']['id']

# DS402 status word, bit -> meaning. Bits 0-6 drive the state machine; the rest
# are condition flags. Bit 5 is inverted (0 means quick stop IS active), which
# is why it is spelled out rather than printed as a bare bit name.
STATUS_BITS = [
    (0,  'ready to switch on'),
    (1,  'switched on'),
    (2,  'operation enabled'),
    (3,  'FAULT'),
    (4,  'voltage enabled'),
    (5,  'quick stop not active'),
    (6,  'switch on disabled'),
    (7,  'WARNING'),
    (9,  'remote'),
    (10, 'target reached'),
    (11, 'internal limit active'),
    (12, 'setpoint ack / mode specific'),
    (13, 'following error'),
]

# (mask, value, state name), evaluated in order - the masks overlap, so the
# order below is the standard DS402 disambiguation and must not be sorted.
DS402_STATES = [
    (0x4F, 0x00, 'Not ready to switch on'),
    (0x4F, 0x40, 'Switch on disabled'),
    (0x6F, 0x21, 'Ready to switch on'),
    (0x6F, 0x23, 'Switched on'),
    (0x6F, 0x27, 'Operation enabled'),
    (0x6F, 0x07, 'Quick stop active'),
    (0x4F, 0x0F, 'Fault reaction active'),
    (0x4F, 0x08, 'Fault'),
]

# 0x6061 display values, in Kollmorgen's numbering (see devices.AKD.mode_dict).
OP_MODES = {0: 'none', 3: 'velocity', 4: 'torque', 7: 'position',
            8: 'cyclic sync position'}

# 0x1001 error register, DS301 bit meanings.
ERROR_REGISTER_BITS = [
    (0, 'generic'), (1, 'current'), (2, 'voltage'), (3, 'temperature'),
    (4, 'communication'), (5, 'device profile'), (7, 'manufacturer specific'),
]

# DS301 emergency error codes, keyed by high byte. The low byte is vendor
# detail - look it up in the AKD fault table or read the drive's front panel.
EMCY_GROUPS = {
    0x00: 'no error', 0x10: 'generic', 0x20: 'current',
    0x21: 'current, device input side', 0x22: 'current inside device',
    0x23: 'current, device output side', 0x30: 'voltage',
    0x31: 'mains voltage', 0x32: 'DC link voltage', 0x33: 'output voltage',
    0x40: 'temperature', 0x41: 'ambient temperature', 0x42: 'device temperature',
    0x50: 'device hardware', 0x60: 'device software', 0x61: 'internal software',
    0x62: 'user software', 0x63: 'data set', 0x70: 'additional modules',
    0x73: 'sensor / feedback', 0x80: 'monitoring', 0x81: 'communication',
    0x82: 'protocol', 0x90: 'external error', 0xF0: 'additional functions',
    0xFF: 'device specific',
}


def ds402_state(statusword):
    for mask, value, name in DS402_STATES:
        if statusword & mask == value:
            return name
    return 'UNDECODABLE'


def decode_bits(statusword):
    return [label for bit, label in STATUS_BITS if statusword & (1 << bit)]


def read_drive_name(slave):
    """DRV.NAME comes back null-padded; trim it rather than trusting length."""
    return slave.sdo_read(0x2031, 0).decode('utf-8', 'replace').rstrip('\x00').strip()


def read_fault_detail(slave):
    """Best-effort fault detail for a drive whose fault bit is set.

    Every read here is individually guarded: a drive that does not implement
    one of these objects must still produce a useful report rather than
    aborting the whole query (which is exactly what 0x603F used to do).
    """
    lines = []
    try:
        reg = int.from_bytes(slave.sdo_read(0x1001, 0, size=1), 'little')
        classes = [label for bit, label in ERROR_REGISTER_BITS if reg & (1 << bit)]
        lines.append(f"error reg    0x{reg:02X}  "
                     f"({', '.join(classes) if classes else 'no bits set'})")
    except Exception as e:
        lines.append(f"error reg    unavailable ({type(e).__name__})")

    try:
        count = int.from_bytes(slave.sdo_read(0x1003, 0, size=1), 'little')
    except Exception as e:
        lines.append(f"error field  unavailable ({type(e).__name__})")
        return lines

    if not count:
        lines.append('error field  empty (fault bit set but no logged emergency)')
    # 0x1003:01 is the most recent. Cap the walk so a drive reporting a large
    # history does not turn one query into dozens of mailbox round trips.
    for i in range(1, min(count, 5) + 1):
        try:
            raw = int.from_bytes(slave.sdo_read(0x1003, i, size=4), 'little')
        except Exception as e:
            lines.append(f"fault [{i}]     unreadable ({type(e).__name__})")
            continue
        # 0x1003:nn is UNSIGNED32: low 16 bits are the DS301 emergency code,
        # high 16 bits are manufacturer-specific additional information. On the
        # AKD that upper half carries the fault detail, so printing only the
        # standard code (which lands in a vendor range DS301 does not define)
        # discards the half that identifies the actual fault.
        emcy = raw & 0xFFFF
        mfr = raw >> 16
        group = EMCY_GROUPS.get(emcy >> 8, 'unknown group')
        recency = ' (most recent)' if i == 1 else ''
        lines.append(f"fault [{i}]     emergency 0x{emcy:04X} - {group}, "
                     f"mfr detail 0x{mfr:04X} ({mfr}){recency}")
    return lines


def report(slave, position):
    name = read_drive_name(slave)
    statusword = int.from_bytes(slave.sdo_read(0x6041, 0, size=2), 'little')
    op_mode = int.from_bytes(slave.sdo_read(0x6061, 0, size=1), 'little', signed=True)

    print(f"\nslave {position}: {name}")
    print(f"  statusword   0x{statusword:04X}  (0b{statusword:016b})")
    print(f"  DS402 state  {ds402_state(statusword)}")
    print(f"  mode of op   {OP_MODES.get(op_mode, 'unknown')} ({op_mode})")
    flags = decode_bits(statusword)
    print(f"  bits set     {', '.join(flags) if flags else '(none)'}")

    if statusword & 0x0008:
        for line in read_fault_detail(slave):
            print(f"  {line}")
        print('  clear with     controlword 0x0080 (fault reset), once the '
              'cause is dealt with')
    return name


def main(argv):
    wanted = [arg.strip() for arg in argv[1:] if arg.strip()]

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

        # SDO mailbox traffic needs PRE-OP; config_init() already leaves the
        # chain there, but say so explicitly so a half-transitioned bus from a
        # previous run cannot silently fail every read below.
        master.state = pysoem.PREOP_STATE
        master.write_state()
        master.state_check(pysoem.PREOP_STATE, timeout=500_000)

        print(f"\n{count} slave(s) on {ifname}; querying AKD drives")

        seen = []
        for i, slave in enumerate(master.slaves):
            if slave.id != AKD_PRODUCT_CODE:
                continue
            try:
                name = read_drive_name(slave)
            except Exception as e:
                print(f"\nslave {i}: AKD present but DRV.NAME read failed: {e}")
                continue
            if wanted and name not in wanted:
                print(f"\nslave {i}: {name} (skipped, not requested)")
                seen.append(name)
                continue
            try:
                seen.append(report(slave, i))
            except Exception as e:
                print(f"\nslave {i}: {name} SDO read failed: {e}")

        if not seen:
            print('\nNo AKD drives found on the chain. Run ./dyno/bus_scan.sh '
                  'to see what is actually there.')
            return 1
        missing = [name for name in wanted if name not in seen]
        if missing:
            print(f"\nNOTE: requested drive(s) not on the bus: {', '.join(missing)}")
            print(f"      found: {', '.join(seen)}")
            return 1
        return 0
    finally:
        master.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
