"""Read-only scaling query for a DS402 drive, for ELMO bringup.

devices.ELMO carries three numbers that cannot be derived from a motor
datasheet because they are properties of how THIS DRIVE is configured. Guessing
any of them produces a rig that runs and logs happily while reporting wrong
numbers. This reads all three off the drive:

    drive_params.i_rated             <- 0x6075 / 0x6076   (torque scaling)
    drive_params.counts_per_rev      <- 0x608F            (position scaling)
    the csp/csv/cst mode choice      <- 0x6502            (supported modes)

Objects read:

    0x1008:00  Manufacturer device name   absorbers.yaml lookup key
    0x1018:nn  Identity                   vendor / product code / revision
    0x6075:00  Motor rated current        mA
    0x6076:00  Motor rated torque         mNm
    0x6073:00  Max current                per-mille of rated current
    0x6072:00  Max torque                 per-mille of rated torque
    0x608F:01  Encoder increments         counts...
    0x608F:02  Motor revolutions          ...per this many motor revolutions
    0x6091:nn  Gear ratio                 motor revs : shaft revs, if present
    0x6502:00  Supported drive modes      bitfield

The headline output is the two candidate values for drive_params.i_rated, which
differ only if the drive's rated torque is not kt x its rated current. See
`interpret_torque_scaling` for why that is the whole question.

Like bus_scan and drive_status, this is deliberately inert: config_init() reads
slave EEPROM and leaves the chain in PRE-OP, where SDO mailbox traffic is legal
but no PDOs are mapped, no process data is exchanged and no drive is enabled.
Nothing is energized and nothing can move. It is safe to run on a live rig.

Usage:  ./dyno/utilities/elmo_scaling.sh --kt 0.983      # every drive that answers
        ./dyno/utilities/elmo_scaling.sh --kt 0.983 --slave 3
"""
import argparse
import math
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

# product code -> model name, so a slave this repo already drives is reported as
# such instead of being offered up as the Elmo. Without this the AKDs on the
# chain each print a confident ELMO_PRODUCT_CODE suggestion, and pasting one in
# would make master.py match the wrong slave.
KNOWN_PRODUCT_CODES = {spec['id']: model for model, spec in DEVICE_CLASSES.items()}

# 0x6502 supported drive modes. Bit -> mode name and the 0x6060 value that
# selects it. devices.ELMO drives the cyclic synchronous three (8/9/10); the
# profile modes are listed so a drive that supports only those is obvious
# rather than just failing to confirm a mode change at runtime.
DRIVE_MODES = [
    (0, 'pp   profile position', 1),
    (1, 'vl   velocity (frequency converter)', 2),
    (2, 'pv   profile velocity', 3),
    (3, 'tq   profile torque', 4),
    (5, 'hm   homing', 6),
    (6, 'ip   interpolated position', 7),
    (7, 'csp  cyclic synchronous position', 8),
    (8, 'csv  cyclic synchronous velocity', 9),
    (9, 'cst  cyclic synchronous torque', 10),
]

# The modes devices.ELMO needs, as (0x6060 value, label).
ELMO_REQUIRED_MODES = [(8, 'csp'), (9, 'csv'), (10, 'cst')]


def read_u(slave, index, subindex, size, signed=False):
    """One unsigned/signed SDO read, or None if the drive does not have it.

    Every object here is individually optional: a drive missing one must still
    produce a useful report for the rest rather than aborting the query.
    """
    try:
        return int.from_bytes(slave.sdo_read(index, subindex, size=size),
                              'little', signed=signed)
    except Exception:
        return None


def read_string(slave, index, subindex=0):
    try:
        raw = slave.sdo_read(index, subindex)
        return raw.split(b'\x00')[0].decode('utf-8', 'replace').strip()
    except Exception:
        return None


def fmt(value, unit='', scale=1.0, nd=3):
    if value is None:
        return 'not implemented'
    return f'{value*scale:.{nd}f} {unit}'.strip() if scale != 1.0 else f'{value} {unit}'.strip()


def interpret_torque_scaling(rated_current_ma, rated_torque_mnm, kt):
    '''The two candidate values for drive_params.i_rated, and whether it matters.

    devices.ELMO.send_command computes

        per_mille = (T_cmd / kt) / i_rated * 1000

    and the drive multiplies that back out against ITS rated reference. Which
    reference decides what i_rated has to be for the two to cancel:

      0x6071 is per-mille of RATED CURRENT   -> i_rated = 0x6075
      0x6071 is per-mille of RATED TORQUE    -> i_rated = 0x6076 / kt

    ...because in the second case the current we computed is only a stand-in for
    "this fraction of rated torque", so the denominator has to be the current
    that WOULD produce the rated torque, not the drive's rated current.

    The two coincide exactly when 0x6076 == kt x 0x6075, which is how a
    consistently commissioned drive is usually set up. When they coincide the
    whole rated-torque-vs-rated-current question is moot and no experiment is
    needed. When they do not, the ratio between them is precisely the factor a
    wrong guess would scale every torque command by.
    '''
    if rated_current_ma is None or not kt:
        return None
    amps = rated_current_ma / 1000.0
    by_current = amps
    by_torque = (rated_torque_mnm / 1000.0) / kt if rated_torque_mnm else None
    return {'by_current': by_current, 'by_torque': by_torque}


def report_torque_scaling(slave, kt):
    rated_current = read_u(slave, 0x6075, 0, 4)
    rated_torque = read_u(slave, 0x6076, 0, 4)
    max_current = read_u(slave, 0x6073, 0, 2)
    max_torque = read_u(slave, 0x6072, 0, 2)

    print('\n  --- torque scaling -------------------------------------------')
    print(f'  0x6075 rated current   {fmt(rated_current, "mA")}'
          + (f'  = {rated_current/1000:.3f} A' if rated_current else ''))
    print(f'  0x6076 rated torque    {fmt(rated_torque, "mNm")}'
          + (f'  = {rated_torque/1000:.3f} Nm' if rated_torque else ''))
    print(f'  0x6073 max current     {fmt(max_current, "per-mille of rated")}')
    print(f'  0x6072 max torque      {fmt(max_torque, "per-mille of rated")}')

    if rated_current is None:
        print('\n  0x6075 is not implemented, so i_rated cannot be read off this\n'
              '  drive. It has to be established experimentally against a torque\n'
              '  cell before any torque command is trusted.')
        return
    if not kt:
        print('\n  pass --kt <motor torque constant> to evaluate i_rated')
        return

    cand = interpret_torque_scaling(rated_current, rated_torque, kt)
    print(f'\n  candidate drive_params.i_rated (kt = {kt} Nm/A):')
    print(f'    if 0x6071 is per-mille of RATED CURRENT:  {cand["by_current"]:.4f} A')
    if cand['by_torque'] is None:
        print('    if 0x6071 is per-mille of RATED TORQUE:   '
              'unknown (0x6076 not implemented)')
        print('\n  With no rated torque to compare against, the two readings\n'
              '  cannot be told apart from the object dictionary alone.')
        return
    print(f'    if 0x6071 is per-mille of RATED TORQUE:   {cand["by_torque"]:.4f} A')

    a, b = cand['by_current'], cand['by_torque']

    # 0x6076 and 0x6075 carrying the SAME RAW NUMBER (9900 mNm / 9900 mA) is
    # not a commissioned motor, it is a drive whose rated torque was left
    # mirroring its rated current - i.e. an internal torque constant of exactly
    # 1 Nm/A, which no real motor has.
    #
    # That is good news rather than bad. Whichever reference 0x6071 uses, the
    # drive converts torque to current through its own 0x6076/0x6075, so with
    # the two equal both readings produce the SAME commanded current and the
    # ambiguity disappears. i_rated is 0x6075 either way. What is NOT
    # established is the absolute torque calibration, because the drive's idea
    # of a Nm is meaningless here - that still comes from the cell.
    if rated_torque and abs(rated_torque - rated_current) / rated_current < 0.001:
        print(f'\n  >>> 0x6076 ({rated_torque} mNm) and 0x6075 ({rated_current} mA) '
              f'hold the same\n      raw value, so the drive is carrying an internal '
              f'torque constant of\n      exactly 1 Nm/A. No motor has that: rated '
              f'torque was never commissioned\n      and is mirroring rated current.\n')
        print(f'      Both readings of 0x6071 therefore command the same current,\n'
              f'      so the rated-torque-vs-rated-current question is MOOT here.\n')
        print(f'      Use  i_rated: {a:.4f}   (0x6075, {rated_current} mA)\n')
        print(f'      But the ABSOLUTE torque calibration is still unverified - the\n'
              f'      drive\'s Nm are fictional, so only the cell can confirm that a\n'
              f'      commanded Nm is a real Nm. Expect to correct i_rated by the\n'
              f'      measured ratio:  i_rated_correct = {a:.4f} x (T_measured / T_commanded)\n'
              f'      That single constant also absorbs any A_peak/A_rms frame error\n'
              f'      in kt, so do not agonise over the sqrt(2) before measuring.')
        return

    spread = abs(a - b) / max(abs(a), 1e-12)
    print(f'\n  0x6076 vs kt x 0x6075:  {rated_torque} vs '
          f'{kt*rated_current:.0f} mNm  ({spread*100:.1f}% apart)')
    if spread < 0.02:
        print('\n  >>> The two candidates agree. The rated-torque-vs-rated-current\n'
              '      question does not affect this drive: either reading gives the\n'
              f'      same command. Use  i_rated: {a:.4f}\n'
              '      No motion experiment is needed.')
    else:
        ratio = max(a, b) / min(a, b)
        print(f'\n  >>> The two candidates DISAGREE by {spread*100:.1f}%. Guessing wrong\n'
              f'      scales every torque command by {ratio:.3f}x.')
        # A ratio sitting on sqrt(2) or 2 is much more likely to be a peak/rms
        # bookkeeping error in --kt than a genuine difference in the drive's
        # commissioning: by_current is in whatever frame 0x6075 uses, while
        # by_torque is in --kt's frame, so feeding a kt from the wrong frame
        # manufactures a disagreement out of nothing.
        for suspect, why in ((math.sqrt(2), 'one sqrt(2), a peak-vs-rms frame mismatch'),
                             (2.0, 'two sqrt(2)s, i.e. a peak/rms conversion applied backwards')):
            if abs(ratio - suspect) / suspect < 0.05:
                print(f'\n      CAUTION: {ratio:.3f} is within 5% of {suspect:.3f} - {why}.\n'
                      f'      --kt must be in the SAME current frame that 0x6075 uses.\n'
                      f'      Remember kt_rms = sqrt(2) x kt_peak (rms amps are FEWER\n'
                      f'      for the same current, so torque per amp is LARGER).\n'
                      f'      Re-run with the other frame before trusting this split.')
                break
        else:
            print('      Resolve against a torque cell: command a small torque and\n'
                  '      compare measured Nm to commanded. Start well under the cell\n'
                  '      full scale - if the guess is wrong it errs in this ratio.')


def report_position_scaling(slave):
    increments = read_u(slave, 0x608F, 1, 4)
    motor_revs = read_u(slave, 0x608F, 2, 4)
    gear_motor = read_u(slave, 0x6091, 1, 4)
    gear_shaft = read_u(slave, 0x6091, 2, 4)

    print('\n  --- position scaling -----------------------------------------')
    print(f'  0x608F:01 encoder increments  {fmt(increments)}')
    print(f'  0x608F:02 motor revolutions   {fmt(motor_revs)}')
    if gear_motor is not None:
        print(f'  0x6091 gear ratio             {gear_motor} : {gear_shaft}')

    # 0 and 1 are both "nobody filled this in" rather than real resolutions: a
    # drive reporting 1 increment per revolution is not a 1-count encoder, it is
    # an uncommissioned object. Treating 1 as real printed "counts_per_rev: 1,
    # = 2^0, 0-bit feedback", which is worse than saying nothing.
    if increments is not None and increments <= 1:
        print(f'\n  0x608F:01 reads {increments}, which is a default rather than '
              f'a resolution.\n'
              '  This drive has no commissioned feedback scaling to read, so\n'
              '  counts_per_rev must come from the encoder datasheet and be\n'
              '  confirmed by turning the shaft on the telemetry profile.')
    elif increments and motor_revs:
        cpr = increments / motor_revs
        print(f'\n  >>> drive_params.counts_per_rev: {cpr:.0f}')
        if cpr.is_integer():
            whole = int(cpr)
            bits = whole.bit_length() - 1
            if 2**bits == whole:
                print(f'      = 2^{bits}, i.e. {bits}-bit feedback')
        else:
            print(f'      NOTE: not a whole number of counts per revolution '
                  f'({increments}/{motor_revs}) - check 0x6091 gearing.')
    else:
        print('\n  0x608F not implemented; counts_per_rev has to come from the\n'
              '  feedback device datasheet, then be confirmed by turning the\n'
              '  shaft one revolution on the telemetry PDO profile.')

    print('\n  NOTE: this does NOT tell you whether 0x6064 accumulates turns or\n'
          '  wraps every revolution - that is devices.ELMO position_counter_bits\n'
          '  and it is settled by watching logged position across one slow turn.')


def report_modes(slave):
    modes = read_u(slave, 0x6502, 0, 4)
    print('\n  --- supported modes (0x6502) ---------------------------------')
    if modes is None:
        print('  not implemented; mode support has to be taken from the ESI/manual')
        return
    print(f'  raw 0x{modes:08X}')
    for bit, label, op_value in DRIVE_MODES:
        mark = 'yes' if modes & (1 << bit) else ' - '
        print(f'    [{mark}] {label:42s} 0x6060 = {op_value}')

    missing = [label for op_value, label in ELMO_REQUIRED_MODES
               for bit, _lbl, ov in DRIVE_MODES if ov == op_value
               and not modes & (1 << bit)]
    if missing:
        print(f'\n  >>> devices.ELMO drives csp/csv/cst and this drive is missing: '
              f'{", ".join(missing)}.')
        print('      _mode_to_op in devices.ELMO.__init__ has to be repointed at\n'
              '      the profile modes it does support (pp=1 / pv=3 / tq=4), and\n'
              '      mode_dict inverted to match.')
    else:
        print('\n  >>> csp/csv/cst all supported; devices.ELMO._mode_to_op '
              '(8/9/10) is correct.')


def report(slave, position, kt):
    name = read_string(slave, 0x1008) or '(no 0x1008 device name)'
    vendor = read_u(slave, 0x1018, 1, 4)
    product = read_u(slave, 0x1018, 2, 4)
    revision = read_u(slave, 0x1018, 3, 4)

    print(f'\n{"="*66}')
    print(f'slave {position}: {name}')
    print(f'{"="*66}')
    rev_txt = f'   revision 0x{revision:08X}' if revision is not None else ''
    print(f'  0x1018 vendor  {vendor}   product {product}{rev_txt}')

    known = KNOWN_PRODUCT_CODES.get(product)
    if known and known != 'ELMO':
        print(f'\n  >>> already registered in DEVICE_CLASSES as {known} - '
              f'not the Elmo.')
        print(f'      Queried because it answers DS402; the scaling below is '
              f'{known}\'s, not the Elmo\'s.')
    elif product is not None:
        print(f'\n  >>> devices.ELMO_PRODUCT_CODE = {product}'
              + ('  (already set)' if known == 'ELMO' else ''))
    print(f'  >>> absorbers.yaml key (0x1008): {name!r}')

    report_torque_scaling(slave, kt)
    report_position_scaling(slave)
    report_modes(slave)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Read-only DS402 scaling query for ELMO bringup.')
    parser.add_argument('--kt', type=float, default=None,
                        help='motor torque constant, Nm/A. Required to evaluate '
                             'the two i_rated candidates.')
    parser.add_argument('--slave', type=int, default=None,
                        help='only query this chain position (default: every '
                             'slave that answers 0x6075 or 0x6502)')
    args = parser.parse_args(argv)

    with open(f'{dyno_paths.dyno_config_directory}/master_config.yaml') as f:
        ifname = yaml.safe_load(f)['interface']

    master = pysoem.Master()
    try:
        master.open(ifname)
    except Exception as e:
        print(f'could not open {ifname}: {e}')
        print('  - link up?      ip -br link show ' + ifname)
        print('  - bring it up?  sudo systemctl start ethercat-link@' + ifname)
        print('  - capabilities? getcap .venv/bin/python')
        return 1

    try:
        count = master.config_init()
        if count <= 0:
            print(f'{ifname} opened, but config_init() found no slaves '
                  f'(returned {count}). Check chain power and the IN/OUT port '
                  f'direction on the first coupler.')
            return 1

        master.state = pysoem.PREOP_STATE
        master.write_state()
        master.state_check(pysoem.PREOP_STATE, timeout=500_000)

        print(f'\n{count} slave(s) on {ifname}')
        if args.kt is None:
            print('NOTE: no --kt given; the i_rated candidates need it and will '
                  'be skipped.')

        # Selected by whether the slave answers a DS402 object rather than by
        # product code, because ELMO_PRODUCT_CODE is one of the things this
        # script exists to find out - requiring it first would be circular.
        queried = 0
        for i, slave in enumerate(master.slaves):
            if args.slave is not None and i != args.slave:
                continue
            if args.slave is None:
                probe = (read_u(slave, 0x6075, 0, 4) is not None
                         or read_u(slave, 0x6502, 0, 4) is not None
                         or read_u(slave, 0x6041, 0, 2) is not None)
                if not probe:
                    continue
            try:
                report(slave, i, args.kt)
                queried += 1
            except Exception as e:
                print(f'\nslave {i}: query failed: {type(e).__name__}: {e}')

        if not queried:
            print('\nNo slave answered a DS402 object. Run '
                  './dyno/utilities/bus_scan.sh to see what is on the chain, '
                  'or pass --slave <n> to force a query.')
            return 1
        return 0
    finally:
        master.close()


if __name__ == '__main__':
    sys.exit(main())
