"""Human-readable experimental setup tables for HDF5 logs.

The root `resolved_config` attribute is the machine-readable record of a run
(post-processing parses it with json.loads). It is unreadable by eye, though:
HDFView renders a scalar string attribute on a single line, so every newline
shows up as a literal \\n and every quote comes back backslashed.

This module renders the same dict into a `/setup` group that HDFView displays
as tables:

    /setup/overview     one line per element -> reads like a cover page
    /setup/parameters   DEVICE | TYPE | PARAMETER | VALUE | UNITS |
                        DESCRIPTION | SOURCE

Both are fixed-length ASCII, which every HDF5 viewer renders as a plain grid.
`resolved_config` is left untouched so nothing downstream breaks.
"""

import json
import os

import h5py
import numpy as np


# Units and plain-English meaning for every parameter we know about, keyed by
# the trailing dotted path. Anything unknown still lands in the table, just
# with blank units/description -- add entries here as configs grow.
PARAM_INFO = {
    # --- AKD / AXON drive params ---
    'flip_torque_sign':          ('',            'Torque sign inverted for this motor\'s wiring'),
    'gear_ratio':                (':1',          'Gearing between motor and output (1 = direct drive)'),
    'type':                      ('',            'Absorber family from absorbers.yaml'),
    'esi_file_name':             ('',            'EtherCAT device description file used for this drive'),
    'motor_params.kt':           ('Nm/A_rms',    'Torque constant - torque produced per amp'),
    'motor_params.k_tanh':       ('',            'Shapes the torque saturation curve (used with kt)'),
    'motor_limits.torque':       ('Nm',          'Torque commands beyond this are clipped'),
    'motor_limits.velocity':     ('rad/s',       'Velocity commands beyond this are clipped'),
    'motor_limits.acceleration': ('rad/s^2',     'Acceleration commands beyond this are clipped'),
    'motor_limits.rotatum':      ('Nm/s',        'Limit on how fast torque may change'),
    'drive_params.i_cont':       ('A_rms',       'Drive continuous current rating (must match drive setting)'),
    # --- sensor channel params ---
    'port':                      ('',            'Front panel connector the sensor plugs into'),
    'supply':                    ('',            'Excitation supply switched on for this sensor'),
    'fs_v':                      ('V',           'Analog input range (full scale voltage)'),
    'fs_pos':                    ('sensor units', 'Reading at full scale voltage (e.g. Nm for a torque cell)'),
    'offset':                    ('sensor units', 'Constant added to every reading (zero offset)'),
    'sensor_type':               ('',            'RTD element type'),
}

# Column order for /setup/parameters.
COLUMNS = ('DEVICE', 'TYPE', 'PARAMETER', 'VALUE', 'UNITS', 'DESCRIPTION', 'SOURCE')

# Drives get listed before sensors before everything else.
_DRIVE_CLASSES = ('AKD', 'AXON')


def _looks_like_channels(params):
    """True when a module's params are keyed by channel (ch1..ch8) rather than
    being parameters of the module itself -- that is how master.py injects
    sensor configs into ADC/RTD modules."""
    if not params:
        return False
    return all(k.startswith('ch') and k[2:].isdigit() and isinstance(v, dict)
               for k, v in params.items())


def _flatten(node, prefix, out):
    if isinstance(node, dict):
        for k, v in node.items():
            _flatten(v, f'{prefix}{k}.' if isinstance(v, dict) else f'{prefix}{k}', out)
    else:
        out[prefix] = node


def _fmt(value):
    """Format a value for a reader who is not going to enjoy `True` or
    `1e-05`."""
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        text = f'{value:.6f}'.rstrip('0').rstrip('.')
        return text or '0'
    return str(value)


def _info(dotted):
    """Look up units/description by the longest matching suffix, so both
    'motor_params.kt' and a bare 'fs_v' resolve."""
    if dotted in PARAM_INFO:
        return PARAM_INFO[dotted]
    parts = dotted.split('.')
    for i in range(1, len(parts)):
        tail = '.'.join(parts[i:])
        if tail in PARAM_INFO:
            return PARAM_INFO[tail]
    return ('', '')


def build_rows(resolved):
    """Flatten the resolved config into (DEVICE, TYPE, PARAMETER, VALUE, UNITS,
    DESCRIPTION, SOURCE) tuples. Sensors are lifted out of their host module so
    they read as 'load_torque / ELM3004 ch1' instead of 'adc_1 / ch1.fs_v'.
    Devices with no parameters are skipped -- see build_overview."""
    drives, sensors, other = [], [], []

    for device_name, entry in resolved.get('devices', {}).items():
        klass = entry.get('class', '?')
        params = entry.get('params') or {}
        provenance = entry.get('provenance') or {}
        if not params:
            continue

        if _looks_like_channels(params):
            for channel, channel_params in sorted(params.items()):
                # master.py stamps the yaml key onto every sensor config as
                # 'name' before routing it to the module.
                label = channel_params.get('name') or f'{device_name} {channel}'
                source = f'dyno config: sensors.{label}'
                for key in sorted(k for k in channel_params if k != 'name'):
                    units, desc = _info(key)
                    sensors.append((label, f'{klass} {channel}', key,
                                    _fmt(channel_params[key]), units, desc, source))
            continue

        flat = {}
        _flatten(params, '', flat)
        bucket = drives if klass in _DRIVE_CLASSES else other
        for key in sorted(flat):
            units, desc = _info(key)
            bucket.append((device_name, klass, key, _fmt(flat[key]), units, desc,
                           provenance.get(key, '')))

    return drives + sensors + other


def build_overview(resolved, meta=None):
    """Cover-page lines: what ran, when, and what hardware was on the bus."""
    meta = meta or {}
    lines = ['EXPERIMENTAL SETUP', '']

    header = [('Test', meta.get('test_name')),
              ('Log folder', meta.get('log_dir')),
              ('Rig mode', resolved.get('mode')),
              ('Config resolved at', resolved.get('timestamp')),
              ('Logged at', meta.get('logged_at')),
              ('Duration', meta.get('duration')),
              ('Samples', meta.get('samples')),
              ('DUT serial', meta.get('dut_serial'))]
    for label, value in header:
        if value not in (None, ''):
            lines.append(f'{label + ":":<20}{value}')

    notes = meta.get('notes')
    if notes:
        lines += ['', 'Operator notes:']
        lines += [f'    {line}' for line in str(notes).strip().splitlines()]

    # Everything present on the bus but carrying no parameters -- worth
    # recording that it was there, not worth four table rows each.
    bare = [f"{name} ({entry.get('class', '?')})"
            for name, entry in resolved.get('devices', {}).items()
            if not (entry.get('params') or {})]
    if bare:
        lines += ['', 'Also on the EtherCAT bus (no configurable parameters):']
        lines += [f'    {name}' for name in bare]

    lines += ['', 'Per-parameter detail is in /setup/parameters.',
              'The exact machine-readable config is in the root '
              '"resolved_config" attribute.']
    return lines


def _string_array(rows, columns=1):
    """Fixed-length ASCII arrays render as a plain grid in every HDF5 viewer;
    variable-length strings inside compound types do not, reliably."""
    if columns == 1:
        width = max((len(r) for r in rows), default=1)
        return np.array([r.encode('ascii', 'replace') for r in rows],
                        dtype=f'S{max(width, 1)}')
    dtype = np.dtype([(name, f'S{max(max((len(r[i]) for r in rows), default=1), len(name))}')
                      for i, name in enumerate(COLUMNS)])
    return np.array([tuple(c.encode('ascii', 'replace') for c in row) for row in rows],
                    dtype=dtype)


def write_setup(h5file, resolved, meta=None):
    """Create (or replace) the /setup group on an open h5py File."""
    if 'setup' in h5file:
        del h5file['setup']
    group = h5file.create_group('setup')

    overview = build_overview(resolved, meta)
    group.create_dataset('overview', data=_string_array(overview))

    rows = build_rows(resolved)
    if rows:
        group.create_dataset('parameters', data=_string_array(rows, columns=len(COLUMNS)))
    return group


def render_text(resolved, meta=None):
    """The same content as plain text, for stdout or a sidecar file."""
    rows = build_rows(resolved)
    widths = [max(len(c), max((len(r[i]) for r in rows), default=0))
              for i, c in enumerate(COLUMNS)]
    out = list(build_overview(resolved, meta)) + ['']
    out.append('  '.join(c.ljust(widths[i]) for i, c in enumerate(COLUMNS)))
    out.append('  '.join('-' * widths[i] for i in range(len(COLUMNS))))
    out += ['  '.join(r[i].ljust(widths[i]) for i in range(len(COLUMNS))) for r in rows]
    return '\n'.join(out)


def _meta_from_log(path, h5file):
    """Best-effort context for a log that already exists on disk."""
    folder = os.path.dirname(os.path.abspath(path))
    base = os.path.splitext(os.path.basename(path))[0]
    meta = {'test_name': base, 'log_dir': os.path.basename(folder)}

    notes_path = os.path.join(folder, base + '.txt')
    if os.path.exists(notes_path):
        with open(notes_path, 'r') as f:
            meta['notes'] = f.read()

    if 'time' in h5file and h5file['time'].shape[0]:
        t = h5file['time']
        meta['samples'] = f'{t.shape[0]:,}'
        meta['duration'] = f'{float(t[-1]) - float(t[0]):.1f} s'
    return meta


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', help='path to a .hdf5 log')
    parser.add_argument('--in-place', action='store_true',
                        help='write /setup into the log itself (default: print only)')
    parser.add_argument('--out', help='write /setup into a copy at this path')
    args = parser.parse_args(argv)

    with h5py.File(args.log, 'r') as f:
        if 'resolved_config' not in f.attrs:
            parser.error(f'{args.log} has no resolved_config attribute')
        resolved = json.loads(f.attrs['resolved_config'])
        meta = _meta_from_log(args.log, f)

    print(render_text(resolved, meta))

    target = None
    if args.out:
        import shutil
        shutil.copyfile(args.log, args.out)
        target = args.out
    elif args.in_place:
        target = args.log
    if target:
        with h5py.File(target, 'a') as f:
            write_setup(f, resolved, meta)
        print(f'\nWrote /setup into {target}')


if __name__ == '__main__':
    main()
