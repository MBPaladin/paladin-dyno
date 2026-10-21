"""Shared helpers for resolving dyno configuration.

The GUI, the Controller (master), and the Logger each load the dyno config in
their own process and must agree EXACTLY on the ordered telemetry key list —
telemetry rows cross the process queues positionally, so any divergence means
silently misaligned plots and logs. This module is the single source of that
ordering (it was previously copy-pasted in all three places).
"""


import os


def _mark_leaves(provenance, path, value, src):
    if isinstance(value, dict):
        for k, v in value.items():
            _mark_leaves(provenance, f'{path}.{k}', v, src)
    else:
        provenance[path] = src


def deep_merge(base, override, base_src='dyno config', override_src='override'):
    """Recursive dict merge — override wins per key, but only for keys it
    actually defines (unlike wholesale dict replacement). Returns
    (merged, provenance) where provenance maps dotted leaf keys to the
    source label each final value came from."""
    provenance = {}

    def _merge(b, o, prefix):
        merged = {}
        for k in list(b.keys()) + [k for k in o.keys() if k not in b]:
            path = f'{prefix}{k}'
            if k in o and isinstance(o[k], dict) and isinstance(b.get(k), dict):
                merged[k] = _merge(b[k], o[k], path + '.')
            elif k in o:
                merged[k] = o[k]
                _mark_leaves(provenance, path, o[k], override_src)
            else:
                merged[k] = b[k]
                _mark_leaves(provenance, path, b[k], base_src)
        return merged

    return _merge(base or {}, override or {}, ''), provenance


def get_dotted(params, dotted_key):
    node = params
    for part in dotted_key.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def validate_required_params(params, required_dotted_keys, context):
    """Raise a clear error naming the config source if required keys are
    missing — instead of a KeyError from deep inside device bring-up."""
    missing = [k for k in required_dotted_keys if get_dotted(params, k) is None]
    if missing:
        raise ValueError(
            f'{context}: missing required parameter(s) {missing}. Check the '
            f'device params in the dyno config and/or its absorbers.yaml entry.')


def print_params_provenance(title, params, provenance):
    """Bring-up provenance table: each resolved parameter, its final value,
    and which source it came from."""
    print(f'\t--- resolved params for {title} ---')
    flat = {}
    _flatten(params, '', flat)
    for key in sorted(flat):
        src = provenance.get(key, '?')
        print(f'\t{key:38s} = {flat[key]!r:<12}  <- {src}')


def _flatten(node, prefix, out):
    if isinstance(node, dict):
        for k, v in node.items():
            _flatten(v, f'{prefix}{k}.' if isinstance(v, dict) else f'{prefix}{k}', out)
    else:
        out[prefix] = node


def augment_log_keys(dyno_params, verbose=False):
    """Append auto-logged sensor entries to dyno_params['log_keys'] in place.

    For every entry in the config's `sensors:` section, derives the telemetry
    attribute path (devices.<module>.<sensor>) via the panel_ports routing and
    appends [sensor_name, path] to log_keys unless already present.
    Idempotent. Returns the ordered list of key names.
    """
    log_keys = dyno_params.setdefault('log_keys', [])
    for sensor_name, config in dyno_params.get('sensors', {}).items():
        if 'port' in config:
            port_map = dyno_params.get('panel_ports', {}).get(config['port'])
            module_name = port_map['signal_module'] if port_map else None
        else:
            module_name = config.get('signal_module')
        if module_name is None:
            continue

        existing_keys = [k[0] for k in log_keys]
        if sensor_name not in existing_keys:
            log_path = f"devices.{module_name}.{sensor_name}"
            log_keys.append([sensor_name, log_path])
            if verbose:
                print(f"\tAuto-logged sensor: {sensor_name} -> {log_path}")

    return [entry[0] for entry in log_keys]


FEEDFORWARD_MODES = ('off', 'fixed', 'quadrant')


def compile_feedforward(dyno_params):
    """The rig's cross-coupling torque feedforward settings, validated.

    Config is either the legacy scalar `feedforward_ratio: r` (mode fixed) or a
    `feedforward:` block -- never both, so there is one place to look:

        feedforward:
          mode: quadrant        # off | fixed | quadrant
          ratio: 0.8            # fixed mode; also the DUT-side term in any mode
          forward_ratio: 1.02   # quadrant: DUT drives the load
          backdrive_ratio: 2.73 # quadrant: the load drives the DUT
          drag_nm: 1.38         # quadrant: LOAD torque along the motion, any torque
          v_blend_rad_s: 0.005  # quadrant: tanh width across a speed reversal
          max_nm: 30            # optional clip on the LOAD feedforward; default
                                # is LOAD's motor_limits.torque

    Neither key means mode fixed at 0.8, the historical code default. See
    Controller._load_feedforward for what each term does and how its sign is
    derived. Pure function of the config, so Master can record the result in
    resolved_config.json before the controller exists.
    """
    spec = dyno_params.get('feedforward')
    legacy = dyno_params.get('feedforward_ratio')
    if spec is None:
        return {'mode': 'fixed', 'ratio': float(0.8 if legacy is None else legacy),
                'source': 'feedforward_ratio' if legacy is not None else 'code default'}
    if legacy is not None:
        raise ValueError('dyno config sets both feedforward: and feedforward_ratio; '
                         'keep one (feedforward.ratio replaces feedforward_ratio)')
    if not isinstance(spec, dict):
        raise ValueError('feedforward: must be a mapping, see config_utils.compile_feedforward')
    mode = spec.get('mode', 'fixed')
    if mode not in FEEDFORWARD_MODES:
        raise ValueError(f'feedforward.mode must be one of {FEEDFORWARD_MODES}, got {mode!r}')
    out = {'mode': mode, 'ratio': float(spec.get('ratio', 0.8)), 'source': 'feedforward'}
    if mode == 'quadrant':
        missing = [k for k in ('forward_ratio', 'backdrive_ratio') if k not in spec]
        if missing:
            raise ValueError(f'feedforward.mode quadrant needs {missing}')
        out.update({'forward_ratio': float(spec['forward_ratio']),
                    'backdrive_ratio': float(spec['backdrive_ratio']),
                    'drag_nm': float(spec.get('drag_nm', 0.0)),
                    'v_blend_rad_s': float(spec.get('v_blend_rad_s', 0.005))})
        if out['v_blend_rad_s'] <= 0:
            raise ValueError('feedforward.v_blend_rad_s must be > 0')
        if out['forward_ratio'] < 0 or out['backdrive_ratio'] < 0 or out['drag_nm'] < 0:
            raise ValueError('feedforward ratios and drag_nm are magnitudes: give them >= 0; '
                             'the sign is derived from the drive flags')
    if spec.get('max_nm') is not None:
        out['max_nm'] = abs(float(spec['max_nm']))
    return out
