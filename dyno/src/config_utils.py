"""Shared helpers for resolving dyno configuration.

The GUI, the Controller (master), and the Logger each load the dyno config in
their own process and must agree EXACTLY on the ordered telemetry key list —
telemetry rows cross the process queues positionally, so any divergence means
silently misaligned plots and logs. This module is the single source of that
ordering (it was previously copy-pasted in all three places).
"""


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
