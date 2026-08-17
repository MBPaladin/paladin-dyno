"""Read-only EtherCAT bus scan for rig bringup.

Opens the NIC named in master_config.yaml, runs config_init() and reports every
slave found, in chain order, with the model name inferred from its product
code. Then prints a ready-to-paste `expected_slave_layout:` block.

This is deliberately inert: config_init() reads slave EEPROM and leaves the
chain in PRE-OP. No PDOs are mapped, no process data is exchanged, no drive is
enabled and nothing is energized. It is safe to run on a live rig.

Usage:  ./dyno/bus_scan.sh
"""
import sys
import yaml
import pysoem

from deployment import dyno_paths
from dyno.src.devices import DEVICE_CLASSES

# Product code -> model name(s). Several logical models can share a class but
# never a product code, so this stays 1:1 in practice; a list is kept anyway so
# an accidental collision is visible rather than silently resolved.
BY_PRODUCT_CODE = {}
for _model, _spec in DEVICE_CLASSES.items():
    BY_PRODUCT_CODE.setdefault(_spec['id'], []).append(_model)

# Slaves the layout names individually rather than by model, because the
# control code reaches for them by name (devices.LOAD, devices.DUT).
AKD_HINTS = ['LOAD', 'DUT']


def main():
    with open(f"{dyno_paths.dyno_config_directory}/master_config.yaml") as f:
        ifname = yaml.safe_load(f)['interface']

    master = pysoem.Master()
    try:
        master.open(ifname)
    except Exception as e:
        print(f"could not open {ifname}: {e}")
        print("  - link up?      ip -br link show " + ifname)
        print("  - capabilities? getcap .venv/bin/python")
        return 1

    try:
        count = master.config_init()
        if count <= 0:
            print(f"{ifname} opened, but config_init() found no slaves "
                  f"(returned {count}). Check chain power and the IN/OUT port "
                  f"direction on the first coupler.")
            return 1

        print(f"\n{count} slave(s) found on {ifname}\n")
        # Revision is worth a column of its own: two AKDs of different hardware
        # revisions share a product code but not a process-data budget, and the
        # Rev B drives are the ones that drop the status word out of the PDO.
        # See AKD_PDO_PROFILES in devices.py.
        print(f"{'pos':>3}  {'model':<12} {'product':>10} {'rev':>10} {'vendor':>8}"
              f"  name reported by slave")
        print('-' * 82)

        layout = []
        akd_seen = 0
        for i, slave in enumerate(master.slaves):
            models = BY_PRODUCT_CODE.get(slave.id, [])
            if models:
                model = '/'.join(models)
                known = True
            else:
                model = 'UNKNOWN'
                known = False
            rev = f'0x{slave.rev:08X}'
            print(f"{i:>3}  {model:<12} {slave.id:>10} {rev:>10} {slave.man:>8}"
                  f"  {slave.name}")

            if not known:
                layout.append((None, f"# pos {i}: UNKNOWN product code {slave.id} "
                                     f"(vendor {slave.man}, reports '{slave.name}') "
                                     f"- add it to DEVICE_CLASSES in devices.py"))
                continue

            chosen = models[0]
            if chosen == 'AKD':
                name = AKD_HINTS[akd_seen] if akd_seen < len(AKD_HINTS) else f'AKD_{akd_seen}'
                akd_seen += 1
                layout.append((chosen, f"  - model: AKD\n    name: {name}\n"
                                       f"    params: {{}}  # TODO: motor_params / motor_limits / drive_params"))
            else:
                layout.append((chosen, f"  - {{ model: {chosen}, name: TODO_{i} }}"))

        print('\n' + '=' * 72)
        print('expected_slave_layout:   # paste into your <mode>_dyno_config.yaml')
        print('=' * 72)
        for _, line in layout:
            print(line)
        print()
        if any(m is None for m, _ in layout):
            print('NOTE: at least one slave was not recognised - see the comments above.')
        if akd_seen < 2:
            print(f'NOTE: {akd_seen} AKD drive(s) found. The stock control loop expects '
                  'both LOAD and DUT; see load_only in the config if DUT is absent.')
        return 0
    finally:
        master.close()


if __name__ == '__main__':
    sys.exit(main())
