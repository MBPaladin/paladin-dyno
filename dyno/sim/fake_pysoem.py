"""Drop-in fake for the pysoem module: a simulated EtherCAT bus.

Activated by DYNO_SIM=<mode> (set by `gui.py --sim`): master.py imports this
module instead of the real pysoem. Everything above the wire — Master's
bring-up sequence, the 1 kHz loop, the device classes in devices.py, the
controller, GUI, and logger — runs unmodified.

The simulated slaves are built from the same `<mode>_dyno_config.yaml`
expected_slave_layout the Controller loads, so config_init always "finds"
exactly the configured hardware. Each slave has a behavior model that fills
its input buffer every cycle, packing bytes with the REAL ctypes TxPDO
structs imported from devices.py — layouts cannot drift from production code.

Phase 1 scope: rig idles with plausible noisy sensor values; AKD drives run a
real DS402 state machine (enable/disable/fault-reset/mode-switch handshakes
all work). No coupled mechanical plant yet — commanded torque does not yet
spin a simulated shaft.
"""
import ctypes
import os
import random
import time

import yaml

# ---- pysoem API constants (values match real pysoem/EtherCAT AL states) ----
NONE_STATE = 0x00
INIT_STATE = 0x01
PREOP_STATE = 0x02
BOOT_STATE = 0x03
SAFEOP_STATE = 0x04
OP_STATE = 0x08
STATE_ERROR = 0x10
STATE_ACK = 0x10

CYCLE_S_DEFAULT = 0.001


# --------------------------------------------------------------------------
# Device behavior models
# --------------------------------------------------------------------------
class Behavior:
    """Base: no IO. Subclasses fill input bytes each cycle from output bytes."""
    input_size = 0
    output_size = 0

    def __init__(self, layout_entry):
        self.entry = layout_entry
        self.sdo_log = []  # (index, subindex, data) writes, for debugging

    def sdo_write(self, index, subindex, data, ca=False):
        self.sdo_log.append((index, subindex, bytes(data)))

    def sdo_read(self, index, subindex, size=None, ca=False):
        return b'\x00' * (size if size else 4)

    def step(self, dt_s, output_buf):
        return b'\x00' * self.input_size


class CouplerBehavior(Behavior):
    pass


class DigitalOutBehavior(Behavior):  # EL2002 / EL2004
    output_size = 1


class DigitalInBehavior(Behavior):  # EL1002
    input_size = 1


class ELM300xBehavior(Behavior):
    """ELM3002/ELM3004 measurement terminal.

    Reproduces the real terminal's observable behavior: a power-on settling
    window with TxPDO State = invalid and 'No of Samples' = 0, then valid
    samples with the 2-bit input-cycle counter advancing each cycle. Channel
    values are zero-mean gaussian noise in raw counts (scaled to engineering
    units by devices.py: eng = fs_pos * counts / 7812500).
    """
    SETTLE_CYCLES = 500  # real ELM3004 takes ~2.2 s; shortened for sim startup

    def __init__(self, layout_entry, channels, noise_counts=4000.0):
        super().__init__(layout_entry)
        self.channels = channels
        self.noise_counts = noise_counts
        self.input_size = 8 * channels  # per ch: u8 nsamp, u8 status, u16 pad, i32 value
        self.cycles = 0

    def step(self, dt_s, output_buf):
        self.cycles += 1
        buf = bytearray(self.input_size)
        settling = self.cycles < self.SETTLE_CYCLES
        for ch in range(self.channels):
            off = ch * 8
            if settling:
                buf[off] = 0          # No of Samples
                buf[off + 1] = 0x20   # TxPDO State: data invalid
            else:
                buf[off] = 1
                buf[off + 1] = (self.cycles & 3) << 6  # 2-bit input cycle counter
                val = int(random.gauss(0.0, self.noise_counts))
                buf[off + 4:off + 8] = val.to_bytes(4, 'little', signed=True)
        return bytes(buf)


class AKDBehavior(Behavior):
    """Kollmorgen AKD servo drive: DS402 state machine + mode echo + idle
    telemetry with noise. Understands the exact RxPDO/TxPDO layouts from
    devices.AKD (imported lazily to avoid import cycles)."""

    # statusword templates; bit masks per devices.AKD.update_modes
    SW_FAULT = 0x0008
    SW_SOD = 0x0050         # switch_on_disabled | voltage_enabled
    SW_READY = 0x0031       # ready | quick_stop(inactive=1) | voltage
    SW_SWITCHED_ON = 0x0033
    SW_ENABLED = 0x0037

    MODE_ECHO_DELAY = 3  # cycles between control_mode write and op_mode echo

    def __init__(self, layout_entry):
        super().__init__(layout_entry)
        from dyno.src.devices import AKD  # real structs, guaranteed layout match
        self._RxPDO = AKD.RxPDO
        self._TxPDO = AKD.TxPDO
        self.input_size = ctypes.sizeof(AKD.TxPDO)
        self.output_size = ctypes.sizeof(AKD.RxPDO)
        self.state = 'switch_on_disabled'
        self.fault = False
        self.op_mode = 0
        self._pending_mode = None
        self._pending_countdown = 0
        self.position_counts = 1000  # non-zero so drive-side zeroing logic runs
        self.drive_name = f"SIM-{layout_entry['name']}"

    def sdo_read(self, index, subindex, size=None, ca=False):
        if index == 0x2031:  # drive name string, keyed into absorbers.yaml
            return (self.drive_name + '\x00').encode()
        if index == 0x3598:  # IL.KP, printed during setup
            return (1000).to_bytes(4, 'little')
        return super().sdo_read(index, subindex, size, ca)

    def _statusword(self):
        base = {'switch_on_disabled': self.SW_SOD,
                'ready_to_switch_on': self.SW_READY,
                'switched_on': self.SW_SWITCHED_ON,
                'operation_enabled': self.SW_ENABLED}[self.state]
        return base | (self.SW_FAULT if self.fault else 0)

    def _apply_controlword(self, cw):
        if cw & 0x0080:  # fault reset
            self.fault = False
            self.state = 'switch_on_disabled'
            return
        masked = cw & 0x0F
        if masked == 0x06:  # shutdown -> ready to switch on
            if not self.fault:
                self.state = 'ready_to_switch_on'
        elif masked == 0x07:  # switch on / disable operation
            if self.state in ('ready_to_switch_on', 'operation_enabled'):
                self.state = 'switched_on'
        elif masked in (0x0F, 0x1F & 0x0F):  # enable operation
            if self.state in ('switched_on', 'operation_enabled'):
                self.state = 'operation_enabled'
        elif masked == 0x00:  # disable voltage
            self.state = 'switch_on_disabled'

    def step(self, dt_s, output_buf):
        rx = self._RxPDO()
        if output_buf and len(output_buf) == self.output_size:
            rx = self._RxPDO.from_buffer_copy(output_buf)
            self._apply_controlword(rx.controlword)
            # op-mode change echoes back after a short, realistic delay
            if rx.control_mode != self.op_mode and rx.control_mode != 0:
                if self._pending_mode != rx.control_mode:
                    self._pending_mode = rx.control_mode
                    self._pending_countdown = self.MODE_ECHO_DELAY
                else:
                    self._pending_countdown -= 1
                    if self._pending_countdown <= 0:
                        self.op_mode = self._pending_mode
                        self._pending_mode = None

        tx = self._TxPDO()
        tx.statusword = self._statusword()
        tx.op_mode = self.op_mode
        tx.actual_position = self.position_counts
        enabled = self.state == 'operation_enabled'
        # Idle telemetry: echo commands when enabled (no plant yet), else noise
        if enabled and self.op_mode == 4:
            tx.actual_current = rx.current_command + int(random.gauss(0, 15))
            tx.actual_velocity = int(random.gauss(0, 3))
        elif enabled and self.op_mode == 3:
            tx.actual_velocity = rx.velocity_command + int(random.gauss(0, 3))
            tx.actual_current = int(random.gauss(0, 15))
        else:
            tx.actual_velocity = int(random.gauss(0, 3))       # milli-RPM
            tx.actual_current = int(random.gauss(0, 15))       # mA
        tx.i2t_counter = 0
        return bytes(tx)


class RTDBehavior(Behavior):
    """EL3208: constant plausible temperatures (only used by production cfg)."""
    def __init__(self, layout_entry):
        super().__init__(layout_entry)
        from dyno.src.devices import EL3208
        self.input_size = ctypes.sizeof(EL3208.TxPDO)


BEHAVIOR_FOR_MODEL = {
    'EK1100': lambda e: CouplerBehavior(e),
    'EL2002': lambda e: DigitalOutBehavior(e),
    'EL2004': lambda e: DigitalOutBehavior(e),
    'EL2004_12v': lambda e: DigitalOutBehavior(e),
    'EL1002': lambda e: DigitalInBehavior(e),
    'ELM3002': lambda e: ELM300xBehavior(e, channels=2),
    'ELM3004': lambda e: ELM300xBehavior(e, channels=4),
    'EL3208': lambda e: RTDBehavior(e),
    'AKD': lambda e: AKDBehavior(e),
}


# --------------------------------------------------------------------------
# pysoem API fakes
# --------------------------------------------------------------------------
class _SimSlave:
    def __init__(self, layout_entry, product_id):
        self.name = layout_entry['model']
        self.id = product_id
        self.man = 0x5A5A5A5A
        self.rev = 0
        self.state = INIT_STATE
        self.al_status = 0
        self.is_lost = False
        model = layout_entry['model']
        if model not in BEHAVIOR_FOR_MODEL:
            raise NotImplementedError(
                f'No sim behavior for device model {model!r} yet — add one to '
                f'dyno/sim/fake_pysoem.py BEHAVIOR_FOR_MODEL')
        self._behavior = BEHAVIOR_FOR_MODEL[model](layout_entry)
        self.input = b''
        self.output = b''

    def sdo_write(self, index, subindex, data, ca=False):
        self._behavior.sdo_write(index, subindex, data, ca)

    def sdo_read(self, index, subindex, size=None, ca=False):
        return self._behavior.sdo_read(index, subindex, size, ca)

    def dc_sync(self, act, sync0_cycle_time, sync0_shift_time=0):
        pass

    def state_check(self, expected_state, timeout=2000):
        return self.state

    def reconfig(self, timeout=500):
        return True

    def recover(self, timeout=500):
        return True


class Master:
    def __init__(self):
        self.slaves = []
        self.state = NONE_STATE
        self._opened = False

    # -- lifecycle ---------------------------------------------------------
    def open(self, ifname, ifname_red=None):
        self._opened = True

    def close(self):
        self._opened = False

    def config_init(self):
        from dyno.src.devices import DEVICE_CLASSES
        from deployment import dyno_paths
        mode = os.environ.get('DYNO_SIM')
        cfg = f'{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml'
        with open(cfg, 'r') as f:
            layout = yaml.safe_load(f)['expected_slave_layout']
        self.slaves = [_SimSlave(entry, DEVICE_CLASSES[entry['model']]['id'])
                       for entry in layout]
        for s in self.slaves:
            s.state = PREOP_STATE
        self.state = PREOP_STATE
        return len(self.slaves)

    # -- state machine -----------------------------------------------------
    def write_state(self):
        target = self.state & 0x0F
        if target:
            for s in self.slaves:
                s.state = target

    def state_check(self, expected_state, timeout=2000):
        return self.state & 0x0F

    def read_state(self):
        return self.state & 0x0F

    # -- configuration -----------------------------------------------------
    def config_map(self):
        for s in self.slaves:
            s.input = b'\x00' * s._behavior.input_size
            s.output = b'\x00' * s._behavior.output_size
        return sum(s._behavior.input_size + s._behavior.output_size
                   for s in self.slaves)

    def config_dc(self):
        return True

    # -- cyclic exchange ----------------------------------------------------
    def send_processdata(self):
        pass

    def receive_processdata(self, timeout=2000):
        for s in self.slaves:
            s.input = s._behavior.step(CYCLE_S_DEFAULT, s.output)
        return self.expected_wkc

    @property
    def expected_wkc(self):
        return 3

    @property
    def dc_time(self):
        # Reference-slave DC clock: monotonic ns is a perfectly synchronized,
        # jitter-free stand-in; the master's DC servo locks to it happily.
        return time.monotonic_ns()
