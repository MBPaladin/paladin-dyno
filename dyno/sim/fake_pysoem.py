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

Phase 2: a shared single-DOF Plant (dyno/sim/plant.py) couples the drives —
AKD behaviors decode their commanded currents/velocities/positions into
torques on the shaft, the plant integrates, and the ADC / encoder / RTD
behaviors read shaft torque, position, and stator temperature back out, with
noise. Tests execute closed-loop. Fake AKDs report drive names SIM-<name>,
which intentionally do NOT match absorbers.yaml, so sim params resolve from
the running rig config alone.
"""
import ctypes
import math
import os
import random
import time

import yaml

from dyno.sim.plant import Plant, load_sim_params
from dyno.src.config_utils import deep_merge

# ---- pysoem API constants (values match real pysoem/EtherCAT AL states) ----
NONE_STATE = 0x00
INIT_STATE = 0x01
PREOP_STATE = 0x02
BOOT_STATE = 0x03
SAFEOP_STATE = 0x04
OP_STATE = 0x08
STATE_ERROR = 0x10
STATE_ACK = 0x10

CYCLE_S = 0.001


def wrap_i32(value):
    return ((int(value) + 2**31) % 2**32) - 2**31


# --------------------------------------------------------------------------
# Device behavior models
# --------------------------------------------------------------------------
class Behavior:
    """Base: no IO. pre_step() consumes output bytes / pushes torques into the
    plant; step() produces the next input bytes after the plant integrates."""
    input_size = 0
    output_size = 0

    def __init__(self, layout_entry):
        self.entry = layout_entry
        self.plant = None
        self.sdo_log = []

    def set_plant(self, plant, is_input_side=False):
        self.plant = plant

    def sdo_write(self, index, subindex, data, ca=False):
        self.sdo_log.append((index, subindex, bytes(data)))

    def sdo_read(self, index, subindex, size=None, ca=False):
        return b'\x00' * (size if size else 4)

    def pre_step(self, output_buf):
        pass

    def step(self, dt_s, output_buf):
        return b'\x00' * self.input_size


class CouplerBehavior(Behavior):
    pass


class DigitalOutBehavior(Behavior):  # EL2002 / EL2004
    output_size = 1


class PWMOutBehavior(Behavior):  # EL2502
    output_size = 4  # 2 x u16 PWM


class DigitalInBehavior(Behavior):  # EL1002
    input_size = 1


class ChannelSensorBehavior(Behavior):
    """Base for terminals whose channels map to named plant quantities via the
    dyno config's sensors/panel_ports routing."""

    def __init__(self, layout_entry):
        super().__init__(layout_entry)
        self.channel_map = {}  # ch_index -> (sensor_name, fs_pos, offset)

    def configure_channel(self, ch_index, sensor_name, fs_pos=1.0, offset=0.0):
        self.channel_map[ch_index] = (sensor_name, fs_pos, offset)

    def read_quantity(self, ch_index):
        if self.plant is None or ch_index not in self.channel_map:
            return None
        name, fs_pos, offset = self.channel_map[ch_index]
        return self.plant.quantity(name), fs_pos, offset


class ELM300xBehavior(ChannelSensorBehavior):
    """ELM3002/ELM3004 measurement terminal: settling window, cycle counter,
    and channel values = mapped plant quantity (inverse of the device scaling
    eng = fs_pos * counts / 7812500 + offset) plus gaussian noise."""
    SETTLE_CYCLES = 500

    def __init__(self, layout_entry, channels, noise_counts=4000.0):
        super().__init__(layout_entry)
        self.channels = channels
        self.noise_counts = noise_counts
        self.input_size = 8 * channels
        self.cycles = 0

    def step(self, dt_s, output_buf):
        self.cycles += 1
        buf = bytearray(self.input_size)
        settling = self.cycles < self.SETTLE_CYCLES
        for ch in range(self.channels):
            off = ch * 8
            if settling:
                buf[off] = 0
                buf[off + 1] = 0x20  # TxPDO State: data invalid
            else:
                buf[off] = 1
                buf[off + 1] = (self.cycles & 3) << 6
                counts = random.gauss(0.0, self.noise_counts)
                mapped = self.read_quantity(ch)
                if mapped is not None:
                    eng, fs_pos, offset = mapped
                    counts += (eng - offset) * 7812500.0 / fs_pos
                val = max(-2**31, min(2**31 - 1, int(counts)))
                buf[off + 4:off + 8] = val.to_bytes(4, 'little', signed=True)
        return bytes(buf)


class RTDBehavior(ChannelSensorBehavior):
    """EL3208: mapped channels read plant temperatures, others ambient (0.1C/count)."""

    def __init__(self, layout_entry):
        super().__init__(layout_entry)
        from dyno.src.devices import EL3208
        self.input_size = ctypes.sizeof(EL3208.TxPDO)

    def step(self, dt_s, output_buf):
        buf = bytearray(self.input_size)
        for ch in range(8):
            mapped = self.read_quantity(ch)
            temp_c = mapped[0] if mapped is not None else 25.0
            counts = int((temp_c + random.gauss(0, 0.05)) * 10)
            buf[ch * 4 + 2:ch * 4 + 4] = counts.to_bytes(2, 'little', signed=True)
        return bytes(buf)


class EnDatEncoderBehavior(Behavior):
    """EL5042: ch1 reads the output shaft position (device negates and scales
    by 2pi/2^32, so we pre-negate to make devices.encoder.position = theta)."""

    def __init__(self, layout_entry):
        super().__init__(layout_entry)
        from dyno.src.devices import EL5042
        self._TxPDO = EL5042.TxPDO
        self.input_size = ctypes.sizeof(EL5042.TxPDO)

    def step(self, dt_s, output_buf):
        tx = self._TxPDO()
        if self.plant is not None:
            tx.ch1_position = int(-self.plant.theta / (2 * math.pi) * 2**32)
        return bytes(tx)


class AKDBehavior(Behavior):
    """Kollmorgen AKD servo drive: DS402 state machine, mode echo, and a
    drive model that converts commands into shaft torque on the shared plant.

    Command decoding mirrors devices.AKD.send_command exactly (tanh current
    saturation with the SAME resolved kt/k_tanh the device class uses), so
    what the controller commands is what the plant feels.
    """
    SW_FAULT = 0x0008
    SW_SOD = 0x0050
    SW_READY = 0x0031
    SW_SWITCHED_ON = 0x0033
    SW_ENABLED = 0x0037
    MODE_ECHO_DELAY = 3

    def __init__(self, layout_entry):
        super().__init__(layout_entry)
        from dyno.src.devices import AKD
        self._RxPDO = AKD.RxPDO
        self._TxPDO = AKD.TxPDO
        self.input_size = ctypes.sizeof(AKD.TxPDO)
        self.output_size = ctypes.sizeof(AKD.RxPDO)

        self.drive_name = f"SIM-{layout_entry['name']}"
        # Resolve params exactly like devices.AKD does (SIM-* names are absent
        # from absorbers.yaml on purpose, so this is the layout params).
        self.params, _ = deep_merge(layout_entry.get('params', {}) or {}, {})
        mp = self.params.get('motor_params', {})
        self.kt = mp.get('kt', 1.0)
        self.k_tanh = mp.get('k_tanh', 0.02)
        self.i_cont = self.params.get('drive_params', {}).get('i_cont', 8) or 8
        self.gear_ratio = self.params.get('gear_ratio', 1)
        self.tau_limit = self.params.get('motor_limits', {}).get('torque', 10)
        self.flip = self.params.get('flip_torque_sign', False)
        self.coupling = -1.0 if self.flip else 1.0  # mounting direction on shaft
        self.is_input_side = layout_entry['name'] == 'DUT'

        self.state = 'switch_on_disabled'
        self.fault = False
        self.op_mode = 0
        self._pending_mode = None
        self._pending_countdown = 0
        self.tau_motor = 0.0

    # -- SDO ---------------------------------------------------------------
    def sdo_read(self, index, subindex, size=None, ca=False):
        if index == 0x2031:
            return (self.drive_name + '\x00').encode()
        if index == 0x3598:
            return (1000).to_bytes(4, 'little')
        return super().sdo_read(index, subindex, size, ca)

    # -- DS402 -------------------------------------------------------------
    def _statusword(self):
        base = {'switch_on_disabled': self.SW_SOD,
                'ready_to_switch_on': self.SW_READY,
                'switched_on': self.SW_SWITCHED_ON,
                'operation_enabled': self.SW_ENABLED}[self.state]
        return base | (self.SW_FAULT if self.fault else 0)

    def _apply_controlword(self, cw):
        if cw & 0x0080:
            self.fault = False
            self.state = 'switch_on_disabled'
            return
        masked = cw & 0x0F
        if masked == 0x06:
            if not self.fault:
                self.state = 'ready_to_switch_on'
        elif masked == 0x07:
            if self.state in ('ready_to_switch_on', 'operation_enabled'):
                self.state = 'switched_on'
        elif masked == 0x0F:
            if self.state in ('switched_on', 'operation_enabled'):
                self.state = 'operation_enabled'
        elif masked == 0x00:
            self.state = 'switch_on_disabled'

    # -- drive model ---------------------------------------------------------
    def _motor_state(self):
        """Motor-frame kinematics from the shared shaft."""
        omega = self.coupling * self.gear_ratio * self.plant.omega
        theta = self.coupling * self.gear_ratio * self.plant.theta
        return omega, theta

    def _current_to_torque(self, current_a):
        return self.kt / self.k_tanh * math.tanh(self.k_tanh * current_a)

    def _torque_to_current(self, tau):
        x = max(-0.999, min(0.999, tau * self.k_tanh / self.kt))
        return math.atanh(x) / self.k_tanh

    def pre_step(self, output_buf):
        rx = self._RxPDO()
        if output_buf and len(output_buf) == self.output_size:
            rx = self._RxPDO.from_buffer_copy(output_buf)
            self._apply_controlword(rx.controlword)
            if rx.control_mode != self.op_mode and rx.control_mode != 0:
                if self._pending_mode != rx.control_mode:
                    self._pending_mode = rx.control_mode
                    self._pending_countdown = self.MODE_ECHO_DELAY
                else:
                    self._pending_countdown -= 1
                    if self._pending_countdown <= 0:
                        self.op_mode = self._pending_mode
                        self._pending_mode = None
        self._rx = rx

        tau = 0.0
        if self.plant is not None and self.state == 'operation_enabled':
            omega_m, theta_m = self._motor_state()
            j_motor = self.plant.J / max(self.gear_ratio ** 2, 1e-9)
            if self.op_mode == 4:  # torque (current) mode
                i_cmd = rx.current_command / 1000.0 + rx.torque_ff * self.i_cont / 1000.0
                if self.flip:
                    i_cmd = -i_cmd  # undo the device-side sign flip
                tau = self._current_to_torque(i_cmd)
            elif self.op_mode == 3:  # velocity mode: stiff first-order servo
                v_cmd = rx.velocity_command / 1000.0 * 2 * math.pi / 60.0
                # gain limited by explicit-Euler stability at the 1 ms cycle
                kv = min(self.tau_limit / 0.5, j_motor * 200.0)
                tau = kv * (v_cmd - omega_m)
            elif self.op_mode == 7:  # position mode: critically damped PD
                theta_cmd = rx.position_command * 2 * math.pi / 2**32
                wn = 50.0
                kp = j_motor * wn * wn
                kd = 2.0 * j_motor * wn
                tau = kp * (theta_cmd - theta_m) - kd * omega_m
            tau = max(-self.tau_limit, min(self.tau_limit, tau))

        self.tau_motor = tau
        if self.plant is not None:
            self.plant.set_drive_torque(
                self.entry['name'],
                self.coupling * self.gear_ratio * tau,
                tau_motor_frame=tau,
                is_input_side=self.is_input_side)

    def step(self, dt_s, output_buf):
        tx = self._TxPDO()
        tx.statusword = self._statusword()
        tx.op_mode = self.op_mode
        if self.plant is not None:
            omega_m, theta_m = self._motor_state()
        else:
            omega_m, theta_m = 0.0, 0.0
        tx.actual_position = wrap_i32(theta_m / (2 * math.pi) * 2**32 + 1000)
        tx.actual_velocity = int(omega_m * 60.0 / (2 * math.pi) * 1000.0
                                 + random.gauss(0, 3))
        current = self._torque_to_current(self.tau_motor)
        if self.flip:
            current = -current
        tx.actual_current = int(current * 1000.0 + random.gauss(0, 15))
        tx.i2t_counter = 0
        return bytes(tx)


BEHAVIOR_FOR_MODEL = {
    'EK1100': lambda e, noise: CouplerBehavior(e),
    'EL2002': lambda e, noise: DigitalOutBehavior(e),
    'EL2004': lambda e, noise: DigitalOutBehavior(e),
    'EL2004_12v': lambda e, noise: DigitalOutBehavior(e),
    'EL2502': lambda e, noise: PWMOutBehavior(e),
    'EL1002': lambda e, noise: DigitalInBehavior(e),
    'ELM3002': lambda e, noise: ELM300xBehavior(e, channels=2, noise_counts=noise),
    'ELM3004': lambda e, noise: ELM300xBehavior(e, channels=4, noise_counts=noise),
    'EL3208': lambda e, noise: RTDBehavior(e),
    'EL5042': lambda e, noise: EnDatEncoderBehavior(e),
    'AKD': lambda e, noise: AKDBehavior(e),
}


# --------------------------------------------------------------------------
# pysoem API fakes
# --------------------------------------------------------------------------
class _SimSlave:
    def __init__(self, layout_entry, product_id, noise_counts):
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
        self._behavior = BEHAVIOR_FOR_MODEL[model](layout_entry, noise_counts)
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


def _channel_index(channel_key):
    return int(str(channel_key).lstrip('ch')) - 1


class Master:
    def __init__(self):
        self.slaves = []
        self.state = NONE_STATE
        self.plant = None
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
        cfg_path = f'{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml'
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        sim_params = load_sim_params()
        noise = sim_params['adc_noise_counts']
        self.plant = Plant(sim_params)

        layout = cfg['expected_slave_layout']
        self.slaves = [
            _SimSlave(entry, DEVICE_CLASSES[entry['model']]['id'], noise)
            for entry in layout]

        # Wire behaviors to the shared plant
        by_name = {}
        for slave, entry in zip(self.slaves, layout):
            slave._behavior.set_plant(self.plant)
            by_name[entry['name']] = slave._behavior

        # Route configured sensors onto terminal channels (same resolution as
        # the controller's sensor routing: port -> panel_ports -> module/ch).
        for sensor_name, scfg in (cfg.get('sensors') or {}).items():
            if 'port' in scfg:
                pmap = (cfg.get('panel_ports') or {}).get(scfg['port'])
                if not pmap:
                    continue
                module, channel = pmap['signal_module'], pmap['signal_channel']
            else:
                module = scfg.get('signal_module')
                channel = scfg.get('signal_channel', 'ch1')
            behavior = by_name.get(module)
            if behavior is not None and hasattr(behavior, 'configure_channel'):
                behavior.configure_channel(
                    _channel_index(channel), sensor_name,
                    fs_pos=scfg.get('fs_pos', 1.0), offset=scfg.get('offset', 0.0))

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
        # 1) drives consume their outputs and push torques into the plant
        for s in self.slaves:
            s._behavior.pre_step(s.output)
        # 2) the shaft integrates one cycle
        if self.plant is not None:
            self.plant.step(CYCLE_S)
        # 3) sensors and drives publish the new state as input bytes
        for s in self.slaves:
            s.input = s._behavior.step(CYCLE_S, s.output)
        return self.expected_wkc

    @property
    def expected_wkc(self):
        return 3

    @property
    def dc_time(self):
        return time.monotonic_ns()
