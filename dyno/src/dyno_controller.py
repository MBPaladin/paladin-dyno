from dyno.src.master import Master
from dyno.src.devices import ScaledChannels
import yaml
import time
import math
import os
import signal
import threading
from operator import attrgetter
from deployment import dyno_paths
from dyno.src.test_manager import TestManager

SCHED_POLICY = os.SCHED_FIFO
SCHED_PRIO = 50

class Controller(Master):
    def __init__(self, telemetry_queue=None, command_queue=None,mode=None):
        # Runs in a child process: ignore SIGINT so a terminal Ctrl-C can't kill
        # the real-time control loop mid-cycle (which would orphan it with the
        # drives still enabled). The parent GUI orchestrates an orderly shutdown
        # via the command queue instead.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.mode = mode

        with open(f"{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml", 'r') as f:
            self.dyno_params = yaml.safe_load(f)

        self._expected_slave_layout = self.dyno_params['expected_slave_layout']

        # Load-only bringup. The DUT drive is still enumerated, read, and logged
        # -- only actuation is withheld: it is never enabled, never commanded,
        # and its fault state never gates or trips a test. Set `load_only: true`
        # in <mode>_dyno_config.yaml. Without this, start_test enables BOTH
        # drives unconditionally, which is not what you want on a rig whose DUT
        # side has no motor or coupling fitted yet.
        #
        # "No coupling fitted" is also why this flag relaxes the reaction-torque
        # check in _get_limits: with an open shaft the load motor's torque is not
        # reacted into the DUT. Clearing the flag re-arms that check, which will
        # start rejecting output torque above DUT.torque x DUT.gear_ratio.
        self._load_only = bool(self.dyno_params.get('load_only', False))
        if self._load_only:
            print('Controller: LOAD-ONLY mode -- DUT will not be enabled or commanded')

        super().__init__(slave_layout = self._expected_slave_layout)

        self._telemetry_queue = telemetry_queue
        self._command_queue = command_queue

        self.data_counter = 0
        self._safe_default_command = {'input_mode':'torque','output_mode':'torque','input_command': 0,'output_command': 0}

        self.t_offset = time.perf_counter()

        self._current_input_mode = None
        self._current_output_mode = None
        self._test_active = False
        # Why the last test ended, published on the sample that turns logging
        # off so the Logger can stamp it into the file it is about to close.
        # See _stop_test.
        self._stop_reason = None
        self.generated_cmd = None
        self.test_definition = None
        self._test_init_thread = None
        # Handoff slot for a test loaded on the init thread: (generation,
        # TestManager). Only _cmd_check (control thread) moves it into
        # test_definition — see _cmd_check.
        self._pending_test_definition = None
        # Bumped every time the armed test changes. A load still running when
        # the operator re-arms or disarms carries a stale generation and is
        # dropped, rather than resurfacing a test that was already dismissed.
        self._test_load_generation = 0
        # (generation, test_file) currently loading, and (generation, message)
        # for one that failed. Both are reported to the GUI and both go stale
        # on their own when the generation moves past them.
        self._loading_test = None
        self._test_load_error = None

        # --- Torque cell tare ---
        # Per-session zeroing of the torque cells: average each cell at rest and
        # carry the negated mean as a bias. Config knobs live under `tare:` in
        # the rig yaml; the defaults below make the feature work on a rig whose
        # config predates it.
        tare_cfg = self.dyno_params.get('tare', {}) or {}
        self._tare_duration_s = float(tare_cfg.get('duration_s', 3.0))
        self._tare_max_velocity = float(tare_cfg.get('max_velocity', 0.05))
        self._tare_warn_frac_fs = float(tare_cfg.get('warn_frac_fs', 0.02))
        # Which sensors get tared. Torque cells are the ones that drift and the
        # ones a zero is meaningful for -- a temperature probe is not something
        # you zero against the room.
        self._tare_sensors = tare_cfg.get('sensors')
        if self._tare_sensors is None:
            self._tare_sensors = [name for name in self.dyno_params.get('sensors', {})
                                  if 'torque' in name]
        # Accumulator for a tare in flight (None when idle), and the committed
        # result: sensor -> {bias, raw_mean, stddev, samples, at, frac_fs}.
        self._tare_state = None
        self._tare_result = {}
        self._tare_message = None

        self._aux_funcs = []
        if self.mode == 'actuator_production':
            self._aux_funcs.append(self._aux_func_A3_Dyno)

        # Compile safety checks from config: each entry names a telemetry
        # source (attrgetter path, same idiom as log_keys) and a limit.
        # safeties: { output_torque: { source: devices.ADC.output_torque, limit: 375 } }
        self._safety_checks = []
        for check_name, spec in self.dyno_params.get('safeties', {}).items():
            if not isinstance(spec, dict) or 'source' not in spec or 'limit' not in spec:
                raise ValueError(
                    f"safeties entry '{check_name}' in {self.mode}_dyno_config.yaml "
                    f"must be a dict with 'source' and 'limit' keys, got: {spec!r}")
            self._safety_checks.append(
                (check_name, attrgetter(spec['source']), abs(spec['limit'])))

        try:
            os.sched_setscheduler(0, SCHED_POLICY, os.sched_param(SCHED_PRIO))
            cpu_set = {1, 2, 3, 4}
            os.sched_setaffinity(0, cpu_set)
            print("Controller: Real-time scheduling enabled")
        except PermissionError:
            print("Controller: Real-time scheduling not permitted, running normally")

        self.run()

        # The drives are released and the bus is closed, but this process still
        # has to exit. By now the GUI is blocked in close_processes and nothing
        # is draining telemetry, so the samples buffered in the queue's feeder
        # thread have no reader: at exit that thread blocks writing to a full
        # pipe and holds the process open until the parent gives up and kills
        # it. The samples are worthless at this point - drop them.
        if self._telemetry_queue is not None:
            self._telemetry_queue.cancel_join_thread()

    def _stop_test(self, reason=None):
        """End the active test and record WHY, for the log that is closing.

        Every caller passes a reason because the alternative is what this
        codebase did before: a safety trip and a clean finish were
        indistinguishable once the run was over. The trip is invisible in the
        data itself -- clearing _test_active gates logging (see
        _send_telemetry), so the sample that breached a limit is the first one
        NOT written, and the log ends healthy just under the threshold with no
        record of what stopped it. The only account of it was a print to a
        terminal nobody kept.

        The reason rides out on the next telemetry sample (the one carrying
        log=False) and the Logger stamps it onto the file as the `stop_reason`
        attribute. A None reason still overwrites the previous one: a stale
        reason on a new run is worse than no reason at all.
        """
        self._stop_reason = reason
        if not self.test_definition == None:
            self.test_definition.reset()
        self.devices.DUT.sw_enable = False
        self.devices.LOAD.sw_enable = False
        self._test_active = False

        # self.devices.DUT.command_operating_mode('torque')
        self.devices.LOAD.command_operating_mode('torque')
        self._safe_default_command['input_mode'] = 'torque'
        self._safe_default_command['output_mode'] = 'torque'
        self._safe_default_command['input_command'] = 0
        self._safe_default_command['output_command'] = 0

    def _control_state(self):
        """Rig state for the GUI's status indicator (telemetry slot -1).

        The GUI can see none of this on its own: test_definition lives in this
        process and drive faults are not in log_keys, so the GUI could only
        report what it ASKED for -- which is wrong for as long as a load is in
        flight, and never learns that a load failed here. Slot -1 was
        previously always None and both consumers slice it off (sample[:-2]),
        so filling it in cannot shift plot or log columns.

        `_loading_test` / `_test_load_error` are reported only while their
        generation is current, so a superseded load reports nothing.
        """
        current = self._test_load_generation
        return {
            'armed': self.test_definition.name if self.test_definition else None,
            'loading': (self._loading_test[1]
                        if self._loading_test and self._loading_test[0] == current
                        else None),
            'load_error': (self._test_load_error[1]
                           if self._test_load_error
                           and self._test_load_error[0] == current else None),
            'test_active': self._test_active,
            'fault': bool(self.devices.LOAD.fault
                          or (not self._load_only and self.devices.DUT.fault)),
            # Identifies the bring-up that wrote resolved_config.json, so the
            # Logger can tell whether the file it picked up describes this run
            # (see Master.run step 3.6). getattr: telemetry must never depend
            # on bring-up having reached that step.
            'session_id': getattr(self, 'session_id', None),
            # The session tare, for the GUI's readout and for the Logger to
            # stamp into the log. Riding on the state dict the Logger already
            # receives means the value is present on every sample, so a log
            # opening at any moment gets it without a separate handshake.
            'tare': self._tare_result or None,
            'tare_active': self._tare_state is not None,
            'tare_message': self._tare_message,
        }

    def _send_telemetry(self):
        self.logging_state = {'log': False} # Default to not logging
        if self._test_active and 'log_flag' in self.current_cmd:
            self.logging_state = {'log': True, 'behavior_id': self.current_cmd['log_flag']}
        elif self._test_active: # If test is active but no specific log_flag
            self.logging_state = {'log': True}
        elif self._stop_reason is not None:
            # _stop_test cleared _test_active earlier in this same step(), so
            # this is the first log=False sample after the run -- exactly the
            # one the Logger closes the file on. It keeps riding along on the
            # idle samples after it, which costs nothing and means a Logger
            # that starts late still has the reason to hand.
            self.logging_state = {'log': False, 'stop_reason': self._stop_reason}

        self.control_state = self._control_state()

        self.time = time.perf_counter() - self.t_offset

        telemetry = [getter(self) for getter in self._telemetry_compiled]
        telemetry.append(self.logging_state)
        telemetry.append(self.control_state)

        self._telemetry_queue.put_nowait(telemetry)

    # recieves and manages commands from the GUI
    def _cmd_check(self):
        # Publish a background-loaded test, on this (the control) thread.
        # TestManager.__init__ leaves the instance unusable until reset(), and
        # the init thread can finish AFTER start_test has already reset the
        # previous test -- assigning from that thread would hand step() a test
        # with no command generator and kill the control loop mid-run. Holding
        # the swap here means test_definition only ever changes between tests.
        if self._pending_test_definition is not None and not self._test_active:
            generation, test = self._pending_test_definition
            self._pending_test_definition = None
            if generation == self._test_load_generation:
                self.test_definition = test
                self._loading_test = None
                print(f'Controller: {test.name} armed')
            else:
                print(f'Controller: dropped superseded load of {test.name}')

        read_queue = True
        while read_queue:
            try:
                cmd = self._command_queue.get_nowait()
                if cmd[0] == 'start_test':
                    if self._test_active:
                        print('Unable to start test: Test already active')
                    elif not self._load_only and self.devices.DUT.fault:
                        print('Unable to start test: DUT in fault state')
                    elif self.devices.LOAD.fault:
                        print('Unable to start test: LOAD in fault state')
                    else:
                        # The previous run's reason must not outlive it into
                        # the next log.
                        self._stop_reason = None
                        self.pull_cmd = True
                        if not self._load_only:
                            self.devices.DUT.sw_enable = True
                        self.devices.LOAD.sw_enable = True
                        if not self.test_definition == None:
                            self._test_active = True
                            self.test_definition.reset()
                        print('Starting test')
                        
                elif cmd[0] == 'stop_test':
                    self._stop_test({'kind': 'operator',
                                     'detail': 'stopped from the GUI by the operator'})
                    print('attempting to stop test, in / out motor commanded to torque mode')

                elif cmd[0] == 'test_def':
                    if not self._test_active:
                        self._get_limits()
                        self._test_load_generation += 1
                        generation = self._test_load_generation
                        self._loading_test = (generation, cmd[1][0])

                        # Define the target function for the thread
                        def load_test(file, mode, limits):
                            # This runs off the control loop, so an exception
                            # here would otherwise die with the thread and
                            # leave the GUI waiting forever. Report it instead
                            # -- these are the limit asserts checked against
                            # the drives' real limits, which the GUI's
                            # config-based pre-check cannot see.
                            try:
                                test = TestManager(file, mode, limits)
                                test.reset()  # usable before the loop sees it
                            except Exception as e:
                                self._test_load_error = (generation,
                                                         f'{type(e).__name__}: {e}')
                                print(f'Controller: FAILED to load {file}: {e}')
                                return
                            self._pending_test_definition = (generation, test)
                            print("Controller: TestManager ready.")

                        self._test_init_thread = threading.Thread(target=load_test, args=(cmd[1][0],cmd[1][1], self.limits))
                        self._test_init_thread.start()
                    else:
                        print('Please re-select test when dyno is not active')

                elif cmd[0] == 'tare':
                    self._start_tare()

                elif cmd[0] == 'clear_tare':
                    self._clear_tare()

                elif cmd[0] == 'clear_test':
                    # Disarm. The GUI sends this whenever it stops vouching for
                    # the armed test (a new pick, a failed load), because it
                    # cannot clear test_definition itself — without this the
                    # previous test stays loaded here and Start would run it.
                    if not self._test_active:
                        self._test_load_generation += 1  # drop any load in flight
                        self._pending_test_definition = None
                        if self.test_definition is not None:
                            print(f'Controller: {self.test_definition.name} disarmed')
                        self.test_definition = None
                    else:
                        print('Unable to disarm: test already active')

                elif cmd[0] == 'shutdown':
                    print('Shutdown command recieved by control loop')
                    self.shutdown = True
                    for device_name in vars(self.devices).keys():
                        device_instance = getattr(self.devices, device_name)
                        if hasattr(device_instance, 'shutdown'):
                            device_instance.shutdown = True

                    self.devices.LOAD.shutdown = True
                    self.devices.DUT.shutdown = True

                    try:
                        self.devices.input_motor.shutdown = True
                    except:
                        pass


            except:
                read_queue = False
                pass

    # stops the test if measured values are outside of an acceptable range.
    # Checks are declared in the config's `safeties:` section (source + limit);
    # add entries there rather than here.
    def _safety_trigger(self):
        """None when everything is in range, else a reason dict for _stop_test.

        The dict carries the breaching value and the limit it broke, because
        those are exactly what the log cannot show: the tripping sample is
        never written (see _stop_test), so without them a reader is left
        extrapolating from the last sample below the threshold.

        `at_s` is self.time, which _send_telemetry last set one cycle ago --
        i.e. the timestamp of the final logged sample. That is the intended
        meaning: the trip happened immediately after the trace ends.
        """
        # Nothing on this path may raise: an exception here propagates out of
        # step() and takes the control loop down, turning a limit breach into a
        # rig left running with no supervision at all. Hence the getattrs.
        at_s = getattr(self, 'time', None)

        for check_name, getter, limit in self._safety_checks:
            value = abs(getter(self))
            if value > limit:
                print(f'Safety triggered, {check_name} of {value} exceeds limit of {limit}')
                return {'kind': 'safety',
                        'check': check_name,
                        'value': float(value),
                        'limit': float(limit),
                        'at_s': at_s,
                        'detail': (f'safety check {check_name!r} measured '
                                   f'{value:.6g}, over its limit of {limit:g}')}

        for name in (() if self._load_only else ('DUT',)) + ('LOAD',):
            drive = getattr(self.devices, name)
            if drive.fault:
                print(f'Safety triggered, {name} is in fault state')
                # The statusword is the drive's own account of the fault, and
                # the one thing that can be cross-referenced against Workbench.
                statusword = getattr(drive, 'statusword', None)
                sw_text = ('' if statusword is None
                           else f' (statusword 0x{int(statusword):04X})')
                return {'kind': 'drive_fault',
                        'drive': name,
                        'statusword': None if statusword is None else int(statusword),
                        'at_s': at_s,
                        'detail': f'{name} drive went into DS402 fault state{sw_text}'}

        return None
    
    # --- Torque cell tare ---------------------------------------------------
    # Averaging happens here rather than in the GUI because this loop sees every
    # sample at the bus cycle rate; the GUI's view is decimated for plotting and
    # would average a fraction of the data, unevenly.

    def _tare_targets(self):
        """{sensor name -> (publishing module, its channel params)}.

        Sensors are not addressable directly -- master.py routes each into its
        ADC module's params under the channel it is wired to -- so rather than
        re-derive that routing, ask each tare-capable module what it publishes."""
        targets = {}
        for module in vars(self.devices).values():
            if not isinstance(module, ScaledChannels):
                continue
            for ch_params in (getattr(module, 'params', None) or {}).values():
                if isinstance(ch_params, dict) and ch_params.get('name') in self._tare_sensors:
                    targets[ch_params['name']] = (module, ch_params)
        return targets

    def _rig_at_rest(self):
        """True when nothing is turning fast enough to be making real torque. A
        cell averaged while the shaft creeps records drag as though it were
        sensor bias, and every reading afterwards inherits the error."""
        speeds = [abs(getattr(self.devices.LOAD, 'velocity', 0.0) or 0.0)]
        if not self._load_only:
            speeds.append(abs(getattr(self.devices.DUT, 'velocity', 0.0) or 0.0))
        return max(speeds) <= self._tare_max_velocity

    def _start_tare(self):
        """Begin averaging, or refuse and say why.

        Refusing is the point: a tare taken under load or mid-test writes a bias
        into every subsequent reading AND shifts the torque safety trip, both
        silently. Better to reject it than to record a confident wrong zero."""
        if self._test_active:
            self._tare_message = 'Tare refused: a test is running'
        elif self._tare_state is not None:
            self._tare_message = 'Tare refused: a tare is already in progress'
        elif not self._tare_targets():
            self._tare_message = 'Tare refused: no tareable sensors on this rig'
        elif not self._rig_at_rest():
            self._tare_message = (f'Tare refused: the rig is moving (limit '
                                  f'{self._tare_max_velocity} rad/s)')
        else:
            targets = self._tare_targets()
            self._tare_state = {
                'targets': targets,
                'sums': {name: 0.0 for name in targets},
                'sumsq': {name: 0.0 for name in targets},
                'counts': {name: 0 for name in targets},
                'deadline': time.perf_counter() + self._tare_duration_s,
            }
            self._tare_message = (f'Taring {", ".join(sorted(targets))} over '
                                  f'{self._tare_duration_s:g} s...')
        print(f'Controller: {self._tare_message}')

    def _tare_step(self):
        """Advance a tare in flight. Called every control cycle and never
        blocks -- the loop keeps its deadline whether or not a tare is running.

        Accumulates the UNTARED reading, so re-taring measures the cell rather
        than walking the previous bias toward zero."""
        st = self._tare_state
        if st is None:
            return

        if self._test_active or not self._rig_at_rest():
            cause = ('a test started' if self._test_active else 'the rig moved')
            self._tare_state = None
            self._tare_message = (f'Tare aborted: {cause} before the averaging '
                                  'window finished')
            print(f'Controller: {self._tare_message}')
            return

        for sensor, (module, _params) in st['targets'].items():
            value = module.untared.get(sensor)
            if value is None:
                continue
            st['sums'][sensor] += value
            st['sumsq'][sensor] += value * value
            st['counts'][sensor] += 1

        if time.perf_counter() >= st['deadline']:
            self._commit_tare(st)

    def _commit_tare(self, st):
        """Turn the accumulated sums into a bias per cell and apply it.

        Records the spread alongside the mean: a tight mean over a noisy window
        is not a zero, it is an average of the disturbance, and the operator
        needs to see that to judge whether the number is worth keeping."""
        self._tare_state = None
        stamped = time.strftime('%Y-%m-%d %H:%M:%S')
        result = {}

        for sensor, (module, ch_params) in st['targets'].items():
            n = st['counts'][sensor]
            if n < 2:
                continue
            mean = st['sums'][sensor] / n
            variance = max(st['sumsq'][sensor] / n - mean * mean, 0.0)
            full_scale = abs(float(ch_params.get('fs_pos', 0) or 0.0))
            module.tare[sensor] = -mean
            result[sensor] = {
                'bias': -mean,
                'raw_mean': mean,
                'stddev': math.sqrt(variance),
                'samples': n,
                'at': stamped,
                'full_scale': full_scale or None,
                # Percent of full scale is the only portable way to judge a
                # bias: 0.5 Nm is noise on a 500 Nm cell and 2.5% on a 20 Nm one.
                'frac_fs': (abs(mean) / full_scale) if full_scale else None,
            }

        if not result:
            self._tare_message = 'Tare failed: no samples collected'
            print(f'Controller: {self._tare_message}')
            return

        self._tare_result = result
        parts = []
        for sensor in sorted(result):
            r = result[sensor]
            frac = ('%.2f%% FS' % (100 * r['frac_fs'])) if r['frac_fs'] is not None \
                else 'FS unknown'
            parts.append(f'{sensor} {r["bias"]:+.4g} ({frac}, sd {r["stddev"]:.3g}, '
                         f'n={r["samples"]})')
        self._tare_message = 'Tare applied: ' + '; '.join(parts)
        print(f'Controller: {self._tare_message}')

        loud = [s for s, r in result.items()
                if r['frac_fs'] is not None and r['frac_fs'] > self._tare_warn_frac_fs]
        if loud:
            print(f'Controller: WARNING - tare on {", ".join(sorted(loud))} exceeds '
                  f'{100 * self._tare_warn_frac_fs:g}% of full scale. That is large '
                  'for a zero offset: check the cell is genuinely unloaded.')

    def _clear_tare(self):
        """Drop every applied bias, back to config offsets alone."""
        for module in vars(self.devices).values():
            if isinstance(module, ScaledChannels):
                module.tare.clear()
        self._tare_state = None
        self._tare_result = {}
        self._tare_message = 'Tare cleared'
        print(f'Controller: {self._tare_message}')

    def _get_limits(self):
        # Per-motor, keyed by the command stream's motor names (DUT -> 'input',
        # LOAD -> 'output'). Each command is checked against the motor it is
        # actually sent to, so a derated DUT no longer caps what the load motor
        # may be commanded to do. What the two motors do to *each other* through
        # the shaft is the `coupled` flag plus test_builder.reaction_torque_issue.
        #
        # These are the drives' real limits (AKD.__init__ merges absorbers.yaml
        # over the config), so they routinely differ from the config values the
        # GUI pre-checks against in test_preview.limits_from_config -- usually
        # higher, since a bring-up config derates what the absorber entry rates.
        def motor_limits(device):
            return {
                'torque': abs(device.torque_limit),
                'velocity': abs(device.velocity_limit),
                'acceleration': abs(device.acceleration_limit),
                'rotatum': abs(device.rotatum_limit),
                # Shaft coupling, for the reaction-torque check the per-motor
                # ceilings cannot express (test_builder.reaction_torque_issue).
                'gear_ratio': abs(device.params.get('gear_ratio', 1)),
            }

        # load_only means the DUT side has no motor or coupling fitted, so the
        # load motor's torque is not reacted into the DUT. See
        # test_builder.reaction_torque_issue.
        self.limits = {'input': motor_limits(self.devices.DUT),
                       'output': motor_limits(self.devices.LOAD),
                       'coupled': not self._load_only}
       
    def step(self):
        self.data_counter += 1

        # Before anything is commanded: a tare only ever runs with the rig idle,
        # and _tare_step abandons it the moment that stops being true.
        self._tare_step()

        # Aligning LOAD's position frame to DUT's only means something when the
        # two are mechanically coupled, which load-only assumes they are not.
        if (not self._load_only
                and self.devices.LOAD.position_offset == 0
                and not self.devices.DUT.position == 0):
            self.devices.LOAD.position_offset = self.devices.DUT.position - self.devices.LOAD.position

        if self._test_active:
            trip = self._safety_trigger()
            if trip:
                self._stop_test(trip)

            if self.pull_cmd:
                self.generated_cmd = self.test_definition.next_command()
                self.pull_cmd = False

            if not self.generated_cmd == None and not self.shutdown:
                if not self._load_only and self.generated_cmd['input_mode'] != self.devices.DUT.mode:
                    if not self.devices.DUT.switching_modes:
                        self.devices.DUT.command_operating_mode(self.generated_cmd['input_mode'])
                        self._safe_default_command['input_mode'] = self.current_cmd['input_mode']
                        self._safe_default_command['input_command'] = self.current_cmd['input_command']
                        

                if self.generated_cmd['output_mode'] != self.devices.LOAD.mode:
                    if not self.devices.LOAD.switching_modes:
                        self.devices.LOAD.command_operating_mode(self.generated_cmd['output_mode'])
                        self._safe_default_command['output_mode'] = self.current_cmd['output_mode']
                        self._safe_default_command['output_command'] = self.current_cmd['output_command']
                        

                # Load-only never commands DUT into a mode, so waiting for it to
                # report one would stall the test at the first command forever.
                dut_ready = self._load_only or (
                    self.devices.DUT.mode == self.generated_cmd['input_mode']
                    and not self.devices.DUT.switching_modes)

                if dut_ready and self.devices.LOAD.mode == self.generated_cmd['output_mode'] and not self.devices.LOAD.switching_modes:
                    self.current_cmd = self.generated_cmd # Use the command from the test
                    self.pull_cmd = True

            else:
                # The normal end of a run: the test generator is out of
                # commands. Recorded like any other ending so that a log with
                # no stop_reason means "written by a build that predates this",
                # not "finished cleanly" -- the two must not look alike.
                self._stop_test({'kind': 'shutdown', 'detail': 'rig shutdown requested'}
                                if self.shutdown else
                                {'kind': 'completed', 'detail': 'test ran to completion'})

        else:
            self.current_cmd = self._safe_default_command

            if self.devices.DUT.sw_enable or self.devices.LOAD.sw_enable:
                self.devices.DUT.sw_enable = False
                self.devices.LOAD.sw_enable = False

        # Both are rebuilt in _send_telemetry, below.
        self.control_state = None
        self.logging_state = None

        ff_ratio = 0.8

        # The feedforward terms exist to cancel the torque the *other* machine is
        # putting through the coupling. Load-only has no DUT contribution to
        # cancel, so LOAD is commanded plain and DUT is not written at all --
        # this is the one place that would otherwise actuate a drive the
        # operator was told stays dormant.
        if not self._load_only and self.devices.DUT.mode == 'torque' and not self.devices.LOAD.mode == 'torque':
            torque_ff = ff_ratio*self.current_cmd['input_command'] * self.devices.DUT.params['gear_ratio']
            self.devices.LOAD.send_command(self.current_cmd['output_command'], torque_ff)
        else:
            self.devices.LOAD.send_command(self.current_cmd['output_command'])

        if not self._load_only:
            if self.devices.LOAD.mode == 'torque' and not self.devices.DUT.mode == 'torque':
                torque_ff = ff_ratio*self.current_cmd['output_command'] / self.devices.DUT.params['gear_ratio']
                self.devices.DUT.send_command(self.current_cmd['input_command'], torque_ff)
            else:
                self.devices.DUT.send_command(self.current_cmd['input_command'])

        if self.current_cmd['input_command'] != getattr(self, '_last_dut_cmd', None):
            self._last_dut_cmd = self.current_cmd['input_command']

        for aux_func in self._aux_funcs:
            aux_func()

        self._send_telemetry()
        self._cmd_check()
        time.sleep(0) #momentarily yeilds the GIL

    def _write_led(self, ch, mode):

        # LED pins on AKD (LOAD)
        pins = {'g':1, 'r':2}
        """Mode can be 'on', 'off', or 'blink'"""
        blink_state = (self.data_counter % 1000) < 500
        val = False
        if mode == 'on': val = True
        elif mode == 'blink': val = blink_state

        try:
            dev = getattr(self.devices, 'LOAD')
            if hasattr(dev, 'set_channel'):
                dev.set_channel(pins[ch], val)
            elif hasattr(dev, 'set_dout'):
                dev.set_dout(pins[ch], val)
        except AttributeError:
            pass

    def _aux_func_A3_Dyno(self):
        # Run specific logic at lower cycle rate
        if self.data_counter % 20 == 0:
            # Stator temperature control
            stator_temp = self.devices.rtd_module.load_stator_temp

            # Grab the 24v_power3 module
            pwr3 = getattr(self.devices, '24v_power3')
            if math.isnan(stator_temp) or stator_temp >= 100:
                pwr3.set_channel(4, False)
                if math.isnan(stator_temp):
                    print('CRITICAL: Stator RTD Invalid (NaN). Test Stopped')
                    self._stop_test({'kind': 'stator_temp',
                                     'value': None,
                                     'at_s': getattr(self, 'time', None),
                                     'detail': 'stator RTD read NaN -- sensor failed '
                                               'or disconnected, so its temperature '
                                               'could not be supervised'})
                else:
                    print(f'CRITICAL: Stator Temp {stator_temp:.1f}C >= 100C. Test Stopped')
                    self._stop_test({'kind': 'stator_temp',
                                     'value': float(stator_temp),
                                     'limit': 100.0,
                                     'at_s': getattr(self, 'time', None),
                                     'detail': f'stator temperature {stator_temp:.1f}C '
                                               'reached the 100C cutout'})
            elif stator_temp >= 60:
                pwr3.set_channel(4, True)
            else:
                pwr3.set_channel(4, False)

            # Status Indication
            if self.devices.LOAD.fault or (not self._load_only and self.devices.DUT.fault):
                self._write_led('r', 'on')
                self._write_led('g', 'off')
            elif self._test_active:
                self._write_led('r', 'off')
                self._write_led('g', 'on')
            else:
                self._write_led('r', 'off')
                self._write_led('g', 'blink')