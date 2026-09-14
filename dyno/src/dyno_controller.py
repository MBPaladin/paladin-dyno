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

        # The preamble behavior's monotonic sample index, exposed for logging
        # (log key 'preamble_sample'). NaN whenever no preamble is generating.
        # Coherent averaging in analysis needs exact period boundaries, which
        # perf_counter timestamps cannot give under cycle jitter -- this is the
        # one piece of plumbing the analysis genuinely cannot work around.
        self.preamble_sample = float('nan')

        # Command value the ramp_break detector fired at, on the one sample it
        # fired (log key 'breakaway_torque'), NaN everywhere else. The release
        # that follows looks identical to one that merely hit the amplitude
        # ceiling, so without this marker a log cannot distinguish the two.
        self.breakaway_torque = float('nan')

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

        # --- Drive fault reset ---
        # A reset in flight: {'drives': [names asked], 'deadline': perf_counter}.
        # None when idle. The outcome is only knowable a few cycles later -- the
        # drive has to see the controlword and answer in its statusword -- so the
        # request and the verdict are separate steps, and the verdict is what the
        # operator actually needs (a fault whose cause is still present clears
        # and re-latches immediately, which from the button alone is
        # indistinguishable from a reset that worked).
        self._fault_clear_state = None
        self._fault_clear_message = None

        # step() used to copy DUT's position into LOAD.position_offset early in
        # every session, ignoring any gear ratio. A rig that reads the output
        # position on its own (the position window below) must turn that off.
        self._align_load_position = bool(
            self.dyno_params.get('align_load_position_to_dut', True))

        # Scale on the cross-coupling torque feedforward in step(). 0 disables it.
        self._feedforward_ratio = float(self.dyno_params.get('feedforward_ratio', 0.8))

        # --- Output position window (limited-travel DUTs) ---
        # Config `position_window:`; absent means the feature is off and every
        # _window_* / jog / touch-off method is a no-op. See the section below.
        self._window = self._compile_window(self.dyno_params.get('position_window'))
        self._window_centre = None      # LOAD-frame position declared as centre
        self._touch_off = None          # last completed touch-off result
        self._touch_off_history = []    # every touch-off this session
        self._jog = None                # jog or touch-off in flight
        self._window_message = None

        self._aux_funcs = []
        if self.mode == 'actuator_production':
            self._aux_funcs.append(self._aux_func_A3_Dyno)

        # Compile safety checks from config: each entry names a telemetry
        # source (attrgetter path, same idiom as log_keys) and a limit.
        # safeties: { output_torque: { source: devices.ADC.output_torque, limit: 375 } }
        # `nan_trips: true` makes a NaN reading trip the check. Off by default
        # because abs(nan) > limit is False, and some rigs list channels (an
        # unfitted RTD) that read NaN in normal use.
        self._safety_checks = []
        for check_name, spec in self.dyno_params.get('safeties', {}).items():
            if not isinstance(spec, dict) or 'source' not in spec or 'limit' not in spec:
                raise ValueError(
                    f"safeties entry '{check_name}' in {self.mode}_dyno_config.yaml "
                    f"must be a dict with 'source' and 'limit' keys, got: {spec!r}")
            # trip_samples: how many CONSECUTIVE breaching cycles it takes to
            # stop the test. Default 1 = the historical instantaneous
            # behaviour. It exists for wkc_error, which master.py recomputes
            # every cycle as `expected_wkc - actual_wkc` (master.py:88) rather
            # than accumulating -- so it is bounded by expected_wkc (8 on this
            # bench) and a raised limit cannot express "tolerate a glitch but
            # stop on sustained loss". Raising the limit past expected_wkc
            # instead disables the check outright, including total bus loss.
            trip_samples = int(spec.get('trip_samples', 1))
            if trip_samples < 1:
                raise ValueError(
                    f"safeties entry '{check_name}' in {self.mode}_dyno_config."
                    f"yaml: trip_samples must be >= 1, got {trip_samples}")
            self._safety_checks.append(
                (check_name, attrgetter(spec['source']), abs(spec['limit']),
                 bool(spec.get('nan_trips', False)), trip_samples))
        # Consecutive-breach counter per check, reset on any in-range read.
        self._safety_streaks = {}

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
            # Where the run is, for the GUI's progress readout. Only meaningful
            # while a test is running -- an armed-but-idle plan has not entered
            # its first segment, and reporting last run's position would read as
            # progress on this one.
            'progress': (self.test_definition.progress()
                         if self._test_active and self.test_definition else None),
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
            # Every faulted DS402 drive on the bus, not just the two the 'fault'
            # flag above gates tests on -- the fault-reset button acts on all of
            # them, so it has to be able to say which ones it would act on.
            'faulted_drives': self._faulted_drives(),
            'fault_clear_active': self._fault_clear_state is not None,
            'fault_clear_message': self._fault_clear_message,
            # Centre, touch-off and jog state for the GUI and the Logger. None
            # when the config has no position_window.
            'window': self._window_state(),
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
                    elif self._window_arming_refusal():
                        self._window_message = ('Start refused: '
                                                + self._window_arming_refusal())
                        print(f'Unable to start test: {self._window_message}')
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
                    if self._jog is not None:
                        self._end_jog('Stopped by the operator', failed=True)
                    self._stop_test({'kind': 'operator',
                                     'detail': 'stopped from the GUI by the operator'})
                    print('attempting to stop test, in / out motor commanded to torque mode')

                elif cmd[0] == 'declare_centre':
                    self._declare_centre()

                elif cmd[0] == 'jog':
                    self._start_jog(cmd[1])

                elif cmd[0] == 'jog_stop':
                    # Hold-to-run release. Only ends a manual jog: a touch-off
                    # is stopped with Stop, not by letting go of a jog button.
                    if self._jog is not None and self._jog['kind'] == 'jog':
                        self._end_jog('Jog released')

                elif cmd[0] == 'touch_off':
                    self._start_touch_off()

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
                                test = TestManager(file, mode, limits,
                                                   sensor_reader=self._sensor_snapshot)
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

                elif cmd[0] == 'clear_faults':
                    self._clear_faults()

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

        for check_name, getter, limit, nan_trips, trip_samples in self._safety_checks:
            value = abs(getter(self))
            if value > limit or (nan_trips and math.isnan(value)):
                streak = self._safety_streaks.get(check_name, 0) + 1
                self._safety_streaks[check_name] = streak
                if streak < trip_samples:
                    continue
                held = ('' if trip_samples == 1 else
                        f' for {streak} consecutive cycles')
                print(f'Safety triggered, {check_name} of {value} exceeds '
                      f'limit of {limit}{held}')
                return {'kind': 'safety',
                        'check': check_name,
                        'value': float(value),
                        'limit': float(limit),
                        'trip_samples': trip_samples,
                        'at_s': at_s,
                        'detail': (f'safety check {check_name!r} read NaN'
                                   if math.isnan(value) else
                                   f'safety check {check_name!r} measured '
                                   f'{value:.6g}, over its limit of '
                                   f'{limit:g}') + held}
            # In range: any run of breaches ends here.
            self._safety_streaks[check_name] = 0

        window_trip = self._window_trip(at_s)
        if window_trip:
            print(f'Safety triggered: {window_trip["detail"]}')
            return window_trip

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

    # --- Drive fault reset --------------------------------------------------
    # The reset itself is the drives' business (AKD.request_fault_reset pulses
    # controlword bit 7 from the process-data loop); what lives here is which
    # drives to ask and how the answer is reported.

    # How long to wait before judging the result. Long enough for the pulse
    # (AKD_FAULT_RESET_CYCLES ms) plus the drive's own reaction, and long enough
    # that a fault whose cause is still present has re-latched by the time it is
    # read -- checking the instant the fault bit drops would report success on a
    # drive that faults again a millisecond later.
    _FAULT_CLEAR_SETTLE_S = 0.5

    def _drives(self):
        """Every DS402 drive on the bus, by name. Duck-typed the same way
        Master._release_drives is: the AXON and RB430 have no controlword."""
        return [(name, device) for name, device in vars(self.devices).items()
                if hasattr(device, 'request_fault_reset')]

    def _faulted_drives(self):
        return sorted(name for name, device in self._drives() if device.fault)

    def _clear_faults(self):
        """Pulse a fault reset at every faulted drive, or refuse and say why.

        Refused mid-test on purpose: a drive fault trips a stop (see
        _safety_trigger), so a reset while _test_active is still set would be
        clearing the evidence of the thing that is in the middle of stopping the
        run. Stop first, then clear."""
        if self._test_active:
            self._fault_clear_message = 'Fault reset refused: a test is running'
        elif self.shutdown:
            self._fault_clear_message = 'Fault reset refused: the rig is shutting down'
        elif self._fault_clear_state is not None:
            self._fault_clear_message = 'Fault reset already in progress'
        else:
            asked = [name for name, device in self._drives()
                     if device.request_fault_reset()]
            if not asked:
                self._fault_clear_message = 'No drive faults to clear'
            else:
                self._fault_clear_state = {
                    'drives': sorted(asked),
                    'deadline': time.perf_counter() + self._FAULT_CLEAR_SETTLE_S,
                }
                self._fault_clear_message = (
                    f'Clearing faults on {", ".join(sorted(asked))}...')
        print(f'Controller: {self._fault_clear_message}')

    def _fault_clear_step(self):
        """Report on a reset once it has had time to take. Never blocks."""
        st = self._fault_clear_state
        if st is None or time.perf_counter() < st['deadline']:
            return

        self._fault_clear_state = None
        faulted = set(self._faulted_drives())
        stuck = [name for name in st['drives'] if name in faulted]
        if not stuck:
            self._fault_clear_message = (
                f'Faults cleared on {", ".join(st["drives"])}')
        else:
            # The reset dropped the latch and the drive put it straight back:
            # whatever caused the fault is still there. The drive's own account
            # of it is on its front panel and in Workbench, which this process
            # cannot read over the PDO map.
            self._fault_clear_message = (
                f'{", ".join(stuck)} still in fault - the cause is still '
                f'present. Check the drive front panel / Workbench.')
        print(f'Controller: {self._fault_clear_message}')

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
        elif self._jog is not None:
            self._tare_message = 'Tare refused: a jog or touch-off is running'
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

    # --- Output position window -------------------------------------------
    # For DUTs whose output can only travel a limited angle (the Archimedes
    # drive: 200 deg between internal endstops). The operator declares centre,
    # a touch-off confirms both bumpers are where the config says, and from
    # then on a test trips if the output leaves centre +/- half_window. Jog and
    # touch-off run with the rig otherwise idle: LOAD in velocity mode, DUT
    # disabled (input shaft free), stopped by a torque cap, the max excursion,
    # an overspeed check, a timeout, a drive fault, or Stop.

    def _compile_window(self, spec):
        if not spec or not spec.get('enabled', True):
            return None
        t = spec.get('touch_off') or {}
        try:
            w = {
                'source': spec['source'],
                'get': attrgetter(spec['source']),
                'half_window': abs(float(spec['half_window_rad'])),
                'require_touch_off': bool(spec.get('require_touch_off', True)),
                'torque_nm': abs(float(t['torque_nm'])),
                'torque_sources': list(t['torque_sources']),
                'velocity': abs(float(t['velocity_rad_s'])),
                'approach_velocity': abs(float(t.get('approach_velocity_rad_s',
                                                     t['velocity_rad_s']))),
                'slow_from': abs(float(t.get('slow_from_rad', 0.0))),
                'expected_half_span': abs(float(t['expected_half_span_rad'])),
                'tolerance': abs(float(t['tolerance_rad'])),
                'max_excursion': abs(float(t['max_excursion_rad'])),
                'arm_distance': abs(float(t.get('arm_distance_rad', 0.03))),
                'trip_samples': max(1, int(t.get('trip_samples', 10))),
                # Trip on where the output would stop, not where it is: moving
                # outward at v, it adds v^2 / (2 * stop_decel). 0 = off.
                'stop_decel': abs(float(spec.get('stop_decel_rad_s2', 0.0))),
                'velocity_source': spec.get('velocity_source'),
            }
        except KeyError as e:
            raise ValueError(f'position_window in {self.mode}_dyno_config.yaml '
                             f'is missing {e}')
        w['velocity_get'] = (attrgetter(w['velocity_source'])
                             if w['velocity_source'] else None)
        if w['stop_decel'] > 0 and w['velocity_get'] is None:
            raise ValueError('position_window: stop_decel_rad_s2 needs velocity_source')
        w['torque_get'] = [attrgetter(p) for p in w['torque_sources']]
        if not w['torque_get']:
            raise ValueError('position_window.touch_off.torque_sources is empty')
        if w['half_window'] >= w['expected_half_span'] - w['tolerance']:
            raise ValueError('position_window: half_window_rad must sit inside the '
                             'bumpers (expected_half_span_rad - tolerance_rad)')
        if w['max_excursion'] <= w['expected_half_span'] + w['tolerance']:
            raise ValueError('position_window: max_excursion_rad must reach past '
                             'expected_half_span_rad + tolerance_rad, or touch-off '
                             'can never find the bumpers')
        return w

    def _window_rel(self):
        """Output position relative to the declared centre; NaN if unknown."""
        if self._window is None or self._window_centre is None:
            return math.nan
        try:
            return float(self._window['get'](self)) - self._window_centre
        except (TypeError, ValueError, AttributeError):
            return math.nan

    def _window_stop_distance(self, rel):
        """How much further the output travels outward if stopped now, at
        stop_decel. 0 when off or moving toward centre; NaN if unreadable."""
        w = self._window
        if w['stop_decel'] <= 0 or math.isnan(rel):
            return 0.0
        try:
            v = float(w['velocity_get'](self))
        except (TypeError, ValueError, AttributeError):
            return math.nan
        if math.isnan(v):
            return math.nan
        outward = v if rel >= 0 else -v
        return outward * outward / (2 * w['stop_decel']) if outward > 0 else 0.0

    def _window_trip(self, at_s):
        w = self._window
        if w is None:
            return None
        rel = self._window_rel()
        stop_distance = self._window_stop_distance(rel)
        if not math.isnan(rel) and abs(rel) + stop_distance <= w['half_window']:
            return None
        if self._window_centre is None:
            detail = 'position window: no centre declared'
        elif math.isnan(rel):
            detail = f'position window: {w["source"]} read NaN'
        elif math.isnan(stop_distance):
            detail = f'position window: {w["velocity_source"]} read NaN'
        elif abs(rel) <= w['half_window']:
            detail = (f'position window: output at {rel:+.3f} rad would stop at '
                      f'{math.copysign(abs(rel) + stop_distance, rel):+.3f} rad '
                      f'(stop_decel {w["stop_decel"]:g} rad/s^2), past '
                      f'+/-{w["half_window"]:g} rad')
        else:
            detail = (f'position window: output {rel:+.3f} rad from centre, '
                      f'outside +/-{w["half_window"]:g} rad')
        return {'kind': 'position_window',
                'check': 'position_window',
                'value': None if math.isnan(rel) else rel,
                'limit': w['half_window'],
                'at_s': at_s,
                'detail': detail}

    def _touch_off_ok(self):
        t = self._touch_off
        return bool(t and t['passed'] and t['centre'] == self._window_centre)

    def _window_arming_refusal(self):
        """Why Start must be refused, or None."""
        w = self._window
        if w is None:
            return None
        if self._jog is not None:
            return 'a jog or touch-off is running'
        if self._window_centre is None:
            return 'no centre declared this session'
        if w['require_touch_off'] and not self._touch_off_ok():
            return 'touch-off has not passed since centre was declared'
        rel = self._window_rel()
        if math.isnan(rel) or abs(rel) > w['half_window']:
            return (f'output is {rel:+.3f} rad from centre, outside the '
                    f'+/-{w["half_window"]:g} rad window')
        return None

    def _window_state(self):
        w = self._window
        if w is None:
            return None
        rel = self._window_rel()
        return {
            'source': w['source'],
            'centre': self._window_centre,
            'rel': None if math.isnan(rel) else rel,
            'half_window': w['half_window'],
            'expected_half_span': w['expected_half_span'],
            'tolerance': w['tolerance'],
            'max_excursion': w['max_excursion'],
            'jog': None if self._jog is None else self._jog['kind'],
            'touch_off': self._touch_off,
            'touch_off_ok': self._touch_off_ok(),
            'touch_off_history': self._touch_off_history,
            'refusal': self._window_arming_refusal(),
            'message': self._window_message,
        }

    def _window_say(self, message):
        self._window_message = message
        print(f'Controller: {message}')

    def _declare_centre(self):
        if self._window is None:
            return self._window_say('Declare centre refused: no position_window in config')
        if self._test_active or self._jog is not None:
            return self._window_say('Declare centre refused: the rig is busy')
        if not self._rig_at_rest():
            return self._window_say('Declare centre refused: the rig is moving')
        try:
            pos = float(self._window['get'](self))
        except (TypeError, ValueError, AttributeError):
            pos = math.nan
        if math.isnan(pos):
            return self._window_say(f'Declare centre refused: {self._window["source"]} '
                                    'is not readable')
        self._window_centre = pos
        # A touch-off is only meaningful relative to the centre it ran against.
        self._touch_off = None
        self._window_say(f'Centre declared at {pos:.4f} rad. Run touch-off before testing.')

    def _jog_refusal(self):
        if self._window is None:
            return 'no position_window in config'
        if self.shutdown:
            return 'the rig is shutting down'
        if self._test_active:
            return 'a test is running'
        if self._jog is not None:
            return 'a jog or touch-off is already running'
        if self._tare_state is not None:
            return 'a tare is running'
        if self._window_centre is None:
            return 'declare centre first'
        if self.devices.LOAD.fault or (not self._load_only and self.devices.DUT.fault):
            return 'a drive is faulted'
        return None

    def _leg(self, direction, expect, stop_at=None):
        """One constant-direction move. `expect` is 'contact' (ends on the
        torque cap), 'target' (ends when rel crosses stop_at) or 'manual'."""
        w = self._window
        span = 2 * w['max_excursion']
        return {'direction': direction, 'expect': expect, 'stop_at': stop_at,
                'start_rel': None, 't0': None, 'over': 0,
                'timeout_s': 2 * span / max(w['velocity'], 1e-3) + 10.0}

    def _start_jog(self, direction):
        refusal = self._jog_refusal()
        direction = 1 if float(direction) > 0 else -1
        if refusal is None and direction * self._window_rel() >= self._window['max_excursion']:
            refusal = 'already at max excursion in that direction'
        if refusal:
            return self._window_say(f'Jog refused: {refusal}')
        self._jog = {'kind': 'jog', 'legs': [self._leg(direction, 'manual')]}
        self._window_say(f'Jogging {"+" if direction > 0 else "-"}')

    def _start_touch_off(self):
        refusal = self._jog_refusal()
        rel = self._window_rel()
        if refusal is None and (math.isnan(rel) or abs(rel) > self._window['half_window']):
            refusal = 'output is outside the window; jog back toward centre first'
        if refusal:
            return self._window_say(f'Touch-off refused: {refusal}')
        self._touch_off = None
        self._jog = {'kind': 'touch_off',
                     'legs': [self._leg(+1, 'contact'), self._leg(-1, 'contact'),
                              self._leg(+1, 'target', stop_at=0.0)],
                     'contacts': [], 'problems': []}
        self._window_say('Touch-off running: + bumper, - bumper, back to centre')

    def _jog_step(self):
        """Advance the jog/touch-off one cycle and return the command to send."""
        j, w = self._jog, self._window
        load, dut = self.devices.LOAD, self.devices.DUT
        hold = {'input_mode': 'torque', 'output_mode': 'velocity',
                'input_command': 0, 'output_command': 0}

        if self.shutdown:
            return self._end_jog('Jog ended: shutdown', failed=True)
        if load.fault or (not self._load_only and dut.fault):
            return self._end_jog('Jog ended: a drive faulted', failed=True)

        # The input shaft is left free throughout.
        dut.sw_enable = False
        if load.mode != 'velocity' or load.switching_modes:
            if not load.switching_modes:
                load.command_operating_mode('velocity')
            load.sw_enable = False
            return hold
        load.sw_enable = True

        rel = self._window_rel()
        if math.isnan(rel):
            return self._end_jog(f'Jog ended: {w["source"]} read NaN', failed=True)

        now = time.perf_counter()
        leg = j['legs'][0]
        if leg['start_rel'] is None:
            leg['start_rel'], leg['t0'] = rel, now
        d = leg['direction']

        if d * rel >= w['max_excursion']:
            return self._leg_done('excursion', rel)
        if leg['stop_at'] is not None and d * (rel - leg['stop_at']) >= 0:
            return self._leg_done('target', rel)
        if abs(load.velocity) > 2 * max(w['approach_velocity'], w['velocity']) + 0.1:
            return self._end_jog(f'Jog ended: output overspeed '
                                 f'({load.velocity:+.3f} rad/s)', failed=True)
        if now - leg['t0'] > leg['timeout_s']:
            return self._end_jog('Jog ended: timed out', failed=True)

        # Torque cap. Waived only while backing away from a bumper (moving toward
        # centre, within arm_distance of where the move began), when the
        # bumper is still unloading. Moving outward it always applies.
        outward = d * rel > 0
        if outward or abs(rel - leg['start_rel']) >= w['arm_distance']:
            torque = max(abs(float(g(self))) for g in w['torque_get'])
            if math.isnan(torque):
                return self._end_jog('Jog ended: torque reading is NaN', failed=True)
            leg['over'] = leg['over'] + 1 if torque > w['torque_nm'] else 0
            if leg['over'] >= w['trip_samples']:
                return self._leg_done('contact', rel)

        slow = outward and abs(rel) > w['slow_from']
        speed = w['velocity'] if slow else w['approach_velocity']
        return dict(hold, output_command=d * speed)

    def _leg_done(self, reason, rel):
        j, w = self._jog, self._window
        leg = j['legs'].pop(0)

        if j['kind'] == 'jog':
            text = {'contact': f'contact ({w["torque_nm"]:g} Nm) at {rel:+.4f} rad',
                    'excursion': f'max excursion reached at {rel:+.4f} rad'}[reason]
            return self._end_jog(f'Jog stopped: {text}')

        side = '+' if leg['direction'] > 0 else '-'
        if leg['expect'] == 'contact':
            if reason == 'contact':
                j['contacts'].append(rel)
            else:
                j['problems'].append(f'no {side} bumper contact before max '
                                     f'excursion ({rel:+.4f} rad)')
                # Skip any remaining bumper and come home.
                j['legs'] = [self._leg(-1 if rel > 0 else 1, 'target', stop_at=0.0)]
        elif reason != 'target':
            j['problems'].append(f'return to centre stopped by {reason} at '
                                 f'{rel:+.4f} rad; jog back manually')
            j['legs'] = []

        if not j['legs']:
            return self._finish_touch_off()
        return {'input_mode': 'torque', 'output_mode': 'velocity',
                'input_command': 0, 'output_command': 0}

    def _finish_touch_off(self, abort_reason=None):
        j, w = self._jog, self._window
        contacts = j['contacts']
        plus = contacts[0] if len(contacts) > 0 else None
        minus = contacts[1] if len(contacts) > 1 else None
        problems = list(j['problems']) + ([abort_reason] if abort_reason else [])
        passed = plus is not None and minus is not None and not abort_reason
        if passed:
            for name, value in (('+', plus), ('-', -minus)):
                if abs(value - w['expected_half_span']) > w['tolerance']:
                    passed = False
                    problems.append(f'{name} bumper at {name}{value:.4f} rad, expected '
                                    f'{w["expected_half_span"]:.4f} +/- {w["tolerance"]:.4f}')
        result = {
            'at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'centre': self._window_centre,
            'plus_rad': plus,
            'minus_rad': minus,
            # Where the bumpers put centre, relative to the declared one.
            'centre_error_rad': None if plus is None or minus is None else (plus + minus) / 2,
            'span_rad': None if plus is None or minus is None else plus - minus,
            'torque_nm': w['torque_nm'],
            'expected_half_span_rad': w['expected_half_span'],
            'tolerance_rad': w['tolerance'],
            # Only the bumper positions decide a pass. A failed return leg is
            # reported, and Start stays refused until the output is back in
            # the window anyway.
            'passed': passed,
            'problems': problems,
        }
        self._touch_off = result
        self._touch_off_history.append(result)
        verdict = 'PASSED' if result['passed'] else 'FAILED'
        detail = (f'+{plus:.4f} / {minus:+.4f} rad' if plus is not None and minus is not None
                  else 'incomplete')
        return self._end_jog(f'Touch-off {verdict}: {detail}'
                             + (f' ({"; ".join(problems)})' if problems else ''),
                             finished=True)

    def _end_jog(self, message, failed=False, finished=False):
        """Stop LOAD and clear the jog. Returns the idle command for this cycle."""
        j = self._jog
        if j is not None and j['kind'] == 'touch_off' and not finished:
            # An interrupted touch-off must not leave an earlier pass standing.
            self._finish_touch_off(abort_reason=message)
            return self._safe_default_command
        self._jog = None
        load = self.devices.LOAD
        load.sw_enable = False
        self.devices.DUT.sw_enable = False
        if load.mode != 'torque' or load.switching_modes:
            load.command_operating_mode('torque')
        self._safe_default_command['input_mode'] = 'torque'
        self._safe_default_command['output_mode'] = 'torque'
        self._safe_default_command['input_command'] = 0
        self._safe_default_command['output_command'] = 0
        self._window_say(message)
        return self._safe_default_command

    def _sensor_snapshot(self):
        """Live torque-cell, position and velocity readings for behaviors that
        need telemetry (the multisine preamble anchors its amplitude to the
        cells' measured noise floor and watches position drift; ramp_break
        watches velocity to catch the shaft breaking free). Runs on the control
        thread inside next_command(), so it must only read attributes."""
        torque = {}
        for module in vars(self.devices).values():
            if not isinstance(module, ScaledChannels):
                continue
            for ch_params in (getattr(module, 'params', None) or {}).values():
                name = ch_params.get('name') if isinstance(ch_params, dict) else None
                if name and 'torque' in name:
                    torque[name] = float(getattr(module, name, 0.0) or 0.0)
        position = {'output': float(getattr(self.devices.LOAD, 'position', 0.0) or 0.0),
                    'input': float(getattr(self.devices.DUT, 'position', 0.0) or 0.0)}
        velocity = {'output': float(getattr(self.devices.LOAD, 'velocity', 0.0) or 0.0),
                    'input': float(getattr(self.devices.DUT, 'velocity', 0.0) or 0.0)}
        return {'torque': torque, 'position': position, 'velocity': velocity}

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
        self._fault_clear_step()

        # Aligning LOAD's position frame to DUT's only means something when the
        # two are mechanically coupled, which load-only assumes they are not.
        if (self._align_load_position
                and not self._load_only
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

        elif self._jog is not None:
            # A jog or touch-off drives LOAD itself; _jog_step owns the enables.
            self.current_cmd = self._jog_step()

        else:
            self.current_cmd = self._safe_default_command

            if self.devices.DUT.sw_enable or self.devices.LOAD.sw_enable:
                self.devices.DUT.sw_enable = False
                self.devices.LOAD.sw_enable = False

        # Both are rebuilt in _send_telemetry, below.
        self.control_state = None
        self.logging_state = None

        # Preamble sample index for the 'preamble_sample' log key; NaN on every
        # sample not generated by a preamble behavior.
        self.preamble_sample = self.current_cmd.get('sample_index', float('nan'))
        # Breakaway marker for the 'breakaway_torque' log key; NaN on every
        # sample except the one a ramp_break detection fired on.
        self.breakaway_torque = self.current_cmd.get('breakaway', float('nan'))

        ff_ratio = self._feedforward_ratio

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