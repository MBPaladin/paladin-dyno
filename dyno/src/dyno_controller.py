from dyno.src.master import Master
from dyno.src.devices import ScaledChannels
import yaml
import time
import math
import os
import signal
import sys
import threading
from operator import attrgetter
from deployment import dyno_paths
from dyno.src.test_manager import TestManager

SCHED_POLICY = os.SCHED_FIFO
SCHED_PRIO = 50

class Controller(Master):
    # A manual jog stops if the GUI's keepalive ('jog_alive', sent every 0.1 s
    # while a jog button is held) is missing for this long.
    JOG_KEEPALIVE_TIMEOUT_S = 0.5

    # Return to centre is refused this close to centre: the move would be over
    # before the drive finished enabling, and "already there" is the useful
    # answer.
    RETURN_CENTRE_DEADBAND_RAD = 0.005

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

        # Loop-time breakdown, for the `step_us` / `telemetry_us` log keys (both
        # optional -- a config that does not list them pays only the clock
        # reads). step_us is this cycle's; telemetry_us is necessarily the
        # PREVIOUS cycle's, since it is not known until after the sample
        # carrying it has been queued. Seeded NaN so the first cycles read as
        # "not measured yet" rather than a fictitious 0.
        self.step_us = float('nan')
        self.telemetry_us = float('nan')

        # Telemetry slot -1 (the control-state dict) is ~75-80% of every
        # sample's pickled bytes, and multiprocessing.Queue is a 64 KiB pipe.
        # At 1 kHz that pipe filled in ~52 ms, blocked the queue's feeder
        # thread, and the burst of pickles the feeder pushed when the GUI next
        # drained held the GIL against this process's SCHED_FIFO 80 cyclic
        # thread. Measured signature: a 52-cycle beat in cycle_time_us, present
        # at lag 52/102/202 in its autocorrelation, identical in all six
        # Archimedes runs, with the overrun 4-7 cycles ahead of every 20 A
        # current spike. See dyno/docs/archimedes_cycle_overrun_handoff.md.
        #
        # Nobody consumes it at 1 kHz: the GUI renders at ~33 Hz and the Logger
        # only reads it when a file opens. So it rides a heartbeat instead --
        # every CONTROL_STATE_PERIOD cycles, plus any cycle where logging_state
        # changed, which is exactly the cycles the Logger opens a file, closes
        # one, or crosses a behavior boundary on. Slot -1 is None in between.
        #
        # 20 cycles = 50 Hz, chosen so the 30 ms GUI tick always drains at
        # least one state. It cannot be change-triggered instead: _window_state
        # carries 'rel', the live shaft position, which moves every cycle.
        self._state_counter = 0
        self._last_logging_state = None
        self._sample_bytes_logged = False

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
        # Resume point after a position-window trip: which armed test and
        # which segment it was in, so the run can be picked up at the START of
        # that segment once the output is back at centre. Set by _stop_test,
        # reported in control_state (the GUI offers it once post_test is over),
        # consumed by the 'resume_test' command, cleared by 'abort_resume' or
        # any ordinary start. None when there is nothing to resume.
        self._resume = None
        # Why the last test ended, published on the sample that turns logging
        # off so the Logger can stamp it into the file it is about to close.
        # See _stop_test.
        self._stop_reason = None
        # The phase between a test ending and the drives coming off: an active
        # brake, then a logging tail. None whenever no run is winding down.
        # See _stop_test / _post_test_step.
        self._post_test = None
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
        # --- End-of-test brake and logging tail ---
        # Config `post_test:`; absent means tail 0 and brake off, which is the
        # old cut-the-power-and-close-the-file behaviour exactly.
        self._post = self._compile_post_test(self.dyno_params.get('post_test'))

        self._window = self._compile_window(self.dyno_params.get('position_window'))
        self._window_centre = None      # LOAD-frame position declared as centre
        self._touch_off = None          # last completed touch-off result
        self._touch_off_history = []    # every touch-off this session
        self._jog = None                # jog or touch-off in flight
        self._window_message = None
        # Measured on input-shaft moves, kept for the session: which way the
        # output turns for a positive input command (+1/-1) and the signed
        # input/output ratio. The sign is needed to know which input direction
        # is outward; config gear_ratio is not trusted for it (see log 16).
        self._input_sign = None
        self._input_ratio = None

        # Ratio-break watch. A slip is the one failure every other safety here
        # sleeps through: the output stops following the input, so it stays
        # still (position window and output velocity happy) while the input
        # spins well inside its own limit. On 2026-09-17 that let the input
        # reach 155 rad/s with the output locked and nothing stopped it -- see
        # docs/archimedes_gearbox_implementation_log.md. These two numbers are
        # the channel a slip actually moves, published for `safeties:` entries
        # (`source: ratio_error_rad` / `ratio_slip_rad_s`) and for the log.
        #   ratio_error_rad  how far the output has fallen behind where the
        #                    input says it should be, in OUTPUT rad, since the
        #                    reference was latched
        #   ratio_slip_rad_s the rate of that, output rad/s, averaged over
        #                    _RATIO_RATE_WINDOW_S so it is not a 1 kHz
        #                    difference of two encoder readings
        # Both NaN until a reference is latched and both positions read.
        self._ratio_ref = None
        self._ratio_rate_mark = None
        self.ratio_error_rad = math.nan
        self.ratio_slip_rad_s = math.nan

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
        self._resumable_checks = set()
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
            # resumable: this breach leaves the rig sound and merely displaced,
            # so the run can be picked up at the start of the segment it died
            # in once the output is back at centre -- the same offer a
            # position-window trip makes. Off by default: most safeties
            # (over-torque, over-temperature) mean stop and go look at it.
            #
            # Held as a set of names rather than a sixth element of the tuple
            # above, because _safety_checks is built by hand outside this class
            # -- dyno/sim/archimedes_window_test.py stubs it -- and widening
            # the tuple breaks every such caller for a flag none of them set.
            if spec.get('resumable', False):
                self._resumable_checks.add(check_name)
        # Consecutive-breach counter per check, reset on any in-range read.
        self._safety_streaks = {}

        # Cap how long the cyclic thread can wait to get the GIL back. The
        # default is 5 ms, and 5 ms is what the worst cycles in the Archimedes
        # logs cost: cycle_time_us max 5517 us, of which 5453 us was outside
        # both Controller.step() and _send_telemetry. The waiter here is the
        # SCHED_FIFO 80 process-data thread and the holder is the telemetry
        # queue's feeder thread, so priority buys nothing -- the GIL is not
        # priority-aware. 500 us trades a little more switching overhead for a
        # 10x shorter worst-case stall.
        #
        # This bounds the symptom, it does not remove the cause; the cause is
        # that the cyclic thread is the producer for a Python queue whose
        # consumer runs at 33 Hz. See the handoff doc's fix list.
        sys.setswitchinterval(0.0005)
        print(f'Controller: GIL switch interval {sys.getswitchinterval()*1e6:.0f} us')

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

        The reason rides out on the telemetry sample that turns logging off and
        the Logger stamps it onto the file as the `stop_reason` attribute. A
        None reason still overwrites the previous one: a stale reason on a new
        run is worse than no reason at all.

        With `post_test:` configured that sample is no longer the next one: the
        run hands over to _post_test_step, which brakes the shaft and then keeps
        logging for log_tail_s before letting the file close. The breaching
        sample is still absent -- the reason dict carries its value and limit
        for that -- but what the rig did NEXT is now in the trace, which for a
        stopping-distance trip is the part that matters. What the brake itself
        did is folded into the same reason dict under 'brake'.
        """
        self._stop_reason = reason
        # Record where a resumable trip left the plan BEFORE reset() clears
        # the segment bookkeeping. A window trip always qualifies -- the output
        # is still sound and merely displaced, and return-to-centre puts it
        # back. A safety may opt in with `resumable: true`, which is how a
        # ratio-break (slip) trip gets the same offer: the customer's drive is
        # built to slip, so a slip displaces the shafts rather than damaging
        # them, and the cure is the same one -- put the output back on centre
        # and restart the segment. Every other stop (operator, an ordinary
        # safety, completion) clears the offer.
        self._resume = None
        if (self._test_active and self.test_definition is not None
                and isinstance(reason, dict)
                and (reason.get('kind') == 'position_window'
                     or reason.get('resumable'))):
            p = self.test_definition.progress()
            # `label` titles the operator's dialog, so it has to name the thing
            # that actually tripped rather than always saying window.
            label = ('Position window trip'
                     if reason.get('kind') == 'position_window'
                     else f'Safety trip: {reason.get("check")}')
            self._resume = {'test': self.test_definition.name,
                            'segment': p['segment'], 'segments': p['segments'],
                            'segment_id': p['segment_id'], 'label': label,
                            'kind': reason.get('kind'), 'check': reason.get('check'),
                            'value': reason.get('value'), 'detail': reason.get('detail')}
            print(f'Controller: {label} in segment {p["segment"]}/{p["segments"]} '
                  f'({p["segment_id"]}); resumable from its start once recentred')
        if not self.test_definition == None:
            self.test_definition.reset()
        # Whether a run was actually in progress decides whether there is
        # anything to wind down. The GUI's Stop button lands here with the rig
        # already idle, and a tail there would open a fresh log full of nothing.
        was_active = self._test_active
        self._test_active = False

        # The absorber comes off immediately in every case. It is at zero
        # command by now and nothing past this point wants it pushing on the
        # output; only the input drive brakes.
        self.devices.LOAD.sw_enable = False
        self.devices.LOAD.command_operating_mode('torque')
        self._safe_default_command['output_mode'] = 'torque'
        self._safe_default_command['output_command'] = 0

        self._post_test = self._begin_post_test(reason) if was_active else None

        # A braking phase is the one case that keeps the input drive energised,
        # and _post_test_step drops it the moment the brake ends. Note the input
        # is deliberately NOT switched to torque mode here: an AKD de-energises
        # to change mode, so a switch would cut the torque exactly when the
        # brake needs it. The brake works in whatever mode the run left behind.
        # In both branches the input's standing command goes to zero. The
        # brake overrides it with its own every cycle, but _brake_step has one
        # path that falls through to the default (torque mode, velocity still
        # reading exactly zero, no direction to oppose yet) -- and there the
        # last command the TEST issued must not keep being applied.
        self._safe_default_command['input_command'] = 0
        if self._post_test is not None and self._post_test['stage'] == 'brake':
            self._safe_default_command['input_mode'] = self.devices.DUT.mode
        else:
            self.devices.DUT.sw_enable = False
            self._safe_default_command['input_mode'] = 'torque'

    def _start_test(self, start_segment=1):
        """The Start button, and the resume path with start_segment > 1."""
        if self._test_active:
            print('Unable to start test: Test already active')
        elif self._post_test_refusal():
            # Into _window_message, not just a print: this is the
            # one refusal an operator meets by pressing Start twice
            # in quick succession, and a button that does nothing
            # silently reads as a broken button.
            self._window_message = ('Start refused: '
                                    + self._post_test_refusal())
            print(f'Unable to start test: {self._window_message}')
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
            resumed = self._resume if start_segment > 1 else None
            self._resume = None
            self.pull_cmd = True
            if not self._load_only:
                self.devices.DUT.sw_enable = True
            self.devices.LOAD.sw_enable = True
            if not self.test_definition == None:
                self._test_active = True
                self.test_definition.reset(start_segment=start_segment)
            if resumed:
                self._window_say(f'Resuming {resumed["test"]} from the start of segment '
                                 f'{resumed["segment"]}/{resumed["segments"]} '
                                 f'({resumed["segment_id"]})')
            self._latch_ratio_reference()
            print('Starting test')

    def _dut_command_for_mode(self, cmd):
        """The input command to actually send, given the mode the drive is IN.

        AKD.send_command dispatches on the drive's own `mode`, not on the mode
        the command was written for, and the two disagree for as long as a mode
        switch is in flight -- an AKD has to walk back through Ready to Switch
        On, which took 18 cycles on this rig. A command of 0 written for torque
        mode ("no torque") is then read as a POSITION of 0, which means "travel
        to wherever the shaft was when position mode was entered".

        That is not theoretical. It fired on all five runs in
        logs/archimedes/gbx_1p2p0/troubleshooting_initial: every stop ended with
        `dut_position_command` stepping to 0 with `dut_mode` still 2, ~21 A (3x
        the 7 A continuous rating), and the input at 300 rad/s within 6 ms. In
        med_speed_5 the drive then disabled and the train coasted 1.19 rad at
        the output into the rubber bumper. _stop_test and _end_brake both ASK
        for torque mode before leaving a zero standing, and both are correct
        about the intent; the gap is that asking is not arriving.

        So while the drive is still in position mode, a command written for any
        other mode is replaced with "hold where you are". Torque and velocity
        modes need no equivalent: zero is genuinely zero in both.
        """
        command = cmd['input_command']
        dut = self.devices.DUT
        if dut.mode != 'position' or cmd.get('input_mode') == 'position':
            return command
        # getattr, because only AKD and ELMO have had their command-frame
        # arithmetic worked out (each differs -- see their properties). A drive
        # class without it falls through to the behaviour it has today rather
        # than being held at a position derived from a frame nobody checked.
        hold = getattr(dut, 'position_command_frame', math.nan)
        if not math.isnan(hold):
            return hold
        # No readable position to hold at. Repeating the last position command
        # keeps the drive where it was last told to be, which is within a
        # following error of the shaft -- still bounded, unlike 0.
        last = getattr(dut, 'position_command', math.nan)
        return command if math.isnan(last) else last

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
            # The wind-down after a run: 'brake', 'tail', or None. The GUI shows
            # it because Start and the jog buttons are refused while it lasts,
            # and an operator who is not told why reads that as a dead button.
            'post_test': (None if self._post_test is None
                          else self._post_test['stage']),
            # Where a window trip left the plan, or None. The GUI waits for
            # post_test to clear before offering recentre-and-resume on it.
            'resume': self._resume,
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
        elif self._post_test is not None:
            # The run is over but the file stays open: the brake and the coast
            # after it are what a stopping-distance trip has to be judged on,
            # and an E-Stop's interesting part starts here too. No behavior_id,
            # so the Logger closes out the last behavior's index range and the
            # tail is not attributed to it.
            self.logging_state = {'log': True}
        elif self._stop_reason is not None:
            # _stop_test cleared _test_active earlier in this same step(), so
            # this is the first log=False sample after the run -- exactly the
            # one the Logger closes the file on. It keeps riding along on the
            # idle samples after it, which costs nothing and means a Logger
            # that starts late still has the reason to hand.
            self.logging_state = {'log': False, 'stop_reason': self._stop_reason}

        # Heartbeat, or any cycle that changed what the Logger keys off.
        send_state = (self._state_counter % self.CONTROL_STATE_PERIOD == 0
                      or self.logging_state != self._last_logging_state)
        self._last_logging_state = self.logging_state
        self._state_counter += 1
        self.control_state = self._control_state() if send_state else None

        self.time = time.perf_counter() - self.t_offset

        telemetry = [getter(self) for getter in self._telemetry_compiled]
        telemetry.append(self.logging_state)
        telemetry.append(self.control_state)

        self._report_sample_bytes(telemetry, send_state)

        self._telemetry_queue.put_nowait(telemetry)

    # How many cycles between control-state heartbeats. See __init__.
    CONTROL_STATE_PERIOD = 20

    def _report_sample_bytes(self, telemetry, has_state):
        """Print the pickled size of both sample variants, once, at bring-up.

        This is the number that sets the overrun period: multiprocessing.Queue
        is a 64 KiB pipe, so 65536 / sample_bytes is how many samples buffer
        before the feeder thread blocks, and at 1 kHz that count IS the beat
        period in ms. Measuring it beats inferring it, and it costs two
        pickles per bring-up rather than two per cycle."""
        if self._sample_bytes_logged or not has_state:
            return
        self._sample_bytes_logged = True
        try:
            import pickle
            full = len(pickle.dumps(telemetry)) + 4   # +4: send_bytes length header
            lean = len(pickle.dumps(telemetry[:-1] + [None])) + 4
            mean = (full + lean * (self.CONTROL_STATE_PERIOD - 1)) / self.CONTROL_STATE_PERIOD
            rate = 1_000_000 / self.process_data_cycle_time_us
            print('#' * 32)
            print('Telemetry sample size')
            print(f'\twith control_state:    {full} bytes')
            print(f'\twithout (slot -1 None): {lean} bytes')
            print(f'\tmean at 1:{self.CONTROL_STATE_PERIOD} heartbeat: {mean:.0f} bytes')
            print(f'\t64 KiB pipe holds {65536 / mean:.0f} samples '
                  f'= {65536 / mean / rate * 1000:.0f} ms of buffer')
            print(f'\t(before this change: {65536 / full:.0f} samples '
                  f'= {65536 / full / rate * 1000:.0f} ms -- the measured beat was 52 ms)')
            print('#' * 32)
        except Exception as e:  # never let instrumentation cost a run
            print(f'Controller: could not size the telemetry sample: {e}')

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
                # An operator action ends the logging tail early rather than
                # being refused by it. The tail is a best-effort recording
                # window, not a lock: an operator who reaches for Jog 300 ms
                # after a trip should get a jog, not a dead button, and the log
                # simply closes at that point instead of two seconds later. The
                # BRAKE is the opposite -- that one blocks, because it is still
                # holding the shaft (see _post_test_refusal).
                if cmd[0] in self._TAIL_YIELDING_CMDS:
                    self._yield_post_test_tail()
                if cmd[0] == 'start_test':
                    self._start_test()

                elif cmd[0] == 'resume_test':
                    # Pick the armed plan up at the start of the segment a
                    # window trip interrupted. Same refusals as Start, plus:
                    # there must be an offer, and the armed plan must still be
                    # the one that tripped (re-arming another test between the
                    # trip and the resume makes the segment number meaningless).
                    r = self._resume
                    if r is None:
                        self._window_say('Resume refused: nothing to resume')
                    elif self.test_definition is None or self.test_definition.name != r['test']:
                        self._window_say(f'Resume refused: {r["test"]} is no longer armed')
                    elif self._jog is not None:
                        self._window_say('Resume refused: a jog or return-to-centre is running')
                    else:
                        self._start_test(start_segment=r['segment'])

                elif cmd[0] == 'abort_resume':
                    if self._resume is not None:
                        self._window_say(f'Resume of {self._resume["test"]} abandoned by the operator')
                    self._resume = None

                elif cmd[0] == 'stop_test':
                    if self._jog is not None:
                        self._end_jog('Stopped by the operator', failed=True)
                    self._stop_test({'kind': 'operator',
                                     'detail': 'stopped from the GUI by the operator'})
                    print('attempting to stop test, in / out motor commanded to torque mode')

                elif cmd[0] == 'declare_centre':
                    self._declare_centre()

                elif cmd[0] == 'jog':
                    # ['jog', direction] or ['jog', direction, 'input'|'output']
                    self._start_jog(cmd[1], cmd[2] if len(cmd) > 2 else 'output')

                elif cmd[0] == 'jog_alive':
                    # Keepalive from the GUI while a jog button is held. Only
                    # refreshes a running manual jog; it never starts one, so a
                    # late keepalive cannot restart a jog that stopped on
                    # contact.
                    if self._jog is not None and self._jog['kind'] == 'jog':
                        self._jog['alive'] = time.perf_counter()

                elif cmd[0] == 'jog_stop':
                    # Hold-to-run release. Only ends a manual jog: a touch-off
                    # is stopped with Stop, not by letting go of a jog button.
                    if self._jog is not None and self._jog['kind'] == 'jog':
                        self._end_jog('Jog released')

                elif cmd[0] == 'touch_off':
                    self._start_touch_off()

                elif cmd[0] == 'return_centre':
                    self._start_return_centre()

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
                        'resumable': check_name in getattr(
                            self, '_resumable_checks', ()),
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
    # Averaging window for ratio_slip_rad_s. Long enough that encoder
    # quantisation does not dominate, short enough that a runaway is caught in
    # a tenth of a second once trip_samples is added on top.
    _RATIO_RATE_WINDOW_S = 0.05

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
        elif self._post_test is not None:
            self._tare_message = 'Tare refused: ' + self._post_test_refusal()
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
    # touch-off run with the rig otherwise idle, driving one shaft in velocity
    # mode with the other disabled (free):
    #   - output drive: LOAD moves, contact is a fixed torque on the output cell.
    #   - input drive (`touch_off.input:`): DUT moves, contact is a RISE above the
    #     drag measured during the move, on the input cell. The gearbox
    #     multiplies whatever the input pushes, so a fixed threshold would have
    #     to sit above the drag and would land ~ratio x that on the bumper.
    # Either is stopped by contact, the max excursion, an overspeed check, a
    # timeout, a drive fault, or Stop. Input moves also stop if the output stops
    # following the input at the configured ratio (coupling slip).
    #
    # Return to centre (`return_centre`) is the leg a touch-off ends with,
    # offered on its own button: one input-driven move home, click to run
    # rather than hold, subject to every check above.

    # --- End of test: active brake, then a logging tail -------------------
    #
    # Both are config (`post_test:`); absent means neither runs and a stop is
    # the old cut-the-power-and-close-the-file.
    #
    # Why the brake exists: on the Archimedes gearbox the free coast is
    # 20.3 rad/s^2 at the output, so a trip at back-drive speed travels 1.91 rad
    # before stopping -- past the window and past the bumpers. The input drive
    # stops it in 0.14 rad at 4 Nm, because the 43.88:1 ratio multiplies its
    # torque while the train's inertia is mostly its own rotor. See the
    # measurement in the config block and log section 1.
    #
    # Why the tail exists: _test_active gates logging, so the file used to close
    # on the sample that ended the run. The coast after a trip -- the one thing
    # a stopping-distance safety is betting on -- was never recorded, and an
    # E-Stop (which arrives here as a drive fault) cut the file at the instant
    # the interesting part started.

    # Endings that must not try to brake. A DS402 fault means the drive will not
    # take a command at all, and shutdown means the process is going away, where
    # holding a tail open risks losing the log entirely.
    _NO_BRAKE_KINDS = frozenset({'drive_fault', 'shutdown'})
    # Endings that skip the tail too.
    _NO_TAIL_KINDS = frozenset({'shutdown'})
    # How far over stop_velocity the output has to be for a settling brake to
    # start pushing again. Hysteresis, not a threshold: it exists so driveline
    # ring-down cannot re-trigger the brake, only real motion can. Sized off the
    # rig -- the chatter this guards against reached 0.163 rad/s at the output
    # (brake_checks/torque_mode_brake_check), so 8 x the 0.05 stop_velocity
    # leaves 2.5x margin over it. Erring high is safe: the shaft cannot creep
    # at all until something beats 14 Nm of reflected static friction, and once
    # it does it will be moving far faster than this.
    _BRAKE_RESUME_FACTOR = 8.0

    def _compile_post_test(self, spec):
        """`post_test:` -> {'log_tail_s': float, 'brake': dict or None}.

        Config-only: this runs from __init__, before Master.run has built
        self.devices, so nothing here may touch a drive. The brake torque is
        checked against the drive's limit at brake entry instead.
        """
        spec = spec or {}
        post = {'log_tail_s': max(0.0, float(spec.get('log_tail_s', 0.0))),
                'brake': None}
        b = spec.get('brake') or {}
        if not b.get('enabled', False):
            return post
        if self._load_only:
            raise ValueError(f'post_test.brake in {self.mode}_dyno_config.yaml '
                             'brakes with the input drive, but load_only is set '
                             '-- that drive is never commanded')
        source = b.get('velocity_source') or 'devices.LOAD.velocity'
        post['brake'] = {
            'torque_nm': abs(float(b['torque_nm'])),
            'velocity_source': source,
            'velocity_get': attrgetter(source),
            'stop_velocity': abs(float(b.get('stop_velocity_rad_s', 0.05))),
            'min_velocity': abs(float(b.get('min_velocity_rad_s', 0.05))),
            'settle_s': max(0.0, float(b.get('settle_s', 0.05))),
            'timeout_s': max(0.0, float(b.get('timeout_s', 0.5))),
            'verify_s': max(0.0, float(b.get('verify_s', 0.010))),
            'verify_rise': abs(float(b.get('verify_rise_frac', 0.10))),
            # 'auto' derives it from the drive's own two sign flags, which is
            # what the rig is verified on. +1/-1 pins it, for a drive whose
            # current loop does not follow the usual convention or a sim that
            # models the flags differently.
            'polarity': b.get('polarity', 'auto'),
        }
        if post['brake']['polarity'] not in ('auto', 1, -1):
            raise ValueError(f'post_test.brake.polarity in {self.mode}_dyno_'
                             f'config.yaml must be auto, 1 or -1, got '
                             f'{post["brake"]["polarity"]!r}')
        if post['brake']['timeout_s'] <= 0:
            raise ValueError(f'post_test.brake in {self.mode}_dyno_config.yaml '
                             'needs timeout_s > 0, or the brake can never run')
        return post

    def _brake_polarity(self):
        """Which way a DUT *torque* command moves DUT.velocity as the log reads
        it: +1 when a positive command speeds the reported velocity up.

        flip_torque_sign and flip_direction_sign are independent knobs on
        different signals -- current goes through the first (devices.py, in
        send_command), position and velocity through the second (in
        process_txpdo) -- so a config with one set and not the other reports a
        shaft accelerating the opposite way to the torque that drove it. On this
        rig's DUT that is exactly the case today, which is why commanding
        +0.5 Nm in the coastdowns logged -192 rad/s.

        Both hardware cases have been measured on this rig, and the derivation
        matches both (2026-09-17 coastdowns, log finding 21):

        | DUT flags (torque, direction) | +0.5 Nm gave | polarity |
        |---|---|---|
        | False, True (`longer_coastdown`) | -0.501 A, -192 rad/s | -1 |
        | True, True (`test_flip_torque_sign_true`) | -0.490 A, +51 rad/s | +1 |

        The current row is the load-bearing one: a positive command really does
        send NEGATIVE current on a flipped drive, and the drive's own loop then
        makes negative torque in its own frame. So flip_torque_sign reaches the
        shaft, it is not cancelled somewhere -- which is exactly what the fake
        drive in dyno/sim/fake_pysoem.py used to assume (see its
        `hw_torque_sign`).

        A torque-mode brake has to know which it is, and getting it wrong
        accelerates the shaft at the bumper. Deriving it from the same two flags
        the drive itself uses means the answer cannot drift out of step with
        them; `post_test.brake.polarity` can pin it, and _brake_step verifies
        whichever answer it gets against the measured velocity and bails out if
        the shaft speeds up.
        """
        pinned = self._post['brake']['polarity']
        if pinned != 'auto':
            return int(pinned)
        dut = self.devices.DUT
        polarity = 1
        if getattr(dut, 'flip_torque_sign', False):
            polarity = -polarity
        if getattr(dut, 'flip_direction_sign', False):
            polarity = -polarity
        return polarity

    def _begin_post_test(self, reason):
        """The phase dict for a run that has just ended, or None for no phase.

        Called from _stop_test with the drives still enabled.
        """
        kind = (reason or {}).get('kind')
        tail = 0.0 if kind in self._NO_TAIL_KINDS else self._post['log_tail_s']
        b = self._post['brake']
        now = time.perf_counter()

        brake = None
        if b is not None and kind not in self._NO_BRAKE_KINDS:
            try:
                speed = abs(float(b['velocity_get'](self)))
            except (TypeError, ValueError, AttributeError):
                speed = math.nan
            # NaN is not "stopped": the output position source has failed, which
            # is itself a window trip, and the shaft may well be moving. Brake.
            if math.isnan(speed) or speed > b['min_velocity']:
                # Whatever mode the drive is already in is the mode the brake
                # uses. An AKD drops to Ready to Switch On to change mode, which
                # would de-energise it for the duration -- the shaft would coast
                # exactly when it must not.
                mode = self.devices.DUT.mode
                torque = min(b['torque_nm'],
                             abs(float(self.devices.DUT.torque_limit)))
                # The input's own feedback has to be readable, or there is
                # nothing to brake against and nothing to check the polarity
                # with. A NaN entry speed would make the speed-up backstop
                # compare against NaN, which is False forever -- the brake
                # would push at the bumper with its one safeguard silently
                # disabled. Position mode also commands hold_position directly.
                entry_input = abs(float(self.devices.DUT.velocity))
                # In the COMMAND frame, not the feedback frame. These differ by
                # the shaft position at the last position-mode entry, and
                # commanding the feedback value steps the shaft by that gap
                # instead of holding it -- see AKD.position_command_frame.
                hold = float(self.devices.DUT.position_command_frame)
                if math.isnan(entry_input) or (mode == 'position'
                                               and math.isnan(hold)):
                    print('Post-test brake skipped: input drive feedback reads '
                          'NaN, so there is nothing to brake against')
                    return {'reason': reason, 'stage': 'tail', 'brake': None,
                            'tail_s': tail, 'tail_ends': now + tail} if tail > 0 else None
                brake = {'mode': mode,
                         # 'slowing' while it is taking energy out, 'settling'
                         # once the output has been seen at rest. See
                         # _brake_step.
                         'phase': 'slowing',
                         'slowed_at': None,
                         'torque_nm': torque,
                         'clipped': torque < b['torque_nm'],
                         'polarity': self._brake_polarity(),
                         'entry_speed': None if math.isnan(speed) else speed,
                         'entry_input_speed': entry_input,
                         'hold_position': hold,
                         'started': now,
                         'deadline': now + b['timeout_s'],
                         'below_since': None,
                         'outcome': None}

        if brake is None and tail <= 0:
            return None
        return {'reason': reason,
                'stage': 'brake' if brake else 'tail',
                'brake': brake,
                'tail_s': tail,
                'tail_ends': now + tail}

    def _post_test_step(self):
        """Advance the brake, then the tail. Returns this cycle's command.

        Never blocks and never raises: this runs after a safety trip, so an
        exception here would strand the rig with the drives still enabled.
        """
        phase = self._post_test
        now = time.perf_counter()

        # Shutdown ends the wind-down on the spot. Master stops calling step()
        # shortly after _release_drives, and a tail still holding `log: True`
        # when that happens never emits the sample the Logger closes the file
        # on -- so insisting on the last two seconds of trace would cost the
        # whole log.
        if self.shutdown:
            if phase['stage'] == 'brake':
                phase['brake']['outcome'] = 'cut short by rig shutdown'
                self._end_brake(phase, now)
            else:
                self.devices.DUT.sw_enable = False
                self.devices.LOAD.sw_enable = False
            self._post_test = None
            return self._safe_default_command

        if phase['stage'] == 'brake':
            cmd = self._brake_step(phase, now)
            if cmd is not None:
                return cmd
            # Brake finished (or gave up): drives off, tail starts now.
            self._end_brake(phase, now)

        if now >= phase['tail_ends']:
            self._post_test = None
        return self._safe_default_command

    def _brake_step(self, phase, now):
        """One cycle of braking, or None once it is done.

        Records how it ended in phase['brake']['outcome'], which rides out on
        the stop_reason so the log says what the stop actually did.
        """
        b, st = self._post['brake'], phase['brake']
        dut = self.devices.DUT

        if dut.fault or not dut.sw_enable:
            st['outcome'] = 'drive came off mid-brake'
            return None
        if now >= st['deadline']:
            st['outcome'] = f'timed out after {b["timeout_s"]:g} s'
            return None

        # A mode switch would de-energise the drive, so if the mode moved out
        # from under us there is nothing safe left to command.
        if dut.switching_modes or dut.mode != st['mode']:
            st['outcome'] = 'input drive changed mode mid-brake'
            return None

        try:
            speed = abs(float(b['velocity_get'](self)))
        except (TypeError, ValueError, AttributeError):
            speed = math.nan

        at_rest = not math.isnan(speed) and speed <= b['stop_velocity']
        if at_rest:
            if st['below_since'] is None:
                st['below_since'] = now
            if now - st['below_since'] >= b['settle_s']:
                st['outcome'] = 'stopped'
                st['stopped_after_s'] = round(now - st['started'], 4)
                return None
        else:
            st['below_since'] = None

        # Sub-phase. The torque-mode brake is bang-bang on sign(DUT.velocity),
        # and once the shaft is stopped that sign is noise: on the rig
        # (2026-09-17, brake_checks/torque_mode_brake_check) it flipped 77 times
        # in 500 ms, pumping +/-0.5 Nm into a 3.6e-4 kg m^2 rotor and rocking
        # the driveline through the gearbox's backlash at +/-0.16 rad/s at the
        # output. That is over stop_velocity, so the settle window never closed
        # and the brake timed out -- with the output already parked, having moved
        # 0.08 deg across the whole half second.
        #
        # Once the output has been seen at rest, braking torque does nothing
        # useful anyway: the gearbox holds the shaft on its own (0.32 Nm of
        # input drag is 14 Nm at the output). So stop commanding and just watch
        # it settle. Only clearly real motion pushes it back to slowing.
        if at_rest:
            if st['phase'] == 'slowing':
                st['phase'] = 'settling'
                st['slowed_at'] = now
        elif speed > self._BRAKE_RESUME_FACTOR * b['stop_velocity']:
            st['phase'] = 'slowing'

        if st['mode'] == 'velocity':
            # Sign-free: zero is zero in either frame, so this path cannot get
            # the polarity wrong. It is NOT stronger than a torque command,
            # though -- measured on the rig 2026-09-17 the AKD's own ramp pulled
            # 0.51 Nm at the input (33 rad/s^2 at the output), against 0.73 Nm
            # (47 rad/s^2) from a 0.5 Nm torque command. And unlike torque mode
            # it does not scale with torque_nm: raising that leaves this path
            # where it is, which is why stop_decel_rad_s2 has to be set from
            # this weaker number unless the drive's own decel ramp is raised.
            return dict(self._safe_default_command, input_mode='velocity',
                        input_command=0.0)
        if st['mode'] == 'position':
            # Also sign-free: hold where the shaft was when the run ended.
            return dict(self._safe_default_command, input_mode='position',
                        input_command=st['hold_position'])

        # Torque mode. Settling: command nothing. Velocity and position mode
        # need no equivalent -- their commands (zero speed, hold position) are
        # closed by the drive's own loop and do not chatter, which the rig
        # confirmed: the velocity-mode check stopped cleanly in 127 ms.
        if st['phase'] == 'settling':
            return dict(self._safe_default_command, input_mode='torque',
                        input_command=0.0)

        # Oppose the input's own motion -- same shaft the command acts on, and
        # 43.88x the resolution of the output channel.
        v_in = float(dut.velocity)
        if math.isnan(v_in):
            st['outcome'] = 'input velocity read NaN'
            return None
        if v_in == 0.0:
            # No direction to oppose yet; wait rather than guess one.
            return self._safe_default_command

        # Backstop on _brake_polarity: if the input is speeding UP the sign is
        # wrong and every further cycle drives the output harder at the bumper.
        elapsed = now - st['started']
        if elapsed >= b['verify_s']:
            rise = abs(v_in) - st['entry_input_speed'] * (1 + b['verify_rise'])
            if rise > 0:
                st['outcome'] = (f'aborted: input sped up from '
                                 f'{st["entry_input_speed"]:.1f} to {abs(v_in):.1f} '
                                 f'rad/s under braking torque -- polarity '
                                 f'{st["polarity"]:+d} is wrong for this drive, '
                                 f'check flip_torque_sign / flip_direction_sign')
                print(f'POST-TEST BRAKE {st["outcome"]}')
                return None

        command = -st['polarity'] * math.copysign(st['torque_nm'], v_in)
        return dict(self._safe_default_command, input_mode='torque',
                    input_command=command)

    def _end_brake(self, phase, now):
        """Drop the drives and fold the brake's outcome into the stop_reason."""
        st = phase['brake']
        self.devices.DUT.sw_enable = False
        self.devices.LOAD.sw_enable = False
        # Back to torque mode before a zero command is left standing, the same
        # way _end_jog leaves the drives. A zero command in POSITION mode means
        # "go to position 0", which on a shaft resting at 10 rad is a lurch --
        # and the position-mode brake was deliberately holding it where it
        # stopped right up to this point.
        if self.devices.DUT.mode != 'torque' or self.devices.DUT.switching_modes:
            self.devices.DUT.command_operating_mode('torque')
        self._safe_default_command['input_mode'] = 'torque'
        self._safe_default_command['input_command'] = 0
        self._safe_default_command['output_command'] = 0
        phase['stage'] = 'tail'
        # The tail is measured from the end of the brake, not from the trip, so
        # a long brake cannot eat into it.
        phase['tail_ends'] = now + phase['tail_s']

        if st is not None and isinstance(self._stop_reason, dict):
            record = {'mode': st['mode'],
                      'torque_nm': st['torque_nm'],
                      'polarity': st['polarity'],
                      'entry_velocity_rad_s': st['entry_speed'],
                      'outcome': st['outcome'] or 'ended'}
            if st['clipped']:
                record['note'] = ('torque_nm clipped to the drive '
                                  'motor_limits.torque')
            if 'stopped_after_s' in st:
                record['stopped_after_s'] = st['stopped_after_s']
            # How long the brake spent actually taking energy out, as distinct
            # from the settle window bolted on after it. This is the number a
            # braked deceleration is measured from.
            if st['slowed_at'] is not None:
                record['slowing_s'] = round(st['slowed_at'] - st['started'], 4)
            self._stop_reason['brake'] = record
            print(f'Post-test brake: {record["outcome"]} '
                  f'({st["mode"]} mode, {st["torque_nm"]:g} Nm)')

    # Operator commands that end a logging tail early instead of being refused
    # by it. Read in _cmd_check, before the command is acted on.
    _TAIL_YIELDING_CMDS = frozenset({'start_test', 'resume_test', 'jog', 'touch_off', 'tare',
                                     'declare_centre', 'clear_tare',
                                     'clear_faults', 'test_def', 'disarm',
                                     'return_centre'})

    def _yield_post_test_tail(self):
        """Close out a logging tail now, so an operator action can proceed.

        Only the tail: a brake is still holding the shaft and is not something
        an operator command may cut short. Dropping _post_test here means the
        next _send_telemetry emits the log=False sample carrying the stop
        reason, so the file closes one cycle later -- the trace is simply
        shorter than log_tail_s, which is the right trade for not making the
        panel feel broken.
        """
        if self._post_test is not None and self._post_test['stage'] == 'tail':
            self._post_test = None

    def _post_test_refusal(self):
        """Why an operator action must wait, or None.

        Braking only. The tail yields instead (_yield_post_test_tail), so by
        the time a command reaches its handler a tail is already gone and this
        speaks for a brake that is still holding the shaft.
        """
        if self._post_test is None or self._post_test['stage'] != 'brake':
            return None
        return 'the last run is still braking to a stop -- try again in a moment'

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
                'slow_from': abs(float(t.get('slow_from_rad', 0.0))),
                'expected_half_span': abs(float(t['expected_half_span_rad'])),
                'tolerance': abs(float(t['tolerance_rad'])),
                'max_excursion': abs(float(t['max_excursion_rad'])),
                # Trip on where the output would stop, not where it is: moving
                # outward at v, it adds v^2 / (2 * stop_decel). 0 = off.
                'stop_decel': abs(float(spec.get('stop_decel_rad_s2', 0.0))),
                'velocity_source': spec.get('velocity_source'),
                'touch_off_drive': str(t.get('drive', 'output')),
                # Move centre to the midpoint of the two bumpers after touch-off.
                'recentre': bool(t.get('recentre', False)),
                'max_recentre': abs(float(t.get('max_recentre_rad', 0.2))),
            }
            # Per-shaft settings: jog buttons for a shaft need its block.
            w['output'] = self._compile_output_drive(t.get('output'))
            w['input'] = self._compile_input_drive(t.get('input'))
        except KeyError as e:
            raise ValueError(f'position_window in {self.mode}_dyno_config.yaml '
                             f'is missing {e}')
        if w['touch_off_drive'] not in ('input', 'output'):
            raise ValueError('position_window.touch_off.drive must be input or output')
        if w[w['touch_off_drive']] is None:
            raise ValueError(f'position_window.touch_off.drive is {w["touch_off_drive"]}, '
                             f'but there is no touch_off.{w["touch_off_drive"]} block')
        w['velocity_get'] = (attrgetter(w['velocity_source'])
                             if w['velocity_source'] else None)
        if w['stop_decel'] > 0 and w['velocity_get'] is None:
            raise ValueError('position_window: stop_decel_rad_s2 needs velocity_source')
        if w['half_window'] >= w['expected_half_span'] - w['tolerance']:
            raise ValueError('position_window: half_window_rad must sit inside the '
                             'bumpers (expected_half_span_rad - tolerance_rad)')
        if w['max_excursion'] <= w['expected_half_span'] + w['tolerance']:
            raise ValueError('position_window: max_excursion_rad must reach past '
                             'expected_half_span_rad + tolerance_rad, or touch-off '
                             'can never find the bumpers')
        return w

    @staticmethod
    def _compile_output_drive(spec):
        """`position_window.touch_off.output`: jog and touch-off on the output
        shaft. Speeds and torques are output-frame. None when absent."""
        if not spec:
            return None
        o = {
            'torque_nm': abs(float(spec['torque_nm'])),
            'torque_sources': list(spec['torque_sources']),
            'velocity': abs(float(spec['velocity_rad_s'])),
            'approach_velocity': abs(float(spec.get('approach_velocity_rad_s',
                                                    spec['velocity_rad_s']))),
            'arm_distance': abs(float(spec.get('arm_distance_rad', 0.03))),
            'trip_samples': max(1, int(spec.get('trip_samples', 10))),
        }
        o['torque_get'] = [attrgetter(p) for p in o['torque_sources']]
        if not o['torque_get']:
            raise ValueError('position_window.touch_off.output.torque_sources is empty')
        return o

    @staticmethod
    def _compile_input_drive(spec):
        """`position_window.touch_off.input`: jog and touch-off on the input
        shaft. Speeds and torques are input-frame. None when absent."""
        if not spec:
            return None
        sources = []
        for entry in spec['torque_sources']:
            # A bare path, or [path, scale] (e.g. drive current x kt).
            path, scale = (entry, 1.0) if isinstance(entry, str) else (entry[0], float(entry[1]))
            sources.append((path, attrgetter(path), scale))
        if not sources:
            raise ValueError('position_window.touch_off.input.torque_sources is empty')
        i = {
            'velocity': abs(float(spec['velocity_rad_s'])),
            'approach_velocity': abs(float(spec.get('approach_velocity_rad_s',
                                                    spec['velocity_rad_s']))),
            'nominal_ratio': abs(float(spec['nominal_ratio'])),
            'ratio_tolerance': abs(float(spec.get('ratio_tolerance', 0.3))),
            'torque_sources': sources,
            'drag_nm': abs(float(spec['drag_nm'])),
            'contact_rise_nm': abs(float(spec['contact_rise_nm'])),
            'ceiling_nm': abs(float(spec['ceiling_nm'])),
            'arm_input_rad': abs(float(spec.get('arm_input_rad', 1.0))),
            'trip_samples': max(1, int(spec.get('trip_samples', 10))),
            'drag_tau_s': abs(float(spec.get('drag_tau_s', 0.5))),
            # Slew limit on the speed command. A step makes the drive's velocity
            # loop kick far over the contact torque as the move starts.
            'acceleration': abs(float(spec['acceleration_rad_s2'])),
        }
        if i['drag_nm'] + i['contact_rise_nm'] > i['ceiling_nm']:
            raise ValueError('position_window.touch_off.input: drag_nm + contact_rise_nm '
                             'must not exceed ceiling_nm')
        return i

    def _window_rel(self):
        """Output position relative to the declared centre; NaN if unknown."""
        if self._window is None or self._window_centre is None:
            return math.nan
        try:
            return float(self._window['get'](self)) - self._window_centre
        except (TypeError, ValueError, AttributeError):
            return math.nan

    def _ratio_positions(self):
        """(input position, output position) in their own raw frames, or None
        when either is unreadable. Load-only has no input shaft to compare
        against, so it never reports a pair."""
        if self._load_only:
            return None
        try:
            din = float(self.devices.DUT.position)
            dout = float(self._window['get'](self)) if self._window is not None \
                else float(self.devices.LOAD.position)
        except (TypeError, ValueError, AttributeError, KeyError):
            return None
        if math.isnan(din) or math.isnan(dout):
            return None
        return din, dout

    def _ratio_value(self):
        """Signed input/output ratio to hold the shafts to, or None.

        Prefers the ratio TOUCH-OFF MEASURED (`_input_ratio`, input rad per
        output rad, signed, in the frame the logged channels use) over the
        config's `gear_ratio`. That is deliberate: as of 2026-09-17 this rig
        carries `gear_ratio: -43.88` with `flip_direction_sign: true` on the
        DUT, so the two flips cancel and the channels actually read +43.88 --
        the config sign is the open item in the log, not a fact. Taking the
        measured one means this check cannot be wrong-footed by that sign,
        whichever way it is eventually settled, and a wrong sign here would
        double every normal motion into a false trip."""
        if self._input_ratio:
            return float(self._input_ratio)
        try:
            r = float(self.devices.DUT.params.get('gear_ratio', 0) or 0)
        except (TypeError, ValueError, AttributeError):
            return None
        return r or None

    def _latch_ratio_reference(self):
        """Zero the ratio-break watch where the shafts are now.

        Called at Declare Centre and at the start of every test, so a run
        measures the slip it causes itself instead of inheriting the last
        one's. Without that, one deliberate slip test would leave the error
        parked past the limit and refuse every run after it."""
        pos = self._ratio_positions()
        ratio = self._ratio_value()
        self._ratio_ref = (None if pos is None or not ratio else
                           {'input': pos[0], 'output': pos[1], 'ratio': ratio})
        self._ratio_rate_mark = None
        self.ratio_error_rad = math.nan
        self.ratio_slip_rad_s = math.nan

    def _update_ratio_error(self):
        """Refresh the two ratio channels. Runs every cycle, before the safety
        checks read them.

        NaN when there is no reference or a position is unreadable. Whether that
        stops a test is the config's call, through `nan_trips` on the entry --
        this does not decide it, the same way the window's own NaN handling is
        declared rather than assumed."""
        ref = self._ratio_ref
        pos = self._ratio_positions() if ref is not None else None
        if pos is None:
            self.ratio_error_rad = math.nan
            self.ratio_slip_rad_s = math.nan
            return
        err = ((pos[1] - ref['output'])
               - (pos[0] - ref['input']) / ref['ratio'])
        self.ratio_error_rad = err
        now = time.perf_counter()
        mark = self._ratio_rate_mark
        if mark is None:
            self._ratio_rate_mark = (now, err)
            self.ratio_slip_rad_s = 0.0
        elif now - mark[0] >= self._RATIO_RATE_WINDOW_S:
            self.ratio_slip_rad_s = (err - mark[1]) / (now - mark[0])
            self._ratio_rate_mark = (now, err)

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
            'jog_drive': None if self._jog is None else self._jog['legs'][0]['drive']
                         if self._jog['legs'] else None,
            'input_jog': w['input'] is not None,
            'output_jog': w['output'] is not None,
            'touch_off_drive': w['touch_off_drive'],
            'input_sign': self._input_sign,
            'input_ratio': self._input_ratio,
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
        self._latch_ratio_reference()
        self._window_say(f'Centre declared at {pos:.4f} rad. Run touch-off before testing.')

    def _jog_refusal(self, drive='output'):
        if self._window is None:
            return 'no position_window in config'
        if self._window[drive] is None:
            return f'no position_window.touch_off.{drive} in config'
        if drive == 'input' and self._load_only:
            return 'the rig is in load-only mode'
        if self.shutdown:
            return 'the rig is shutting down'
        if self._test_active:
            return 'a test is running'
        if self._post_test is not None:
            return self._post_test_refusal()
        if self._jog is not None:
            return 'a jog or touch-off is already running'
        if self._tare_state is not None:
            return 'a tare is running'
        if self._window_centre is None:
            return 'declare centre first'
        if self.devices.LOAD.fault or (not self._load_only and self.devices.DUT.fault):
            return 'a drive is faulted'
        return None

    def _leg(self, direction, expect, stop_at=None, drive='output'):
        """One constant-direction move of one shaft. `direction` is the sign of
        the command to that shaft's drive. `expect` is 'contact' (ends on
        contact), 'target' (ends when rel crosses stop_at) or 'manual'."""
        w = self._window
        span = 2 * w['max_excursion']
        if drive == 'input':
            i = w['input']
            timeout_s = 2 * span * i['nominal_ratio'] / max(i['velocity'], 1e-3) + 10.0
        else:
            timeout_s = 2 * span / max(w['output']['velocity'], 1e-3) + 10.0
        return {'direction': direction, 'expect': expect, 'stop_at': stop_at,
                'drive': drive, 'start_rel': None, 't0': None, 'over': 0,
                'timeout_s': timeout_s,
                # input drive only
                'start_input': None, 'baselines': None, 'last_t': None,
                'drag_samples': 0, 'contact_torque': None, 'speed_cmd': 0.0,
                'cmd_t': None}

    def _leg_out_dir(self, leg):
        """Direction the output turns on this leg (+1/-1), or None while an
        input move's direction is not yet known."""
        if leg['drive'] == 'output':
            return leg['direction']
        return None if self._input_sign is None else leg['direction'] * self._input_sign

    def _start_jog(self, direction, drive='output'):
        drive = 'input' if drive == 'input' else 'output'
        refusal = self._jog_refusal(drive)
        direction = 1 if float(direction) > 0 else -1
        leg = self._leg(direction, 'manual', drive=drive) if refusal is None else None
        if refusal is None:
            out_dir = self._leg_out_dir(leg)
            if out_dir is not None and out_dir * self._window_rel() >= self._window['max_excursion']:
                refusal = 'already at max excursion in that direction'
        name = 'input' if drive == 'input' else 'output'
        if refusal:
            return self._window_say(f'Jog {name} refused: {refusal}')
        self._jog = {'kind': 'jog', 'legs': [leg], 'alive': time.perf_counter()}
        self._window_say(f'Jogging {name} {"+" if direction > 0 else "-"}')

    def _start_touch_off(self):
        drive = self._window['touch_off_drive'] if self._window else 'output'
        refusal = self._jog_refusal(drive)
        rel = self._window_rel()
        if refusal is None and (math.isnan(rel) or abs(rel) > self._window['half_window']):
            refusal = 'output is outside the window; jog back toward centre first'
        if refusal:
            return self._window_say(f'Touch-off refused: {refusal}')
        self._touch_off = None
        # The return leg is built once both bumpers are found: on the input
        # shaft, which command direction heads home is only known by then.
        self._jog = {'kind': 'touch_off', 'drive': drive,
                     'legs': [self._leg(+1, 'contact', drive=drive),
                              self._leg(-1, 'contact', drive=drive)],
                     'contacts': [], 'problems': [], 'contact_torques': [],
                     # Contacts are relative to the centre the run started
                     # from; `shift` is how far recentring moved it (None if not).
                     'run_centre': self._window_centre, 'shift': None}
        home = 'the bumper midpoint' if self._window['recentre'] else 'centre'
        self._window_say(f'Touch-off on the {drive} shaft running: one bumper, '
                         f'the other, then to {home}')

    def _start_return_centre(self):
        """Drive the input shaft back to centre. Click-to-run: there is no
        keepalive, so it ends on the target, on contact, on any of the jog
        checks, or on Stop."""
        refusal = self._jog_refusal('input')
        rel = self._window_rel()
        if refusal is None and math.isnan(rel):
            refusal = f'{self._window["source"]} is not readable'
        if refusal is None and abs(rel) <= self.RETURN_CENTRE_DEADBAND_RAD:
            refusal = f'already at centre ({rel:+.4f} rad)'
        if refusal:
            return self._window_say(f'Return to centre refused: {refusal}')
        leg = self._home_leg(rel, 'input')
        if leg is None:
            # No input move yet this session, so which way the output follows
            # input + is unmeasured. Set off on the assumption it follows +;
            # _jog_step turns the leg around once the first 0.05 rad of travel
            # settles it. Both directions are held to max_excursion and the
            # contact checks meanwhile, so the guess costs travel, not safety.
            leg = self._leg(-1 if rel > 0 else 1, 'target', stop_at=0.0,
                            drive='input')
            leg['probe'] = True
        self._jog = {'kind': 'return_centre', 'legs': [leg]}
        self._window_say(f'Returning to centre on the input shaft from '
                         f'{rel:+.4f} rad')

    def _home_leg(self, rel, drive):
        """A leg back to centre from rel, or None if the direction is unknown."""
        if drive == 'output':
            return self._leg(-1 if rel > 0 else 1, 'target', stop_at=0.0)
        if self._input_sign is None:
            return None
        return self._leg((-1 if rel > 0 else 1) * self._input_sign, 'target',
                         stop_at=0.0, drive='input')

    def _jog_step(self):
        """Advance the jog/touch-off one cycle and return the command to send."""
        j, w = self._jog, self._window
        load, dut = self.devices.LOAD, self.devices.DUT
        leg = j['legs'][0]
        drive = leg['drive']
        moving, free = (dut, load) if drive == 'input' else (load, dut)
        hold = {'input_mode': 'velocity' if drive == 'input' else 'torque',
                'output_mode': 'torque' if drive == 'input' else 'velocity',
                'input_command': 0, 'output_command': 0}

        if self.shutdown:
            return self._end_jog('Jog ended: shutdown', failed=True)
        if load.fault or (not self._load_only and dut.fault):
            return self._end_jog('Jog ended: a drive faulted', failed=True)
        # Dead-man for hold-to-run. The button's release is not guaranteed to
        # reach us: Qt drops `released` when the mouse is let go off the button
        # without a move event first, and a hung GUI sends nothing at all.
        if (j['kind'] == 'jog'
                and time.perf_counter() - j['alive'] > self.JOG_KEEPALIVE_TIMEOUT_S):
            return self._end_jog(f'Jog stopped: no keepalive from the GUI for '
                                 f'{self.JOG_KEEPALIVE_TIMEOUT_S:g} s (button '
                                 f'release lost?)', failed=True)

        # The other shaft is left free throughout.
        free.sw_enable = False
        if free.mode != 'torque' and not free.switching_modes:
            free.command_operating_mode('torque')
        if moving.mode != 'velocity' or moving.switching_modes:
            if not moving.switching_modes:
                moving.command_operating_mode('velocity')
            moving.sw_enable = False
            return hold
        moving.sw_enable = True

        rel = self._window_rel()
        if math.isnan(rel):
            return self._end_jog(f'Jog ended: {w["source"]} read NaN', failed=True)

        now = time.perf_counter()
        if leg['start_rel'] is None:
            leg['start_rel'], leg['t0'] = rel, now
            if drive == 'input':
                leg['start_input'] = float(dut.position)
                leg['baselines'] = [w['input']['drag_nm']] * len(w['input']['torque_sources'])
        d = leg['direction']

        if drive == 'input':
            problem = self._learn_input_direction(leg, rel)
            if problem:
                return self._end_jog(f'Jog ended: {problem}', failed=True)
        out_dir = self._leg_out_dir(leg)

        if leg.get('probe') and out_dir is not None:
            # A return to centre that started before the output direction was
            # known: now that it is, turn around if the guess was backwards.
            # The replacement leg re-initialises, so the speed slew and the drag
            # baselines start again from the reversal.
            leg['probe'] = False
            if out_dir * rel > 0:
                j['legs'][0] = self._home_leg(rel, 'input')
                self._window_say('Return to centre: the output turns the other '
                                 'way, reversing')
                return hold

        if out_dir is None:
            # Direction unknown: stop past max excursion as soon as the output
            # is seen moving further out.
            past = (abs(rel) >= w['max_excursion']
                    and abs(rel) > abs(leg['start_rel']) + 0.005)
        else:
            past = out_dir * rel >= w['max_excursion']
        if past:
            return self._leg_done('excursion', rel)
        if (leg['stop_at'] is not None and out_dir is not None
                and out_dir * (rel - leg['stop_at']) >= 0):
            return self._leg_done('target', rel)

        if drive == 'input':
            i = w['input']
            fastest = max(i['approach_velocity'], i['velocity'])
            if abs(dut.velocity) > 2 * fastest + 1.0:
                return self._end_jog(f'Jog ended: input overspeed '
                                     f'({dut.velocity:+.2f} rad/s)', failed=True)
        elif abs(load.velocity) > 2 * max(w['output']['approach_velocity'],
                                          w['output']['velocity']) + 0.1:
            return self._end_jog(f'Jog ended: output overspeed '
                                 f'({load.velocity:+.3f} rad/s)', failed=True)
        if now - leg['t0'] > leg['timeout_s']:
            return self._end_jog('Jog ended: timed out', failed=True)

        # Unknown direction counts as outward: slow speed, full contact checks.
        outward = out_dir is None or out_dir * rel > 0
        if drive == 'input':
            result = self._input_contact(leg, outward, now)
            if result is not None:
                return result
        else:
            # Torque cap. Waived only while backing away from a bumper (moving
            # toward centre, within arm_distance of where the move began), when
            # the bumper is still unloading. Moving outward it always applies.
            o = w['output']
            if outward or abs(rel - leg['start_rel']) >= o['arm_distance']:
                torque = max(abs(float(g(self))) for g in o['torque_get'])
                if math.isnan(torque):
                    return self._end_jog('Jog ended: torque reading is NaN', failed=True)
                leg['over'] = leg['over'] + 1 if torque > o['torque_nm'] else 0
                if leg['over'] >= o['trip_samples']:
                    return self._leg_done('contact', rel)

        slow = outward and abs(rel) > w['slow_from']
        if drive == 'input':
            i = w['input']
            target = d * (i['velocity'] if slow else i['approach_velocity'])
            # Cycle time from the previous command, clamped so a stalled loop
            # cannot hand the slew one large step.
            dt = 0.001 if leg['cmd_t'] is None else min(max(now - leg['cmd_t'], 0.0), 0.01)
            leg['cmd_t'] = now
            step = i['acceleration'] * dt
            leg['speed_cmd'] += max(-step, min(step, target - leg['speed_cmd']))
            return dict(hold, input_command=leg['speed_cmd'])
        speed = w['output']['velocity'] if slow else w['output']['approach_velocity']
        return dict(hold, output_command=d * speed)

    def _learn_input_direction(self, leg, rel):
        """Once an input move has turned the output far enough to measure, learn
        (or confirm) which way the output follows the input and at what ratio.
        Returns a reason to stop, or None."""
        i = self._window['input']
        d_rel = rel - leg['start_rel']
        # Wait for enough travel that backlash and bumper unloading at the
        # start of the move are a small part of it.
        if abs(d_rel) < 0.05:
            return None
        d_in = float(self.devices.DUT.position) - leg['start_input']
        sign = 1 if d_rel * leg['direction'] > 0 else -1
        if self._input_sign is not None and sign != self._input_sign:
            return (f'output turned the opposite way to earlier input moves '
                    f'({d_rel:+.3f} rad for input {d_in:+.2f} rad)')
        ratio = d_in / d_rel
        if abs(abs(ratio) - i['nominal_ratio']) > i['ratio_tolerance'] * i['nominal_ratio']:
            return (f'input/output ratio {ratio:+.1f}, expected '
                    f'{i["nominal_ratio"]:g} +/- {100 * i["ratio_tolerance"]:.0f}% '
                    f'-- coupling slipping or encoder wrong?')
        self._input_sign = sign
        self._input_ratio = ratio
        return None

    def _input_contact(self, leg, outward, now):
        """Contact check for an input-shaft move. Contact is torque above the
        drag baseline by contact_rise_nm, or above ceiling_nm outright, on any
        source for trip_samples cycles. The baseline starts at drag_nm and
        tracks the measured drag while the shaft is moving freely. The rise
        check waits arm_input_rad into each move (breakaway at the start reads
        as a rise); the ceiling never waits. Backing away from a bumper, the
        bumper unloading helps the move, so only the ceiling applies until
        arm_input_rad clears it."""
        i, dut = self._window['input'], self.devices.DUT
        dt = 0.0 if leg['last_t'] is None else now - leg['last_t']
        leg['last_t'] = now
        armed = abs(float(dut.position) - leg['start_input']) >= i['arm_input_rad']
        # Free running: armed, and moving at least half the slow speed.
        free = armed and abs(dut.velocity) >= 0.5 * i['velocity']
        alpha = min(1.0, dt / i['drag_tau_s']) if i['drag_tau_s'] > 0 else 1.0

        hit = None
        for k, (path, get, scale) in enumerate(i['torque_sources']):
            torque = abs(float(get(self)) * scale)
            if math.isnan(torque):
                return self._end_jog(f'Jog ended: {path} read NaN', failed=True)
            base = leg['baselines'][k]
            limit = i['ceiling_nm'] if not armed else min(i['ceiling_nm'],
                                                          base + i['contact_rise_nm'])
            if torque > limit and hit is None:
                hit = (path, torque, base)
            elif free and torque < base + 0.5 * i['contact_rise_nm']:
                leg['baselines'][k] = min(base + alpha * (torque - base),
                                          i['ceiling_nm'] - i['contact_rise_nm'])
        if free:
            leg['drag_samples'] += 1

        leg['over'] = leg['over'] + 1 if hit else 0
        if leg['over'] >= i['trip_samples']:
            path, torque, base = hit
            leg['contact_torque'] = {'source': path, 'torque_nm': torque,
                                     'drag_nm': base}
            return self._leg_done('contact', self._window_rel())
        return None

    def _input_drag_text(self, leg):
        if leg['drive'] != 'input' or not leg['drag_samples'] or not leg['baselines']:
            return ''
        names = [p.split('.')[-1] for p, _, _ in self._window['input']['torque_sources']]
        drag = ', '.join(f'{n} {b:.2f}' for n, b in zip(names, leg['baselines']))
        ratio = ('' if self._input_ratio is None
                 else f', ratio {self._input_ratio:+.1f}')
        return f' [drag Nm: {drag}{ratio}]'

    def _leg_done(self, reason, rel):
        j, w = self._jog, self._window
        leg = j['legs'].pop(0)

        if j['kind'] == 'jog':
            if reason == 'contact' and leg['drive'] == 'input':
                c = leg['contact_torque']
                text = (f'contact ({c["source"].split(".")[-1]} {c["torque_nm"]:.2f} Nm '
                        f'over drag {c["drag_nm"]:.2f}) at {rel:+.4f} rad')
            elif reason == 'contact':
                text = f'contact ({w["output"]["torque_nm"]:g} Nm) at {rel:+.4f} rad'
            else:
                text = f'max excursion reached at {rel:+.4f} rad'
            return self._end_jog(f'Jog stopped: {text}{self._input_drag_text(leg)}')

        if j['kind'] == 'return_centre':
            drag = self._input_drag_text(leg)
            if reason == 'target':
                return self._end_jog(f'Back at centre ({rel:+.4f} rad){drag}')
            if reason == 'contact':
                c = leg['contact_torque']
                text = (f'contact ({c["source"].split(".")[-1]} {c["torque_nm"]:.2f} Nm '
                        f'over drag {c["drag_nm"]:.2f})')
            else:
                text = 'max excursion'
            return self._end_jog(f'Return to centre stopped by {text} at '
                                 f'{rel:+.4f} rad; jog back manually{drag}',
                                 failed=True)

        side = '+' if leg['direction'] > 0 else '-'
        if leg['expect'] == 'contact':
            if reason == 'contact':
                j['contacts'].append(rel)
                j['contact_torques'].append(leg['contact_torque'])
            else:
                j['problems'].append(f'no bumper contact moving {j["drive"]} {side} before '
                                     f'max excursion ({rel:+.4f} rad)')
                # Skip any remaining bumper and come home.
                j['legs'] = []
        elif reason != 'target':
            j['problems'].append(f'return to centre stopped by {reason} at '
                                 f'{rel:+.4f} rad; jog back manually')
            j['legs'] = []
            return self._finish_touch_off()

        if not j['legs'] and leg['expect'] == 'contact':
            rel -= self._recentre(j)
            home = self._home_leg(rel, j['drive'])
            if home is None:
                j['problems'].append('output direction for an input move is unknown; '
                                     'jog back to centre manually')
            else:
                j['legs'] = [home]

        if not j['legs']:
            return self._finish_touch_off()
        return {'input_mode': 'torque', 'output_mode': 'torque',
                'input_command': 0, 'output_command': 0}

    @staticmethod
    def _bumpers(contacts):
        """(plus, minus) from the contact positions, by which side of centre
        they are on -- not by which command found them: on the input shaft that
        mapping is measured. Either is None if missing."""
        return (max([c for c in contacts if c > 0], default=None),
                min([c for c in contacts if c < 0], default=None))

    def _recentre(self, j):
        """With both bumpers found, move centre to their midpoint (config
        touch_off.recentre). Returns the shift applied, 0.0 if none. Skipped
        when the span is out of tolerance (the result reports that) or the shift
        is over max_recentre_rad: a centre that far out means a bad contact or
        a bad declaration, and the window must not follow it."""
        w = self._window
        plus, minus = self._bumpers(j['contacts'])
        if not w['recentre'] or plus is None or minus is None:
            return 0.0
        mid, half = (plus + minus) / 2, (plus - minus) / 2
        if abs(half - w['expected_half_span']) > w['tolerance']:
            return 0.0
        if abs(mid) > w['max_recentre']:
            j['problems'].append(f'not recentred: bumper midpoint {mid:+.4f} rad from '
                                 f'centre, over max_recentre_rad {w["max_recentre"]:g}')
            return 0.0
        self._window_centre = j['run_centre'] + mid
        j['shift'] = mid
        return mid

    def _finish_touch_off(self, abort_reason=None):
        j, w = self._jog, self._window
        contacts = j['contacts']
        raw_plus, raw_minus = self._bumpers(contacts)
        both = raw_plus is not None and raw_minus is not None
        # Reported relative to the centre in force after this run: the new one
        # if recentred (so the check below is the half-span, symmetric), else
        # the one the run started from.
        shift = j['shift'] or 0.0
        plus = None if raw_plus is None else raw_plus - shift
        minus = None if raw_minus is None else raw_minus - shift
        problems = list(j['problems']) + ([abort_reason] if abort_reason else [])
        if len(contacts) == 2 and not both:
            problems.append(f'both contacts on the same side of centre '
                            f'({contacts[0]:+.4f}, {contacts[1]:+.4f} rad)')
        passed = both and not abort_reason
        if both:
            for name, value in (('+', plus), ('-', -minus)):
                if abs(value - w['expected_half_span']) > w['tolerance']:
                    passed = False
                    problems.append(f'{name} bumper at {name}{value:.4f} rad, expected '
                                    f'{w["expected_half_span"]:.4f} +/- {w["tolerance"]:.4f}')
            if w['recentre'] and j['shift'] is None:
                passed = False  # _recentre refused; its reason is in problems
        result = {
            'at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'drive': j['drive'],
            'centre': self._window_centre,
            'run_centre': j['run_centre'],
            'centre_shift_rad': j['shift'],
            'plus_rad': plus,
            'minus_rad': minus,
            # Where the bumpers put centre, relative to the centre this run
            # started from. On a repeat after recentring this is the
            # repeatability of the sensed centre.
            'centre_error_rad': (raw_plus + raw_minus) / 2 if both else None,
            'span_rad': raw_plus - raw_minus if both else None,
            # Encoder frame (LOAD position), comparable across runs whatever
            # the centre was: the repeatability numbers come from these.
            'plus_abs_rad': None if raw_plus is None else j['run_centre'] + raw_plus,
            'minus_abs_rad': None if raw_minus is None else j['run_centre'] + raw_minus,
            'midpoint_abs_rad': j['run_centre'] + (raw_plus + raw_minus) / 2 if both else None,
            'torque_nm': w['output']['torque_nm'] if j['drive'] == 'output' else None,
            'contact_torques': j['contact_torques'] if j['drive'] == 'input' else None,
            'input_ratio': self._input_ratio if j['drive'] == 'input' else None,
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
        detail = (f'+{plus:.4f} / {minus:+.4f} rad, midpoint '
                  f'{result["centre_error_rad"]:+.4f} rad from run centre' if both
                  else 'incomplete')
        if j['shift'] is not None:
            detail += f', centre moved {j["shift"]:+.4f} rad to {self._window_centre:.4f}'
        if j['drive'] == 'input' and j['contact_torques']:
            detail += ' (contact at ' + ', '.join(
                f'{c["torque_nm"]:.2f} Nm over {c["drag_nm"]:.2f} drag'
                for c in j['contact_torques']) + ')'
        return self._end_jog(f'Touch-off {verdict}: {detail}'
                             + (f' ({"; ".join(problems)})' if problems else ''),
                             finished=True)

    def _end_jog(self, message, failed=False, finished=False):
        """Stop both drives and clear the jog. Returns the idle command for this cycle."""
        j = self._jog
        if j is not None and j['kind'] == 'touch_off' and not finished:
            # An interrupted touch-off must not leave an earlier pass standing.
            self._finish_touch_off(abort_reason=message)
            return self._safe_default_command
        self._jog = None
        for drive in (self.devices.LOAD, self.devices.DUT):
            drive.sw_enable = False
            if drive.mode != 'torque' or drive.switching_modes:
                drive.command_operating_mode('torque')
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
        # How long this method takes, for the `step_us` log key. The 2026-09-17
        # Archimedes runs show cycle_time_us reaching 2.6-3.1 ms with wkc_error
        # flat at 0 -- so frames arrived and the bus was clean, and the late
        # cycles were spent on THIS side of the wire. Every 20 A current slam in
        # those logs sits within a few cycles of one. Splitting step() out of
        # cycle_time_us says whether the time goes here (command generation,
        # telemetry, safeties) or in the master loop around it, which is the one
        # thing those logs cannot answer. Costs two clock reads a cycle.
        step_start = time.perf_counter()
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

        # Before _safety_trigger, which reads the channels it publishes.
        self._update_ratio_error()

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

        elif self._post_test is not None:
            # A run winding down: brake, then hold the log open for the tail.
            # Ahead of the jog and idle branches because both would force the
            # drives off, and the brake needs the input drive energised.
            self.current_cmd = self._post_test_step()

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
            dut_command = self._dut_command_for_mode(self.current_cmd)
            if self.devices.LOAD.mode == 'torque' and not self.devices.DUT.mode == 'torque':
                torque_ff = ff_ratio*self.current_cmd['output_command'] / self.devices.DUT.params['gear_ratio']
                self.devices.DUT.send_command(dut_command, torque_ff)
            else:
                self.devices.DUT.send_command(dut_command)

        if self.current_cmd['input_command'] != getattr(self, '_last_dut_cmd', None):
            self._last_dut_cmd = self.current_cmd['input_command']

        for aux_func in self._aux_funcs:
            aux_func()

        # Set before _send_telemetry so the sample carries this cycle's own
        # figure. It excludes _send_telemetry and _cmd_check themselves, which
        # is deliberate: the queue put is timed separately below.
        self.step_us = (time.perf_counter() - step_start) * 1e6

        telemetry_start = time.perf_counter()
        self._send_telemetry()
        self._cmd_check()
        self.telemetry_us = (time.perf_counter() - telemetry_start) * 1e6
        # There was a time.sleep(0) here "to momentarily yield the GIL". It is
        # removed deliberately. step() is called from Master._processdata_loop
        # between process_txpdo and write_rxpdo, so a yield on this line hands
        # the GIL to the telemetry feeder thread in the one place it hurts
        # most: immediately before the outbound setpoint is marshalled into the
        # frame. Reacquiring it costs up to one sys.setswitchinterval, and it
        # lands in the region the log calls "unaccounted" (median 1.5 ms, max
        # 5.45 ms against a 5 ms default switch interval).
        #
        # It was also redundant. hybrid_sleep_until spends ~700 us of every
        # cycle inside clock_nanosleep with the GIL released, which is a far
        # better window for the feeder than a yield mid-cycle, and far more
        # than the ~5 us it needs to pickle one sample.

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