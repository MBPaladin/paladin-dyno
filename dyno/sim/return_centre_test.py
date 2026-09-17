"""Return to Centre (GUI button -> controller `return_centre`), on the fake bus.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/return_centre_test.py [mode]

Narrow and separate from archimedes_window_test.py for the same reason
post_test_brake_test.py is: that test's touch-off steps are mid-rework, and
their failures move `centre` somewhere later steps cannot reach, which would
hide whether this worked. Nothing here touches off -- it declares centre, jogs
the OUTPUT away from it, and asks the input shaft to bring it back.

The two paths that matter are the two the button meets on the bench:
  1. Nothing has driven the input yet, so which way the output follows input +
     is unmeasured. The move sets off on a guess and turns itself around once
     the first 0.05 rad of travel settles it.
  2. The direction is known (any earlier input jog or touch-off measured it),
     so the move heads home directly.
"""
import json
import multiprocessing
import os
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else 'inhouse_archimedes'
os.environ['DYNO_SIM'] = MODE
os.environ['DYNO_SIM_PARAMS'] = json.dumps({'hardware_torque_sign': True})

from dyno.src.dyno_controller import Controller  # noqa: E402

ok = True


def fail(msg):
    global ok
    ok = False
    print(f'FAIL: {msg}')


def check(cond, msg):
    if not cond:
        fail(msg)
    return cond


class Rig:
    def __init__(self):
        self.tq = multiprocessing.Queue()
        self.cq = multiprocessing.Queue()
        self.proc = multiprocessing.Process(target=Controller,
                                            args=[self.tq, self.cq, MODE])
        self.proc.start()
        self.state = {}
        self.holding = False      # a jog button is held: keep the dead-man fed
        self._alive = 0.0

    def send(self, *cmd):
        if cmd[0] == 'jog':
            self.holding = True
        elif cmd[0] == 'jog_stop':
            self.holding = False
        self.cq.put_nowait(list(cmd) if len(cmd) > 1 else [cmd[0], 0])

    def pump(self, seconds, until=None):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self.holding and time.time() - self._alive > 0.1:
                self.cq.put_nowait(['jog_alive', 0])
                self._alive = time.time()
            try:
                s = self.tq.get(timeout=0.5)
            except Exception:
                if not self.proc.is_alive():
                    fail('controller process died')
                    return False
                continue
            self.state = s[-1]
            if until is not None and until(self):
                return True
        return False

    @property
    def w(self):
        return self.state.get('window') or {}

    def jog_output_to(self, rel, timeout=20):
        """Hold the output jog until rel passes the target, then release."""
        direction = 1 if rel > (self.w.get('rel') or 0.0) else -1
        reached = (lambda r: (r.w.get('rel') or -9) > rel) if direction > 0 else \
                  (lambda r: (r.w.get('rel') or 9) < rel)
        self.send('jog', direction)
        got = self.pump(timeout, until=reached)
        self.send('jog_stop')
        self.pump(3, until=lambda r: r.w.get('jog') is None)
        return got

    def close(self):
        self.send('shutdown')
        self.proc.join(timeout=15)
        if self.proc.is_alive():
            self.proc.terminate()


print(f'--- return to centre ({MODE}) ---')
rig = Rig()
try:
    print('  bring-up...')
    rig.pump(6)

    rig.send('return_centre')
    rig.pump(1.5)
    print(f'  1 no centre declared: {rig.w.get("message")}')
    check('declare centre first' in (rig.w.get('message') or ''),
          'return to centre was not refused without a centre')

    rig.send('declare_centre')
    if not rig.pump(3, until=lambda r: r.w.get('centre') is not None):
        fail('centre was never declared')
    print(f'  2 centre at {rig.w.get("centre"):+.4f} rad')

    rig.send('return_centre')
    rig.pump(1.5)
    print(f'  3 standing on centre: {rig.w.get("message")}')
    check('already at centre' in (rig.w.get('message') or ''),
          'return to centre was not refused inside the deadband')

    # 4-5: the unmeasured-direction path. The OUTPUT jog is what moves the rig
    # off centre here, precisely so nothing has driven the input yet.
    check(rig.jog_output_to(+0.4), 'output jog never reached +0.4 rad')
    print(f'  4 output-jogged to rel {rig.w.get("rel"):+.4f}, '
          f'input_sign={rig.w.get("input_sign")}')
    check(rig.w.get('input_sign') is None,
          'the input direction is already measured: step 5 no longer covers '
          'the guess')

    t0 = time.time()
    rig.send('return_centre')
    check(rig.pump(3, until=lambda r: r.w.get('jog') == 'return_centre'),
          'return to centre never started')
    check(rig.w.get('jog_drive') == 'input',
          f'return to centre ran on the {rig.w.get("jog_drive")} shaft, not input')
    done = rig.pump(90, until=lambda r: r.w.get('jog') is None)
    print(f'  5 returned in {time.time() - t0:.1f} s to rel {rig.w.get("rel"):+.4f}, '
          f'sign {rig.w.get("input_sign")}, ratio {rig.w.get("input_ratio"):+.1f}')
    print(f'    {rig.w.get("message")}')
    check(done and abs(rig.w.get('rel') or 9) < 0.05,
          'return to centre did not bring the output back to centre')
    check('Back at centre' in (rig.w.get('message') or ''),
          f'unexpected end message: {rig.w.get("message")}')

    # 6-7: the same move the other way with the direction now known, so the
    # leg heads home directly instead of probing.
    check(rig.jog_output_to(-0.5), 'output jog never reached -0.5 rad')
    print(f'  6 output-jogged to rel {rig.w.get("rel"):+.4f}')
    t0 = time.time()
    rig.send('return_centre')
    rig.pump(3, until=lambda r: r.w.get('jog') == 'return_centre')
    done = rig.pump(90, until=lambda r: r.w.get('jog') is None)
    print(f'  7 returned in {time.time() - t0:.1f} s to rel {rig.w.get("rel"):+.4f}')
    print(f'    {rig.w.get("message")}')
    check(done and abs(rig.w.get('rel') or 9) < 0.05,
          'return to centre with a known direction did not reach centre')

    # 8: Stop ends it mid-move. There is no keepalive on this one, so Stop is
    # the operator's only way out of it.
    check(rig.jog_output_to(+0.8), 'output jog never reached +0.8 rad')
    rig.send('return_centre')
    check(rig.pump(3, until=lambda r: r.w.get('jog') == 'return_centre'),
          'return to centre never started before the Stop')
    rig.pump(1.0)
    rig.send('stop_test')
    stopped = rig.pump(5, until=lambda r: r.w.get('jog') is None)
    print(f'  8 Stop mid-return at rel {rig.w.get("rel"):+.4f}: {rig.w.get("message")}')
    check(stopped, 'Stop did not end the return')
    check(abs(rig.w.get('rel') or 0) > 0.05,
          'the Stop landed after the move had already finished; it proved nothing')
finally:
    rig.close()

print('\nRETURN TO CENTRE', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
