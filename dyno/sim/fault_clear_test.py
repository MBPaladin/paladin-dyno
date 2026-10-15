"""Validation of the drive fault reset: the controller's clear_faults command
against the fake bus, end to end.

Two cases, because they are the two the operator has to be able to tell apart:
a fault whose cause has gone (the reset works) and one whose cause is still
present (the reset drops the latch and the drive re-faults immediately). The
sim injects both via DYNO_SIM_FAULT -- see dyno/sim/fake_pysoem._fault_injection.

Run from repo root:  PYTHONPATH=. .venv/bin/python dyno/sim/fault_clear_test.py [mode]
(mode defaults to actuator_production)
"""
import multiprocessing
import os
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else 'actuator_production'
os.environ['DYNO_SIM'] = MODE

ok = True


def fail(msg):
    global ok
    ok = False
    print(f'FAIL: {msg}')


def run_case(title, fault_spec, expect_cleared):
    """Bring the rig up with a scheduled fault, hit Clear Drive Faults, and
    report what the GUI would have shown."""
    global ok
    print(f'\n--- {title} (DYNO_SIM_FAULT={fault_spec}) ---')
    os.environ['DYNO_SIM_FAULT'] = fault_spec

    # Imported here, after DYNO_SIM is set, and re-read per case because the
    # child inherits the environment at fork time.
    from dyno.src.dyno_controller import Controller

    telemetry_q = multiprocessing.Queue()
    command_q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=Controller,
                                   args=[telemetry_q, command_q, MODE], name='ctrl')
    proc.start()

    def drain(seconds, until=None):
        """Collect control-state dicts (telemetry slot -1), stopping early once
        `until(state)` holds."""
        states, deadline = [], time.time() + seconds
        while time.time() < deadline:
            try:
                sample = telemetry_q.get(timeout=0.5)
            except Exception:
                if not proc.is_alive():
                    fail('controller process died')
                    return states
                continue
            if isinstance(sample, list) and isinstance(sample[-1], dict):
                states.append(sample[-1])
                if until is not None and until(sample[-1]):
                    break
        return states

    try:
        print('  waiting for bring-up and the injected fault...')
        states = drain(20.0, until=lambda s: s.get('faulted_drives'))
        if not states:
            fail('no telemetry from the controller')
            return
        faulted = states[-1].get('faulted_drives')
        print(f'  faulted_drives: {faulted}')
        if not faulted:
            fail('the injected fault never reached the controller')
            return
        if not states[-1].get('fault'):
            fail("control state's 'fault' flag did not follow the faulted drive")

        print('  sending clear_faults...')
        command_q.put_nowait(['clear_faults', 0])
        states = drain(6.0, until=lambda s: (
            not s.get('fault_clear_active')
            and (s.get('fault_clear_message') or '').startswith(('Faults cleared',
                                                                 faulted[0]))))
        if not states:
            fail('no telemetry after clear_faults')
            return

        saw_active = any(s.get('fault_clear_active') for s in states)
        final = states[-1]
        print(f'  saw fault_clear_active during the window: {saw_active}')
        print(f'  final message: {final.get("fault_clear_message")}')
        print(f'  final faulted_drives: {final.get("faulted_drives")}')

        if not saw_active:
            fail('fault_clear_active was never reported, so the GUI could not '
                 'have shown the reset in flight')

        cleared = not final.get('faulted_drives')
        if cleared != expect_cleared:
            fail(f'expected cleared={expect_cleared}, got {cleared} '
                 f'(message {final.get("fault_clear_message")!r})')
        elif expect_cleared:
            if 'cleared' not in (final.get('fault_clear_message') or '').lower():
                fail(f'cleared, but the message does not say so: '
                     f'{final.get("fault_clear_message")!r}')
        else:
            if 'still in fault' not in (final.get('fault_clear_message') or ''):
                fail(f'still faulted, but the message does not say so: '
                     f'{final.get("fault_clear_message")!r}')

        # A reset with nothing left to clear must say so rather than imply it
        # did something.
        if cleared:
            print('  sending clear_faults again, with nothing faulted...')
            command_q.put_nowait(['clear_faults', 0])
            states = drain(4.0, until=lambda s: (s.get('fault_clear_message') or '')
                           .startswith('No drive faults'))
            message = states[-1].get('fault_clear_message') if states else None
            print(f'  message: {message}')
            if message != 'No drive faults to clear':
                fail(f'a no-op reset reported {message!r}')
    finally:
        command_q.put_nowait(['shutdown', 0])
        proc.join(timeout=15)
        if proc.is_alive():
            proc.terminate()


# The drive is faulted a few seconds in, well after bring-up: a fault present
# during bring-up is a different (and separately handled) situation.
run_case('1. Fault whose cause has gone', 'LOAD@4', expect_cleared=True)
run_case('2. Fault whose cause is still present', 'LOAD@4:sticky',
         expect_cleared=False)

print('\nFAULT CLEAR TEST', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
