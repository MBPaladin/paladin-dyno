"""
Generate a torque-position sweep trace in the style of cogging_trace_backward.csv.

For each output-motor torque setpoint in OUTPUT_TORQUES, the input motor sweeps
its position from 0 -> +AMPLITUDE -> 0 (a forward then backward revolution sweep),
holding the torque constant during the position sweep. Between torque levels the
position is parked at 0 while the output torque ramps to the next level.

Columns (consumed by TestTrace in test_manager.py via np.interp over `time`):
    time                  [s]    monotonically increasing, starts at 0
    output_motor_torque   [Nm]   starts at 0 and ends at 0
    input_motor_position  [rad]  starts at 0, ends at 0 (CSV unit is radians)

All values are written as floats. AMPLITUDE = REVOLUTIONS * 2*pi is irrational,
so integer columns would be incorrect here.

Pair this CSV with a test_trace behavior whose settings are:
    input_motor:  { control_mode: position }
    output_motor: { control_mode: torque   }
    trace_file:   "<OUTPUT_NAME>"
"""

import csv
import math
import os

# --- Tunables --------------------------------------------------------------
OUTPUT_TORQUES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]  # [Nm] output-motor torque levels
REVOLUTIONS = 30.0          # input-motor position sweep amplitude, in revolutions
POSITION_SPEED = 0.5       # [rad/s] target input-motor sweep speed
TORQUE_ROTATUM = 1.0        # [Nm/s] rate for changing the output torque between levels
PEAK_DWELL_S = 1.0          # [s] hold at peak position (mirrors the reference's doubled row)
BOTTOM_DWELL_S = 1.0        # [s] optional hold at position 0 before each torque change

OUTPUT_NAME = "torque_position_sweep.csv"
# ---------------------------------------------------------------------------

AMPLITUDE = REVOLUTIONS * 2.0 * math.pi   # [rad]
SWEEP_TIME = AMPLITUDE / POSITION_SPEED   # [s] per direction


def build_rows():
    """Return a list of (time, output_motor_torque, input_motor_position) tuples."""
    rows = [(0.0, 0.0, 0.0)]  # all columns must start at 0
    t = 0.0
    prev_torque = 0.0

    for i, torque in enumerate(OUTPUT_TORQUES):
        # Ramp the output torque to this level while parked at position 0.
        if torque != prev_torque:
            t += abs(torque - prev_torque) / TORQUE_ROTATUM
            rows.append((t, torque, 0.0))

        # Forward sweep: 0 -> +AMPLITUDE at constant torque.
        t += SWEEP_TIME
        rows.append((t, torque, AMPLITUDE))

        # Dwell at the peak.
        if PEAK_DWELL_S > 0.0:
            t += PEAK_DWELL_S
            rows.append((t, torque, AMPLITUDE))

        # Backward sweep: +AMPLITUDE -> 0.
        t += SWEEP_TIME
        rows.append((t, torque, 0.0))

        # Optional settle at 0 before the next torque change (not after the last level).
        if BOTTOM_DWELL_S > 0.0 and i < len(OUTPUT_TORQUES) - 1:
            t += BOTTOM_DWELL_S
            rows.append((t, torque, 0.0))

        prev_torque = torque

    # Torque trace must end at 0.
    if prev_torque != 0.0:
        t += abs(prev_torque) / TORQUE_ROTATUM
        rows.append((t, 0.0, 0.0))

    return rows


def main():
    rows = build_rows()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "output_motor_torque", "input_motor_position"])
        # str() on a float gives shortest round-trip precision (full float64 fidelity).
        for t, torque, pos in rows:
            writer.writerow([t, torque, pos])

    # Sanity summary.
    times = [r[0] for r in rows]
    assert all(b > a for a, b in zip(times, times[1:])), "time column is not strictly increasing"
    assert rows[0] == (0.0, 0.0, 0.0), "trace must start at 0,0,0"
    assert rows[-1][1] == 0.0, "torque must end at 0"
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"  amplitude        = {AMPLITUDE:.6f} rad ({REVOLUTIONS} rev)")
    print(f"  per-direction    = {SWEEP_TIME:.3f} s at {POSITION_SPEED} rad/s")
    print(f"  torque levels    = {OUTPUT_TORQUES} Nm at {TORQUE_ROTATUM} Nm/s")
    print(f"  total duration   = {times[-1]:.1f} s ({times[-1] / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
