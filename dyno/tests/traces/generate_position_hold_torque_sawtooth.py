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
OUTPUT_TORQUES = [70,-70]
ITERATIONS = 1    
TORQUE_ROTATUM = 46.666

OUTPUT_NAME = "torque_sawtooth.csv"
# ---------------------------------------------------------------------------

dt_initial = abs(OUTPUT_TORQUES[0])/TORQUE_ROTATUM
dt_traverse = abs(OUTPUT_TORQUES[0]-OUTPUT_TORQUES[1])/TORQUE_ROTATUM
dt_final = abs(OUTPUT_TORQUES[1]/TORQUE_ROTATUM)

def build_rows():
    """Return a list of (time, output_motor_position, input_motor_torque) tuples."""
    rows = [(0.0, 0.0, 0.0)]  # all columns must start at 0
    t = 0.0
    prev_torque = 0.0

    for i in range(ITERATIONS):
        # Ramp the output torque to this level while parked at position 0.
        t += dt_initial
        rows.append((t,0,OUTPUT_TORQUES[0]))
        t += dt_traverse
        rows.append((t,0,OUTPUT_TORQUES[1]))
        t+= dt_final
        rows.append((t,0,0))

    return rows


def main():
    rows = build_rows()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "output_motor_position", "input_motor_torque"])
        # str() on a float gives shortest round-trip precision (full float64 fidelity).
        for t, torque, pos in rows:
            writer.writerow([t, torque, pos])


if __name__ == "__main__":
    main()
