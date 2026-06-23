"""
Generate a velocity-sweep trace.

The profile accelerates the input motor to a target velocity, holds it,
decelerates through zero to the negative target velocity, holds it, and
finally decelerates back to zero exactly at the 20-second mark.
Output motor torque is always held at 0.

Columns:
    time                  [s]      monotonically increasing, starts at 0
    input_motor_velocity  [rad/s]  starts at 0, ends at 0
    output_motor_torque   [Nm]     always 0
"""

import csv
import os

# --- Tunables --------------------------------------------------------------
MAX_SPEED = 30.0        # Target maximum speed in rad/s
ACCELERATION = 10.0     # Acceleration/deceleration in rad/s^2
TOTAL_TIME = 20.0       # Total duration of the trace in seconds

OUTPUT_NAME = "encoder_trace.csv"
# ---------------------------------------------------------------------------


def build_rows(max_speed, acceleration, total_time):
    """Return a list of (time, input_motor_velocity, output_motor_torque) tuples."""
    
    # Calculate how much time is spent ramping
    t_ramp_up = max_speed / acceleration
    t_ramp_reverse = (2 * max_speed) / acceleration
    t_ramp_down = max_speed / acceleration
    
    total_ramp_time = t_ramp_up + t_ramp_reverse + t_ramp_down
    
    if total_ramp_time > total_time:
        raise ValueError("Acceleration is too slow to reach the target speeds within the total time.")
        
    # Distribute the remaining time equally between the positive and negative holds
    total_hold_time = total_time - total_ramp_time
    hold_time = total_hold_time / 2.0
    
    # Calculate the exact timestamp for each transition
    t0 = 0.0
    t1 = t0 + t_ramp_up
    t2 = t1 + hold_time
    t3 = t2 + t_ramp_reverse
    t4 = t3 + hold_time
    t5 = total_time 
    
    # Generate the 6 critical points of the profile
    rows = [
        (t0, 0.0, 0.0),
        (t1, max_speed, 0.0),
        (t2, max_speed, 0.0),
        (t3, -max_speed, 0.0),
        (t4, -max_speed, 0.0),
        (t5, 0.0, 0.0)
    ]
    
    return rows


def main():
    rows = build_rows(MAX_SPEED, ACCELERATION, TOTAL_TIME)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "output_motor_velocity", "input_motor_torque"])
        
        for t, vel, trq in rows:
            # Drop the decimal if it's a whole number to cleanly match your requested format
            t_out = int(t) if float(t).is_integer() else t
            vel_out = int(vel) if float(vel).is_integer() else vel
            trq_out = int(trq) if float(trq).is_integer() else trq
            
            writer.writerow([t_out, vel_out, trq_out])


if __name__ == "__main__":
    main()