import csv
import os
import math

# --- Tunables --------------------------------------------------------------
NUM_TESTS = 10                  # Number of times to repeat test for statistical analysis
ROTATUM = 0.075                   # Slow torque increase
RAMP_DURATION = 10              # [s] Duration of ramp. Tune for a longer stiction behavior
REST_DURATION = 2               # [s] Duration of rest before reverse test. Allow stiction factors to settle
SLOWDOWN_DURATION = 1.0         # [s] Duration to slowdown motor back to 0.    
MOTOR_CHOICE = "input"          # "input" or "output". Test controls single motor only
GEAR_RATIO = 1                 # gear_ratio : 1  | output torque: input torque


OUTPUT_NAME = "stiction_generated.csv"
# ---------------------------------------------------------------------------



def build_rows(num_tests, accel, t_ramp, t_rest, t_slowdown, motor_choice, gear_ratio):
    # rows = [time, input_torque, dummy position (unused since test is one input at a time)]
    rows = [[0,0,0]]
    prev_t = 0 
    for i in range(num_tests):
        
        # Slowly ramp torque for t_ramp [s]
        prev_t += t_ramp
        rows.append([prev_t, accel * t_ramp,0])
        
        # Slowdown back to 0 torque
        prev_t += t_slowdown
        rows.append([prev_t, 0,0])
        
        # Pause to let stiction factors settle
        prev_t += t_rest
        rows.append([prev_t, 0,0])
        
        # Slowly ramp torque in other direction for t_ramp [s]
        prev_t += t_ramp
        rows.append([prev_t, -accel * t_ramp,0])
        
        # Slowdown back to 0 torque
        prev_t += t_slowdown
        rows.append([prev_t, 0,0])
        
        # Pause to let stiction factors settle 
        prev_t += t_rest
        rows.append([prev_t, 0,0])

    return rows


def main():
    rows = build_rows(NUM_TESTS, ROTATUM, RAMP_DURATION, REST_DURATION, SLOWDOWN_DURATION, MOTOR_CHOICE, GEAR_RATIO)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        if MOTOR_CHOICE == "input":
            writer.writerow(["time", "input_motor_torque", "output_motor_torque"])
        else:
            writer.writerow(["time", "output_motor_torque", "input_motor_torque"])

        
        # Note: Corrected the variable names here to match the data mapping
        for t, vel, dummy in rows:
            # Round slightly first to clean up math generation floating-point errors
            t = round(t, 5)
            vel = round(vel, 5)
            
            # Drop the decimal if it's a whole number
            t_out = int(t) if float(t).is_integer() else t
            vel_out = int(vel) if float(vel).is_integer() else vel
            
            writer.writerow([t_out, vel_out, dummy])


if __name__ == "__main__":
    main()