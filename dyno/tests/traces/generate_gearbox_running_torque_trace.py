import csv
import os
import math

# --- Tunables --------------------------------------------------------------
ACCEL = 0.5                      # [rad/s^2] Rate of velocity increase of the OUTPUT
PEAK_VELOCITY = 50             # [rad/s] Peak Velocity of the OUTPUT. 
REST_DURATION = 3               # [s] Duration of rest before reverse test   
MOTOR_CHOICE = "input"          # "input" or "output". Test controls single motor only
GEAR_RATIO = 1                 # gear_ratio : 1  | output torque : input torque

OUTPUT_NAME = "gearbox_running_torque_generated.csv"
# ---------------------------------------------------------------------------



def build_rows(accel, peak_vel, t_rest, motor_choice, gear_ratio):
    # rows = [time, input_velocity, dummy position (unused since test is one input at a time)]
    rows = [[0,0,0]]
    prev_t = 0 
    
    if motor_choice == "input":
        accel *= gear_ratio
        peak_vel *= gear_ratio
    t_ramp_duration = peak_vel/accel
        
    # Slowly ramp velocity until peak
    prev_t += t_ramp_duration
    rows.append([prev_t, peak_vel, 0])
    
    # Slowdown and Inverse Direction
    prev_t += t_ramp_duration *2 
    rows.append([prev_t, -peak_vel,0])
    
    # Slowdown to 0 velocity
    prev_t += t_ramp_duration
    rows.append([prev_t, 0,0])
    return rows


def main():
    rows = build_rows(ACCEL, PEAK_VELOCITY, REST_DURATION, MOTOR_CHOICE, GEAR_RATIO)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        if MOTOR_CHOICE == "input":
            writer.writerow(["time", "input_motor_velocity", "output_motor_torque"])
        else:
            writer.writerow(["time", "output_motor_velocity", "input_motor_torque"])

        
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