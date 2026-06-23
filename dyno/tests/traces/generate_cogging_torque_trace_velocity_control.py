import csv
import os
import math

# --- Tunables --------------------------------------------------------------
SPEEDS = [4,8,12,16,20,24,28,30]    # [rad/s] Speeds to test
TORQUES = [0,10,20]              # [Nm] Torques to test
MIN_TIME = 20.0                     # [s] Minimum time per cycle
MIN_ROT = 20.0                      # [rev] Minimum number of revolutions per cycle
MAX_ACCEL = 10                      # [rad/s^2] Maximum rotational acceleration
MAX_ROTATUM  = 160                  # [Nm/s] Maximum Torque increase per second 
MIN_DWELL_TIME = 10                  # [s] Minimum time to sit at given speed
ZERO_DWELL_TIME = 1                 # [s] Seconds to sit at 0 velocity, 0 torque

OUTPUT_NAME = "cogging_trace_vc.csv"
# ---------------------------------------------------------------------------



def build_rows(speeds, torques, min_time, min_rot, max_accel, min_dwell_time, zero_dwell_time):
    rows = [[0,0,0]]
    prev_t = 0 
    twenty_rot = 20 * 2 * math.pi # [rad]
    for s in speeds:
        
        # Determine how long to spin for to reach min(20 rev, 20s) for each cycle
        t_spin = 10
        if twenty_rot / s < 20 :
            t_spin = twenty_rot/s/2
        t_ramp = s/max_accel
        
        # Check if min dwell time is reached:
        dwell_time = t_spin - 2 * t_ramp
        if t_spin - 2* t_ramp < min_dwell_time:
            dwell_time = min_dwell_time
        
        # Check if max rotatum is exceeded:
        assert max(torques)/ t_ramp < MAX_ROTATUM, "Max Rotatum Exceeded" 
        for trq in torques:
            
            # Start each cycle with a pause at 0 torque, 0 speed
            prev_t += zero_dwell_time
            rows.append([prev_t, 0, 0]) 
            
            # Ramp up to velocity based on max acceleration
            prev_t += t_ramp
            rows.append([prev_t, trq, s])
            
            # Hold velocity
            prev_t += dwell_time
            rows.append([prev_t, trq, s])
            
            # Reverse velocity
            prev_t += 2 * t_ramp
            rows.append([prev_t, trq, -s])
            
            # Hold velocity
            prev_t += dwell_time
            rows.append([prev_t, trq, -s])
            
            # Brake to zero velocity
            prev_t += t_ramp
            rows.append([prev_t, trq, 0])
            
    # Zero velocity, Zero Torque as last command for test.        
    prev_t += zero_dwell_time
    rows.append([prev_t, 0, 0])
    return rows


def main():
    rows = build_rows(SPEEDS, TORQUES, MIN_TIME, MIN_ROT, MAX_ACCEL, MIN_DWELL_TIME, ZERO_DWELL_TIME)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "input_motor_torque", "output_motor_velocity"])
        
        # Note: Corrected the variable names here to match the data mapping
        for t, trq, pos in rows:
            # Round slightly first to clean up math generation floating-point errors
            t = round(t, 5)
            pos = round(pos, 5)
            
            # Drop the decimal if it's a whole number
            t_out = int(t) if float(t).is_integer() else t
            trq_out = int(trq) if float(trq).is_integer() else trq
            pos_out = int(pos) if float(pos).is_integer() else pos
            
            writer.writerow([t_out, trq_out, pos_out])


if __name__ == "__main__":
    main()