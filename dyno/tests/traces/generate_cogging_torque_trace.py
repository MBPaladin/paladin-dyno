import csv
import os
import math

# --- Tunables --------------------------------------------------------------
SPEEDS = [4,8,12,16,20,24,28,30]    # [rad/s] Speeds to test
TORQUES = [0,10,20,30]              # [Nm] Torques to test
MIN_TIME = 20.0                     # [s] Minimum time per cycle
MIN_ROT = 20.0                      # [rev] Minimum number of revolutions per cycle
MAX_ROTATUM = 10                    # [rad/s^2] Maximum rotational acceleration
DWELL_TIME = 1                      # [s] Seconds to sit at maximum position
ZERO_DWELL_TIME = 1                 # [s] Seconds to sit at zero position

RAMP_RESOLUTION = 0.1               # [s] Time step for the acceleration ramps
OUTPUT_NAME = "cogging_trace_generated.csv"
# ---------------------------------------------------------------------------


def generate_trapezoidal_move(t_start, p_start, p_end, speed, max_a, torque, resolution):
    """
    Generates a list of [time, torque, position] points representing a 
    trapezoidal velocity profile. Only the parabolic acceleration and 
    deceleration curves are discretized.
    """
    rows = []
    direction = 1 if p_end > p_start else -1
    distance = abs(p_end - p_start)
    
    # Calculate time and distance for the acceleration ramp
    t_a = speed / max_a
    d_a = 0.5 * max_a * t_a**2
    
    # Check if the move is too short to reach the target speed
    if 2 * d_a > distance:
        d_a = distance / 2
        t_a = math.sqrt(2 * d_a / max_a)
        actual_speed = max_a * t_a
        t_c = 0
        d_c = 0
    else:
        actual_speed = speed
        d_c = distance - 2 * d_a
        t_c = d_c / actual_speed
        
    # 1. Discretize Acceleration Phase
    t = resolution
    while t < t_a - 1e-6: # 1e-6 prevents floating-point overshoot
        pos = p_start + direction * (0.5 * max_a * t**2)
        rows.append([t_start + t, torque, pos])
        t += resolution
        
    # Exact end of acceleration / Start of cruise phase
    rows.append([t_start + t_a, torque, p_start + direction * d_a])
    
    # 2. Cruise Phase (Single jump because linear interpolation handles constant velocity perfectly)
    if t_c > 0:
        rows.append([t_start + t_a + t_c, torque, p_start + direction * (d_a + d_c)])
        
    # 3. Discretize Deceleration Phase
    t = resolution
    while t < t_a - 1e-6:
        pos = p_start + direction * (d_a + d_c + actual_speed * t - 0.5 * max_a * t**2)
        rows.append([t_start + t_a + t_c + t, torque, pos])
        t += resolution
        
    # Exact end of movement
    t_end = t_start + 2 * t_a + t_c
    rows.append([t_end, torque, p_end])
    
    return rows, t_end


def build_rows(speeds, torques, min_time, min_rot, max_rotatum, dwell_time, zero_dwell_time, ramp_res):
    rows = [[0, 0, 0]] 
    prev_t = 0 
    for s in speeds:
        for trq in torques:
            # 1. Zero Dwell (Ramps up torque while position remains stationary)
            prev_t += zero_dwell_time
            rows.append([prev_t, trq, 0])
            
            # Calculate peak position
            peak_pos = (min_rot / 2) * 2 * math.pi 
            dt = peak_pos / s
            
            if dt < min_time / 2:
                dt = 10
                peak_pos = s * dt
                
            # 2. Move forward with trapezoidal velocity
            ramp_rows, prev_t = generate_trapezoidal_move(
                prev_t, 0, peak_pos, s, max_rotatum, trq, ramp_res
            )
            rows.extend(ramp_rows)
            
            # 3. Peak Dwell
            prev_t += dwell_time
            rows.append([prev_t, trq, peak_pos])
            
            # 4. Move backward with trapezoidal velocity
            ramp_rows, prev_t = generate_trapezoidal_move(
                prev_t, peak_pos, 0, s, max_rotatum, trq, ramp_res
            )
            rows.extend(ramp_rows)
            
    rows.append([prev_t + 1, 0, 0]) # End at zero torque 
    return rows


def main():
    rows = build_rows(SPEEDS, TORQUES, MIN_TIME, MIN_ROT, MAX_ROTATUM, DWELL_TIME, ZERO_DWELL_TIME, RAMP_RESOLUTION)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "input_motor_torque", "output_motor_position"])
        
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