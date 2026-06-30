import csv
import os
import math

# --- Tunables --------------------------------------------------------------
NUM_LOC = 20                            # Number of locations to test backlash 
PEAK_OUTPUT_TORQUE = 0.25 * 110          # [Nm] Peak torque on OUTPUT side
CYCLE_TIME = 10                         # [s] Time per cycle
MAX_ROTATUM  = 160                      # [Nm/s] Maximum Torque increase per second on OUTPUT side
GEAR_RATIO = 26                         # gear_ratio : 1  | output torque: input torque 
ZERO_DWELL_TIME = 1                     # [s] Rest time between each cycle. Uses time to move to next location

OUTPUT_NAME = "backlash_generated.csv"
# ---------------------------------------------------------------------------



def build_rows(num_loc, output_trq, t_cycle, max_rotatum, gear_ratio, zero_dwell_time):
    # rows = [time, input torque, output pos]
    rows = [[0,0,0]]
    prev_t = 0 
    
    # Locations in radians to test backlash from 0 - 360 degrees of the input side
    locs = [2*math.pi / num_loc * i for i in range(0,num_loc)]
    locs.insert(1, 2*math.pi/num_loc) # repeat first test to remove any initial condition dependancy
    
    t_step = t_cycle/4
    assert output_trq/t_step < max_rotatum, "Rotatum exceeds maximum. Slow down cycle time or lower torque."
    
    # Calculate input torque thru gear ratio
    input_trq = output_trq / gear_ratio
    
    for l in locs:
        # Moves to next location with zero torque
        prev_t += zero_dwell_time
        rows.append([prev_t, 0, l])
        
        # Add a delay before torque ramp to read 0 velocity, 0 torque, tare value
        prev_t += zero_dwell_time
        rows.append([prev_t,0,l])
        
        # Ramp up Torque
        prev_t += t_step    
        rows.append([prev_t, input_trq, l])
        
        # Reverse Torque
        prev_t += t_step * 2
        rows.append([prev_t, -input_trq, l])
        
        # Ramp down Torque
        prev_t += t_step
        rows.append([prev_t, 0, l])
        
    # Ends test at 0 torque
    prev_t += zero_dwell_time
    rows.append([prev_t, 0, l])    

    return rows


def main():
    rows = build_rows(NUM_LOC, PEAK_OUTPUT_TORQUE, CYCLE_TIME, MAX_ROTATUM, GEAR_RATIO, ZERO_DWELL_TIME)
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