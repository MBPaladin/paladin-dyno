import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import argparse
import json
import os
from datetime import datetime, timezone
try:
    from deployment import dyno_paths
except ImportError:
    dyno_paths = None

# Post processor that evaluates backlash, positional accuracy, efficiency,
# peak effort, and encoder (CONSTANT_TORQUE) tests from a single log.

def dut_input_pos(f, start, end):
    # Actuator-production logs name this 'dut_projected_input_encoder';
    # gearbox/actuator-dev logs name it 'dut_input_position'.
    key = 'dut_input_position' if 'dut_input_position' in f else 'dut_projected_input_encoder'
    return f[key][start:end]

g = 1

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Dyno Post Processor')
parser.add_argument('log_folder', nargs='?', default=None,
                    help='Path to log folder (e.g., logs/1770060719 or just 1770060719)')
parser.add_argument('--headless', action='store_true',
                    help='Run in headless mode (save plots but do not display them)')
parser.add_argument('--interactive', action='store_true',
                    help='Show interactive matplotlib plots (zoom/pan enabled)')
parser.add_argument('--skip-start', type=float, default=3.0,
                    help='Skip first N seconds of data for CONSTANT_TORQUE analysis (default: 3.0s)')
args = parser.parse_args()

# Get log folder from command line or file dialog
if args.log_folder:
    log_folder = os.path.abspath(args.log_folder.rstrip('/'))
    print(f'Using log folder: {log_folder}')
else:
    print('Select test folder to process: ')
    root = tk.Tk()
    root.withdraw()
    log_folder = filedialog.askdirectory()
    print(log_folder)

# If the chosen folder doesn't have log.hdf5 directly, pick the most recent
# timestamped subfolder that does. Lets the user select either the parent
# 'logs' folder or a specific run.
if not os.path.isfile(os.path.join(log_folder, 'log.hdf5')):
    candidates = [
        os.path.join(log_folder, d) for d in os.listdir(log_folder)
        if os.path.isfile(os.path.join(log_folder, d, 'log.hdf5'))
    ]
    if not candidates:
        raise FileNotFoundError(f"No log.hdf5 found in {log_folder} or its subfolders")
    log_folder = max(candidates, key=os.path.getmtime)
    print(f'Resolved to most recent run: {log_folder}')

test_folder_name = os.path.basename(log_folder)

if args.headless:
    print('Running in headless mode - plots will be saved but not displayed')


# Function to unwrap encoder signals % 2*pi
def unwrapped(encoder_signal, modulo_value=2 * np.pi):
    unwrapped_signal = np.copy(encoder_signal).astype(float)
    num_wraps = 0

    for i in range(1, len(encoder_signal)):
        current_value = encoder_signal[i]
        previous_value = encoder_signal[i-1]

        if current_value < previous_value - modulo_value / 2:
            num_wraps += 1
        elif current_value > previous_value + modulo_value / 2:
            num_wraps -= 1

        unwrapped_signal[i] = encoder_signal[i] + num_wraps * modulo_value

    return unwrapped_signal

# Function to convert HDF5 time to ISO 8601 timestamp
def hdf5_time_to_iso8601(time_seconds, base_timestamp):
    absolute_time = base_timestamp + float(time_seconds)
    dt = datetime.fromtimestamp(absolute_time, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

# Function to write test results to JSON file
def write_test_results_to_file(test_results, test_folder_name):
    output_path = f'{log_folder}/test_results.json'
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    return output_path

def save(fig, path):
    fig.savefig(path, dpi=200)
    print(f'  Saved: {os.path.basename(path)}')
    if not args.interactive:
        plt.close(fig)


# Hard coded file path
with h5.File(os.path.join(log_folder, 'log.hdf5'), 'r') as f:
    test_results = []
    # Extract numeric timestamp from folder name (handles names like '1772154942_post_calibration')
    import re
    ts_match = re.match(r'(\d+)', test_folder_name)
    base_timestamp = int(ts_match.group(1)) if ts_match else 0

    # -------------------------------------------------------------------------
    # POSITIONAL ACCURACY (ISO 230-2 style)
    # -------------------------------------------------------------------------
    # Collects settled position error at each step across multiple cycles.
    # Reports per-position accuracy, repeatability, reversal error, and
    # system-level bidirectional accuracy.
    # -------------------------------------------------------------------------
    print('Processing Positional Accuracy')

    # Gather all POSITIONAL_ACCURACY runs (one per cycle)
    pos_acc_runs = []
    for ui, behavior_id in enumerate(f['behavior_ids']):
        bid = behavior_id.decode("utf-8")
        if bid.split('-')[0].startswith('POSITIONAL_ACCURACY'):
            pos_acc_runs.append((bid, f['behavior_indices'][ui, 0], f['behavior_indices'][ui, 1]))

    if len(pos_acc_runs) > 0:
        num_cycles = len(pos_acc_runs)
        print(f'  Found {num_cycles} cycle(s)')

        # -- Helper: detect step holds from command, refine with measured position --
        # Command stationarity finds coarse hold boundaries; measured position
        # velocity threshold determines when the actuator has actually settled.
        trim_pct = 0.10           # trim last 10% of each hold to exclude pre-ramp transients
        settle_vel_threshold = 1e-4  # rad/sample (~0.1 rad/s at 1kHz) — measured velocity must drop below this
        settle_window = 20        # number of consecutive samples below threshold to confirm settling
        min_hold_samples = 200    # at 1kHz, minimum 200ms hold to count as a step

        def find_step_holds(dut_cmd, dut_measured=None):
            """Find regions where the position command is constant (step holds).
            If dut_measured is provided, the start of each hold is refined to
            the point where measured velocity drops below settle_vel_threshold."""
            cmd_diff = np.zeros(len(dut_cmd))
            cmd_diff[1:] = np.abs(np.diff(dut_cmd))
            cmd_valid = ~np.isnan(dut_cmd)
            cmd_diff[~cmd_valid] = 1.0  # mark NaN regions as moving

            cmd_stationary = cmd_diff < 1e-6

            coarse_regions = []
            in_region = False
            region_start = 0
            for i in range(len(cmd_stationary)):
                if cmd_stationary[i] and not in_region:
                    region_start = i
                    in_region = True
                elif not cmd_stationary[i] and in_region:
                    if i - region_start >= min_hold_samples:
                        coarse_regions.append((region_start, i))
                    in_region = False
            if in_region and len(cmd_stationary) - region_start >= min_hold_samples:
                coarse_regions.append((region_start, len(cmd_stationary)))

            if dut_measured is None:
                return coarse_regions

            # Refine each region start using measured position velocity
            meas_vel = np.zeros(len(dut_measured))
            meas_vel[1:] = np.abs(np.diff(dut_measured))

            regions = []
            for start, end in coarse_regions:
                # Find first index in [start, end) where measured velocity stays
                # below threshold for settle_window consecutive samples
                settled_start = None
                consecutive = 0
                for i in range(start, end):
                    if meas_vel[i] < settle_vel_threshold:
                        consecutive += 1
                        if consecutive >= settle_window:
                            settled_start = i - settle_window + 1
                            break
                    else:
                        consecutive = 0
                if settled_start is not None:
                    # Skip first half of the settled window for extra margin
                    midpoint = settled_start + (end - settled_start) // 2
                    if (end - midpoint) >= min_hold_samples:
                        regions.append((midpoint, end))
            return regions

        def extract_step_measurements(regions, dut_output_pos, dut_input_pos, dut_cmd_aligned, load_pos_tared, dut_input_raw):
            """Return list of dicts with per-step settled measurements.
            Computes five error signals:
              - output_tracking_error: APS - command  (output tracking quality)
              - input_tracking_error:  INC - command  (servo tracking quality)
              - output_vs_load:        APS - LOAD     (output encoder vs ground truth)
              - input_vs_load:         INC - LOAD     (input encoder vs ground truth)
              - input_vs_output:       INC(tared to APS) - APS
            """
            steps = []
            for start, end in regions:
                region_length = end - start
                trim_end_samples = int(region_length * trim_pct)
                analysis_start = start
                analysis_end = end - trim_end_samples
                if analysis_end <= analysis_start:
                    continue
                s = slice(analysis_start, analysis_end)
                target_pos = np.nanmean(dut_cmd_aligned[s])
                output_tracking_err = np.mean(dut_output_pos[s] - dut_cmd_aligned[s])
                input_tracking_err = np.mean(dut_input_pos[s] - dut_cmd_aligned[s])
                output_vs_load_err = np.mean(dut_output_pos[s] - load_pos_tared[s])
                input_vs_load_err = np.mean(dut_input_pos[s] - load_pos_tared[s])
                input_vs_output_err = np.mean(dut_input_raw[s] - dut_output_pos[s])
                steps.append({
                    'target_rad': target_pos,
                    'output_tracking_error_rad': output_tracking_err,
                    'input_tracking_error_rad': input_tracking_err,
                    'output_vs_load_error_rad': output_vs_load_err,
                    'input_vs_load_error_rad': input_vs_load_err,
                    'input_vs_output_error_rad': input_vs_output_err,
                    'region': (start, end),
                })
            return steps

        def classify_direction(steps):
            """Label each step as 'up' or 'down' based on target position change.
            The first step is classified based on the second step's direction
            (if the trace starts by stepping up, the first hold is 'up')."""
            for i, step in enumerate(steps):
                if i == 0:
                    # Determine from next step: if next target is higher, first is 'up'
                    if len(steps) > 1 and steps[1]['target_rad'] > step['target_rad'] + 0.01:
                        step['direction'] = 'up'
                    elif len(steps) > 1 and steps[1]['target_rad'] < step['target_rad'] - 0.01:
                        step['direction'] = 'down'
                    else:
                        step['direction'] = 'up'
                elif step['target_rad'] > steps[i - 1]['target_rad'] + 0.01:
                    step['direction'] = 'up'
                elif step['target_rad'] < steps[i - 1]['target_rad'] - 0.01:
                    step['direction'] = 'down'
                else:
                    step['direction'] = 'hold'

        # -- Process each cycle --
        all_cycle_steps = []  # list of lists
        first_time = None
        last_time = None

        for cycle_idx, (bid, idx_start, idx_end) in enumerate(pos_acc_runs):
            time_data = f['time'][idx_start:idx_end]
            dut_position_command = f['dut_position_command'][idx_start:idx_end]
            dut_output_position = f['dut_output_position'][idx_start:idx_end]
            dut_input_position = dut_input_pos(f, idx_start, idx_end)
            load_position = f['load_position'][idx_start:idx_end]

            if first_time is None:
                first_time = time_data[0]
            last_time = time_data[-1]

            load_position_unwrapped = unwrapped(load_position)

            # Align command to output (handles multi-turn offset)
            if not np.all(np.isnan(dut_position_command)):
                dut_command = np.copy(dut_position_command)
                valid_mask = ~np.isnan(dut_command)
                if np.any(valid_mask):
                    mean_offset = np.mean(dut_output_position[valid_mask] - dut_command[valid_mask])
                    rotations_offset = np.round(mean_offset / (2 * np.pi))
                    dut_command_aligned = dut_command + rotations_offset * 2 * np.pi
                else:
                    dut_command_aligned = dut_command
            else:
                dut_command_aligned = dut_position_command

            # Detect hold regions first so we can tare on the first settled bin
            regions = find_step_holds(dut_command_aligned, dut_output_position)

            # Tare load and input using the settled portion of the first static bin
            # so that the first bin's error is always zero.
            # Load is tared to APS; INC is tared to LOAD (so INC-LOAD starts at 0).
            # INC vs APS comparison uses raw (untared) signals.
            if regions:
                r_start, r_end = regions[0]
                r_len = r_end - r_start
                trim_end = int(r_len * trim_pct)
                tare_s = slice(r_start, r_end - trim_end)
                if tare_s.stop > tare_s.start:
                    load_offset = np.mean(dut_output_position[tare_s]) - np.mean(load_position_unwrapped[tare_s])
                    input_offset = np.mean(load_position_unwrapped[tare_s] + load_offset) - np.mean(dut_input_position[tare_s])
                else:
                    load_offset = dut_output_position[0] - load_position_unwrapped[0]
                    input_offset = (load_position_unwrapped[0] + load_offset) - dut_input_position[0]
            else:
                load_offset = dut_output_position[0] - load_position_unwrapped[0]
                input_offset = (load_position_unwrapped[0] + load_offset) - dut_input_position[0]
            load_position_tared = load_position_unwrapped + load_offset
            dut_input_tared = dut_input_position + input_offset
            steps = extract_step_measurements(regions, dut_output_position, dut_input_tared, dut_command_aligned, load_position_tared, dut_input_position)
            classify_direction(steps)

            print(f'  Cycle {cycle_idx}: {len(steps)} settled steps detected')
            all_cycle_steps.append(steps)

        # -- Build per-position, per-direction data across cycles --
        # Round target positions to group them (within 0.05 rad tolerance)
        def round_target(rad, tol=0.05):
            return round(rad / tol) * tol

        # Collect: position_key -> direction -> list of errors across cycles
        # Track all four error types separately
        from collections import defaultdict
        error_keys = ['output_tracking_error_rad', 'input_tracking_error_rad',
                      'output_vs_load_error_rad', 'input_vs_load_error_rad',
                      'input_vs_output_error_rad']
        pos_dir_errors = defaultdict(lambda: {d: {ek: [] for ek in error_keys} for d in ('up', 'down')})

        for cycle_steps in all_cycle_steps:
            for step in cycle_steps:
                if step['direction'] in ('up', 'down'):
                    key = round_target(step['target_rad'])
                    for ek in error_keys:
                        pos_dir_errors[key][step['direction']][ek].append(step[ek])

        # Sort positions
        sorted_positions = sorted(pos_dir_errors.keys())

        # -- Compute ISO 230-2 metrics per position --
        # Primary metric for ISO accuracy is output tracking error (APS - command)
        # but we also report all four encoder pair comparisons.
        per_position = []
        for pos in sorted_positions:
            up_data = pos_dir_errors[pos]['up']
            down_data = pos_dir_errors[pos]['down']
            # Primary: output tracking error (APS - cmd) for ISO 230-2
            up_errors = np.array(up_data['output_tracking_error_rad'])
            down_errors = np.array(down_data['output_tracking_error_rad'])

            entry = {'target_degrees': float(np.degrees(pos))}

            # Mean positional error per direction (output tracking = APS - cmd)
            if len(up_errors) > 0:
                entry['mean_error_up_degrees'] = float(np.degrees(np.mean(up_errors)))
                entry['std_up_degrees'] = float(np.degrees(np.std(up_errors, ddof=1))) if len(up_errors) > 1 else 0.0
            if len(down_errors) > 0:
                entry['mean_error_down_degrees'] = float(np.degrees(np.mean(down_errors)))
                entry['std_down_degrees'] = float(np.degrees(np.std(down_errors, ddof=1))) if len(down_errors) > 1 else 0.0

            # Reversal error (B_i): difference between mean error from each direction
            if len(up_errors) > 0 and len(down_errors) > 0:
                entry['reversal_error_degrees'] = float(np.degrees(np.mean(up_errors) - np.mean(down_errors)))

            # Repeatability per direction (R_i = 4 * sigma, covers ~95.4%)
            if len(up_errors) > 1:
                entry['repeatability_up_degrees'] = float(np.degrees(4 * np.std(up_errors, ddof=1)))
            if len(down_errors) > 1:
                entry['repeatability_down_degrees'] = float(np.degrees(4 * np.std(down_errors, ddof=1)))

            # Encoder comparison errors (all four pairs, per direction)
            for ek, label in [('input_tracking_error_rad', 'input_tracking'),
                              ('output_vs_load_error_rad', 'output_vs_load'),
                              ('input_vs_load_error_rad', 'input_vs_load'),
                              ('input_vs_output_error_rad', 'input_vs_output')]:
                up_vals = np.array(up_data[ek])
                dn_vals = np.array(down_data[ek])
                if len(up_vals) > 0:
                    entry[f'{label}_up_degrees'] = float(np.degrees(np.mean(up_vals)))
                if len(dn_vals) > 0:
                    entry[f'{label}_down_degrees'] = float(np.degrees(np.mean(dn_vals)))

            per_position.append(entry)

        # -- System-level metrics --
        # Collect all mean errors and repeatabilities
        mean_errors_up = [e['mean_error_up_degrees'] for e in per_position if 'mean_error_up_degrees' in e]
        mean_errors_down = [e['mean_error_down_degrees'] for e in per_position if 'mean_error_down_degrees' in e]
        reversal_errors = [e['reversal_error_degrees'] for e in per_position if 'reversal_error_degrees' in e]
        repeatabilities_up = [e['repeatability_up_degrees'] for e in per_position if 'repeatability_up_degrees' in e]
        repeatabilities_down = [e['repeatability_down_degrees'] for e in per_position if 'repeatability_down_degrees' in e]

        # Unidirectional accuracy: range of mean errors in each direction
        accuracy_up = (max(mean_errors_up) - min(mean_errors_up)) if len(mean_errors_up) > 1 else 0
        accuracy_down = (max(mean_errors_down) - min(mean_errors_down)) if len(mean_errors_down) > 1 else 0

        # Mean reversal error
        mean_reversal = np.mean(np.abs(reversal_errors)) if reversal_errors else 0

        # Bidirectional accuracy: range of (all mean errors from both directions combined)
        all_means = mean_errors_up + mean_errors_down
        bidirectional_accuracy = (max(all_means) - min(all_means)) if len(all_means) > 1 else 0

        # Max repeatability
        max_repeatability_up = max(repeatabilities_up) if repeatabilities_up else 0
        max_repeatability_down = max(repeatabilities_down) if repeatabilities_down else 0
        max_repeatability = max(max_repeatability_up, max_repeatability_down)

        # Encoder pair aggregate metrics (RMSE, 95th percentile, and max absolute error across all settled positions)
        # For INC comparisons (input_vs_load, input_vs_output), discard first up and first down bin
        # (backlash not preloaded at start and turnaround)
        skip_first_both = {'input_vs_load', 'input_vs_output'}
        encoder_pair_metrics = {}
        for label in ['output_vs_load', 'input_vs_load', 'input_tracking', 'input_vs_output']:
            up_vals = [e[f'{label}_up_degrees'] for e in per_position if f'{label}_up_degrees' in e]
            dn_vals = [e[f'{label}_down_degrees'] for e in per_position if f'{label}_down_degrees' in e]
            if label in skip_first_both:
                if len(up_vals) > 1:
                    up_vals = up_vals[1:]
                if len(dn_vals) > 1:
                    dn_vals = dn_vals[1:]
            all_vals = up_vals + dn_vals
            if all_vals:
                arr = np.array(all_vals)
                encoder_pair_metrics[f'rmse_{label}_degrees'] = float(np.sqrt(np.mean(arr ** 2)))
                encoder_pair_metrics[f'p95_abs_{label}_degrees'] = float(np.percentile(np.abs(arr), 95))
                encoder_pair_metrics[f'max_abs_{label}_degrees'] = float(np.max(np.abs(arr)))

        # Per-direction INC-APS nonlinearity: deviation from the direction's mean
        io_up_vals = [e['input_vs_output_up_degrees'] for e in per_position if 'input_vs_output_up_degrees' in e]
        io_dn_vals = [e['input_vs_output_down_degrees'] for e in per_position if 'input_vs_output_down_degrees' in e]
        if len(io_up_vals) > 1:
            arr_up = np.array(io_up_vals[1:])  # discard first up bin (backlash outlier)
            dev_up = arr_up - np.mean(arr_up)
            encoder_pair_metrics['nonlinearity_rmse_inc_aps_up_degrees'] = float(np.sqrt(np.mean(dev_up ** 2)))
            encoder_pair_metrics['nonlinearity_p95_inc_aps_up_degrees'] = float(np.percentile(np.abs(dev_up), 95))
        if len(io_dn_vals) > 1:
            arr_dn = np.array(io_dn_vals[1:])  # discard first down bin (backlash at turnaround)
            dev_dn = arr_dn - np.mean(arr_dn)
            encoder_pair_metrics['nonlinearity_rmse_inc_aps_down_degrees'] = float(np.sqrt(np.mean(dev_dn ** 2)))
            encoder_pair_metrics['nonlinearity_p95_inc_aps_down_degrees'] = float(np.percentile(np.abs(dev_dn), 95))

        system_metrics = {
            'num_cycles': num_cycles,
            'num_positions': len(sorted_positions),
            **encoder_pair_metrics,
            'unidirectional_tracking_accuracy_up_degrees': float(accuracy_up),
            'unidirectional_tracking_accuracy_down_degrees': float(accuracy_down),
            'bidirectional_tracking_accuracy_degrees': float(bidirectional_accuracy),
            'mean_reversal_error_degrees': float(mean_reversal),
            'max_reversal_error_degrees': float(max(abs(np.array(reversal_errors)))) if reversal_errors else 0,
            'max_repeatability_degrees': float(max_repeatability),
        }

        print(f'\n  === Positional Accuracy Results ({num_cycles} cycles) ===')
        print(f'  --- Encoder vs Ground Truth (settled) ---')
        for label, desc in [('output_vs_load', 'APS - LOAD'),
                            ('input_vs_load', 'INC - LOAD'),
                            ('input_vs_output', 'INC - APS'),
                            ('input_tracking', 'INC - Cmd')]:
            if f'rmse_{label}_degrees' in encoder_pair_metrics:
                print(f'  {desc}: RMSE={encoder_pair_metrics[f"rmse_{label}_degrees"]:.4f} '
                      f'P95={encoder_pair_metrics[f"p95_abs_{label}_degrees"]:.4f} '
                      f'max={encoder_pair_metrics[f"max_abs_{label}_degrees"]:.4f} deg')
        print(f'  --- Tracking Accuracy (APS - Command) ---')
        print(f'  Unidirectional tracking accuracy (up):   {accuracy_up:.4f} deg')
        print(f'  Unidirectional tracking accuracy (down): {accuracy_down:.4f} deg')
        print(f'  Bidirectional tracking accuracy:         {bidirectional_accuracy:.4f} deg')
        print(f'  Mean reversal error:            {mean_reversal:.4f} deg')
        print(f'  Max repeatability (4*sigma):    {max_repeatability:.4f} deg')

        test_results.append({
            'unit_test_step_type': 'POSITIONAL_ACCURACY',
            'description': 'ISO 230-2 style positional accuracy. Settled position error measured at each step target across multiple bidirectional cycles. Encoder-vs-LOAD RMSE (hardware quality), tracking accuracy (controller quality), reversal error (directional backlash), and repeatability (4*sigma spread across cycles).',
            'system_metrics': system_metrics,
            'per_position': per_position,
            'start_time': hdf5_time_to_iso8601(first_time, base_timestamp),
            'end_time': hdf5_time_to_iso8601(last_time, base_timestamp)
        })

        # -- Plot 1: Per-position analysis (3x2 grid) --
        # Row 0: Encoder vs LOAD (hardware quality)
        # Row 1: Reversal error + summary table
        # Row 2: Repeatability + INC-APS nonlinearity
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))

        # (0,0) Output vs LOAD per position (APS - LOAD)
        ax_ol = axes[0, 0]
        ol_up_pos = [e['target_degrees'] for e in per_position if 'output_vs_load_up_degrees' in e]
        ol_up_val = [e['output_vs_load_up_degrees'] for e in per_position if 'output_vs_load_up_degrees' in e]
        ol_dn_pos = [e['target_degrees'] for e in per_position if 'output_vs_load_down_degrees' in e]
        ol_dn_val = [e['output_vs_load_down_degrees'] for e in per_position if 'output_vs_load_down_degrees' in e]
        if ol_up_pos:
            ax_ol.plot(ol_up_pos, ol_up_val, 'o-', color='blue', label='Up', markersize=5)
        if ol_dn_pos:
            ax_ol.plot(ol_dn_pos, ol_dn_val, 's-', color='red', label='Down', markersize=5)
        ax_ol.set_xlabel('Target Position (deg)')
        ax_ol.set_ylabel('Error (deg)')
        ax_ol.set_title('Output Encoder vs LOAD: APS - LOAD')
        ax_ol.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax_ol.legend()
        ax_ol.grid(True)

        # (0,1) Input vs LOAD per position (INC - LOAD)
        # Skip first up and first down bin (backlash outliers)
        ax_il = axes[0, 1]
        il_up_pos = [e['target_degrees'] for e in per_position if 'input_vs_load_up_degrees' in e]
        il_up_val = [e['input_vs_load_up_degrees'] for e in per_position if 'input_vs_load_up_degrees' in e]
        il_dn_pos = [e['target_degrees'] for e in per_position if 'input_vs_load_down_degrees' in e]
        il_dn_val = [e['input_vs_load_down_degrees'] for e in per_position if 'input_vs_load_down_degrees' in e]
        if len(il_up_pos) > 1:
            ax_il.plot(il_up_pos[1:], il_up_val[1:], 'o-', color='blue', label='Up', markersize=5)
        if len(il_dn_pos) > 1:
            ax_il.plot(il_dn_pos[1:], il_dn_val[1:], 's-', color='red', label='Down', markersize=5)
        il_all = il_up_val[1:] + il_dn_val[1:]
        if il_all:
            il_min, il_max = min(il_all), max(il_all)
            il_margin = (il_max - il_min) * 0.05 if il_max != il_min else 0.01
            ax_il.set_ylim(il_min - il_margin, il_max + il_margin)
        ax_il.set_xlabel('Target Position (deg)')
        ax_il.set_ylabel('Error (deg)')
        ax_il.set_title('Input Encoder vs LOAD: INC - LOAD')
        ax_il.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax_il.legend()
        ax_il.grid(True)

        # (1,0) Reversal error per position
        ax2 = axes[1, 0]
        if reversal_errors:
            rev_pos = [e['target_degrees'] for e in per_position if 'reversal_error_degrees' in e]
            rev_err = [e['reversal_error_degrees'] for e in per_position if 'reversal_error_degrees' in e]
            ax2.bar(rev_pos, rev_err, width=8, color='purple', alpha=0.7)
            ax2.axhline(y=mean_reversal, color='r', linestyle='--', label=f'Mean |B| = {mean_reversal:.4f} deg')
            ax2.axhline(y=-mean_reversal, color='r', linestyle='--')
        ax2.set_xlabel('Target Position (deg)')
        ax2.set_ylabel('Reversal Error (deg)')
        ax2.set_title('Reversal Error (directional backlash)')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax2.legend()
        ax2.grid(True)

        # (1,1) System summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        table_data = [
            ['Metric', 'Value'],
            ['Cycles', f'{num_cycles}'],
            ['Positions', f'{len(sorted_positions)}'],
            ['RMSE APS-LOAD', f'{encoder_pair_metrics.get("rmse_output_vs_load_degrees", 0):.4f} deg'],
            ['P95 |APS-LOAD|', f'{encoder_pair_metrics.get("p95_abs_output_vs_load_degrees", 0):.4f} deg'],
            ['Max |APS-LOAD|', f'{encoder_pair_metrics.get("max_abs_output_vs_load_degrees", 0):.4f} deg'],
            ['RMSE INC-LOAD', f'{encoder_pair_metrics.get("rmse_input_vs_load_degrees", 0):.4f} deg'],
            ['P95 |INC-LOAD|', f'{encoder_pair_metrics.get("p95_abs_input_vs_load_degrees", 0):.4f} deg'],
            ['Max |INC-LOAD|', f'{encoder_pair_metrics.get("max_abs_input_vs_load_degrees", 0):.4f} deg'],
            ['RMSE INC-APS', f'{encoder_pair_metrics.get("rmse_input_vs_output_degrees", 0):.4f} deg'],
            ['P95 |INC-APS|', f'{encoder_pair_metrics.get("p95_abs_input_vs_output_degrees", 0):.4f} deg'],
            ['Max |INC-APS|', f'{encoder_pair_metrics.get("max_abs_input_vs_output_degrees", 0):.4f} deg'],
            ['NL RMSE INC-APS (up)', f'{encoder_pair_metrics.get("nonlinearity_rmse_inc_aps_up_degrees", 0):.4f} deg'],
            ['NL P95 INC-APS (up)', f'{encoder_pair_metrics.get("nonlinearity_p95_inc_aps_up_degrees", 0):.4f} deg'],
            ['NL RMSE INC-APS (down)', f'{encoder_pair_metrics.get("nonlinearity_rmse_inc_aps_down_degrees", 0):.4f} deg'],
            ['NL P95 INC-APS (down)', f'{encoder_pair_metrics.get("nonlinearity_p95_inc_aps_down_degrees", 0):.4f} deg'],
            ['Max Repeatability', f'{max_repeatability:.4f} deg'],
        ]
        table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)
        for i in range(len(table_data[0])):
            table[0, i].set_facecolor('#4472C4')
            table[0, i].set_text_props(color='white', fontweight='bold')
        ax4.set_title('Positional Accuracy Summary', fontsize=13, fontweight='bold', pad=20)

        # (2,0) Repeatability per position
        ax3 = axes[2, 0]
        if repeatabilities_up or repeatabilities_down:
            width = 6
            if repeatabilities_up:
                rep_up_pos = [e['target_degrees'] for e in per_position if 'repeatability_up_degrees' in e]
                rep_up_val = [e['repeatability_up_degrees'] for e in per_position if 'repeatability_up_degrees' in e]
                ax3.bar([p - width/2 for p in rep_up_pos], rep_up_val, width=width, color='blue', alpha=0.6, label='Up (4*sigma)')
            if repeatabilities_down:
                rep_dn_pos = [e['target_degrees'] for e in per_position if 'repeatability_down_degrees' in e]
                rep_dn_val = [e['repeatability_down_degrees'] for e in per_position if 'repeatability_down_degrees' in e]
                ax3.bar([p + width/2 for p in rep_dn_pos], rep_dn_val, width=width, color='red', alpha=0.6, label='Down (4*sigma)')
            ax3.axhline(y=max_repeatability, color='orange', linestyle='--', label=f'Max R = {max_repeatability:.4f} deg')
        ax3.set_xlabel('Target Position (deg)')
        ax3.set_ylabel('Repeatability (deg)')
        ax3.set_title(f'Repeatability per Position ({num_cycles} cycles)')
        ax3.legend()
        ax3.grid(True)

        # (2,1) Input vs Output per position (INC - APS)
        # Skip first up and first down bin (backlash outliers)
        ax_io = axes[2, 1]
        io_up_pos = [e['target_degrees'] for e in per_position if 'input_vs_output_up_degrees' in e]
        io_up_val = [e['input_vs_output_up_degrees'] for e in per_position if 'input_vs_output_up_degrees' in e]
        io_dn_pos = [e['target_degrees'] for e in per_position if 'input_vs_output_down_degrees' in e]
        io_dn_val = [e['input_vs_output_down_degrees'] for e in per_position if 'input_vs_output_down_degrees' in e]
        if len(io_up_pos) > 1:
            ax_io.plot(io_up_pos[1:], io_up_val[1:], 'o-', color='blue', label='Up', markersize=5)
        if len(io_dn_pos) > 1:
            ax_io.plot(io_dn_pos[1:], io_dn_val[1:], 's-', color='red', label='Down', markersize=5)
        ax_io.set_xlabel('Target Position (deg)')
        ax_io.set_ylabel('Error (deg)')
        ax_io.set_title('Input vs Output Encoder: INC - APS (Raw FW)')
        ax_io.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax_io.legend()
        ax_io.grid(True)

        plt.tight_layout()
        save(fig, f'{log_folder}/POSITIONAL_ACCURACY_analysis.png')

        # -- Plot 2: Per-cycle time series with segmentation lines (first cycle) --
        bid_first, idx_s, idx_e = pos_acc_runs[0]
        time_plot = f['time'][idx_s:idx_e] - f['time'][idx_s]
        dut_cmd_plot = f['dut_position_command'][idx_s:idx_e]
        dut_out_plot = f['dut_output_position'][idx_s:idx_e]
        dut_inp_plot = dut_input_pos(f, idx_s, idx_e)
        load_plot = unwrapped(f['load_position'][idx_s:idx_e])

        if not np.all(np.isnan(dut_cmd_plot)):
            valid = ~np.isnan(dut_cmd_plot)
            if np.any(valid):
                offset = np.round(np.mean(dut_out_plot[valid] - dut_cmd_plot[valid]) / (2 * np.pi))
                dut_cmd_aligned_plot = dut_cmd_plot + offset * 2 * np.pi
            else:
                dut_cmd_aligned_plot = dut_cmd_plot
        else:
            dut_cmd_aligned_plot = dut_cmd_plot

        # Tare load and input using first settled bin (consistent with analysis)
        # Load tared to APS; INC tared to LOAD (so INC-LOAD starts at 0)
        plot_regions = find_step_holds(dut_cmd_aligned_plot, dut_out_plot)
        if plot_regions:
            pr_start, pr_end = plot_regions[0]
            pr_len = pr_end - pr_start
            pr_trim = int(pr_len * trim_pct)
            pr_s = slice(pr_start, pr_end - pr_trim)
            if pr_s.stop > pr_s.start:
                load_off = np.mean(dut_out_plot[pr_s]) - np.mean(load_plot[pr_s])
                inp_off = np.mean(load_plot[pr_s] + load_off) - np.mean(dut_inp_plot[pr_s])
            else:
                load_off = dut_out_plot[0] - load_plot[0]
                inp_off = (load_plot[0] + load_off) - dut_inp_plot[0]
        else:
            load_off = dut_out_plot[0] - load_plot[0]
            inp_off = (load_plot[0] + load_off) - dut_inp_plot[0]
        load_tared_plot = load_plot + load_off
        dut_inp_tared_plot = dut_inp_plot + inp_off

        # Five error signals
        output_tracking_err_plot = dut_out_plot - dut_cmd_aligned_plot       # APS - cmd
        input_tracking_err_plot = dut_inp_tared_plot - dut_cmd_aligned_plot  # INC - cmd
        output_vs_load_err_plot = dut_out_plot - load_tared_plot             # APS - LOAD
        input_vs_load_err_plot = dut_inp_tared_plot - load_tared_plot        # INC - LOAD
        input_vs_output_err_plot = dut_inp_plot - dut_out_plot               # INC - APS (raw)

        # Recompute step holds for cycle 0 to draw segmentation lines
        cycle0_regions = find_step_holds(dut_cmd_aligned_plot, dut_out_plot)

        fig2, axes2 = plt.subplots(4, 1, figsize=(16, 16))

        # Panel 1: Position time series
        ax_ts1 = axes2[0]
        if not np.all(np.isnan(dut_cmd_aligned_plot)):
            ax_ts1.plot(time_plot, dut_cmd_aligned_plot, 'b-', label='Command', linewidth=1.5, alpha=0.7)
        ax_ts1.plot(time_plot, dut_out_plot, 'r-', label='DUT Output (APS)', linewidth=1)
        ax_ts1.plot(time_plot, dut_inp_tared_plot, 'm-', label='DUT Input (INC, tared)', linewidth=1, alpha=0.7)
        ax_ts1.plot(time_plot, load_tared_plot, 'g-', label='Load (tared)', linewidth=1, alpha=0.7)

        # Draw analysis windows as shaded regions
        for start, end in cycle0_regions:
            region_length = end - start
            trim_end_samples = int(region_length * trim_pct)
            a_start = start
            a_end = end - trim_end_samples
            if a_end > a_start:
                t_start = time_plot[a_start]
                t_end = time_plot[min(a_end - 1, len(time_plot) - 1)]
                ax_ts1.axvspan(t_start, t_end, alpha=0.15, color='orange')
            ax_ts1.axvline(x=time_plot[start], color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            ax_ts1.axvline(x=time_plot[min(end - 1, len(time_plot) - 1)], color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        ax_ts1.set_ylabel('Position (rad)')
        ax_ts1.set_title('Cycle 0 Time Series (orange = analysis windows)')
        ax_ts1.legend()
        ax_ts1.grid(True)

        # Panel 2: Measured velocity (used for settle detection)
        ax_vel = axes2[1]
        meas_vel_plot = np.zeros(len(dut_out_plot))
        meas_vel_plot[1:] = np.abs(np.diff(dut_out_plot))
        ax_vel.semilogy(time_plot, meas_vel_plot, 'r-', label='|DUT Output Velocity| (APS)', linewidth=0.8, alpha=0.8)
        ax_vel.axhline(y=settle_vel_threshold, color='k', linestyle='--', linewidth=1, label=f'Settle threshold ({settle_vel_threshold:.0e} rad/sample)')

        for start, end in cycle0_regions:
            region_length = end - start
            trim_end_samples = int(region_length * trim_pct)
            a_start = start
            a_end = end - trim_end_samples
            if a_end > a_start:
                ax_vel.axvspan(time_plot[a_start], time_plot[min(a_end - 1, len(time_plot) - 1)], alpha=0.15, color='orange')

        ax_vel.set_ylabel('|Velocity| (rad/sample)')
        ax_vel.set_title('Measured Velocity — Settle Detection')
        ax_vel.legend()
        ax_vel.grid(True)

        # Panel 3: Tracking errors (each encoder vs command)
        ax_ts2 = axes2[2]
        ax_ts2.plot(time_plot, output_tracking_err_plot * 1000, 'r-', label='Output Tracking (APS - Cmd)', linewidth=1, alpha=0.8)
        ax_ts2.plot(time_plot, input_tracking_err_plot * 1000, 'm-', label='Input Tracking (INC - Cmd)', linewidth=1, alpha=0.8)

        for start, end in cycle0_regions:
            region_length = end - start
            trim_end_samples = int(region_length * trim_pct)
            a_start = start
            a_end = end - trim_end_samples
            if a_end > a_start:
                ax_ts2.axvspan(time_plot[a_start], time_plot[min(a_end - 1, len(time_plot) - 1)], alpha=0.15, color='orange')

        ts2_all = np.concatenate([output_tracking_err_plot * 1000, input_tracking_err_plot * 1000])
        ts2_min, ts2_max = np.nanmin(ts2_all), np.nanmax(ts2_all)
        ts2_margin = (ts2_max - ts2_min) * 0.05
        ax_ts2.set_ylim(ts2_min - ts2_margin, ts2_max + ts2_margin)
        ax_ts2.set_ylabel('Tracking Error (mrad)')
        ax_ts2.set_title('Tracking Errors — Encoder vs Command')
        ax_ts2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax_ts2.legend()
        ax_ts2.grid(True)

        # Panel 3: Encoder vs ground truth (LOAD) — INC-APS on secondary y-axis
        ax_ts3 = axes2[2]
        ax_ts3.plot(time_plot, output_vs_load_err_plot * 1000, 'r-', label='APS - LOAD (output encoder error)', linewidth=1, alpha=0.8)
        ax_ts3.plot(time_plot, input_vs_load_err_plot * 1000, 'm-', label='INC - LOAD (input encoder error)', linewidth=1, alpha=0.8)

        ax_ts3_r = ax_ts3.twinx()
        ax_ts3_r.plot(time_plot, input_vs_output_err_plot * 1000, 'c-', label='INC - APS (input vs output)', linewidth=1, alpha=0.8)

        for start, end in cycle0_regions:
            region_length = end - start
            trim_end_samples = int(region_length * trim_pct)
            a_start = start
            a_end = end - trim_end_samples
            if a_end > a_start:
                ax_ts3.axvspan(time_plot[a_start], time_plot[min(a_end - 1, len(time_plot) - 1)], alpha=0.15, color='orange')

        ts3_left = np.concatenate([output_vs_load_err_plot * 1000, input_vs_load_err_plot * 1000])
        ts3_l_min, ts3_l_max = np.nanmin(ts3_left), np.nanmax(ts3_left)
        ts3_l_margin = (ts3_l_max - ts3_l_min) * 0.05
        ax_ts3.set_ylim(ts3_l_min - ts3_l_margin, ts3_l_max + ts3_l_margin)

        ts3_r_data = input_vs_output_err_plot * 1000
        ts3_r_min, ts3_r_max = np.nanmin(ts3_r_data), np.nanmax(ts3_r_data)
        ts3_r_margin = (ts3_r_max - ts3_r_min) * 0.05
        ax_ts3_r.autoscale(enable=False)
        ax_ts3_r.set_ylim(ts3_r_min - ts3_r_margin, ts3_r_max + ts3_r_margin)
        ax_ts3_r.set_ylabel('INC - APS Error (mrad)', color='c')
        ax_ts3_r.tick_params(axis='y', labelcolor='c')

        ax_ts3.set_xlabel('Time (s)')
        ax_ts3.set_ylabel('Encoder Error vs LOAD (mrad)')
        ax_ts3.set_title('Encoder Comparison Errors')
        ax_ts3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        lines_l, labels_l = ax_ts3.get_legend_handles_labels()
        lines_r, labels_r = ax_ts3_r.get_legend_handles_labels()
        ax_ts3.legend(lines_l + lines_r, labels_l + labels_r)
        ax_ts3.grid(True)

        plt.tight_layout()
        save(fig2, f'{log_folder}/POSITIONAL_ACCURACY_cycle0_timeseries.png')

    # -------------------------------------------------------------------------
    # BACKLASH
    # -------------------------------------------------------------------------
    print('Processing Backlash')
    # post process any backlash behaviors in the log
    # per BPD ERD the behavior is run once in one location
    # per BPD ERD the backlash is reported in units of Arc-Min
    for ui, behavior_id in enumerate(f['behavior_ids']):
        behavior_id = behavior_id.decode("utf-8")
        if behavior_id[:8] == 'BACKLASH':
            minimum_measured_torque_ratio = 0.25 #Fraction of peak torque command that must be measured by the torque cell for a datapoint to be used for the linear fit

            idx_start = f['behavior_indices'][ui,0]
            idx_end = f['behavior_indices'][ui,1]

            # unwap encoder traces (these traces should be in output corodinates)
            input_position = unwrapped(f['dut_projected_input_encoder'][idx_start:idx_end]) #From self.devices.DUT.position , which is 
            output_position = unwrapped(f['load_position'][idx_start:idx_end])
            time = f['time'][idx_start:idx_end]


            # try to use the second encoder, if present (this is absent on the gearbox dyno)
            try:
                output_position_2 = unwrapped(f['load_position_2'][idx_start:idx_end])
                output_position = (output_position + output_position_2) / 2
                dual_encoders_used = True
                print('\tUsing dual encoders for backlash test')
            except:
                dual_encoders_used = False

            # zero out the average output position, and align the input position to it
            output_position -= np.mean(output_position)
            input_position -= np.mean(input_position - output_position)

            # load input torque and create deflection traces
            measured_torque = f['load_torque'][idx_start:idx_end]
            minimum_measured_torque = max(f['dut_torque_command'][idx_start:idx_end]) * minimum_measured_torque_ratio

            deflection = input_position - output_position
            deflection *= 3437.75

            # polyfit the left side of the deflection curve
            negative_torque_deflection = deflection[measured_torque < -1*minimum_measured_torque]
            negative_torque_torque = measured_torque[measured_torque < -1*minimum_measured_torque]
            negative_fit = np.poly1d(np.polyfit(negative_torque_torque, negative_torque_deflection, 1))
            negative_fit_line = [[min(measured_torque),-1*minimum_measured_torque],[negative_fit(min(measured_torque)), negative_fit(-1*minimum_measured_torque)]]

            # polyfit the right side of the deflection curve
            positive_torque_deflection = deflection[measured_torque > minimum_measured_torque]
            positive_torque_torque = measured_torque[measured_torque > minimum_measured_torque]
            positive_fit = np.poly1d(np.polyfit(positive_torque_torque, positive_torque_deflection, 1))
            positive_fit_line = [[minimum_measured_torque, max(measured_torque)],[positive_fit(minimum_measured_torque), positive_fit(max(measured_torque))]]

            backlash_arcmin = positive_fit(0) - negative_fit(0)
            backlash_arcsec = backlash_arcmin * 60

            print(f'  {behavior_id}:')
            print(f'    Backlash: {backlash_arcmin:.2f} arc-min = {backlash_arcsec:.2f} arc-sec')

            test_results.append({
                'unit_test_step_type': 'BACKLASH',
                'description': 'Measures mechanical backlash by comparing DUT input position after positive vs negative torque loading cycles. Results are averaged over two cycles.',
                'measurements': {
                    'backlash_arcmin': float(backlash_arcmin)
                },
                'start_time': hdf5_time_to_iso8601(time[0], base_timestamp),
                'end_time': hdf5_time_to_iso8601(time[-1], base_timestamp)
            })

            fig = plt.figure(figsize=(16, 10))

            plt.plot(deflection, 'r-', label='Measured Deflection', linewidth=1.5)
            plt.plot(measured_torque, 'b-', label='Measured Torque', linewidth=1.5)
            plt.ylabel('Deflection (arc-min)')
            plt.title('Backlash: Deflection Time Series')
            plt.grid(True)
            plt.legend()
            
            save(fig, f'{log_folder}/{behavior_id}_deflection_trace.png')

            fig = plt.figure(figsize=(16, 10))
            colors = ['green', 'orange', 'green', 'orange']

            plt.plot(measured_torque, deflection, 'k-', label='DUT Data Trace', linewidth=1.5)
            plt.plot(positive_fit_line[0], positive_fit_line[1], 'r-', label='Positive Torque Fit', linewidth=1.5)
            plt.plot(0, positive_fit(0), 'r.', label='Positive Intercept', linewidth=1.5)
            plt.plot(negative_fit_line[0], negative_fit_line[1], 'b-', label='Negative Torque Fit', linewidth=1.5)
            plt.plot(0, negative_fit(0), 'b.', label='Negative Intercept', linewidth=1.5)
            plt.xlabel('Measured Torque (Nm)')
            plt.ylabel('Deflection (arc-min)')
            plt.title('Backlash: Torque Deflection Plot')
            plt.grid(True)
            plt.legend()
            
            save(fig, f'{log_folder}/{behavior_id}_backlash_analysis.png')
            print(f'  Saved: {behavior_id}_backlash_analysis.png')

    # -------------------------------------------------------------------------
    # BACKLASH
    # -------------------------------------------------------------------------
    print('Processing Cogging')
    pole_pairs = 30
    for ui, behavior_id in enumerate(f['behavior_ids']):
        behavior_id = behavior_id.decode("utf-8")
        if behavior_id[:7] == 'COGGING':
            idx_start = f['behavior_indices'][ui,0]
            idx_end = f['behavior_indices'][ui,1]

            torque_command = f['dut_torque_command'][idx_start:idx_end]
            position = unwrapped(f['dut_output_position'][idx_start:idx_end]) #From self.devices.DUT.position , which is 
            torque = f['load_torque'][idx_start:idx_end] #From self.devices.DUT.position , which is 
            time = f['time'][idx_start:idx_end]

            unique_values, counts = np.unique(torque_command, return_counts=True)
            value_counts_dict = dict(zip(unique_values, counts))
            command_counts_dict = {k:v for (k,v) in value_counts_dict.items() if v > 100}
            setpoints = [float(k) for (k,v) in command_counts_dict.items()]

            x_lines = []
            y_lines = []

            for uj, setpoint in enumerate(setpoints):
                setpoint_torque = torque[torque_command == setpoint]
                setpoint_position = position[torque_command == setpoint]

                setpoint_torque = setpoint_torque[-12000:]
                setpoint_position = setpoint_position[-12000:]

                setpoint_position /= (2*np.pi/pole_pairs)
                setpoint_position -= 0.075
                setpoint_position = setpoint_position % 1

                sort_indices = np.argsort(setpoint_position)
                setpoint_torque = setpoint_torque[sort_indices]
                setpoint_position = setpoint_position[sort_indices]

                window_size = 500 
                window = np.ones(window_size) / window_size

                # Apply the moving average
                y_line = np.convolve(setpoint_torque, window, mode='valid')

                # Adjust x to match the shrunken y array from 'valid' convolution
                x_line = setpoint_position[(window_size-1)//2 : -(window_size-1)//2]

                x_lines.append(x_line)
                y_lines.append(y_line)

                fig = plt.figure(figsize=(16, 10))

                plt.scatter(setpoint_position, setpoint_torque, label='Measured Deflection', linewidth=1.5)
                plt.plot(x_line, y_line, color='r')
                plt.ylabel('Measured Torque (Nm)')
                plt.title('Electrical Phase')
                plt.grid(True)
                plt.legend()
                
                save(fig, log_folder+'/'+str(uj)+'_deflection_trace.png')

            
            fig = plt.figure(figsize=(16, 10))
            for uj in range(len(x_lines)):
                y_lines[uj] -= np.mean(y_lines[uj])
                plt.plot(x_lines[uj], y_lines[uj],label=str(np.round(setpoints[uj]))+' Nm, torque ripple')
                plt.ylabel('Measured Torque (Nm)')
                plt.title('Electrical Phase')
                plt.grid(True)
                plt.legend()

            # for uj in range(1,3):
            #     y_lines[uj] -= y_lines[0]
            #     plt.plot(x_lines[uj], y_lines[uj],label=str(np.round(setpoints[uj]))+' Nm, torque ripple delta vs. 0 Nm')
            #     plt.ylabel('Measured Torque (Nm)')
            #     plt.title('Electrical Phase')
            #     plt.grid(True)
            #     plt.legend()
            
            save(fig, log_folder+'/ripple.png')



    # -------------------------------------------------------------------------
    # EFFICIENCY
    # -------------------------------------------------------------------------
    print('Calculating Efficiency')
    runs = []
    for ui, behavior_id in enumerate(f['behavior_ids']):
        behavior_id = behavior_id.decode("utf-8")
        if behavior_id.split('-')[0] == 'EFFICIENCY':
            run_id = behavior_id.split('-')[1]
            if run_id not in runs:
                runs.append(run_id)

    for run in runs:
        setpoints = []
        for i, behavior_id in enumerate(f['behavior_ids']):
            behavior_id = behavior_id.decode("utf-8")
            if behavior_id.split('-')[0] == 'EFFICIENCY' and behavior_id.split('-')[1] == run:
                setpoints.append((behavior_id, f['behavior_indices'][i, 0], f['behavior_indices'][i, 1]))

        torque_cmds = []
        velocity_cmds = []
        for setpoint in setpoints:
            print('Setpoints: ', setpoint)
            tc = int(np.round(np.mean(f['load_torque_command'][setpoint[1]:setpoint[2]])))
            vc = int(np.round(np.mean(f['dut_velocity_command'][setpoint[1]:setpoint[2]])))
            if tc not in torque_cmds:
                torque_cmds.append(tc)
            if vc not in velocity_cmds:
                velocity_cmds.append(vc)

        print('Torque commands: ', torque_cmds)
        print('Velocity commands: ', velocity_cmds)

        velocity_arr = np.zeros((len(velocity_cmds), len(torque_cmds)))
        torque_arr = np.zeros((len(velocity_cmds), len(torque_cmds)))
        efficiency_arr = np.zeros((len(velocity_cmds), len(torque_cmds)))

        i = 0
        for ui, velocity_cmd in enumerate(velocity_cmds):
            for uj, torque_cmd in enumerate(torque_cmds):
                setpoint = setpoints[i]
                print(ui, velocity_cmd, uj, torque_cmd)

                input_torque = np.mean(f['dut_effort'][setpoint[1]:setpoint[2]])
                output_torque = np.mean(f['load_torque'][setpoint[1]:setpoint[2]])
                input_velocity = np.mean(f['dut_velocity'][setpoint[1]:setpoint[2]])
                output_velocity = np.mean(f['load_velocity'][setpoint[1]:setpoint[2]])
                input_velocity_command = np.mean(f['dut_velocity_command'][setpoint[1]:setpoint[2]])
                output_torque_command = np.mean(f['load_torque_command'][setpoint[1]:setpoint[2]])

                backdriving = np.sign(input_velocity * output_torque) == -1
                if backdriving:
                    efficiency = input_torque * g / output_torque
                else:
                    efficiency = output_torque / (input_torque * g)

                if efficiency < 0:
                    efficiency = 0

                torque_arr[ui, uj] = output_torque_command
                velocity_arr[ui, uj] = input_velocity_command
                efficiency_arr[ui, uj] = efficiency
                print(efficiency_arr)
                i += 1

        measurements = {}
        for ui, velocity_cmd in enumerate(velocity_cmds):
            velocity_key = f"velocity_{int(velocity_cmd)}_rad_s"
            measurements[velocity_key] = {}
            for uj, torque_cmd in enumerate(torque_cmds):
                torque_key = f"torque_{int(-1 * torque_cmd)}_Nm"
                measurements[velocity_key][torque_key] = float(efficiency_arr[ui, uj])

        test_results.append({
            'unit_test_step_type': f'EFFICIENCY-{run}',
            'description': f'Measures transmission efficiency across a swept grid of input velocity and output torque setpoints. Efficiency is computed as output power / input power, with backdrive direction handled separately.',
            'measurements': measurements,
            'start_time': hdf5_time_to_iso8601(f['time'][setpoints[0][1]], base_timestamp),
            'end_time': hdf5_time_to_iso8601(f['time'][setpoints[-1][2]-1], base_timestamp)
        })

        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(111)
        cax = ax.matshow(efficiency_arr, interpolation='nearest')
        for ui in range(len(velocity_cmds)):
            for uj in range(len(torque_cmds)):
                ax.text(uj, ui, str(np.round(efficiency_arr[ui, uj] * 100, 1)) + ' %',
                        fontdict={'fontsize': 7}, va='center', ha='center')
        ax.set_xticks(np.arange(len(torque_cmds)))
        ax.set_yticks(np.arange(len(velocity_cmds)))
        ax.set_xticklabels([str(-1 * cmd) for cmd in torque_cmds])
        ax.set_yticklabels([str(cmd) for cmd in velocity_cmds])
        plt.xlabel('Output Torque (nm) negative torque = backdriving')
        plt.ylabel('Input Velocity (rad/s)')
        plt.title('efficiency test, ' + run)
        save(fig, f'{log_folder}/EFFIENCY-{run}.png')

    # -------------------------------------------------------------------------
    # PEAK EFFORT
    # -------------------------------------------------------------------------
    print('Processing PEAK EFFORT')
    for ui, behavior_id in enumerate(f['behavior_ids']):
        behavior_id = behavior_id.decode("utf-8")
        if behavior_id[:11] == 'PEAK_EFFORT':
            idx_start = f['behavior_indices'][ui, 0]
            idx_end = f['behavior_indices'][ui, 1]

            time = f['time'][idx_start:idx_end] - f['time'][idx_start]
            dut_effort = f['dut_effort'][idx_start:idx_end]
            load_torque = f['load_torque'][idx_start:idx_end]

            absolute_error = load_torque - dut_effort
            percent_error = np.zeros_like(absolute_error)
            nonzero_mask = np.abs(dut_effort) > 0.01
            percent_error[nonzero_mask] = 100 * absolute_error[nonzero_mask] / np.abs(dut_effort[nonzero_mask])

            mean_abs_error = np.mean(np.abs(absolute_error))
            rms_error = np.sqrt(np.mean(absolute_error**2))
            max_error = np.max(np.abs(absolute_error))
            std_error = np.std(absolute_error)
            mean_percent_error = np.mean(np.abs(percent_error[nonzero_mask])) if np.any(nonzero_mask) else 0
            error_bounds_95 = 1.96 * std_error

            print(f'  {behavior_id}:')
            print(f'    Mean Absolute Error: {mean_abs_error:.4f} Nm')
            print(f'    RMS Error: {rms_error:.4f} Nm')
            print(f'    Max Error: {max_error:.4f} Nm')
            print(f'    Std Dev: {std_error:.4f} Nm')
            print(f'    95% Error Bounds: ±{error_bounds_95:.4f} Nm')
            print(f'    Mean Percent Error: {mean_percent_error:.2f}%')

            test_results.append({
                'unit_test_step_type': 'PEAK_EFFORT',
                'description': 'Measures peak torque output accuracy by comparing DUT commanded effort against load-measured torque. Reports peak values and tracking error statistics.',
                'measurements': {
                    'maximum_peak_Nm': float(np.max(load_torque)),
                    'minimum_peak_Nm': float(np.min(load_torque)),
                    'mean_absolute_error_Nm': float(mean_abs_error),
                    'rms_error_Nm': float(rms_error)
                },
                'start_time': hdf5_time_to_iso8601(time[0], base_timestamp),
                'end_time': hdf5_time_to_iso8601(time[-1], base_timestamp)
            })

            fig = plt.figure(figsize=(16, 10))

            plt.subplot(3, 2, 1)
            plt.plot(time, dut_effort, 'b-', label='DUT Effort (Commanded)', linewidth=1.5, alpha=0.7)
            plt.plot(time, load_torque, 'r-', label='Load Torque (Measured)', linewidth=1)
            plt.fill_between(time, dut_effort - error_bounds_95, dut_effort + error_bounds_95,
                             alpha=0.2, color='blue', label=f'95% Error Bounds (±{error_bounds_95:.3f} Nm)')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (Nm)')
            plt.title('Commanded vs Measured Torque')
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 2, 2)
            plt.plot(time, absolute_error, 'g-', linewidth=1)
            plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            plt.axhline(y=mean_abs_error, color='r', linestyle='--', label=f'Mean: {mean_abs_error:.3f} Nm', linewidth=1.5)
            plt.axhline(y=-mean_abs_error, color='r', linestyle='--', linewidth=1.5)
            plt.axhline(y=error_bounds_95, color='orange', linestyle='--', label=f'95% CI: ±{error_bounds_95:.3f} Nm', linewidth=1.5)
            plt.axhline(y=-error_bounds_95, color='orange', linestyle='--', linewidth=1.5)
            plt.xlabel('Time (s)')
            plt.ylabel('Error (Nm)')
            plt.title(f'Tracking Error (RMS: {rms_error:.4f} Nm)')
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 2, 3)
            plt.scatter(dut_effort, load_torque, s=10, alpha=0.5, label='Measured')
            torque_range = [np.min(dut_effort), np.max(dut_effort)]
            plt.plot(torque_range, torque_range, 'k--', linewidth=2, label='Ideal (1:1)')
            plt.plot(torque_range, [t + np.mean(absolute_error) for t in torque_range],
                     'r--', linewidth=1.5, label=f'Mean Offset: {np.mean(absolute_error):.3f} Nm')
            plt.xlabel('Commanded Torque (Nm)')
            plt.ylabel('Measured Torque (Nm)')
            plt.title('Torque Tracking')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')

            plt.subplot(3, 2, 4)
            plt.scatter(dut_effort, absolute_error, s=10, alpha=0.5)
            plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            plt.axhline(y=mean_abs_error, color='r', linestyle='--', linewidth=1.5)
            plt.axhline(y=-mean_abs_error, color='r', linestyle='--', linewidth=1.5)
            plt.xlabel('Commanded Torque (Nm)')
            plt.ylabel('Error (Nm)')
            plt.title('Error vs Command Level')
            plt.grid(True)

            plt.tight_layout()
            save(fig, f'{log_folder}/{behavior_id}_peak_effort.png')
            print(f'  Saved: {behavior_id}_peak_effort.png')

    # -------------------------------------------------------------------------
    # CONSTANT TORQUE (Encoder Analysis)
    # -------------------------------------------------------------------------
    print('Processing Encoder Analysis (CONSTANT_TORQUE)')
    for ui, behavior_id in enumerate(f['behavior_ids']):
        behavior_id = behavior_id.decode("utf-8")
        if behavior_id[:15] == 'CONSTANT_TORQUE':
            idx_start = f['behavior_indices'][ui, 0]
            idx_end = f['behavior_indices'][ui, 1]

            print(f'\nProcessing {behavior_id}...')

            if 'time' in f.keys():
                time_data = f['time'][idx_start:idx_end]
                sample_rate = 1.0 / np.mean(np.diff(time_data)) if len(time_data) > 1 else 1000.0
            else:
                sample_rate = 1000.0

            skip_samples = int(args.skip_start * sample_rate)
            total_samples = idx_end - idx_start

            if skip_samples >= total_samples:
                print(f'ERROR: Skip duration ({args.skip_start}s = {skip_samples} samples) exceeds total samples ({total_samples})')
                continue

            print(f'Skipping first {args.skip_start}s ({skip_samples} samples) for constant velocity analysis')
            print(f'Analyzing samples {skip_samples}-{total_samples} ({total_samples - skip_samples} samples)')

            idx_start_adjusted = idx_start + skip_samples

            load_position_full = unwrapped(f['load_position'][idx_start_adjusted:idx_end])
            dut_aps_position_full = unwrapped(f['dut_output_position'][idx_start_adjusted:idx_end])
            dut_inc_position_full = unwrapped(dut_input_pos(f, idx_start_adjusted, idx_end))

            aps_saturation_threshold = -15.75
            raw_aps_data = f['dut_output_position'][idx_start_adjusted:idx_end]
            valid_indices = raw_aps_data > aps_saturation_threshold

            if np.sum(valid_indices) < len(valid_indices):
                saturation_sample = np.where(~valid_indices)[0][0] if len(np.where(~valid_indices)[0]) > 0 else len(valid_indices)
                print(f'Warning: APS saturation detected. Analyzing {np.sum(valid_indices)}/{len(valid_indices)} samples before saturation (sample {saturation_sample}).')
                load_position_aps = load_position_full[valid_indices]
                dut_aps_position = dut_aps_position_full[valid_indices]
                dut_inc_position_aps = dut_inc_position_full[valid_indices]
            else:
                load_position_aps = load_position_full
                dut_aps_position = dut_aps_position_full
                dut_inc_position_aps = dut_inc_position_full

            load_position_inc = load_position_full
            dut_inc_position_inc = dut_inc_position_full

            rotations_aps = load_position_aps / (2 * np.pi)
            rotations_inc = load_position_inc / (2 * np.pi)

            aps_error = load_position_aps - dut_aps_position
            inc_error = load_position_inc - dut_inc_position_inc
            inc_aps_error = dut_inc_position_aps - dut_aps_position

            # Row 1: Linear fits
            A_aps = np.vstack([load_position_aps, np.ones(len(load_position_aps))]).T
            m_aps, c_aps = np.linalg.lstsq(A_aps, aps_error, rcond=None)[0]
            aps_fitted_error = m_aps * load_position_aps + c_aps
            aps_detrended = aps_error - aps_fitted_error

            A_inc = np.vstack([load_position_inc, np.ones(len(load_position_inc))]).T
            m_inc, c_inc = np.linalg.lstsq(A_inc, inc_error, rcond=None)[0]
            inc_fitted_error = m_inc * load_position_inc + c_inc
            inc_detrended = inc_error - inc_fitted_error

            A_inc_aps = np.vstack([load_position_aps, np.ones(len(load_position_aps))]).T
            m_inc_aps, c_inc_aps = np.linalg.lstsq(A_inc_aps, inc_aps_error, rcond=None)[0]
            inc_aps_fitted_error = m_inc_aps * load_position_aps + c_inc_aps
            inc_aps_detrended = inc_aps_error - inc_aps_fitted_error

            # Row 3: Corrections
            coeffs_inc = np.polyfit(dut_inc_position_inc, load_position_inc, 1)
            gear_ratio_correction = coeffs_inc[0]
            offset_inc = coeffs_inc[1]
            inc_pos_corrected_full = dut_inc_position_inc * gear_ratio_correction + offset_inc
            inc_pos_corrected_aps = dut_inc_position_aps * gear_ratio_correction + offset_inc
            slope_inc = gear_ratio_correction - 1.0

            angle_aps = np.mod(dut_aps_position, 2 * np.pi)
            X_fourier_aps = np.column_stack([
                np.ones_like(angle_aps),
                np.sin(angle_aps), np.cos(angle_aps),
                np.sin(2 * angle_aps), np.cos(2 * angle_aps),
            ])

            error_load_aps = load_position_aps - dut_aps_position
            fourier_coeffs_aps_load = np.linalg.lstsq(X_fourier_aps, error_load_aps, rcond=None)[0]
            offset_ecc_aps_load, A1_aps_load, B1_aps_load, A2_aps_load, B2_aps_load = fourier_coeffs_aps_load
            amp1_aps_load = np.sqrt(A1_aps_load**2 + B1_aps_load**2)
            phase1_aps_load = np.arctan2(B1_aps_load, A1_aps_load)
            amp2_aps_load = np.sqrt(A2_aps_load**2 + B2_aps_load**2)
            phase2_aps_load = np.arctan2(B2_aps_load, A2_aps_load)
            aps_pos_corrected_load = dut_aps_position + X_fourier_aps @ fourier_coeffs_aps_load

            error_inc_aps = inc_pos_corrected_aps - dut_aps_position
            fourier_coeffs_aps_inc = np.linalg.lstsq(X_fourier_aps, error_inc_aps, rcond=None)[0]
            offset_ecc_aps_inc, A1_aps_inc, B1_aps_inc, A2_aps_inc, B2_aps_inc = fourier_coeffs_aps_inc
            amp1_aps_inc = np.sqrt(A1_aps_inc**2 + B1_aps_inc**2)
            phase1_aps_inc = np.arctan2(B1_aps_inc, A1_aps_inc)
            amp2_aps_inc = np.sqrt(A2_aps_inc**2 + B2_aps_inc**2)
            phase2_aps_inc = np.arctan2(B2_aps_inc, A2_aps_inc)
            aps_pos_corrected_inc = dut_aps_position + X_fourier_aps @ fourier_coeffs_aps_inc

            error_load_aps_final = load_position_aps - aps_pos_corrected_load
            error_load_inc_final = load_position_inc - inc_pos_corrected_full
            error_inc_aps_final = inc_pos_corrected_aps - aps_pos_corrected_inc

            # Print corrections
            print(f'\n{"="*80}')
            print(f'CORRECTION 1: INC Encoder (Gear Ratio - Linear)')
            print(f'{"="*80}')
            print(f'Transform: INC → INC* = INC * {gear_ratio_correction:.9f} + {offset_inc:.6f}')
            print(f'  Fitted drift slope: {slope_inc:.9f} rad/rad ({slope_inc*1e6:.3f} ppm)')
            print(f'  Fitted offset:      {offset_inc:.6f} rad ({np.rad2deg(offset_inc):.3f} deg, {offset_inc*1000:.3f} mrad)')

            print(f'\n{"="*80}')
            print(f'CORRECTION 2A: APS Encoder (Eccentricity vs Load)')
            print(f'{"="*80}')
            print(f'  DC offset:              {offset_ecc_aps_load:.6f} rad')
            print(f'  1st harmonic amplitude: {amp1_aps_load:.6f} rad ({amp1_aps_load*1000:.3f} mrad)')
            print(f'  1st harmonic phase:     {phase1_aps_load:.3f} rad ({np.rad2deg(phase1_aps_load):.1f} deg)')
            print(f'  2nd harmonic amplitude: {amp2_aps_load:.6f} rad ({amp2_aps_load*1000:.3f} mrad)')
            print(f'  2nd harmonic phase:     {phase2_aps_load:.3f} rad ({np.rad2deg(phase2_aps_load):.1f} deg)')

            print(f'\n{"="*80}')
            print(f'CORRECTION 2B: APS Encoder (Eccentricity vs INC*)')
            print(f'{"="*80}')
            print(f'  DC offset:              {offset_ecc_aps_inc:.6f} rad')
            print(f'  1st harmonic amplitude: {amp1_aps_inc:.6f} rad ({amp1_aps_inc*1000:.3f} mrad)')
            print(f'  1st harmonic phase:     {phase1_aps_inc:.3f} rad ({np.rad2deg(phase1_aps_inc):.1f} deg)')
            print(f'  2nd harmonic amplitude: {amp2_aps_inc:.6f} rad ({amp2_aps_inc*1000:.3f} mrad)')
            print(f'  2nd harmonic phase:     {phase2_aps_inc:.3f} rad ({np.rad2deg(phase2_aps_inc):.1f} deg)')

            print(f'\n{"="*80}')
            print(f'APS PARAMETER COMPARISON (Validation)')
            print(f'{"="*80}')
            print(f'  Amplitude difference: {abs(amp1_aps_load - amp1_aps_inc)*1000:.3f} mrad')
            print(f'  Phase difference:     {abs(phase1_aps_load - phase1_aps_inc):.3f} rad ({abs(np.rad2deg(phase1_aps_load - phase1_aps_inc)):.1f} deg)')
            if abs(amp1_aps_load - amp1_aps_inc) * 1000 < 0.5:
                print(f'  ✓ Parameters match well! APS eccentricity is consistent.')
            else:
                print(f'  ⚠ Parameters differ. May indicate reference-dependent errors.')

            print(f'\n{"="*80}')
            print(f'COMPARISON 1: Load vs APS*  |  Original: {np.std(aps_error)*1000:.3f} mrad  →  Corrected: {np.std(error_load_aps_final)*1000:.3f} mrad  ({(1 - np.std(error_load_aps_final)/np.std(aps_error))*100:.1f}% improvement)')
            print(f'COMPARISON 2: INC* vs APS*  |  Original: {np.std(inc_aps_error)*1000:.3f} mrad  →  Corrected: {np.std(error_inc_aps_final)*1000:.3f} mrad  ({(1 - np.std(error_inc_aps_final)/np.std(inc_aps_error))*100:.1f}% improvement)')
            print(f'COMPARISON 3: Load vs INC*  |  Original: {np.std(inc_error)*1000:.3f} mrad  →  Corrected: {np.std(error_load_inc_final)*1000:.3f} mrad  ({(1 - np.std(error_load_inc_final)/np.std(inc_error))*100:.1f}% improvement)')

            # 3x3 plot
            fig_compare, axes = plt.subplots(3, 3, figsize=(20, 18))
            ax1, ax2, ax3 = axes[0, :]
            ax4, ax5, ax6 = axes[1, :]
            ax7, ax8, ax9 = axes[2, :]

            ax1.plot(rotations_inc, inc_error * 1000, 'r-', linewidth=0.5, alpha=0.8)
            ax1.plot(rotations_inc, inc_fitted_error * 1000, 'k-', linewidth=2, label='Linear Fit')
            ax1.set_ylabel('Position Error (mrad)')
            ax1.set_title(f'INC Error (Load - INC)\nσ={np.std(inc_error)*1000:.2f} mrad ({np.rad2deg(np.std(inc_error)):.3f}°)')
            ax1.text(0.02, 0.98, f'Slope: {m_inc*1e6:.1f} ppm\nOffset: {c_inc*1000:.1f} mrad ({np.rad2deg(c_inc):.3f}°)',
                     transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
            ax1.legend()
            ax1.grid(True)

            ax2.plot(rotations_aps, aps_error * 1000, 'b-', linewidth=0.5, alpha=0.8)
            ax2.plot(rotations_aps, aps_fitted_error * 1000, 'r-', linewidth=2, label='Linear Fit')
            ax2.set_title(f'APS Error (Load - APS)\nσ={np.std(aps_error)*1000:.2f} mrad ({np.rad2deg(np.std(aps_error)):.3f}°)')
            ax2.text(0.02, 0.98, f'Slope: {m_aps*1e6:.1f} ppm\nOffset: {c_aps*1000:.1f} mrad ({np.rad2deg(c_aps):.3f}°)',
                     transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
            ax2.legend()
            ax2.grid(True)

            ax3.plot(rotations_aps, inc_aps_error * 1000, 'g-', linewidth=0.5, alpha=0.8)
            ax3.plot(rotations_aps, inc_aps_fitted_error * 1000, 'k-', linewidth=2, label='Linear Fit')
            ax3.set_title(f'INC-APS Error (INC - APS)\nσ={np.std(inc_aps_error)*1000:.2f} mrad ({np.rad2deg(np.std(inc_aps_error)):.3f}°)')
            ax3.text(0.02, 0.98, f'Slope: {m_inc_aps*1e6:.1f} ppm\nOffset: {c_inc_aps*1000:.1f} mrad ({np.rad2deg(c_inc_aps):.3f}°)',
                     transform=ax3.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
            ax3.legend()
            ax3.grid(True)

            ax4.plot(rotations_inc, inc_detrended * 1000, 'r-', linewidth=0.5, alpha=0.8)
            ax4.set_ylabel('Detrended Error (mrad)')
            ax4.set_title(f'INC Detrended\nσ={np.std(inc_detrended)*1000:.2f} mrad ({np.rad2deg(np.std(inc_detrended)):.3f}°)')
            ax4.grid(True)

            ax5.plot(rotations_aps, aps_detrended * 1000, 'b-', linewidth=0.5, alpha=0.8)
            ax5.set_title(f'APS Detrended\nσ={np.std(aps_detrended)*1000:.2f} mrad ({np.rad2deg(np.std(aps_detrended)):.3f}°)')
            ax5.grid(True)

            ax6.plot(rotations_aps, inc_aps_detrended * 1000, 'g-', linewidth=0.5, alpha=0.8)
            ax6.set_title(f'INC-APS Detrended\nσ={np.std(inc_aps_detrended)*1000:.2f} mrad ({np.rad2deg(np.std(inc_aps_detrended)):.3f}°)')
            ax6.grid(True)

            error_load_inc_final_demean = error_load_inc_final - np.mean(error_load_inc_final)
            ax7.plot(rotations_inc, error_load_inc_final_demean * 1000, 'r-', linewidth=0.5, alpha=0.8)
            ax7.set_xlabel('Rotations')
            ax7.set_ylabel('Detrended Error (mrad)')
            ax7.set_title(f'Load - INC* (de-meaned)\nσ={np.std(error_load_inc_final_demean)*1000:.2f} mrad ({np.rad2deg(np.std(error_load_inc_final_demean)):.3f}°)')
            ax7.text(0.02, 0.98, f'Load: no correction\nINC gear ratio: {gear_ratio_correction:.6f} ({(gear_ratio_correction-1)*1e6:.0f} ppm)',
                     transform=ax7.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=8)
            ax7.grid(True)
            ax7.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)

            error_load_aps_final_demean = error_load_aps_final - np.mean(error_load_aps_final)
            ax8.plot(rotations_aps, error_load_aps_final_demean * 1000, 'b-', linewidth=0.5, alpha=0.8)
            ax8.set_xlabel('Rotations')
            ax8.set_title(f'Load - APS* (de-meaned)\nσ={np.std(error_load_aps_final_demean)*1000:.2f} mrad ({np.rad2deg(np.std(error_load_aps_final_demean)):.3f}°)')
            ax8.text(0.02, 0.98, f'Load: no correction\nAPS ecc. 1st harmonic: {amp1_aps_load*1000:.2f} mrad ({np.rad2deg(amp1_aps_load):.3f}°)',
                     transform=ax8.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=8)
            ax8.grid(True)
            ax8.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)

            error_inc_aps_final_demean = error_inc_aps_final - np.mean(error_inc_aps_final)
            ax9.plot(rotations_aps, error_inc_aps_final_demean * 1000, 'g-', linewidth=0.5, alpha=0.8)
            ax9.set_xlabel('Rotations')
            ax9.set_title(f'INC* - APS* (de-meaned)\nσ={np.std(error_inc_aps_final_demean)*1000:.2f} mrad ({np.rad2deg(np.std(error_inc_aps_final_demean)):.3f}°)')
            ax9.text(0.02, 0.98, f'INC gear ratio: {gear_ratio_correction:.6f} ({(gear_ratio_correction-1)*1e6:.0f} ppm)\nAPS ecc. 1st harmonic: {amp1_aps_inc*1000:.2f} mrad ({np.rad2deg(amp1_aps_inc):.3f}°)',
                     transform=ax9.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=8)
            ax9.grid(True)
            ax9.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)

            fig_compare.suptitle(f'Encoder Analysis with Corrections - {behavior_id}', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0)
            save(fig_compare, f'{log_folder}/{behavior_id}_encoder_analysis_3x3.png')

    # -------------------------------------------------------------------------
    # Write consolidated JSON results
    # -------------------------------------------------------------------------
    if test_results:
        output_path = write_test_results_to_file(test_results, test_folder_name)
        print(f'\nTest results written to: {output_path}')

print('\nPost processing complete!')
