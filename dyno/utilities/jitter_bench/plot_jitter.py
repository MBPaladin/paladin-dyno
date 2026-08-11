"""Regenerate the EtherCAT jitter-investigation figures (July 2026 bench work).

Figure 1 (jitter_compare.png): cycle-period histogram + period-error timeline
    comparing pacing strategies, measured by the ELM3004's DC hardware
    timestamps where available.
Figure 2 (adc_sync_compare.png): master cycle timing vs. the ADC's
    self-reported invalid-data flag (TxPDO State bit) — the 'ADC out of sync'
    failure under old pacing, and its absence under patched pacing.

Data: .npz captures recorded by the bench testbed scripts, one file per run,
with arrays send_t (master send timestamps, ns), dc_t (slave DC timestamps,
ns), wkc (working counter), and for sync runs status (per-channel PAI status
bytes) — plus a json 'meta' blob.

Usage (Windows, needs numpy + matplotlib — anaconda base has both):
    "C:\\Users\\Nathan Justus\\anaconda3\\python.exe" plot_jitter.py
Run from anywhere; data files are located relative to this script. Figures
are written next to the script. Pass --show to open interactive windows.
"""
import argparse
import json
import os

import matplotlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Each entry lists candidate files, first existing wins: fresh captures from
# collect_jitter.py take priority over the original July 2026 60s captures.
JITTER_RUNS = [  # (files, label) for figure 1
    (['fiveMinTrial_baseline.npz', 'baseline_60s.npz'], 'baseline: sleep pacing (production-style)'),
    #(['hybrid_60s.npz'], 'hybrid sleep+spin'),
    #(['spin_prio2_60s.npz'], 'spin + high prio + GC off'),
    (['fiveMinTrial_fixed.npz', 'spin_rxlong_60s.npz'], 'spin + high priority + GC off, 5ms rx timeout'),
]
SYNC_RUNS = [  # (files, label, color) for figure 2
    (['baseline.npz', 'sync_bad.npz'], 'old pacing', 'tab:red'),
    (['fixed.npz', 'sync_fixed.npz'], 'patched pacing', 'tab:blue'),
]
SETTLING_CYCLES = 2300  # ELM3004 power-on settling window to exclude


def load(candidates):
    if isinstance(candidates, str):
        candidates = [candidates]
    for name in candidates:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            return z, json.loads(str(z['meta']))
    raise FileNotFoundError(f'none of {candidates} found in {HERE}')


def fig_jitter_compare(plt):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for name, label in JITTER_RUNS:
        z, meta = load(name)
        src = z['dc_t'] if not meta.get('no_bus', False) else z['send_t']
        d = np.diff(src).astype(np.float64) / 1000.0
        d = d[d > 0]  # drop cycles where dc_time did not update (lost frame)
        bins = np.linspace(0, max(2500, d.max() * 1.05), 400)
        axes[0].hist(d, bins=bins, histtype='step', label=label)
        axes[1].plot(np.arange(len(d)) / 1000.0, d - 1000.0, lw=0.3, label=label)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('cycle period (us)')
    axes[0].set_ylabel('count')
    axes[0].axvline(1000, color='k', ls='--', lw=0.8)
    axes[0].legend(fontsize=8)
    axes[0].set_title('Cycle-period distribution (slave DC hardware timestamps)')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('period error (us)')
    axes[1].legend(fontsize=8)
    axes[1].set_title('Period error over time')
    fig.tight_layout()
    out = os.path.join(HERE, 'jitter_compare.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)


def fig_adc_sync_compare(plt):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for name, label, color in SYNC_RUNS:
        z, _ = load(name)
        st = z['status'][:, 0]
        send = z['send_t']
        t = (send - send[0]) / 1e9
        inv = ((st >> 5) & 1).astype(float)  # TxPDO State bit: 1 = data invalid
        inv[:SETTLING_CYCLES] = 0
        axes[0].plot(t[1:], np.diff(send) / 1000.0 - 1000.0, lw=0.3,
                     label=label, color=color, alpha=0.7)
        axes[1].fill_between(t, 0, inv, step='mid', alpha=0.6,
                             label=label, color=color)
    axes[0].set_ylabel('cycle period error (us)')
    axes[0].set_ylim(-900, 2000)
    axes[0].legend()
    axes[0].set_title('Master cycle timing (top) and ADC self-reported '
                      'invalid data (bottom)')
    axes[1].set_ylabel('ADC data invalid')
    axes[1].set_yticks([0, 1])
    axes[1].set_xlabel('time (s)')
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(HERE, 'adc_sync_compare.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--show', action='store_true', help='open windows instead of only saving PNGs')
    args = p.parse_args()
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_jitter_compare(plt)
    fig_adc_sync_compare(plt)
    if args.show:
        plt.show()
