"""Figure helpers, so every plot in the pack reads the same way.

Deliberately small. The customer asked for raw data plus quick plots, not a
house style -- what matters is that a figure says which unit, which direction
and which units it is in, because these packs are compared side by side across
three drives.
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker                 # noqa: E402,F401

FORWARD_C = '#1f77b4'
BACKDRIVE_C = '#d62728'
GRID = dict(alpha=0.3, linewidth=0.6)

# One colour per direction, everywhere.
DIR_COLOR = {'forward': FORWARD_C, 'backdrive': BACKDRIVE_C}
DIR_LABEL = {'forward': 'forward driving', 'backdrive': 'back-driving'}


def figure(title, subtitle='', size=(9, 5.5)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title + (f'\n{subtitle}' if subtitle else ''), fontsize=11)
    ax.grid(True, **GRID)
    return fig, ax


def grid_figure(nrows, ncols, title, subtitle='', size=None):
    # A single-column grid gets the full width of a normal figure: the panel
    # titles on these carry the measured result ("slip at -50.9 Nm output"),
    # and at 5 inches they clip. Width per panel tapers as columns are added so
    # a four-across grid stays inside a printable page.
    per_col = 9.0 if ncols == 1 else max(4.6, 11.0 / ncols)
    size = size or (per_col * ncols, 3.9 * nrows + 0.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    for ax in axes.ravel():
        ax.grid(True, **GRID)
    fig.suptitle(title + (f'\n{subtitle}' if subtitle else ''), fontsize=12)
    return fig, axes


def finish_grid(fig):
    """tight_layout with room left for the suptitle."""
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def finish(fig):
    fig.tight_layout()
    return fig


def stamp(ax, text):
    """A corner note for the caveat a figure must not be read without."""
    ax.text(0.99, 0.01, text, transform=ax.transAxes, fontsize=7,
            ha='right', va='bottom', color='#555555')


BAND_ALPHA = 0.18


def band(ax, x, lo, hi, color, label=None, alpha=BAND_ALPHA):
    """Shade lo..hi behind a series, in place of per-point error bars.

    Error bars on a 21-point sweep turn into a picket fence and the eye reads
    the caps rather than the trend; a filled band carries the same spread and
    leaves the line on top of it readable. It also composes -- several series
    can be drawn over each other without their whiskers colliding.

    `x` is sorted here rather than assumed sorted: fill_between joins points in
    the order given, so a series that runs right-to-left (a mirrored flank on a
    signed axis, say) folds the band back over itself into a bow tie.
    """
    import numpy as _np

    x = _np.asarray(x, dtype=float)
    lo = _np.asarray(lo, dtype=float)
    hi = _np.asarray(hi, dtype=float)
    if x.size == 0:
        return
    order = _np.argsort(x)
    ax.fill_between(x[order], lo[order], hi[order], color=color, alpha=alpha,
                    lw=0, label=label, zorder=1)


def csv(rows, header):
    """Rows of scalars -> CSV text. The customer gets these beside the PNGs."""
    out = [','.join(header)]
    for r in rows:
        out.append(','.join('' if v is None else
                            (f'{v:.6g}' if isinstance(v, float) else str(v))
                            for v in r))
    return '\n'.join(out) + '\n'


def heatmap(ax, xvals, yvals, grid, cmap='viridis', vmin=0.0, vmax=100.0,
            fmt='{:.0f}'):
    """An annotated grid of one value over two swept axes.

    The efficiency map the analysis tool draws, with the axes the way this
    report reads them: `xvals` increasing to the right, `yvals` increasing
    upward, and `grid` indexed [y, x]. Cells with no measurement are grey
    rather than the bottom colour of the scale, which would read as a measured
    zero.

    `vmin`/`vmax` are fixed by the CALLER and not by the data. An efficiency
    map scaled to its own maximum makes a drive that peaks at 60% look like one
    that peaks at 100%, and makes two units drawn the same way uncomparable --
    which is the whole point of the map.
    """
    import numpy as _np

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad('lightgray')
    g = _np.ma.masked_invalid(_np.asarray(grid, dtype=float))
    im = ax.imshow(g, origin='lower', aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xvals)))
    ax.set_xticklabels([f'{v:g}' for v in xvals])
    ax.set_yticks(range(len(yvals)))
    ax.set_yticklabels([f'{v:g}' for v in yvals])
    ax.grid(False)
    # Ink colour from the cell's own luminance rather than from the value: a
    # diverging map is dark at BOTH ends, so a threshold on the value gets half
    # of them wrong.
    for i in range(len(yvals)):
        for j in range(len(xvals)):
            v = _np.asarray(grid, dtype=float)[i, j]
            if not _np.isfinite(v):
                continue
            r, gg, b, _ = cmap((v - vmin) / (vmax - vmin or 1))
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color='white' if 0.299 * r + 0.587 * gg + 0.114 * b < 0.55
                    else 'black')
    return im
