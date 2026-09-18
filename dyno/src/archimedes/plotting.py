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


def csv(rows, header):
    """Rows of scalars -> CSV text. The customer gets these beside the PNGs."""
    out = [','.join(header)]
    for r in rows:
        out.append(','.join('' if v is None else
                            (f'{v:.6g}' if isinstance(v, float) else str(v))
                            for v in r))
    return '\n'.join(out) + '\n'
