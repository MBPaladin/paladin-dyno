r"""LaTeX formatting primitives for the generated report snippets.

Every number that reaches a .tex file goes through here, for one reason: a
metric that is None or NaN must never reach a PDF as the literal text "None" or
"nan". That is not a defensive hypothetical -- running_torque returns nan for
coulomb_drag_Nm whenever the drag fit finds only one usable direction, and the
mac_motors drive-on entry fitted against load_torque does exactly that.

Missing values degrade in two steps:

  * one missing table cell prints as an em-dash, so the surrounding rows still
    carry their numbers;
  * a missing *headline* -- the number a section exists to report -- collapses
    that block into the report's \placeholder box, which is red, loud, and
    cannot be mistaken for a result.

The second is the important one. A table quietly full of em-dashes reads as a
formatting problem; a placeholder box reads as missing data, which is what it
is.
"""

import math

MISSING = '---'

# How a drive reports current, mapped to the siunitx macros the report preamble
# declares. Labels only: nothing here rescales a number. Two drives on one bench
# disagree today (the Elmo reports Iq amplitude, the AKDs report A_rms), and the
# near-term goal is to make that visible in the PDF rather than to unify it --
# unifying means touching every processor's current path.
CURRENT_UNITS = {'peak': r'\Apeak', 'rms': r'\Arms'}
DEFAULT_CURRENT_UNITS = 'rms'

_ESCAPES = (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
            ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'),
            ('}', r'\}'), ('~', r'\textasciitilde{}'),
            ('^', r'\textasciicircum{}'))


def ok(v):
    """True when `v` is a number worth printing.

    None, NaN, and infinity all mean 'this was not measured' somewhere in the
    catalog, and all three must take the same path.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    return True


def all_ok(*vals):
    return all(ok(v) for v in vals)


def escape(s):
    r"""Escape LaTeX specials in free text.

    Channel and device names are the reason this exists: `dut_current` reaching
    a .tex unescaped is a math-mode subscript, and it fails to compile rather
    than merely looking wrong -- honest, but still an hour gone.
    """
    out = str(s)
    for ch, rep in _ESCAPES:
        out = out.replace(ch, rep)
    return out


def tt(s):
    """A channel or device name, monospaced and escaped."""
    return r'\texttt{%s}' % escape(s)


def raw(v, sig=3):
    """A bare number, no siunitx wrapper. MISSING when unmeasured.

    Significant figures are kept rather than trimmed -- 0.140 Nm/krpm, not
    0.14 -- because a column of results where some rows show three digits and
    others two reads as sloppy rather than as precise. The '#' flag is what
    holds the trailing zeros; it also leaves a bare '100.' for a value that
    lands on a round number, which is what the rstrip removes.
    """
    if not ok(v):
        return MISSING
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    return f'{float(v):#.{sig}g}'.rstrip('.')


def sci(v, sig=3):
    """A bare number forced into scientific notation.

    %g picks fixed or exponential per value, which puts 0.0002 next to 8.7e-05
    in the same column of the same table and makes them look like different
    kinds of quantity. Inertias span decades, so they are always exponential.
    """
    return MISSING if not ok(v) else f'{float(v):.{sig}e}'


def num(v, sig=3):
    r"""\num{...}, or an em-dash."""
    return MISSING if not ok(v) else r'\num{%s}' % raw(v, sig)


def si(v, unit, sig=3):
    r"""\SI{value}{unit}, or an em-dash.

    `unit` is a siunitx unit macro string, e.g. r'\newton\meter'.
    """
    return MISSING if not ok(v) else r'\SI{%s}{%s}' % (raw(v, sig), unit)


def pct(v, sig=3, signed=True):
    """A percentage. Signed by default: every percentage in this report is a
    delta from something, and an unsigned -8.9 reads as a magnitude."""
    if not ok(v):
        return MISSING
    s = f'{float(v):+.{sig}g}' if signed else f'{float(v):.{sig}g}'
    return r'\SI{%s}{\percent}' % s


def pm(value, err, unit, sig=4):
    r"""value with its uncertainty, or the value alone when the bar is missing.

    Emitted in siunitx's compact form -- \SI{8.704(171)e-5}{...} renders as
    8.704(171) x 10^-5 -- because that is the only uncertainty syntax siunitx
    v3 accepts inside \SI. Both `\SI{a \pm b}{u}` and `\SI{a +- b}{u}` are hard
    errors there, and the second used to work, so this is a trap worth staying
    inside one function.

    A missing uncertainty must not silently become '+/- nan': inertia hits that
    whenever only one acceleration was run.
    """
    if not ok(value):
        return MISSING
    if not ok(err) or value == 0:
        return si(value, unit, sig)
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10.0 ** exponent
    decimals = max(sig - 1, 0)
    digits = round(abs(err) / 10.0 ** exponent * 10 ** decimals)
    if digits < 1 or digits >= 10 ** sig or abs(err) >= abs(value):
        # Below the printed resolution, or so large the bracket stops reading
        # as an uncertainty -- 1.000(5000) is technically 1.0 +/- 5.0 but every
        # reader parses a bracket as a last-digit tolerance. Spell those out
        # instead; a result whose error bar exceeds it should look alarming.
        return r'$%s \pm %s$' % (si(value, unit, sig), si(err, unit, 2))
    return r'\SI{%.*f(%d)e%d}{%s}' % (decimals, mantissa, digits, exponent, unit)


def placeholder(message):
    """The report's red 'data pending' box. Used for a whole block whose
    headline number is missing, never for one cell."""
    return '\\placeholder{%s}\n' % escape(message)


def current_unit(units):
    """siunitx macro for a drive's current convention.

    Unknown or absent conventions fall back to rms, which is what
    setup_summary.PARAM_INFO has always labelled `motor_params.kt` as. Callers
    are expected to have surfaced the fallback already -- see
    Segment.current_units, which reports whether the value was declared.
    """
    return CURRENT_UNITS.get(str(units).lower(),
                             CURRENT_UNITS[DEFAULT_CURRENT_UNITS])


def kt_unit(units):
    """Nm per amp, in whichever amp the drive reports."""
    return r'\newton\meter\per' + current_unit(units)


def per_current_unit(units):
    """Inverse amps -- the unit of the tanh saturation rate."""
    return r'\per' + current_unit(units)


def table(caption, header, rows, label=None, spec=None):
    """A booktabs table in an [H] float, matching the report's existing style.

    `rows` holds already-formatted cell strings; nothing here formats numbers,
    so a caller cannot accidentally bypass the MISSING handling above.
    """
    if spec is None:
        spec = 'l' + 'r' * (len(header) - 2) + 'l' if len(header) > 2 \
            else 'l' * len(header)
    out = ['\\begin{table}[H]', '  \\centering', '  \\caption{%s}' % caption]
    if label:
        out.append('  \\label{%s}' % label)
    out += ['  \\begin{tabular}{%s}' % spec, '    \\toprule',
            '    ' + ' & '.join(header) + '\\\\', '    \\midrule']
    out += ['    ' + ' & '.join(r) + '\\\\' for r in rows]
    out += ['    \\bottomrule', '  \\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)
