r"""One renderer per report section.

A section is a pure function of the results.json entries its manifest roles name
plus the manifest itself. Nothing here opens a log, runs a processor, or plots;
figures are *declared* against a collector and copied afterwards.

Sections are keyed by report section, not by processor, because three of the
four need two logs at once: inertia compares the full chain against the
decoupled structure, and torque ripple and running torque each pair a drive-on
run with a drive-off one. Hanging the renderer off a processor would leave the
pairing with nowhere to live.

The prose here is deliberately thin. It states what was commanded and what came
back, and stops. Interpretation belongs in the driver, beside the \input line,
where a person writes it and no generator overwrites it.
"""

import math
import os

from . import texfmt as tf

# siunitx unit strings used often enough that spelling them inline inside an
# f-string means escaping backslashes by hand and getting it wrong once.
NM = r'\newton\meter'
RAD_S = r'\radian\per\second'
RAD_S2 = r'\radian\per\second\squared'
NM_PER_S = r'\newton\meter\per\second'
VOLT = r'\volt'
RPM = r'\rpm'

RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)


def _rpm(v):
    return v * RPM_PER_RAD_S if tf.ok(v) else None


def _signed_span(lo, hi, unit, sig=3):
    r"""\SI{\pm X}{unit} for a symmetric command, a range otherwise.

    Sawtooth commands come back as -5.000 / +4.999 -- the ramp simply never
    lands exactly on its turning point -- so symmetry is judged with a
    tolerance rather than by equality, which would print a range for every
    sawtooth ever logged.
    """
    if not tf.all_ok(lo, hi):
        return tf.MISSING
    amp = max(abs(lo), abs(hi))
    if amp > 0 and abs(abs(lo) - abs(hi)) / amp < 0.02 and lo < 0 < hi:
        return r'\SI{\pm %s}{%s}' % (tf.raw(amp, sig), unit)
    return r'\SIrange{%s}{%s}{%s}' % (tf.raw(lo, sig), tf.raw(hi, sig), unit)


def _article(word):
    """'a' or 'an' for a manufacturer name pulled from the manifest.

    Spelling, not phonetics: the manifest holds names like 'Elmo Gold Drum',
    and getting "a Elmo" into the introduction of a customer report is the kind
    of thing that makes the whole document look unproofed.
    """
    return 'an' if str(word)[:1].lower() in 'aeiou' else 'a'


def _levels(values, unit, sig=3):
    """A commanded setpoint grid as a readable list."""
    if not values:
        return tf.MISSING
    body = ', '.join(tf.raw(v, sig) for v in values)
    return r'\SIlist{%s}{%s}' % (body.replace(', ', ';'), unit)


class Context:
    """What a section renderer is handed.

    `roles` maps a manifest role ('main', 'full', 'drive_on', ...) to
    {'entry', 'log_dir', 'rel'}. Everything else is a lookup convenience so a
    renderer reads as prose rather than as dictionary walking.
    """

    def __init__(self, key, roles, manifest, figures):
        self.key = key
        self.roles = roles
        self.man = manifest
        self.figures = figures
        self.notes = []

    def has(self, role):
        return role in self.roles

    def entry(self, role):
        got = self.roles.get(role)
        return got['entry'] if got else None

    def m(self, role, key, default=None):
        e = self.entry(role)
        if e is None:
            return default
        v = e.get('metrics', {}).get(key, default)
        return default if v is None else v

    def p(self, role, key, default=None):
        e = self.entry(role)
        if e is None:
            return default
        v = e.get('params', {}).get(key, default)
        return default if v is None else v

    def units(self, role, motor_key='motor'):
        """The current convention to label this role's numbers with.

        Manifest first, then the metric the processor recorded, then rms. The
        manifest override exists for logs recorded before drive_params carried
        the field: those results.json files cannot be regenerated without
        re-running the analysis, and a report should not be blocked on that.
        """
        motor = self.m(role, motor_key)
        override = (self.man.get('current_units') or {}).get(motor)
        if override:
            return str(override).lower()
        recorded = self.m(role, 'current_units')
        if recorded:
            if not self.m(role, 'current_units_declared', False):
                self.note(f'{self.key}/{role}: {motor} did not declare '
                          f'drive_params.current_units; currents labelled '
                          f'{recorded} by default')
            return str(recorded).lower()
        self.note(f'{self.key}/{role}: no current_units recorded (analysis '
                  f'predates the field); labelling as '
                  f'{tf.DEFAULT_CURRENT_UNITS}')
        return tf.DEFAULT_CURRENT_UNITS

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    def fig(self, role, slug, alias, caption, label, width='.9'):
        r"""A \resultfig line, or a placeholder box when the figure is absent.

        The alias is what the .tex will reference forever, so it is prefixed by
        role whenever a section has more than one -- two logs both emitting
        `inertia.png` must not race for one filename.
        """
        got = self.roles.get(role)
        if got is None:
            return tf.placeholder(f'{caption}: no {role} log configured for '
                                  f'the {self.key} section.')
        path = self.figures.declare(got['log_dir'], slug, alias)
        if path is None:
            return tf.placeholder(
                f'{caption}: the {role} analysis produced no {slug} figure.')
        return ('\\resultfig{%s}{%s}{%s}{%s}\n\n'
                % (path, width, caption, label))


# ------------------------------------------------------------------ sections

class Section:
    key = ''
    title = ''
    processor = ''

    def render(self, ctx):
        raise NotImplementedError

    def missing(self, ctx, role, label):
        return (f'\\section{{{self.title}}}\\label{{sec:{self.key}}}\n\n'
                + tf.placeholder(
                    f'No {self.processor} result for the {label} log. Run the '
                    f'analysis on it, then re-render.'))


class TorqueResponse(Section):
    key = 'torque_response'
    title = 'Torque Response'
    processor = 'current_torque'

    def render(self, ctx):
        if not ctx.has('main'):
            return self.missing(ctx, 'main', 'blocked-rotor')
        u = ctx.units('main')
        amp_unit = tf.current_unit(u)
        out = [f'\\section{{{self.title}}}\\label{{sec:torque}}\n']
        out.append(self._setup(ctx, u, amp_unit))
        out.append(self._kt(ctx, u))
        out.append(self._km(ctx))
        return '\n'.join(out)

    def _setup(self, ctx, u, amp_unit):
        cmd = _signed_span(ctx.m('main', 'cmd_torque_min_Nm'),
                           ctx.m('main', 'cmd_torque_max_Nm'),
                           r'\newton\meter')
        rate = ctx.m('main', 'cmd_rotatum_Nm_per_s')
        cycles = ctx.m('main', 'n_torque_cycles')
        angles = ctx.m('main', 'n_park_angles', 1)
        cell_fs = ctx.m('main', 'torque_cell_fs_Nm')
        v_bus = ctx.m('main', 'km_supply_voltage_V')
        clamp = ctx.m('main', 'km_channel')

        s = ['The absorber was commanded to a stationary hold while the device '
             f'under test was commanded a torque sawtooth of {cmd}']
        if tf.ok(rate):
            s.append(f' at up to {tf.si(rate, NM_PER_S)}')
        if tf.ok(cycles) and cycles:
            s.append(f', repeated over {tf.num(cycles)} cycles')
        if tf.ok(angles) and angles > 1:
            s.append(f' at {tf.num(angles)} rotor park angles')
        s.append('.\n')

        s.append('The quadrature current $I_q$ was collected from the servo '
                 f'drive on {tf.tt(ctx.m("main", "current_channel"))} and the '
                 'transmitted torque $T$ was measured with the in-line torque '
                 'cell on '
                 f'{tf.tt(ctx.m("main", "torque_channel"))}')
        if tf.ok(cell_fs):
            s.append(f' (full scale {tf.si(cell_fs, NM)})')
        if clamp:
            s.append(f'. The bus current $I_{{\\mathrm{{Bus}}}}$ was measured '
                     f'with a current clamp on {tf.tt(clamp)}')
            if tf.ok(v_bus):
                s.append(f', fed from a {tf.si(v_bus, VOLT)} DC supply')
        s.append('.\n\n')

        # The winding relations are convention-dependent, and this is exactly
        # where the peak/rms divergence surfaces in prose rather than in a unit
        # label: with an rms-reporting drive the same equations carry a sqrt 2.
        s.append(f'Motor-side current is reported in \\si{{{amp_unit}}}, ')
        if u == 'peak':
            s.append("the amplitude of the drive's quadrature-axis current. "
                     'The individual winding currents follow from the '
                     'electrical angle $\\theta_e$ as\n'
                     '\\begin{align}\n'
                     '  I_U &= I_q\\cos(\\theta_e)\\\\\n'
                     '  I_V &= I_q\\cos(\\theta_e + 2\\pi/3)\\\\\n'
                     '  I_W &= I_q\\cos(\\theta_e - 2\\pi/3)\n'
                     '\\end{align}\n')
        else:
            s.append('the rms value of the quadrature-axis current. The '
                     'individual winding currents follow from the electrical '
                     'angle $\\theta_e$ as\n'
                     '\\begin{align}\n'
                     '  I_U &= \\sqrt{2}\\,I_q\\cos(\\theta_e)\\\\\n'
                     '  I_V &= \\sqrt{2}\\,I_q\\cos(\\theta_e + 2\\pi/3)\\\\\n'
                     '  I_W &= \\sqrt{2}\\,I_q\\cos(\\theta_e - 2\\pi/3)\n'
                     '\\end{align}\n')
        s.append('so the winding currents are stationary DC values when the '
                 'rotor is held and sinusoidal when it turns, while $I_q$ is '
                 'the same reported quantity in both cases.\n')
        return ''.join(s)

    def _kt(self, ctx, u):
        kt = ctx.m('main', 'drive_kt_Nm_per_A')
        k_tanh = ctx.m('main', 'drive_k_tanh_per_A')
        t_peak = ctx.m('main', 'usable_peak_torque_Nm')
        droop = ctx.m('main', 'peak_torque_droop_pct')

        head = ('\\subsection{Torque Constant} \\label{sec:kt}\n'
                'The data were fit to a saturating model,\n'
                '\\begin{equation}\n'
                '  T(I_{q}) = \\frac{K_t}{K_{tanh}}\\tanh(K_{tanh} I_{q}),\n'
                '\\end{equation}\n'
                'where the $\\tanh$ captures magnetic saturation. $K_t$ is '
                'quoted as the small-signal slope of that fit at the origin.\n\n')
        if not tf.ok(kt):
            return head + tf.placeholder(
                'The tanh fit did not converge, so no torque constant is '
                'available from this run.')

        rows = [['Torque constant $K_t$', tf.raw(kt), f'\\si{{{tf.kt_unit(u)}}}'],
                ['Saturation rate $K_{tanh}$', tf.raw(k_tanh, 2),
                 f'\\si{{{tf.per_current_unit(u)}}}']]
        peak_label = 'Peak torque $T_{peak}$'
        if tf.ok(droop):
            peak_label += (f' (at {tf.raw(droop, 2)}\\% droop)')
        rows.append([peak_label, tf.raw(t_peak), r'\si{\newton\meter}'])

        body = tf.table('Torque constant fit parameters.',
                        ['Parameter', 'Value', 'Unit'], rows,
                        label='tab:kt', spec='lrl')
        body += ctx.fig('main', 'pooled_fit', 'current_torque_fit',
                        'Current vs Torque Fit', 'fig:torque_constant')
        if not ctx.m('main', 'saturation_exercised', False):
            util = ctx.m('main', 'current_utilisation_pct')
            body += ('\\noindent The sweep reached '
                     f'{tf.pct(util, signed=False)} of the drive\'s continuous '
                     'current rating, so the saturation asymptote is '
                     'extrapolated and only the small-signal $K_t$ is '
                     'constrained by this data.\n\n')
        return head + body

    def _km(self, ctx):
        head = ('\\newpage\n\n\\subsection{Motor Constant} \\label{sec:km}\n\n'
                'The motor constant $K_m$ was fit to the data through the '
                'relationship\n\n'
                '\\begin{equation}\n'
                '    K_m = \\frac{\\abs{T}}{\\sqrt{V_{\\mathrm{Bus}}'
                'I_{\\mathrm{Bus}}}}\n'
                '\\end{equation}\n\n')
        km = ctx.m('main', 'km_Nm_per_sqrtW')
        if not tf.ok(km):
            return head + tf.placeholder(
                'This run carried no DC-supply current clamp channel, so bus '
                'power -- and with it $K_m$ -- could not be measured.')

        frac = ctx.p('main', 'km_torque_frac')
        gate = ctx.m('main', 'km_torque_gate_Nm')
        scope = ctx.m('main', 'km_scope')
        drift = ctx.m('main', 'km_drift') or {}

        body = ['\\noindent where $T$ is the transmitted torque and '
                '$V_{\\mathrm{Bus}}I_{\\mathrm{Bus}}$ is the electrical power '
                'input at that torque.\n'
                "The drive's fixed and switching losses are largely "
                'independent of output torque, so they make up a growing '
                'fraction of bus power as torque falls and depress the '
                'apparent $K_m$ at low torque. To minimize their effect we '
                'only consider samples where ']
        if tf.ok(frac):
            body.append(f'$\\abs{{T}}\\ge{tf.raw(frac, 2)}T_{{peak}}$')
            if tf.ok(gate):
                body.append(f' ({tf.si(gate, NM)})')
        else:
            body.append('the torque is above the low-torque gate')
        body.append('.\n')
        if scope == 'first_cycle':
            body.append('We report $K_m$ as the mean value over the first '
                        'torque cycle of the experiment and record the rate at '
                        'which $K_m$ decays under thermal load.\n\n')
        else:
            body.append('We report $K_m$ as the mean value over the whole '
                        'gated run and record the rate at which it decays '
                        'under thermal load.\n\n')

        rows = [['Motor constant $K_m$', tf.raw(km),
                 r'$\frac{\si{\newton\meter}}{\sqrt{\si{W}}}$']]
        if drift:
            rows.append(['Total drift', tf.raw(drift.get('km_pct'), 2),
                         r'\%'])
            rows.append(['Drift rate',
                         tf.raw(drift.get('km_rate_pct_per_s'), 2),
                         r'$\frac{\%}{\si{\second}}$'])
            rows.append(['Drift window', tf.raw(drift.get('span_s'), 3),
                         r'\si{\second}'])
        out = ''.join(body) + tf.table(
            'Motor constant fit parameters.', ['Parameter', 'Value', 'Unit'],
            rows, label='tab:km', spec='lrl')
        out += ctx.fig('main', 'motor_constant_vs_time', 'km_vs_time',
                       'Motor Constant vs Time', 'fig:km_v_time')
        out += ctx.fig('main', 'motor_constant_vs_torque', 'km_vs_torque',
                       'Motor Constant vs Torque', 'fig:km_v_torque')
        return head + out


class Inertia(Section):
    key = 'inertia'
    title = 'Inertia'
    processor = 'inertia'

    def render(self, ctx):
        if not ctx.has('full'):
            return self.missing(ctx, 'full', 'full-chain inertia')
        out = [f'\\section{{{self.title}}}\\label{{sec:inertia}}\n']
        out.append(self._setup(ctx))
        out.append(_INERTIA_METHOD)
        out.append(self._table(ctx))
        out.append(ctx.fig('full', 'inertia', 'inertia',
                           'Motor Inertia, Full Chain', 'fig:inertia'))
        if ctx.has('structure'):
            out.append(ctx.fig('structure', 'inertia', 'inertia_structure',
                               'Structure Inertia, DUT Disconnected',
                               'fig:inertia_structure'))
        return '\n'.join(out)

    def _setup(self, ctx):
        v_cmd = ctx.m('full', 'cmd_peak_velocity_rad_s')
        a_lo = ctx.m('full', 'alpha_min_rad_s2')
        a_hi = ctx.m('full', 'alpha_max_rad_s2')
        pairs = ctx.m('full', 'n_ramp_pairs')
        accs = ctx.m('full', 'n_accelerations')
        window = ctx.m('full', 'speed_window_rad_s') or [None, None]

        s = ['To calculate inertia, the system was driven up to ']
        s.append(tf.si(v_cmd, r'\radian\per\second') if tf.ok(v_cmd)
                 else 'its commanded peak speed')
        if tf.ok(v_cmd):
            s.append(f' ({tf.si(_rpm(v_cmd), RPM, 3)})')
        s.append(' and back down to zero velocity in a series of '
                 'constant acceleration/deceleration motions')
        if tf.all_ok(a_lo, a_hi):
            s.append(f' at {tf.raw(a_lo)} to '
                     f'{tf.si(a_hi, RAD_S2)}')
        s.append('.\n')
        s.append('The resulting shaft motion and torque response were '
                 'recorded. We isolate each acceleration and deceleration '
                 'pair, reject the jerk transients, and fit the remaining '
                 'data')
        if tf.all_ok(*window):
            s.append(f' over {tf.raw(window[0])} to '
                     f'{tf.si(window[1], RAD_S)}')
        s.append('.')
        if tf.all_ok(pairs, accs):
            s.append(f' {tf.num(pairs)} ramp pairs across {tf.num(accs)} '
                     f'distinct accelerations survived that filtering.')
        s.append('\n')
        return ''.join(s)

    def _table(self, ctx):
        j_test = ctx.m('full', 'J_test_kgm2')
        j_motor = ctx.m('full', 'J_motor_kgm2')
        unc = ctx.m('full', 'J_quoted_uncertainty_kgm2')
        j_param = ctx.m('full', 'J_structure_kgm2')
        unit = r'\kilo\gram\meter\squared'

        if not tf.ok(j_test):
            return tf.placeholder(
                'No usable ramp pair survived, so no inertia was measured.')

        rows = [['Full chain $J_{test}$', tf.sci(j_test, 3), r'\si{%s}' % unit]]

        # Provenance, not just the value: J_structure in the full-chain result
        # is whatever the operator typed into the structure_inertia parameter.
        # The decoupled run is what actually measured it, and the two silently
        # disagreeing is the failure this row exists to expose.
        if ctx.has('structure'):
            j_meas = ctx.m('structure', 'J_test_kgm2')
            rows.append(['Structure $J_{struct}$ (decoupled run)',
                         tf.sci(j_meas, 3), r'\si{%s}' % unit])
            if tf.all_ok(j_meas, j_param) and j_meas:
                delta = 100 * (j_param - j_meas) / j_meas
                if abs(delta) > 1.0:
                    ctx.note(
                        f'inertia: structure_inertia parameter '
                        f'{j_param:.3e} differs from the decoupled run\'s '
                        f'measured {j_meas:.3e} kg.m^2 by {delta:+.1f}% -- '
                        f'J_motor below was computed with the parameter')
        rows.append(['Structure used in the subtraction', tf.sci(j_param, 3),
                     r'\si{%s}' % unit])
        rows.append(['Motor $J_{motor}$', tf.sci(j_motor, 3),
                     r'\si{%s}' % unit])
        if tf.ok(unc):
            rows.append(['Uncertainty (between accelerations)',
                         tf.sci(unc, 2), r'\si{%s}' % unit])

        return tf.table('Measured inertia. $J_{motor}$ is the full chain less '
                        'the structure measured with the DUT disconnected.',
                        ['Quantity', 'Value', 'Unit'], rows,
                        label='tab:inertia', spec='lrl')


class TorqueRipple(Section):
    key = 'torque_ripple'
    title = 'Torque Ripple'
    processor = 'cogging'

    def render(self, ctx):
        primary = 'drive_on' if ctx.has('drive_on') else 'main'
        if not ctx.has(primary):
            return self.missing(ctx, primary, 'torque ripple')
        # "with Servo Drive On" only earns its place when an Off figure follows
        # it. Alone on the page it names a contrast the report does not draw,
        # and the reader has to go looking for the comparison it implies.
        paired = ctx.has('drive_off')
        out = [f'\\section{{{self.title}}}\\label{{sec:torque_ripple}}\n']
        out.append(self._setup(ctx, primary))
        out.append(self._table(ctx, primary))
        out.append(ctx.fig(primary, 'vs_mech_angle', 'torque_ripple_drive_on',
                           'Torque Ripple with Servo Drive On' if paired
                           else 'Torque Ripple',
                           'fig:torque_ripple_elmo_on'))
        if paired:
            out.append(ctx.fig('drive_off', 'vs_mech_angle',
                               'torque_ripple_drive_off',
                               'Torque Ripple with Servo Drive Off',
                               'fig:torque_ripple_elmo_off'))
        return '\n'.join(out)

    def _setup(self, ctx, primary):
        speeds = ctx.m(primary, 'speeds_rad_s') or []
        loads = ctx.m(primary, 'loads_Nm') or []
        s = ['Torque ripple characterization was performed by commanding the '
             'dynamometer absorber to a set velocity and commanding the DUT '
             'through a series of constant-torque holds. ']
        if speeds and loads:
            s.append(f'The grid spanned {tf.num(len(speeds))} speeds '
                     f'({_levels(speeds, RAD_S)}) and '
                     f'{tf.num(len(loads))} load levels '
                     f'({_levels(loads, NM)}). ')
        if ctx.has('drive_off'):
            off_speeds = ctx.m('drive_off', 'speeds_rad_s') or []
            s.append('This test was repeated with the motor disconnected from '
                     'the servo drive to isolate the effect of drive torque '
                     'ripple compensation')
            if off_speeds:
                s.append(f', over {_levels(off_speeds, RAD_S)}')
            s.append('. ')

        order = ctx.m(primary, 'ripple_order')
        detected = ctx.m(primary, 'detected_ripple_order')
        det_amp = ctx.m(primary, 'detected_ripple_amplitude_Nm')
        det_speed = ctx.m(primary, 'detected_at_speed_rad_s')
        if tf.ok(order):
            s.append(f'\nRipple is folded at order {tf.num(order)}')
            if ctx.m(primary, 'ripple_order_source') == 'specified':
                s.append(' (specified)')
            if tf.ok(detected):
                s.append(f'; the dominant spatial order measured in the '
                         f'{tf.si(det_speed, RAD_S)} no-load '
                         f'dwell is {tf.num(detected)}')
                if tf.ok(det_amp):
                    s.append(f' at {tf.si(det_amp, NM)}')
            s.append('.\n')
        return ''.join(s)

    def _table(self, ctx, primary):
        on = ctx.m(primary, 'ripple_pkpk_at_slowest_Nm') or {}
        on_mech = ctx.m(primary, 'mech_ripple_pkpk_Nm') or {}
        off = ctx.m('drive_off', 'ripple_pkpk_at_slowest_Nm') or {}
        off_mech = ctx.m('drive_off', 'mech_ripple_pkpk_Nm') or {}
        if not on and not off:
            return tf.placeholder(
                'The ripple-order fold produced no peak-to-peak values, so '
                'there is nothing to tabulate.')

        keys = list(on) or list(on_mech)
        for k in list(off) + list(off_mech):
            if k not in keys:
                keys.append(k)

        if ctx.has('drive_off'):
            header = ['Load', 'On, order fold', 'On, mechanical',
                      'Off, order fold', 'Off, mechanical']
        else:
            header = ['Load', 'Order fold', 'Mechanical']
        rows = []
        for k in keys:
            row = [tf.escape(k.replace('Nm', ' Nm')),
                   tf.raw(on.get(k), 3), tf.raw(on_mech.get(k), 3)]
            if ctx.has('drive_off'):
                row += [tf.raw(off.get(k), 3), tf.raw(off_mech.get(k), 3)]
            rows.append(row)

        slow_on = ctx.m(primary, 'slowest_speed_rad_s')
        cap = ('Peak-to-peak torque ripple in \\si{\\newton\\meter} at the '
               'slowest speed plateau')
        if tf.ok(slow_on):
            cap += f' ({tf.si(slow_on, RAD_S)})'
        cap += ('. The order fold averages over the ripple order; the '
                'mechanical column is the full-revolution fold and so '
                'includes once-per-rev content.')
        return tf.table(cap, header, rows, label='tab:ripple',
                        spec='l' + 'r' * (len(header) - 1))


class RunningTorque(Section):
    key = 'running_torque'
    title = 'Running Torque'
    processor = 'running_torque'

    def render(self, ctx):
        primary = 'drive_on' if ctx.has('drive_on') else 'main'
        if not ctx.has(primary):
            return self.missing(ctx, primary, 'running torque')
        out = [f'\\section{{{self.title}}}\\label{{sec:running}}\n']
        out.append(self._setup(ctx, primary))
        out.append(_RUNNING_METHOD)
        out.append(self._table(ctx, primary))
        out.append(ctx.fig(primary, 'drag_vs_velocity_krpm',
                           'drag_vs_velocity', 'Running Torque',
                           'fig:running_torque_elmo_on'))
        out.append(ctx.fig(primary, 'spectrogram_speed', 'running_spectrogram',
                           'Running Torque Spectrogram',
                           'fig:running_spectrogram'))
        return '\n'.join(out)

    def _setup(self, ctx, primary):
        v_cmd = ctx.m(primary, 'cmd_peak_velocity_rad_s')
        v_meas = ctx.m(primary, 'peak_speed_rad_s')
        s = ['To calculate running torque, the dynamometer absorber was driven '
             'in a triangular sawtooth pattern to ']
        if tf.ok(v_cmd):
            s.append(f'{_signed_span(-v_cmd, v_cmd, RAD_S)} '
                     f'({tf.si(_rpm(v_cmd), RPM, 3)})')
        else:
            s.append('its commanded peak speed')
        s.append(' while the DUT was held at a constant zero torque command')
        if tf.ok(v_meas):
            s.append(f'. The shaft reached {tf.si(v_meas, RAD_S)} '
                     f'({tf.si(_rpm(v_meas), RPM, 3)})')
        s.append('.\n')
        return ''.join(s)

    def _table(self, ctx, primary):
        # One column needs no on/off label: there is nothing to contrast it
        # with, and heading it "Servo drive on" implies a missing companion.
        if ctx.has('drive_off'):
            cols = [('Servo drive on', primary), ('Servo drive off',
                                                  'drive_off')]
        else:
            cols = [('Value', primary)]

        def row(label, key, unit, sig=3):
            cells = [label]
            for _, role in cols:
                cells.append(tf.raw(ctx.m(role, key), sig))
            cells.append(unit)
            return cells

        rows = [
            row('Coulomb drag', 'coulomb_drag_Nm', r'\si{\newton\meter}'),
            row('Viscous drag', 'viscous_drag_Nm_per_krpm',
                r'\si{\newton\meter}/krpm'),
            row('Total at peak speed', 'drag_at_peak_speed_Nm',
                r'\si{\newton\meter}'),
            row('Peak speed', 'peak_speed_rad_s',
                r'\si{\radian\per\second}'),
            row('Dominant ripple line', 'dominant_line_hz', r'\si{\hertz}'),
            row('Dominant line amplitude', 'dominant_line_Nm',
                r'\si{\newton\meter}', 2),
        ]
        if not tf.ok(ctx.m(primary, 'coulomb_drag_Nm')):
            ctx.note('running_torque: the drive-on drag fit returned no '
                     'Coulomb term (only one direction was usable); its '
                     'column will read as em-dashes')
        header = ['Quantity'] + [c for c, _ in cols] + ['Unit']
        return tf.table(
            'Loss torque split into a speed-independent Coulomb term and a '
            'speed-proportional viscous term.', header, rows,
            label='tab:running', spec='l' + 'r' * len(cols) + 'l')


_INERTIA_METHOD = r"""
With $T^+$ and $T^-$ representing the measured torque during acceleration and
deceleration, respectively, and $\alpha$ as the magnitude of the angular
acceleration, we can estimate the inertia $J$ using

\begin{equation}
  T^+ = J\alpha - T_{loss}
\end{equation}

\begin{equation}
  T^- = -J\alpha - T_{loss}
\end{equation}

\begin{equation}
  J = \frac{T^+ - T^-}{2\alpha}
\end{equation}

In this way, we can determine the inertia $J$ while cancelling loss torque
$T_{loss}$. We repeat this experiment for both the full dyno chain and for the
chain with the DUT disconnected, and calculate the motor inertia $J_{motor}$ as
the difference between the two.
"""

_RUNNING_METHOD = r"""
Similar to the inertia measurement, the resulting loss torque can be estimated
by averaging the torque measured during acceleration and deceleration at each
velocity, effectively cancelling the inertia term and isolating losses:

\begin{equation}
  T_{loss}(\omega) = \frac{T^+(\omega) + T^-(\omega)}{2}
\end{equation}
"""


SECTIONS = {s.key: s() for s in
            (TorqueResponse, Inertia, TorqueRipple, RunningTorque)}


# ------------------------------------------------------------ summary table

def render_summary(ctx, loaded):
    """Table 1: one row per headline, pulled across every section.

    This is the only renderer that reaches across sections, which is why it
    cannot live on a processor: no single results.json holds more than one of
    these rows.
    """
    def pick(section, role, key, default=None):
        got = (loaded.get(section) or {}).get(role)
        if not got:
            return default
        v = got['entry'].get('metrics', {}).get(key)
        return default if v is None else v

    def role_of(section, preferred, fallback='main'):
        avail = loaded.get(section) or {}
        return preferred if preferred in avail else fallback

    def ref(section, target):
        r"""\ref to a section, but only if that section was rendered.

        Every row is listed whether or not its test was run -- a summary table
        that silently omits what is missing hides it -- but the cross-reference
        has to be dropped along with the section. A \ref to a label that was
        never emitted compiles to '??' and leaves LaTeX warning about undefined
        references forever after.
        """
        return r'\ref{%s}' % target if loaded.get(section) else tf.MISSING

    rows = []

    tr = role_of('torque_response', 'main')
    units_metric = pick('torque_response', tr, 'current_units',
                        tf.DEFAULT_CURRENT_UNITS)
    override = (ctx.man.get('current_units') or {}).get(
        pick('torque_response', tr, 'motor'))
    units = str(override or units_metric).lower()
    kt = pick('torque_response', tr, 'drive_kt_Nm_per_A')
    rows.append(['Torque constant $K_t$',
                 f'{tf.num(kt)} \\si{{{tf.kt_unit(units)}}}',
                 ref('torque_response', 'sec:kt')])
    rows.append(['Peak torque $T_{peak}$',
                 tf.si(pick('torque_response', tr, 'usable_peak_torque_Nm'),
                       r'\newton\meter'), ref('torque_response', 'sec:kt')])
    rows.append(['Motor constant $K_m$',
                 tf.num(pick('torque_response', tr, 'km_Nm_per_sqrtW'))
                 + r' \si{\newton\meter}$/\sqrt{\si{\watt}}$',
                 ref('torque_response', 'sec:km')])

    rows.append(['Motor inertia $J$',
                 tf.pm(pick('inertia', 'full', 'J_motor_kgm2'),
                       pick('inertia', 'full', 'J_quoted_uncertainty_kgm2'),
                       r'\kilo\gram\meter\squared'),
                 ref('inertia', 'sec:inertia')])

    # Which ripple log feeds the headline is a report decision, not a data one:
    # the drive-on and drive-off runs measure genuinely different things.
    ripple_role = role_of(
        'torque_ripple',
        str((ctx.man.get('summary') or {}).get('ripple_from', 'drive_on')))
    order = pick('torque_ripple', ripple_role, 'ripple_order')
    pkpk = (pick('torque_ripple', ripple_role,
                 'ripple_pkpk_at_slowest_Nm') or {}).get('0Nm')
    label = 'Torque ripple (p-p @ \\SI{0}{\\newton\\meter}'
    if tf.ok(order):
        label += f', order {tf.raw(order)}'
    label += ')'
    # Name the role only when it is a deliberate, non-obvious choice. 'drive_on'
    # is the default and 'main' means the report supplied one log with no
    # variants at all -- appending either to the row label just adds noise.
    if ripple_role not in ('drive_on', 'main'):
        label += f', {tf.escape(ripple_role)}'
    rows.append([label, tf.si(pkpk, r'\newton\meter', 2),
                 ref('torque_ripple', 'sec:torque_ripple')])

    run_role = role_of(
        'running_torque',
        str((ctx.man.get('summary') or {}).get('running_from', 'drive_on')))
    coulomb = pick('running_torque', run_role, 'coulomb_drag_Nm')
    viscous = pick('running_torque', run_role, 'viscous_drag_Nm_per_krpm')
    if tf.all_ok(coulomb, viscous):
        running = (f'{tf.si(coulomb, NM)} '
                   f'$+$ {tf.num(viscous)} '
                   f'\\si{{\\newton\\meter}}/krpm')
    else:
        running = tf.MISSING
    rows.append(['Running torque', running, ref('running_torque', 'sec:running')])

    return tf.table(
        "Summary of measured results.",
        ['Quantity', 'Measured', 'Section'], rows,
        label='tab:summary', spec='llc')


# ------------------------------------------------------------------- driver

_PREAMBLE = r"""\documentclass[11pt]{article}

\usepackage[letterpaper,margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{siunitx}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}
\hypersetup{linkcolor=blue!50!black}

% siunitx shortcuts for the units that show up everywhere in these tests
\DeclareSIUnit\arcmin{arcmin}
\DeclareSIUnit\rpm{rpm}
\DeclareSIUnit{\Arms}{A_{\mathrm{rms}}}
\DeclareSIUnit{\Apeak}{A_{\mathrm{peak}}}
\DeclareSIUnit{\Adc}{A_{\mathrm{dc}}}
\sisetup{per-mode=symbol,detect-weight=true,detect-family=true,inter-unit-product={}}

% A clearly-styled placeholder box for results that do not exist yet.
\newcommand{\placeholder}[1]{%
  \begin{center}
  \fcolorbox{red!60!black}{red!5}{%
    \parbox{0.92\linewidth}{\small\textbf{[Placeholder --- data pending]}\\#1}}
  \end{center}}

% Standard figure helper: \resultfig{path}{width-fraction}{caption}{label}
\newcommand{\resultfig}[4]{%
  \begin{figure}[H]
    \centering
    \includegraphics[width=#2\linewidth]{#1}
    \caption{#3}
    \label{#4}
  \end{figure}}

\newcommand{\abs}[1]{\left\lvert#1\right\rvert}
"""

_SECTION_ORDER = ('torque_response', 'inertia', 'torque_ripple',
                  'running_torque')


def render_driver(man, section_keys, root=None):
    """A compilable report that \\inputs every generated section.

    Written as `report_generated.tex` rather than over any hand-maintained
    master: the generated sections are meant to be \\input, and a report whose
    commentary someone spent an afternoon on must not be the thing this
    overwrites to prove the point.
    """
    dut = man.get('dut') or {}
    title = man.get('title') or 'Dynamometer Test Report'
    ordered = ([k for k in _SECTION_ORDER if k in section_keys]
               + [k for k in section_keys if k not in _SECTION_ORDER])

    intro = []
    intro.append(f'This report summarizes dynamometer testing on the '
                 f'{tf.escape(title)}')
    if dut.get('supplier'):
        intro.append(f' provided to {tf.escape(dut["supplier"])}')
    intro.append('. The device under test (DUT) is a')
    if dut.get('topology'):
        intro.append(f' {tf.escape(dut["topology"])}')
    intro.append(' motor')
    if dut.get('pole_pairs'):
        intro.append(f' with {tf.num(dut["pole_pairs"])} pole pairs')
    intro.append(', driven on the Paladin dynamometer against an absorber '
                 'motor.')
    if dut.get('torque_cell_Nm'):
        intro.append(f' Between the DUT and the absorber sits a '
                     f'$\\pm{tf.raw(dut["torque_cell_Nm"])}$ Nm in-line torque '
                     f'cell.')
    if dut.get('drive'):
        intro.append(f' The DUT was controlled using {_article(dut["drive"])} '
                     f'{tf.escape(dut["drive"])} servo drive.')
    if dut.get('supply_V'):
        intro.append(f' The drive was powered by a '
                     f'{tf.si(dut["supply_V"], VOLT)} DC supply.')

    # A logo that is not on disk is a *fatal* pdflatex error, not a missing
    # image, so a brand-new report root with nothing in pics/ yet would fail to
    # build for a reason that has nothing to do with the data. Omit the
    # \includegraphics when the file is not there and let the report compile;
    # dropping the logo in later needs no regeneration of anything else.
    logo = man.get('logo', 'pics/paladinLogo.png')
    if root is not None and logo:
        have_logo = os.path.isfile(os.path.join(root, logo))
    else:
        have_logo = bool(logo)
    author = tf.escape(man.get('author', 'Paladin Engineering'))
    byline = (f'{author}\\hspace{{0.6em}}%\n'
              f'  \\raisebox{{-0.4\\height}}{{\\includegraphics[height=2.2em]'
              f'{{{logo}}}}}' if have_logo else author)

    body = [
        '% report_generated.tex -- generated by dyno.src.analysis.tex',
        '% Regenerated on every render. To keep commentary, put it in its own',
        '% file and \\input it beside the section it belongs to, or copy this',
        '% driver to a name of your own and edit that.',
        '',
        _PREAMBLE,
        '',
        f'\\title{{\\textbf{{{tf.escape(title)}: Dynamometer Test Report}}}}',
        f'\\author{{{byline}}}',
        '\\date{\\today}',
        '',
        '\\begin{document}',
        '\\maketitle',
        '',
        '\\section{Introduction}',
        ''.join(intro),
        '',
        '\\input{sections/summary_table}',
        '',
    ]
    for key in ordered:
        body.append(f'% ---- {key} '
                    f'{"-" * max(0, 60 - len(key))}')
        body.append(f'\\input{{sections/{key}}}')
        body.append('')
    body.append('\\end{document}')
    return '\n'.join(body) + '\n'
