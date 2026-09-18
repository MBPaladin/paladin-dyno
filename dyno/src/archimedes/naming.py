"""Segment-ID grammar for the Archimedes plans.

The plans in tests/traces/generate_archimedes_tests.py name every span after
what it measures, and that name is the only place the commanded speed and
torque level survive into the log -- the test YAML is not carried in the HDF5,
only the resolved bench config is. So the whole analysis keys off these IDs:

    V0300           velocity ramp step, 300 rpm INPUT-referred
    V0300_RC        the recentre move after it (not data)
    E1500_P40       efficiency, 1500 rpm input, +40 Nm at the OUTPUT
    E1500_N40_C3    ... third cycle of the -40 Nm level
    K45 / K90       stiffness, at 45% / 90% of measured static slip
    STICT           input breakaway stiction ramp
    SLIP            static slip hunt (far shaft position-held)
    RAMP            breakaway ramp, standalone plan
    SLIP_P1 / _N2   back-drive slip, ramp 1 positive / ramp 2 negative
    BRK_P1 / _N3    back-drive breakaway, likewise
    SLOW / FAST     gravity + drag map traverses

The `_P<n>` / `_N<n>` forms come from `ramp_and_recentre`, which gives every
attempt its own segment so a recentre can sit between them -- a back-drive ramp
that lets go leaves the output metres from centre in output-referred terms, and
a bipolar segment has nowhere to put the move back. The suffix is therefore the
attempt number and its rotation sense, not a different measurement: all of
`SLIP_P1`, `SLIP_N1`, `SLIP_P2` ... are repeats of the one static slip test.

`BWD_` is a PLAN tag, not a segment prefix: a back-drive velocity ramp logs
`V0300-RUN0` exactly like the forward one does. Direction is therefore never
read from an ID -- see dataset.direction_of, which reads it off the data.
"""

import re
from dataclasses import dataclass

from ..analysis.segment import parse_behavior_id

# Recognised behaviours, in the order they are tried.
_VELOCITY = re.compile(r'^V(\d{3,5})$')
_EFFICIENCY = re.compile(r'^E(\d{3,5})_([PN])(\d{1,4})(?:_C(\d+))?$')
_STIFFNESS = re.compile(r'^K(\d{1,3})$')
_GRAVITY = re.compile(r'^(SLOW|FAST)$')
# The ramp tests, with the optional per-attempt suffix ramp_and_recentre adds.
# Bare `SLIP`/`STICT`/`RAMP` are the forward plans, which run their attempts as
# one bipolar segment and so need no suffix.
_RAMP = re.compile(r'^(SLIP|BRK|STICT|RAMP)(?:_([PN])(\d{1,2}))?$')
_RAMP_KINDS = {'SLIP': 'slip', 'BRK': 'breakaway',
               'STICT': 'breakaway', 'RAMP': 'breakaway'}

# Kinds. One per client-facing test, plus the two housekeeping ones.
VELOCITY = 'velocity'
EFFICIENCY = 'efficiency'
STIFFNESS = 'stiffness'
SLIP = 'slip'
BREAKAWAY = 'breakaway'
GRAVITY = 'gravity'
RECENTRE = 'recentre'
UNKNOWN = 'unknown'

# What each kind answers in the customer's test list, for report headings.
KIND_TITLES = {
    VELOCITY: 'Velocity ramp up',
    EFFICIENCY: 'Efficiency',
    STIFFNESS: 'Stiffness',
    SLIP: 'Static slip torque',
    BREAKAWAY: 'Breakaway / stiction',
    GRAVITY: 'Gravity and drag map',
}


@dataclass(frozen=True)
class Point:
    """What one segment ID says the segment is."""

    raw: str                    # 'E1500_P40_C3-RUN0', exactly as logged
    behavior: str               # 'E1500_P40_C3'
    kind: str
    run: int = None
    setpoint: int = None
    rpm: float = None           # commanded INPUT-referred speed, rpm
    torque_nm: float = None     # commanded OUTPUT torque level, signed Nm
    cycle: int = None           # _C<n>, when a level was split per cycle
    pct: int = None             # K45 -> 45
    attempt: int = None         # SLIP_P2 -> 2; which ramp of the repeat set
    sense: int = None           # SLIP_N1 -> -1; which way the ramp pushed

    @property
    def is_data(self):
        """Recentre moves are travel between measurements, never measurements."""
        return self.kind not in (RECENTRE, UNKNOWN)

    @property
    def level_key(self):
        """Identifies an efficiency LEVEL across its cycles: (rpm, torque)."""
        return (self.rpm, self.torque_nm)

    def label(self):
        if self.kind == VELOCITY:
            return f'{self.rpm:g} rpm'
        if self.kind == EFFICIENCY:
            c = f' c{self.cycle}' if self.cycle else ''
            return f'{self.rpm:g} rpm, {self.torque_nm:+g} Nm{c}'
        if self.kind == STIFFNESS:
            return f'{self.pct}% of slip'
        if self.attempt is not None:
            return (f'ramp {self.attempt}, '
                    f'{"positive" if self.sense > 0 else "negative"}')
        return self.behavior


def parse(raw_id):
    """'E1500_P40_C3-RUN0' -> Point. Never raises; unknown IDs come back
    kind=UNKNOWN so a hand-named span cannot abort a whole batch."""
    behavior, run, setpoint = parse_behavior_id(raw_id)
    common = dict(raw=raw_id, behavior=behavior, run=run, setpoint=setpoint)

    # Recentres are tagged by suffix on whatever they follow, so this is first.
    if behavior.endswith('_RC'):
        return Point(kind=RECENTRE, **common)

    m = _VELOCITY.match(behavior)
    if m:
        return Point(kind=VELOCITY, rpm=float(m.group(1)), **common)

    m = _EFFICIENCY.match(behavior)
    if m:
        rpm, sign, mag, cycle = m.groups()
        return Point(kind=EFFICIENCY, rpm=float(rpm),
                     torque_nm=float(mag) * (1 if sign == 'P' else -1),
                     cycle=int(cycle) if cycle else None, **common)

    m = _STIFFNESS.match(behavior)
    if m:
        return Point(kind=STIFFNESS, pct=int(m.group(1)), **common)

    m = _GRAVITY.match(behavior)
    if m:
        return Point(kind=GRAVITY, **common)

    # Ramp tests last, because `_RAMP` also matches the suffixed back-drive
    # forms and the bare forward ones, and nothing above can be confused with
    # either.
    m = _RAMP.match(behavior)
    if m:
        stem, sign, attempt = m.groups()
        return Point(kind=_RAMP_KINDS[stem],
                     attempt=int(attempt) if attempt else None,
                     sense=None if sign is None else (1 if sign == 'P' else -1),
                     **common)

    return Point(kind=UNKNOWN, **common)
