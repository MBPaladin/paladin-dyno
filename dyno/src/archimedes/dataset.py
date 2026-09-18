"""One unit's whole test campaign, however many files it landed in.

The problem this solves: a single experiment is routinely split across several
logs, and the split is an accident of the bench day rather than a property of
the test. The 1.2.0 efficiency sweep is four files because a safety tripped
three times; the back-drive velocity ramp is five because the output kept
walking into its position window. The next unit will break in different places.

So nothing here keys off folder names. A Dataset points at a ROOT, finds every
.hdf5 beneath it, and classifies each SPAN by its own behaviour ID and its own
data. Re-splitting the same test over ten files or one changes nothing.

Two pieces of bookkeeping make that safe:

  * Direction is read off the data, not the filename. `BWD_` is a plan tag that
    never reaches a segment ID -- a back-drive ramp logs `V0300-RUN0` exactly
    like the forward one -- so the discriminator is which shaft's command
    SWEEPS (see direction_of). Forward drives from the input, back-drive from
    the output. A filename that disagrees is reported, not obeyed.

  * Restarts overlap. When a run is resumed, the level it died on is repeated,
    so the same behaviour ID appears in two files: once truncated, once whole.
    Keeping both would double-count that point and weight the report toward the
    failure. The longer span wins; every drop is recorded in `duplicates`.
    Ramp tests are exempt -- see `Span.key`: there a repeated ID is a second
    measurement, not a second copy of one.
"""

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from ..analysis.segment import Log
from . import naming

FORWARD = 'forward'
BACKDRIVE = 'backdrive'

# Spans shorter than this carry no usable plateau at any test speed; they are
# aborted starts, not measurements.
MIN_SAMPLES = 200


# A command channel this flat is a HOLD, not a traverse. Both thresholds sit
# far above the observed noise on a held command (exactly 0.0 span on the
# 2026-09-17 ramp plans) and far below any real sweep.
POS_SWEEP_RAD = 0.05
TORQUE_SWEEP_NM = 0.01
VEL_SWEEP_RAD_S = 0.05


def _sweep(seg, channel):
    """How far a command channel actually travels, or None if it is inactive.

    A command channel is all-NaN whenever its mode is inactive, so 'active' and
    'commanded' are the same question. The span is what separates a shaft that
    is being driven from one that is merely being held -- an absorber grounding
    the train sits at a constant position command, and a far shaft parked at
    0 Nm sits at a constant torque command.
    """
    if not seg.is_active(channel):
        return None
    v = seg[channel]
    v = v[np.isfinite(v)]
    return float(np.ptp(v)) if v.size else None


def direction_of(seg):
    """Which shaft is driving: FORWARD (input) or BACKDRIVE (output).

    Read from the data because the segment ID cannot say -- `BWD_` is a plan
    tag that never reaches a segment ID, so a back-drive ramp logs `V0300-RUN0`
    exactly like the forward one.

    The driver is the shaft whose command SWEEPS. That one rule covers both
    shapes of test on this bench, which is why it is preferred to 'whichever
    shaft is position-commanded':

      * A shuttle (velocity ramp, efficiency) position-commands the driver
        through a sawtooth and torque-holds the far side.
      * A ramp (slip, breakaway, stiffness) torque-ramps the driver and GROUNDS
        the far side -- with a constant position command for slip and
        stiffness. Reading 'position-commanded' as 'driving' inverts every one
        of those tests: it calls the locked shaft the driver.

    Position is tried before torque because an efficiency segment sweeps both
    (the held level ramps in at the start), and the traverse is the thing that
    names the driver. Returns None when nothing sweeps.
    """
    for channels, floor in ((('dut_position_command', 'load_position_command'),
                             POS_SWEEP_RAD),
                            (('dut_torque_command', 'load_torque_command'),
                             TORQUE_SWEEP_NM),
                            (('dut_velocity_command', 'load_velocity_command'),
                             VEL_SWEEP_RAD_S)):
        spans = {}
        for role, ch in zip((FORWARD, BACKDRIVE), channels):
            span = _sweep(seg, ch)
            if span is not None and span > floor:
                spans[role] = span
        if spans:
            return max(spans, key=spans.get)
    return None


@dataclass
class Span:
    """One measured segment, with everything needed to interpret it."""

    point: naming.Point
    seg: object                      # analysis.segment.Segment
    direction: str
    source: str                      # log path, relative to the root
    n: int
    mtime: float = 0.0               # of the source file; breaks dedup ties

    @property
    def kind(self):
        return self.point.kind

    def key(self):
        """Identity for de-duplication across a restart boundary.

        The raw ID carries the RUN index, so genuinely repeated ramps
        (RAMP-RUN0, RAMP-RUN1) stay distinct while a resumed level
        (E1500_P40_C2 in two files) collapses.

        RAMP TESTS ARE NEVER COLLAPSED, which is why the source is in the key
        for them. A velocity step or an efficiency level is one operating
        point: recorded twice, the second copy is the same number again and
        the report wants one of them. A slip or breakaway ramp is not -- the
        release is stochastic, the analyzer's output for it is a median and a
        spread over N attempts, and each attempt is a measurement. The IDs
        collide anyway: every fresh launch of the back-drive slip plan starts
        at ramp 1 and logs `SLIP_P1-RUN0`, so six separate runs on 2026-09-18
        collapsed to two and four measured slips went missing from the
        customer's table.
        """
        if self.kind in (naming.SLIP, naming.BREAKAWAY):
            return (self.direction, self.point.raw, self.source)
        return (self.direction, self.point.raw)


@dataclass
class Dataset:
    root: str
    spans: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    # -- selection ---------------------------------------------------------
    def select(self, kind=None, direction=None):
        out = self.spans
        if kind:
            kinds = (kind,) if isinstance(kind, str) else tuple(kind)
            out = [s for s in out if s.kind in kinds]
        if direction:
            out = [s for s in out if s.direction == direction]
        return out

    def kinds(self):
        """{(kind, direction): count} -- what this campaign actually holds."""
        tally = {}
        for s in self.spans:
            tally[(s.kind, s.direction)] = tally.get((s.kind, s.direction), 0) + 1
        return tally

    def ratio(self, default=43.88):
        """|input rad per output rad|, measured, falling back to the config.

        Measured in preference to configured because the configured value is a
        nameplate and the report quotes a real one.

        Measured on the NO-LOAD sweep wherever one exists, and only on it. This
        is the kinematic ratio, and creep is defined against it -- fitting it
        through the loaded efficiency spans as well would fold the creep into
        the ratio and then subtract it from itself, driving the reported creep
        toward zero no matter what the drive did.
        """
        for kinds in ((naming.VELOCITY,), (naming.VELOCITY, naming.EFFICIENCY)):
            got = self._ratio_votes(kinds)
            if got is not None:
                return got, True
        return float(default), False

    def _ratio_votes(self, kinds):
        votes = []
        for s in self.spans:
            if s.kind not in kinds:
                continue
            try:
                wi, wo = s.seg['dut_velocity'], s.seg['load_velocity']
            except KeyError:
                continue
            m = np.isfinite(wi) & np.isfinite(wo) & (np.abs(wo) > 0.02)
            if m.sum() < 200:
                continue
            votes.append(abs(np.polyfit(wo[m], wi[m], 1)[0]))
        return float(np.median(votes)) if votes else None

    def close(self):
        for log in self.logs:
            log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _find_logs(root, exclude):
    """Every .hdf5 under root, minus the excluded subtrees."""
    hits = []
    for path in sorted(glob.glob(os.path.join(root, '**', '*.hdf5'),
                                 recursive=True)):
        rel = os.path.relpath(path, root)
        if any(part in exclude for part in rel.split(os.sep)):
            continue
        hits.append(path)
    return hits


def load(root, exclude=(), keep_kinds=None, duplicates='longest'):
    """Open every log under `root` and build one campaign-wide Dataset.

    `exclude` is a set of path components to skip -- 'initial_setup' and
    'troubleshooting_initial' hold bring-up runs that share the plans' segment
    names and would otherwise land in the customer's numbers.

    `duplicates` decides what to do when the same span appears in more than one
    file, which nothing here can tell apart automatically:

      'longest'  (default) one copy survives. Right for a RESUMED run, where
                 the repeat is the level the previous file died on and the
                 truncated copy would drag the point's average toward the
                 failure.
      'all'      every copy is kept and pooled. Right for a DELIBERATELY
                 repeated sweep, where the copies are independent measurements
                 and collapsing them throws away most of the data -- the
                 back-drive ramp was run end to end three times on gbx_1p2p0.

    Either way the report lists what happened, so the choice is never silent.
    """
    if duplicates not in ('longest', 'all'):
        raise ValueError(f"duplicates must be 'longest' or 'all', "
                         f"not {duplicates!r}")
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise NotADirectoryError(root)
    exclude = set(exclude)

    ds = Dataset(root=root)
    candidates = {}

    for path in _find_logs(root, exclude):
        rel = os.path.relpath(path, root)
        mtime = os.path.getmtime(path)
        try:
            log = Log(os.path.dirname(path), path)
        except Exception as exc:
            ds.skipped.append((rel, f'{type(exc).__name__}: {exc}'))
            continue
        ds.logs.append(log)

        for seg in log.segments:
            point = naming.parse(seg.raw_id)
            if not point.is_data:
                continue
            if keep_kinds and point.kind not in keep_kinds:
                continue
            if len(seg) < MIN_SAMPLES:
                ds.skipped.append((f'{rel}:{seg.raw_id}',
                                   f'only {len(seg)} samples'))
                continue
            direction = direction_of(seg)
            if direction is None:
                ds.skipped.append((f'{rel}:{seg.raw_id}',
                                   'no shaft is commanded; cannot tell '
                                   'forward from back-drive'))
                continue
            span = Span(point=point, seg=seg, direction=direction,
                        source=rel, n=len(seg), mtime=mtime)
            candidates.setdefault(span.key(), []).append(span)

    ds.spans = _resolve(candidates, ds, duplicates)
    _check_filenames(ds)
    return ds


def _resolve(candidates, ds, mode='longest'):
    """Pick one span per identity (or keep them all), recording what happened.

    Resolved in one pass over all the copies rather than pairwise as they
    arrive: a level that survived three restarts has three copies, and a
    running pairwise fight reports the intermediate winners as if they had been
    kept, which reads as a contradiction in the report.
    """
    kept = []
    for key, spans in candidates.items():
        if len(spans) == 1:
            kept.append(spans[0])
            continue
        # The truncated copies are the ones that died, so the longest wins.
        # Equal lengths mean the sweep was re-run rather than resumed -- the
        # plans emit fixed-length traces, so a repeat lands on the same sample
        # count -- and there the later file supersedes.
        winner = max(spans, key=lambda s: (s.n, s.mtime))
        losers = [s for s in spans if s is not winner]
        if mode == 'all':
            kept.extend(spans)
            ds.duplicates.append({
                'segment': winner.point.raw, 'direction': winner.direction,
                'kept': 'all copies pooled', 'kept_n': sum(s.n for s in spans),
                'why': f'duplicates: all ({len(spans)} copies)',
                'dropped': [],
                'pooled': [{'source': s.source, 'n': s.n} for s in spans],
            })
            continue
        kept.append(winner)
        ds.duplicates.append({
            'segment': winner.point.raw, 'direction': winner.direction,
            'kept': winner.source, 'kept_n': winner.n,
            'why': ('longer' if any(l.n < winner.n for l in losers)
                    else 'later re-run'),
            'dropped': [{'source': l.source, 'n': l.n} for l in losers],
        })
    # Sorted, not in discovery order: the same campaign re-filed into different
    # folders must produce byte-identical tables, and a dict built by walking
    # the filesystem does not give that on its own.
    ds.duplicates.sort(key=lambda d: (d['direction'], d['segment']))
    ds.skipped.sort()
    return sorted(kept, key=lambda s: (s.kind, s.direction, s.point.rpm or 0.0,
                                       s.point.torque_nm or 0.0,
                                       s.point.cycle or 0, s.point.raw))


def _check_filenames(ds):
    """Report -- never act on -- a filename that disagrees with the data.

    `archimedes_bwd_*` in a path is the operator saying 'this is back-drive'.
    The data says which shaft drove. When they disagree one of them is wrong,
    and it matters enough to the customer's numbers to be said out loud.
    """
    for span in ds.spans:
        low = span.source.lower()
        claimed = (BACKDRIVE if '_bwd' in low or 'bkw' in low
                   else FORWARD if '_fwd' in low or 'fwd_' in low else None)
        if claimed and claimed != span.direction:
            ds.notes.append(
                f'{span.source}:{span.point.raw} is named {claimed} but the '
                f'{"input" if span.direction == FORWARD else "output"} shaft is '
                f'the one commanded -- read as {span.direction}')
