"""Processor contract: what an analysis declares and what it returns.

Processors are pure with respect to the filesystem. They receive Segments and
return numbers, figures, and findings; the runner names and writes everything.
That is what keeps output naming consistent across the catalog and lets every
processor be unit-tested against synthetic arrays with no temp directory.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .segment import Segment


# Granularity: how the runner groups a log's spans before calling the processor.
#   'setpoint' - one Segment per logged span
#   'run'      - all spans sharing (behavior, run); one complete grid sweep
#   'behavior' - all runs of a behavior, i.e. across loop iterations
GRANULARITIES = ('setpoint', 'run', 'behavior')


@dataclass
class Applicability:
    """Whether a processor can run on a group of segments, and why.

    `reason` is mandatory even when the verdict is 'yes' -- it becomes the UI
    tooltip and the report's justification line, so a run is always
    self-explaining.

    `params` lets the predicate hand its detection work forward (which motor,
    which channels) so `run` does not repeat it.
    """

    verdict: str                                    # 'yes' | 'maybe' | 'no'
    reason: str
    params: dict = field(default_factory=dict)

    @property
    def applicable(self):
        return self.verdict in ('yes', 'maybe')


@dataclass
class Finding:
    """A structured diagnostic. These are the product, not an afterthought --
    a confidently-reported wrong number is the failure mode being engineered
    against."""

    level: str      # 'info' | 'warn' | 'error'
    code: str       # stable slug, e.g. 'sign_flipped'
    message: str


@dataclass
class Result:
    metrics: dict = field(default_factory=dict)
    figures: list = field(default_factory=list)      # [(slug, Figure)]
    tables: list = field(default_factory=list)       # [(slug, csv_text)]
    findings: list = field(default_factory=list)     # [Finding]
    summary: str = ''

    def add(self, level, code, message):
        self.findings.append(Finding(level, code, message))


class Processor:
    """Base class. Subclasses set the class attributes and implement both
    methods. Register with @register from .registry."""

    name: str = ''            # stable slug, used in output filenames
    title: str = ''
    description: str = ''
    granularity: str = 'run'
    params: dict = {}         # {key: (default, type, help)}

    def defaults(self):
        return {k: v[0] for k, v in self.params.items()}

    def applies_to(self, segs: 'list[Segment]') -> Applicability:
        raise NotImplementedError

    def run(self, segs: 'list[Segment]', params: dict) -> Result:
        raise NotImplementedError
