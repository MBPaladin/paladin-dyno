"""Post-processing framework: a catalog of analyses that run against a log.

Design notes live in dyno/docs/post_processing_design.md.

The short version: `Segment` owns all HDF5 access, processors are pure
(they return figures and numbers, never touch the filesystem), and the runner
does grouping, dispatch, and all file writing.

Run it with:
    dyno/utilities/analyze.sh <log_dir> [--list] [--only NAME] ...
"""

from .processor import Applicability, Finding, Processor, Result
from .registry import all_processors, get_processor, register
from .segment import Segment, open_log

__all__ = [
    'Applicability', 'Finding', 'Processor', 'Result',
    'all_processors', 'get_processor', 'register',
    'Segment', 'open_log',
]
