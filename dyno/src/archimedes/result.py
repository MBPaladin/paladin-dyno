"""What an Archimedes analyzer returns.

Same shape as the main framework's analysis.processor.Result -- figures,
tables, metrics, findings -- so the two stay legible side by side. It is a
separate class rather than an import because these analyzers are not
Processors: they take a whole multi-log Dataset instead of a segment group, and
sharing the base class would imply a contract they do not keep.
"""

from dataclasses import dataclass, field


@dataclass
class Finding:
    level: str          # 'info' | 'warn' | 'error'
    code: str
    message: str


@dataclass
class Result:
    name: str
    title: str
    metrics: dict = field(default_factory=dict)
    figures: list = field(default_factory=list)      # [(slug, Figure)]
    tables: list = field(default_factory=list)       # [(slug, csv_text)]
    findings: list = field(default_factory=list)
    summary: str = ''

    def add(self, level, code, message):
        self.findings.append(Finding(level, code, message))

    @property
    def failed(self):
        return any(f.level == 'error' for f in self.findings)
