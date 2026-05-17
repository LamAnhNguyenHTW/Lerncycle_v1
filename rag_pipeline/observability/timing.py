from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Literal

TimingStatus = Literal["ok", "error", "skipped", "timeout"]

_current_report: ContextVar["TimingReport | None"] = ContextVar(
    "rag_pipeline_timing_report",
    default=None,
)


@dataclass(frozen=True)
class TimingSpan:
    stage: str
    started_at: float
    duration_ms: float
    status: TimingStatus = "ok"

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
        }


class TimingReport:
    """Per-request timing report for ordered RAG pipeline stage spans."""

    def __init__(self) -> None:
        self.started_at = perf_counter()
        self._spans: list[TimingSpan] = []
        self._sequence: list[tuple[int, TimingSpan]] = []

    def add_span(
        self,
        stage: str,
        *,
        started_at: float,
        duration_ms: float,
        status: TimingStatus = "ok",
    ) -> None:
        span = TimingSpan(
            stage=stage,
            started_at=started_at,
            duration_ms=max(0, duration_ms),
            status=status,
        )
        self._spans.append(span)
        self._sequence.append((len(self._sequence), span))

    def to_dict(self) -> dict:
        ordered = [
            span
            for _, span in sorted(
                self._sequence,
                key=lambda item: (item[1].started_at, item[0]),
            )
        ]
        return {
            "started_at": self.started_at,
            "total_ms": max(0, (perf_counter() - self.started_at) * 1000),
            "stages": [span.to_dict() for span in ordered],
            "parallel_groups": [],
        }


class Timer:
    """Context manager that records a timing span on the current report."""

    def __init__(self, stage: str, *, status: TimingStatus = "ok") -> None:
        self.stage = stage
        self.status: TimingStatus = status
        self._started_at: float | None = None

    def __enter__(self) -> "Timer":
        self._started_at = perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._started_at is None:
            return False
        status = "error" if exc_type is not None and self.status == "ok" else self.status
        report = current_timing_report()
        if report is not None:
            report.add_span(
                self.stage,
                started_at=self._started_at,
                duration_ms=(perf_counter() - self._started_at) * 1000,
                status=status,
            )
        return False


def SkippedSpan(stage: str) -> None:
    """Record a stable zero-duration skipped span on the current report."""

    report = current_timing_report()
    if report is not None:
        report.add_span(
            stage,
            started_at=report.started_at,
            duration_ms=0,
            status="skipped",
        )


def current_timing_report() -> TimingReport | None:
    return _current_report.get()


@contextmanager
def timing_report_context(report: TimingReport) -> Iterator[TimingReport]:
    token = _current_report.set(report)
    try:
        yield report
    finally:
        _current_report.reset(token)
