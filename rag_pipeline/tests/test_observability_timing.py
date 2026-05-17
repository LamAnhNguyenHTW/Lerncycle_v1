from __future__ import annotations

import asyncio
import time

import pytest

from rag_pipeline.observability.timing import SkippedSpan
from rag_pipeline.observability.timing import Timer
from rag_pipeline.observability.timing import TimingReport
from rag_pipeline.observability.timing import current_timing_report
from rag_pipeline.observability.timing import timing_report_context


def test_timer_records_stage_start_and_duration() -> None:
    report = TimingReport()

    with timing_report_context(report):
        with Timer("stage"):
            time.sleep(0.001)

    payload = report.to_dict()

    assert payload["total_ms"] >= 0
    assert len(payload["stages"]) == 1
    stage = payload["stages"][0]
    assert stage["stage"] == "stage"
    assert stage["started_at"] >= report.started_at
    assert stage["duration_ms"] >= 0
    assert stage["status"] == "ok"


def test_nested_timers_record_both_spans_in_start_order() -> None:
    report = TimingReport()

    with timing_report_context(report):
        with Timer("outer"):
            with Timer("inner"):
                time.sleep(0.001)

    stages = report.to_dict()["stages"]

    assert [stage["stage"] for stage in stages] == ["outer", "inner"]
    assert stages[0]["started_at"] <= stages[1]["started_at"]


def test_parallel_timers_record_overlapping_intervals() -> None:
    report = TimingReport()

    async def run_stage(stage: str) -> None:
        with Timer(stage):
            await asyncio.sleep(0.02)

    async def run_parallel() -> None:
        with timing_report_context(report):
            await asyncio.gather(run_stage("left"), run_stage("right"))

    asyncio.run(run_parallel())

    stages = {stage["stage"]: stage for stage in report.to_dict()["stages"]}
    left = stages["left"]
    right = stages["right"]
    left_end = left["started_at"] + (left["duration_ms"] / 1000)
    right_end = right["started_at"] + (right["duration_ms"] / 1000)

    assert left["started_at"] < right_end
    assert right["started_at"] < left_end


def test_skipped_span_records_zero_duration_without_context_body() -> None:
    report = TimingReport()

    with timing_report_context(report):
        SkippedSpan("graph_retrieve")

    assert report.to_dict()["stages"] == [
        {
            "stage": "graph_retrieve",
            "started_at": pytest.approx(report.started_at),
            "duration_ms": 0,
            "status": "skipped",
        }
    ]


def test_current_report_is_context_bound_and_restored() -> None:
    outer = TimingReport()
    inner = TimingReport()

    with timing_report_context(outer):
        assert current_timing_report() is outer
        with timing_report_context(inner):
            assert current_timing_report() is inner
        assert current_timing_report() is outer

    assert current_timing_report() is None
