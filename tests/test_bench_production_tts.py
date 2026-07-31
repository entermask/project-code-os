from typing import Any

import pytest

from scripts import bench_production_tts as bench


class StubResponse:
    status_code = 202
    text = ""

    @staticmethod
    def json() -> dict[str, Any]:
        return {
            "request_id": "request-1",
            "status_url": "/v1/tts/jobs/request-1",
            "status": "queued",
            "created_at": 100.0,
            "updated_at": 100.0,
            "lane": "short",
            "chunks_failed": 0,
            "chunks_degraded": 0,
        }


class StubClient:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> StubResponse:
        assert url == "http://api.test/v1/tts"
        assert headers == {"Authorization": "Bearer token"}
        self.payload = json
        return StubResponse()


def test_chunks_per_job_default_remains_ten() -> None:
    args = bench.build_parser().parse_args([])

    assert args.chunks_per_job == 10
    assert args.run_marker == ""
    assert args.start_at_monotonic_ns == 0
    assert bench.chunks_for_job("short-en", 0, False) == bench.DEFAULT_SHORT_CHUNKS


@pytest.mark.parametrize("value", ["0", "11", "not-an-integer"])
def test_chunks_per_job_rejects_values_outside_fixture_range(value: str) -> None:
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--chunks-per-job", value])


@pytest.mark.parametrize(
    "value",
    ["short", "marker-too-long", "unicodeé", "bad mark"],
)
def test_run_marker_requires_fixed_length_safe_ascii(value: str) -> None:
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--run-marker", value])


def test_run_marker_changes_text_without_changing_paired_text_length() -> None:
    first = bench.chunks_for_job(
        "short-en",
        3,
        True,
        chunks_per_job=4,
        run_marker="blockA01",
    )
    second = bench.chunks_for_job(
        "short-en",
        3,
        True,
        chunks_per_job=4,
        run_marker="blockB01",
    )

    assert first != second
    assert [len(chunk) for chunk in first] == [len(chunk) for chunk in second]
    assert all("Batch marker blockA01-0003." in chunk for chunk in first)
    assert all("Batch marker blockB01-0003." in chunk for chunk in second)


def test_start_at_monotonic_ns_rejects_negative_values() -> None:
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--start-at-monotonic-ns", "-1"])


@pytest.mark.asyncio
async def test_wait_until_monotonic_ns_releases_at_target(monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter([100, 200])
    sleeps: list[float] = []

    monkeypatch.setattr(bench.time, "monotonic_ns", lambda: next(readings))

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(bench.asyncio, "sleep", fake_sleep)

    assert await bench.wait_until_monotonic_ns(0) is None
    assert await bench.wait_until_monotonic_ns(200) == 200
    assert sleeps == pytest.approx([100 / 1_000_000_000])


@pytest.mark.asyncio
async def test_submit_one_sends_requested_short_job_chunk_count() -> None:
    args = bench.build_parser().parse_args(
        ["--chunks-per-job", "4", "--text-profile", "short-en"]
    )
    args.ref_base_url = "http://refs.test"
    client = StubClient()

    result = await bench.submit_one(
        client,
        args,
        0,
        [{"file": "ref.mp3", "text": "Reference transcript."}],
        "http://api.test",
        {"Authorization": "Bearer token"},
    )

    assert result.accepted is True
    assert result.submit_started_monotonic_ns is not None
    assert result.submit_started_wall_time_ns is not None
    assert result.created_at == 100.0
    assert result.updated_at == 100.0
    assert result.lane == "short"
    assert result.chunks_failed == 0
    assert result.chunks_degraded == 0
    assert client.payload is not None
    assert client.payload["chunks"] == bench.DEFAULT_SHORT_CHUNKS[:4]
    assert len(client.payload["chunks"]) == 4
    assert set(client.payload) == {
        "chunks",
        "ref_audio_url",
        "ref_text",
        "format",
    }


@pytest.mark.asyncio
async def test_submit_one_forwards_audio_filter_when_requested() -> None:
    args = bench.build_parser().parse_args(
        [
            "--chunks-per-job",
            "1",
            "--audio-format",
            "mp3",
            "--af-filter",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
        ]
    )
    args.ref_base_url = "http://refs.test"
    client = StubClient()

    await bench.submit_one(
        client,
        args,
        0,
        [{"file": "ref.mp3", "text": "Reference transcript."}],
        "http://api.test",
        {"Authorization": "Bearer token"},
    )

    assert client.payload is not None
    assert client.payload["format"] == "mp3"
    assert client.payload["af_filter"] == "loudnorm=I=-16:TP=-1.5:LRA=11"


class PollResponse:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict[str, Any]:
        return {
            "status": "succeeded",
            "chunks_completed": 4,
            "chunks_failed": 0,
            "chunks_degraded": 1,
            "created_at": 200.0,
            "started_at": 200.25,
            "updated_at": 201.75,
            "lane": "short",
            "cache_hit": True,
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 40,
                "total_tokens": 60,
                "engine_time_s": 1.25,
            },
        }


class PollClient:
    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
    ) -> PollResponse:
        assert url == "http://api.test/v1/tts/jobs/request-1"
        assert headers == {"Authorization": "Bearer token"}
        return PollResponse()


@pytest.mark.asyncio
async def test_poll_one_captures_server_job_timestamps_and_counters() -> None:
    result = bench.JobResult(
        index=0,
        accepted=True,
        status_code=202,
        submit_latency_s=0.01,
        submit_started_s=10.0,
        request_id="request-1",
        status_url="/v1/tts/jobs/request-1",
        final_status="queued",
    )

    completed = await bench.poll_one(
        PollClient(),
        result,
        "http://api.test",
        {"Authorization": "Bearer token"},
        poll_interval=0,
        timeout_s=1,
    )

    assert completed.final_status == "succeeded"
    assert completed.chunks_completed == 4
    assert completed.chunks_failed == 0
    assert completed.chunks_degraded == 1
    assert completed.created_at == 200.0
    assert completed.started_at == 200.25
    assert completed.updated_at == 201.75
    assert completed.lane == "short"
    assert completed.server_job_latency_s == pytest.approx(1.75)
    assert completed.server_queue_latency_s == pytest.approx(0.25)
    assert completed.server_service_latency_s == pytest.approx(1.5)


def test_summary_reports_actual_submission_and_server_timing() -> None:
    target_ns = 5_000_000_000
    args = bench.build_parser().parse_args(
        [
            "--jobs",
            "1",
            "--run-marker",
            "blockA01",
            "--start-at-monotonic-ns",
            str(target_ns),
        ]
    )
    result = bench.JobResult(
        index=0,
        accepted=True,
        status_code=202,
        submit_latency_s=0.01,
        submit_started_s=10.0,
        finished_s=12.0,
        final_status="succeeded",
        chunks_completed=10,
        completion_tokens=100,
        submit_started_monotonic_ns=target_ns + 1_000_000,
        submit_started_wall_time_ns=10_000_000_000,
        created_at=300.0,
        started_at=300.25,
        updated_at=302.0,
        lane="long",
        chunks_failed=0,
        chunks_degraded=0,
    )

    summary = bench.summarize_results(
        args,
        started=10.0,
        finished=12.0,
        results=[result],
        gpu_samples=[],
        barrier_released_monotonic_ns=target_ns + 500_000,
    )

    assert summary["config"]["run_marker"] == "blockA01"
    assert summary["timing"]["server_window_s"] == pytest.approx(2.0)
    assert summary["timing"]["submission"] == {
        "target_monotonic_ns": target_ns,
        "barrier_released_monotonic_ns": target_ns + 500_000,
        "barrier_lateness_ms": pytest.approx(0.5),
        "first_submit_monotonic_ns": target_ns + 1_000_000,
        "last_submit_monotonic_ns": target_ns + 1_000_000,
        "submit_fanout_ms": pytest.approx(0.0),
        "first_submit_offset_from_target_ms": pytest.approx(1.0),
        "first_submit_wall_time_ns": 10_000_000_000,
        "last_submit_wall_time_ns": 10_000_000_000,
    }
    assert summary["latency_s"]["server_job"]["avg"] == pytest.approx(2.0)
    assert summary["latency_s"]["server_queue"]["avg"] == pytest.approx(0.25)
    assert summary["latency_s"]["server_service"]["avg"] == pytest.approx(1.75)
    assert summary["rates"]["server_completion_tokens_per_s"] == pytest.approx(50.0)
