"""Tests for Trenni supervisor: unified spawn model, checkpoint, and replay."""
import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trenni.config import TrenniConfig
from trenni.isolation import JobProcess
from trenni.pasloe_client import Event
from trenni.supervisor import SpawnedJob, Supervisor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evt(id_: str, type_: str, data: dict | None = None) -> Event:
    return Event(id=id_, source_id="test", type=type_,
                 ts=datetime.utcnow(), data=data or {})


def _make_supervisor(**overrides) -> Supervisor:
    return Supervisor(TrenniConfig(**overrides))


# ---------------------------------------------------------------------------
# SpawnedJob basics
# ---------------------------------------------------------------------------

def test_spawned_job_defaults():
    j = SpawnedJob("j1", "e1", "task", "default", "/r", "main", None)
    assert j.depends_on == frozenset()


def test_spawned_job_with_deps():
    j = SpawnedJob("j1", "e1", "task", "default", "/r", "main", None,
                   depends_on=frozenset({"a", "b"}))
    assert j.depends_on == frozenset({"a", "b"})


# ---------------------------------------------------------------------------
# UUID v7
# ---------------------------------------------------------------------------

def test_generate_job_id_is_uuid_v7():
    sup = _make_supervisor()
    job_id = sup._generate_job_id()
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        job_id,
    ), f"Not a UUID v7: {job_id}"


# ---------------------------------------------------------------------------
# Queue and dedup state
# ---------------------------------------------------------------------------

def test_supervisor_has_ready_queue_and_dedup():
    sup = _make_supervisor()
    assert isinstance(sup._ready_queue, asyncio.Queue)
    assert isinstance(sup._launched_event_ids, set)
    assert isinstance(sup._pending, dict)
    assert isinstance(sup._completed_jobs, set)


# ---------------------------------------------------------------------------
# task.submit → enqueue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_task_submit_enqueues():
    sup = _make_supervisor()
    event = _evt("evt-abc", "task.submit",
                 {"task": "do X", "role": "default", "repo": "/r", "init_branch": "main"})

    await sup._handle_task_submit(event)

    assert sup._ready_queue.qsize() == 1
    job = sup._ready_queue.get_nowait()
    assert job.source_event_id == "evt-abc"
    assert job.task == "do X"
    assert job.depends_on == frozenset()


@pytest.mark.asyncio
async def test_handle_task_submit_deduplicates():
    sup = _make_supervisor()
    sup._launched_event_ids.add("evt-dup")
    event = _evt("evt-dup", "task.submit",
                 {"task": "do X", "role": "default"})

    await sup._handle_task_submit(event)
    assert sup._ready_queue.qsize() == 0


@pytest.mark.asyncio
async def test_handle_task_submit_ignores_empty_task():
    sup = _make_supervisor()
    event = _evt("evt-empty", "task.submit", {"task": "", "role": "default"})

    await sup._handle_task_submit(event)
    assert sup._ready_queue.qsize() == 0
    assert "evt-empty" not in sup._launched_event_ids


# ---------------------------------------------------------------------------
# job.spawn.request → fan-out children (no continuation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_spawn_creates_children_no_continuation():
    sup = _make_supervisor()
    event = _evt("spawn-1", "job.spawn.request", {
        "job_id": "parent-1",
        "tasks": [
            {"task": "child A", "role_file": "roles/worker.py", "repo": "/r", "branch": "main"},
            {"task": "child B", "role_file": "roles/worker.py", "repo": "/r", "branch": "main"},
        ],
        "wait_for": "all_complete",
    })

    await sup._handle_spawn(event)

    # 2 children should be in ready queue
    assert sup._ready_queue.qsize() == 2
    c0 = sup._ready_queue.get_nowait()
    c1 = sup._ready_queue.get_nowait()
    assert c0.job_id == "parent-1-c0"
    assert c1.job_id == "parent-1-c1"
    assert c0.role == "worker"
    assert c0.depends_on == frozenset()

    # No continuation should be created
    assert len(sup._pending) == 0


@pytest.mark.asyncio
async def test_handle_spawn_empty_tasks_ignored():
    sup = _make_supervisor()
    event = _evt("spawn-empty", "job.spawn.request", {
        "job_id": "parent-2",
        "tasks": [],
    })
    await sup._handle_spawn(event)
    assert sup._ready_queue.qsize() == 0
    assert len(sup._pending) == 0


# ---------------------------------------------------------------------------
# _enqueue: dependency routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enqueue_immediate_goes_to_ready():
    sup = _make_supervisor()
    job = SpawnedJob("j1", "e1", "t", "r", "/r", "main", None)
    await sup._enqueue(job)
    assert sup._ready_queue.qsize() == 1
    assert "j1" not in sup._pending


@pytest.mark.asyncio
async def test_enqueue_with_deps_goes_to_pending():
    sup = _make_supervisor()
    job = SpawnedJob("j1", "e1", "t", "r", "/r", "main", None,
                     depends_on=frozenset({"dep-1"}))
    await sup._enqueue(job)
    assert sup._ready_queue.qsize() == 0
    assert "j1" in sup._pending


@pytest.mark.asyncio
async def test_enqueue_with_satisfied_deps_goes_to_ready():
    sup = _make_supervisor()
    sup._completed_jobs.add("dep-1")
    job = SpawnedJob("j1", "e1", "t", "r", "/r", "main", None,
                     depends_on=frozenset({"dep-1"}))
    await sup._enqueue(job)
    assert sup._ready_queue.qsize() == 1
    assert "j1" not in sup._pending


# ---------------------------------------------------------------------------
# job.completed / job.failed → dependency resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_job_done_resolves_pending():
    sup = _make_supervisor()

    # Set up: a job waiting on two dependencies
    dep_job = SpawnedJob("p-join", "e1", "join task", "default", "/r", "main", None,
                         depends_on=frozenset({"p-c0", "p-c1"}))
    sup._pending["p-join"] = dep_job

    # Complete first dependency
    await sup._handle_job_done(_evt("d1", "job.completed", {
        "job_id": "p-c0", "summary": "done A",
    }))
    assert "p-c0" in sup._completed_jobs
    assert sup._ready_queue.qsize() == 0   # still waiting
    assert "p-join" in sup._pending

    # Complete second dependency
    await sup._handle_job_done(_evt("d2", "job.completed", {
        "job_id": "p-c1", "summary": "done B",
    }))
    assert sup._ready_queue.qsize() == 1   # released
    assert "p-join" not in sup._pending

    resolved = sup._ready_queue.get_nowait()
    assert resolved.job_id == "p-join"


@pytest.mark.asyncio
async def test_handle_job_failed_propagates_to_dependents():
    sup = _make_supervisor()

    # Mock client.emit for failure propagation
    emitted = []
    async def fake_emit(type_, data):
        emitted.append((type_, data))
    sup.client.emit = fake_emit

    # Set up: a job waiting on two dependencies
    dep_job = SpawnedJob("p-join", "e1", "join task", "default", "/r", "main", None,
                         depends_on=frozenset({"p-c0", "p-c1"}))
    sup._pending["p-join"] = dep_job

    # First child fails
    await sup._handle_job_done(_evt("d1", "job.failed", {
        "job_id": "p-c0", "error": "crash",
    }))

    # Failed job should be tracked
    assert "p-c0" in sup._completed_jobs
    assert "p-c0" in sup._failed_jobs

    # Dependent job should be propagated as failed, not enqueued
    assert sup._ready_queue.qsize() == 0
    assert "p-join" not in sup._pending
    assert "p-join" in sup._failed_jobs

    # Should have emitted a job.failed event for the dependent
    fail_events = [(t, d) for t, d in emitted if t == "job.failed"]
    assert len(fail_events) == 1
    assert fail_events[0][1]["job_id"] == "p-join"
    assert fail_events[0][1]["code"] == "dependency_failed"


@pytest.mark.asyncio
async def test_handle_job_done_caches_summary():
    sup = _make_supervisor()
    await sup._handle_job_done(_evt("d1", "job.completed", {
        "job_id": "j1", "summary": "all good",
    }))
    assert sup._job_summaries["j1"] == "all good"


@pytest.mark.asyncio
async def test_handle_job_done_removes_from_jobs():
    sup = _make_supervisor()
    fake_proc = AsyncMock()
    fake_proc.returncode = 0
    sup.jobs["j1"] = JobProcess("j1", fake_proc, Path("/tmp"), Path("/tmp/c.yaml"))

    await sup._handle_job_done(_evt("d1", "job.completed", {"job_id": "j1"}))
    assert "j1" not in sup.jobs


# ---------------------------------------------------------------------------
# Drain queue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_drain_queue_launches_when_capacity():
    sup = _make_supervisor(max_workers=2)
    job = SpawnedJob("job-1", "evt-1", "task", "default", "/repo", "main", None)
    await sup._ready_queue.put(job)

    launched = []
    async def fake_launch(j):
        launched.append(j.job_id)
    sup._launch_from_spawned = fake_launch

    task = asyncio.create_task(sup._drain_queue())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert launched == ["job-1"]


@pytest.mark.asyncio
async def test_drain_queue_waits_when_at_capacity():
    sup = _make_supervisor(max_workers=1)

    fake_proc = AsyncMock()
    fake_proc.returncode = None
    sup.jobs["existing"] = JobProcess("existing", fake_proc, Path("/tmp"), Path("/tmp/c.yaml"))

    job = SpawnedJob("job-2", "evt-2", "task", "default", "/repo", "main", None)
    await sup._ready_queue.put(job)

    launched = []
    async def fake_launch(j):
        launched.append(j.job_id)
    sup._launch_from_spawned = fake_launch

    task = asyncio.create_task(sup._drain_queue())
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert launched == []


# ---------------------------------------------------------------------------
# _launch emits source_event_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_launch_emits_source_event_id():
    sup = _make_supervisor()

    emitted = []
    async def fake_emit(type_, data):
        emitted.append((type_, data))
    sup.client.emit = fake_emit

    fake_proc = AsyncMock()
    fake_proc.returncode = None
    fake_proc.pid = 9999
    fake_jp = MagicMock()
    fake_jp.proc = fake_proc

    with patch("trenni.supervisor.launch_job", new_callable=AsyncMock, return_value=fake_jp):
        await sup._launch("job-xyz", "test", "default", "/repo", "main", None,
                          source_event_id="evt-src-123")

    assert emitted
    type_, data = emitted[0]
    assert type_ == "supervisor.job.launched"
    assert data["source_event_id"] == "evt-src-123"
    assert data["job_id"] == "job-xyz"


# ---------------------------------------------------------------------------
# Checkpoint + reap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mark_exited_processes():
    sup = _make_supervisor()
    fake_proc = AsyncMock()
    fake_proc.returncode = 1
    jp = JobProcess("j1", fake_proc, Path("/tmp"), Path("/tmp/c.yaml"))
    sup.jobs["j1"] = jp
    assert jp.exited_at is None

    sup._mark_exited_processes()
    assert jp.exited_at is not None

    # Calling again doesn't overwrite
    first_ts = jp.exited_at
    sup._mark_exited_processes()
    assert jp.exited_at == first_ts


@pytest.mark.asyncio
async def test_checkpoint_reaps_timed_out_processes():
    sup = _make_supervisor()
    sup._reap_timeout = 0.0  # immediate reap for test

    emitted = []
    async def fake_emit(type_, data):
        emitted.append((type_, data))
    sup.client.emit = fake_emit

    fake_proc = AsyncMock()
    fake_proc.returncode = 137
    jp = JobProcess("j1", fake_proc, Path("/tmp"), Path("/tmp/c.yaml"),
                    exited_at=time.monotonic() - 1.0)
    sup.jobs["j1"] = jp

    await sup._checkpoint()

    assert "j1" not in sup.jobs
    # Should have emitted job.failed + supervisor.checkpoint
    types = [e[0] for e in emitted]
    assert "job.failed" in types
    assert "supervisor.checkpoint" in types
    fail_data = next(d for t, d in emitted if t == "job.failed")
    assert fail_data["code"] == "process_lost"


@pytest.mark.asyncio
async def test_checkpoint_emits_state():
    sup = _make_supervisor()

    emitted = []
    async def fake_emit(type_, data):
        emitted.append((type_, data))
    sup.client.emit = fake_emit

    sup.event_cursor = "2026-01-01T00:00:00|abc"
    sup._pending["p1"] = SpawnedJob("p1", "e", "t", "r", "/r", "m", None)

    await sup._checkpoint()

    cp = next(d for t, d in emitted if t == "supervisor.checkpoint")
    assert cp["cursor"] == "2026-01-01T00:00:00|abc"
    assert cp["pending_jobs"] == ["p1"]


# ---------------------------------------------------------------------------
# _fetch_all pagination
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_all_paginates():
    sup = _make_supervisor()
    page1 = ([_evt("e1", "job.started", {"job_id": "e1"}),
              _evt("e2", "job.started", {"job_id": "e2"})], "cursor-2")
    page2 = ([_evt("e3", "job.started", {"job_id": "e3"})], None)
    pages = [page1, page2]
    call_count = 0

    async def fake_poll(cursor=None, source=None, type_=None, limit=100):
        nonlocal call_count
        result = pages[call_count]
        call_count += 1
        return result

    sup.client.poll = fake_poll
    events = await sup._fetch_all("job.started")
    assert [e.id for e in events] == ["e1", "e2", "e3"]
    assert call_count == 2


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_replay_enqueues_not_launched():
    sup = _make_supervisor()

    async def fake_fetch_all(type_, source=None):
        if type_ == "task.submit":
            return [_evt("sub-1", "task.submit",
                         {"task": "do X", "role": "default", "repo": "/r", "branch": "main"})]
        return []
    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup._ready_queue.qsize() == 1


@pytest.mark.asyncio
async def test_replay_skips_completed():
    sup = _make_supervisor()

    async def fake_fetch_all(type_, source=None):
        if type_ == "task.submit":
            return [_evt("sub-1", "task.submit",
                         {"task": "do X", "role": "default"})]
        if type_ == "supervisor.job.launched":
            return [_evt("l1", "supervisor.job.launched",
                         {"source_event_id": "sub-1", "job_id": "job-A"})]
        if type_ == "job.started":
            return [_evt("s1", "job.started", {"job_id": "job-A"})]
        if type_ == "job.completed":
            return [_evt("c1", "job.completed", {"job_id": "job-A", "summary": "ok"})]
        return []
    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup._ready_queue.qsize() == 0
    assert "sub-1" in sup._launched_event_ids
    assert "job-A" in sup._completed_jobs


@pytest.mark.asyncio
async def test_replay_reenqueues_launched_not_started():
    sup = _make_supervisor()

    async def fake_fetch_all(type_, source=None):
        if type_ == "task.submit":
            return [_evt("sub-1", "task.submit",
                         {"task": "do X", "role": "default", "repo": "/r", "branch": "main"})]
        if type_ == "supervisor.job.launched":
            return [_evt("l1", "supervisor.job.launched",
                         {"source_event_id": "sub-1", "job_id": "job-A"})]
        return []
    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup._ready_queue.qsize() == 1


@pytest.mark.asyncio
async def test_replay_uses_checkpoint_cursor():
    sup = _make_supervisor()

    async def fake_fetch_all(type_, source=None):
        if type_ == "supervisor.checkpoint":
            return [_evt("cp1", "supervisor.checkpoint",
                         {"cursor": "2026-01-01T00:00:00|saved-cursor"})]
        if type_ == "task.submit":
            return []
        return []
    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup.event_cursor == "2026-01-01T00:00:00|saved-cursor"


# ---------------------------------------------------------------------------
# start() integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_calls_replay_and_drain():
    sup = _make_supervisor()

    replay_called = False
    async def fake_replay():
        nonlocal replay_called
        replay_called = True
    sup._replay_unfinished_tasks = fake_replay

    with patch.object(sup, "_run_loop", new_callable=AsyncMock) as mock_run_loop, \
         patch.object(sup.client, "register_source", new_callable=AsyncMock), \
         patch.object(sup.client, "close", new_callable=AsyncMock), \
         patch("asyncio.create_task") as mock_create_task:

        async def _noop(): pass
        class AwaitableTask(AsyncMock):
            def __await__(self):
                return _noop().__await__()

        mock_task = AwaitableTask()
        mock_create_task.return_value = mock_task

        await sup.start()

    assert replay_called
    mock_run_loop.assert_called_once()
    mock_create_task.assert_called_once()
    mock_task.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# Status property
# ---------------------------------------------------------------------------

def test_status_reflects_unified_state():
    sup = _make_supervisor()
    sup._pending["p1"] = SpawnedJob("p1", "e", "t", "r", "/r", "m", None)
    st = sup.status
    assert st["pending_jobs"] == 1
    assert "ready_queue_size" in st
    assert "fork_joins_active" not in st  # old field removed
