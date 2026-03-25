"""Tests for Trenni supervisor: queueing, Podman launch, checkpoint, and replay."""
import asyncio
import re
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trenni.config import TrenniConfig
from trenni.pasloe_client import Event
from trenni.runtime_types import ContainerState, JobHandle, JobRuntimeSpec
from trenni.supervisor import SpawnDefaults, SpawnedJob, Supervisor


def _evt(id_: str, type_: str, data: dict | None = None) -> Event:
    return Event(id=id_, source_id="test", type=type_,
                 ts=datetime.utcnow(), data=data or {})


def _make_supervisor(**overrides) -> Supervisor:
    return Supervisor(TrenniConfig(**overrides))


def test_spawned_job_defaults():
    j = SpawnedJob("j1", "e1", "task", "default", "/r", "main", None)
    assert j.depends_on == frozenset()


def test_spawned_job_with_deps():
    j = SpawnedJob("j1", "e1", "task", "default", "/r", "main", None,
                   depends_on=frozenset({"a", "b"}))
    assert j.depends_on == frozenset({"a", "b"})


def test_generate_job_id_is_uuid_v7():
    sup = _make_supervisor()
    job_id = sup._generate_job_id()
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        job_id,
    ), f"Not a UUID v7: {job_id}"


def test_supervisor_has_ready_queue_and_dedup():
    sup = _make_supervisor()
    assert isinstance(sup._ready_queue, asyncio.Queue)
    assert isinstance(sup._processed_event_ids, set)
    assert isinstance(sup._launched_event_ids, set)
    assert isinstance(sup._pending, dict)
    assert isinstance(sup._completed_jobs, set)
    assert sup.runtime_defaults.kind == "podman"


@pytest.mark.asyncio
async def test_handle_trigger_enqueues():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    event = _evt("evt-abc", "trigger.external",
                 {"goal": "do X", "context": {"role": "default", "repo": "/r", "init_branch": "main"}})

    await sup._handle_trigger(event)

    assert sup._ready_queue.qsize() == 1
    job = sup._ready_queue.get_nowait()
    assert job.source_event_id == "evt-abc"
    assert job.task == "do X"
    assert job.depends_on == frozenset()


@pytest.mark.asyncio
async def test_handle_trigger_deduplicates():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup._launched_event_ids.add("evt-dup")
    event = _evt("evt-dup", "trigger.external",
                 {"goal": "do X", "context": {"role": "default"}})

    await sup._handle_trigger(event)
    assert sup._ready_queue.qsize() == 0


@pytest.mark.asyncio
async def test_handle_trigger_ignores_empty_goal():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    event = _evt("evt-empty", "trigger.external", {"goal": "", "context": {"role": "default"}})

    await sup._handle_trigger(event)
    assert sup._ready_queue.qsize() == 0
    assert "evt-empty" not in sup._launched_event_ids


@pytest.mark.asyncio
async def test_handle_spawn_creates_children_no_continuation():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup._spawn_defaults_by_job["parent-1"] = SpawnDefaults(
        repo="/parent-repo",
        init_branch="main",
        role="default",
        evo_sha="parent-sha",
        task_id="task-1",
    )
    event = _evt("spawn-1", "job.spawn.request", {
        "job_id": "parent-1",
        "tasks": [
            {"prompt": "child A", "job_spec": {"role": "worker", "repo": "/r", "init_branch": "main"}},
            {"prompt": "child B", "job_spec": {"role": "worker", "repo": "/r", "init_branch": "main"}},
        ],
        "wait_for": "all_complete",
    })

    await sup._handle_spawn(event)

    assert sup._ready_queue.qsize() == 2
    c0 = sup._ready_queue.get_nowait()
    c1 = sup._ready_queue.get_nowait()
    assert c0.job_id == "parent-1-c0"
    assert c1.job_id == "parent-1-c1"
    assert c0.task == "child A"
    assert c0.role == "worker"
    assert c0.depends_on == frozenset()
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


@pytest.mark.asyncio
async def test_handle_event_deduplicates_spawn_request():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup._spawn_defaults_by_job["parent-1"] = SpawnDefaults(
        repo="/parent-repo",
        init_branch="main",
        role="default",
        evo_sha="parent-sha",
        task_id="task-1",
    )
    event = _evt("spawn-dup", "job.spawn.request", {
        "job_id": "parent-1",
        "tasks": [
            {"prompt": "child A", "job_spec": {"role": "worker", "repo": "/r", "init_branch": "main"}},
            {"prompt": "child B", "job_spec": {"role": "worker", "repo": "/r", "init_branch": "main"}},
        ],
    })

    await sup._handle_event(event, realtime=True)
    await sup._handle_event(event)

    assert sup._ready_queue.qsize() == 2


@pytest.mark.asyncio
async def test_handle_spawn_inherits_parent_defaults_for_missing_job_spec_fields():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup._spawn_defaults_by_job["parent-1"] = SpawnDefaults(
        repo="/parent-repo",
        init_branch="parent-branch",
        role="parent-role",
        evo_sha="parent-sha",
        task_id="task-1",
        llm_overrides={"model": "kimi-parent"},
        workspace_overrides={"depth": 2},
        publication_overrides={"branch_prefix": "parent/job"},
    )
    event = _evt("spawn-inherit", "job.spawn.request", {
        "job_id": "parent-1",
        "tasks": [
            {"prompt": "child task", "job_spec": {"role": "worker", "workspace": {"depth": 3}}},
        ],
    })

    await sup._handle_event(event, realtime=True)

    child = sup._ready_queue.get_nowait()
    assert child.task == "child task"
    assert child.role == "worker"
    assert child.repo == "/parent-repo"
    assert child.init_branch == "parent-branch"
    assert child.evo_sha == "parent-sha"
    assert child.workspace_overrides["depth"] == 3
    assert child.llm_overrides["model"] == "kimi-parent"
    assert child.publication_overrides["branch_prefix"] == "parent/job"


@pytest.mark.asyncio
async def test_handle_event_realtime_advances_cursor_and_delays_poll():
    sup = _make_supervisor()
    sup._webhook_active = True
    sup.event_cursor = "2026-01-01T00:00:00|old"
    event = _evt("evt-new", "job.started", {"job_id": "job-1"})

    await sup._handle_event(event, realtime=True)

    assert sup.event_cursor.endswith("|evt-new")
    assert sup._webhook_poll_not_before > time.monotonic()


@pytest.mark.asyncio
async def test_handle_event_marks_processed_only_after_success():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    event = _evt(
        "evt-retry",
        "trigger.external",
        {"goal": "do X", "context": {"role": "default"}},
    )

    original = sup._handle_trigger
    first = True

    async def flaky_handle_trigger(evt, *, replay=False):
        nonlocal first
        if first:
            first = False
            raise RuntimeError("boom")
        return await original(evt, replay=replay)

    sup._handle_trigger = flaky_handle_trigger

    with pytest.raises(RuntimeError):
        await sup._handle_event(event)

    assert "evt-retry" not in sup._processed_event_ids

    await sup._handle_event(event)
    assert "evt-retry" in sup._processed_event_ids
    assert sup._ready_queue.qsize() == 1


@pytest.mark.asyncio
async def test_poll_and_handle_advances_cursor_only_after_successful_intake():
    sup = _make_supervisor()
    t0 = datetime.fromisoformat("2026-01-01T00:00:00")
    t1 = datetime.fromisoformat("2026-01-01T00:00:01")
    event_ok = Event(id="evt-1", source_id="test", type="job.started", ts=t0, data={"job_id": "j1"})
    event_fail = Event(id="evt-2", source_id="test", type="job.started", ts=t1, data={"job_id": "j2"})

    async def fake_poll(cursor=None, source=None, type_=None, limit=100):
        return [event_ok, event_fail], "ignored-next-cursor"

    original_handle_event = sup._handle_event

    async def flaky_handle_event(event, *, realtime=False, replay=False):
        if event.id == "evt-2":
            raise RuntimeError("boom")
        return await original_handle_event(event, realtime=realtime, replay=replay)

    sup.client.poll = fake_poll
    sup._handle_event = flaky_handle_event

    with pytest.raises(RuntimeError):
        await sup._poll_and_handle()

    assert sup.event_cursor == "2026-01-01T00:00:00|evt-1"


@pytest.mark.asyncio
async def test_enqueue_emits_supervisor_job_enqueued():
    sup = _make_supervisor()
    emitted = []

    async def fake_emit(event_type, data, *, timeout=None, idempotency_key=None):
        emitted.append((event_type, data, idempotency_key))
        return "evt-id"

    sup.client.emit = fake_emit
    job = SpawnedJob("j1", "e1", "task", "default", "/repo", "main", None, task_id="t1")

    await sup._enqueue(job)

    assert emitted
    event_type, data, idem = emitted[0]
    assert event_type == "supervisor.job.enqueued"
    assert data["job_id"] == "j1"
    assert data["task_id"] == "t1"
    assert data["queue_state"] == "ready"
    assert idem == "trenni:e1:supervisor.job.enqueued:j1"


@pytest.mark.asyncio
async def test_enqueue_immediate_goes_to_ready():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    job = SpawnedJob("j1", "e1", "t", "r", "/r", "main", None)
    await sup._enqueue(job)
    assert sup._ready_queue.qsize() == 1
    assert "j1" not in sup._pending


@pytest.mark.asyncio
async def test_enqueue_with_deps_goes_to_pending():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    job = SpawnedJob("j1", "e1", "t", "r", "/r", "main", None,
                     depends_on=frozenset({"dep-1"}))
    await sup._enqueue(job)
    assert sup._ready_queue.qsize() == 0
    assert "j1" in sup._pending


@pytest.mark.asyncio
async def test_enqueue_with_satisfied_deps_goes_to_ready():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup._completed_jobs.add("dep-1")
    job = SpawnedJob("j1", "e1", "t", "r", "/r", "main", None,
                     depends_on=frozenset({"dep-1"}))
    await sup._enqueue(job)
    assert sup._ready_queue.qsize() == 1
    assert "j1" not in sup._pending


@pytest.mark.asyncio
async def test_handle_job_done_resolves_pending():
    sup = _make_supervisor()
    dep_job = SpawnedJob("p-join", "e1", "join task", "default", "/r", "main", None,
                         depends_on=frozenset({"p-c0", "p-c1"}))
    sup._pending["p-join"] = dep_job

    await sup._handle_job_done(_evt("d1", "job.completed", {
        "job_id": "p-c0", "summary": "done A",
    }))
    assert "p-c0" in sup._completed_jobs
    assert sup._ready_queue.qsize() == 0
    assert "p-join" in sup._pending

    await sup._handle_job_done(_evt("d2", "job.completed", {
        "job_id": "p-c1", "summary": "done B",
    }))
    assert sup._ready_queue.qsize() == 1
    assert "p-join" not in sup._pending

    resolved = sup._ready_queue.get_nowait()
    assert resolved.job_id == "p-join"


@pytest.mark.asyncio
async def test_handle_job_failed_propagates_to_dependents():
    sup = _make_supervisor()

    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))

    sup.client.emit = fake_emit
    dep_job = SpawnedJob("p-join", "e1", "join task", "default", "/r", "main", None,
                         depends_on=frozenset({"p-c0", "p-c1"}))
    sup._pending["p-join"] = dep_job

    await sup._handle_job_done(_evt("d1", "job.failed", {
        "job_id": "p-c0", "error": "crash",
    }))

    assert "p-c0" in sup._completed_jobs
    assert "p-c0" in sup._failed_jobs
    assert sup._ready_queue.qsize() == 0
    assert "p-join" not in sup._pending
    assert "p-join" in sup.state.cancelled_jobs

    cancel_events = [(t, d) for t, d in emitted if t == "job.cancelled"]
    assert len(cancel_events) == 1
    assert cancel_events[0][1]["job_id"] == "p-join"
    assert cancel_events[0][1]["code"] == "condition_unsatisfied"


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
    sup.client.emit = AsyncMock()
    sup.scheduler.record_job_terminal = AsyncMock(return_value=(None, []))
    sup.backend.stop = AsyncMock()
    sup.backend.remove = AsyncMock()
    sup.jobs["j1"] = JobHandle("j1", "ctr-1", "yoitsu-job-j1")

    await sup._handle_job_done(_evt("d1", "job.completed", {"job_id": "j1"}))
    assert "j1" not in sup.jobs
    sup.backend.stop.assert_awaited_once()
    sup.backend.remove.assert_awaited_once()


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

    sup.jobs["existing"] = JobHandle("existing", "ctr-existing", "yoitsu-job-existing")
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


@pytest.mark.asyncio
async def test_launch_emits_container_identity():
    sup = _make_supervisor()

    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))

    sup.client.emit = fake_emit
    sup.runtime_builder.build = MagicMock(return_value=JobRuntimeSpec(
        job_id="job-xyz",
        source_event_id="evt-src-123",
        container_name="yoitsu-job-job-xyz",
        image="localhost/yoitsu-palimpsest-job:dev",
        pod_name="yoitsu-dev",
        labels={},
        env={"PALIMPSEST_JOB_CONFIG_B64": "abc"},
        command=("palimpsest", "container-entrypoint"),
        config_payload_b64="abc",
    ))
    sup.backend.prepare = AsyncMock(return_value=JobHandle("job-xyz", "ctr-9999", "yoitsu-job-job-xyz"))
    sup.backend.start = AsyncMock()

    await sup._launch("job-xyz", "test", "default", "/repo", "main", None,
                      source_event_id="evt-src-123")

    assert emitted
    type_, data = emitted[0]
    assert type_ == "supervisor.job.launched"
    assert data["source_event_id"] == "evt-src-123"
    assert data["job_id"] == "job-xyz"
    assert data["runtime_kind"] == "podman"
    assert data["container_id"] == "ctr-9999"
    assert "pid" not in data


@pytest.mark.asyncio
async def test_launch_removes_container_when_start_fails():
    sup = _make_supervisor()
    sup.runtime_builder.build = MagicMock(return_value=JobRuntimeSpec(
        job_id="job-xyz",
        source_event_id="evt-src-123",
        container_name="yoitsu-job-job-xyz",
        image="localhost/yoitsu-palimpsest-job:dev",
        pod_name="yoitsu-dev",
        labels={},
        env={"PALIMPSEST_JOB_CONFIG_B64": "abc"},
        command=("palimpsest", "container-entrypoint"),
        config_payload_b64="abc",
    ))
    sup.backend.prepare = AsyncMock(return_value=JobHandle("job-xyz", "ctr-9999", "yoitsu-job-job-xyz"))
    sup.backend.start = AsyncMock(side_effect=RuntimeError("boom"))
    sup.backend.remove = AsyncMock()

    with pytest.raises(RuntimeError):
        await sup._launch("job-xyz", "test", "default", "/repo", "main", None,
                          source_event_id="evt-src-123")

    sup.backend.remove.assert_awaited_once()


@pytest.mark.asyncio
async def test_mark_exited_jobs():
    sup = _make_supervisor()
    handle = JobHandle("j1", "ctr-1", "yoitsu-job-j1")
    sup.jobs["j1"] = handle
    sup.backend.inspect = AsyncMock(return_value=ContainerState(
        exists=True, status="exited", running=False, exit_code=1,
    ))
    assert handle.exited_at is None

    await sup._mark_exited_jobs()
    assert handle.exited_at is not None
    assert handle.exit_code == 1

    first_ts = handle.exited_at
    await sup._mark_exited_jobs()
    assert handle.exited_at == first_ts


@pytest.mark.asyncio
async def test_checkpoint_reaps_timed_out_containers():
    sup = _make_supervisor()
    sup._reap_timeout = 0.0

    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))

    sup.client.emit = fake_emit
    sup.backend.logs = AsyncMock(return_value="container log line")
    sup.backend.stop = AsyncMock()
    sup.backend.remove = AsyncMock()

    from trenni.state import SpawnedJob, TaskRecord
    sup.state.tasks["t1"] = TaskRecord(task_id="t1", goal="...")
    sup.state.jobs_by_id["j1"] = SpawnedJob(
        job_id="j1", source_event_id="e1", task="t", role="default",
        repo="r", init_branch="b", evo_sha="s", llm_overrides={},
        workspace_overrides={}, publication_overrides={}, task_id="t1"
    )
    handle = JobHandle(
        "j1",
        "ctr-1",
        "yoitsu-job-j1",
        exit_code=137,
        exited_at=time.monotonic() - 1.0,
    )
    sup.jobs["j1"] = handle

    await sup._checkpoint()

    assert "j1" not in sup.jobs
    types = [e[0] for e in emitted]
    assert "job.failed" in types
    assert "task.failed" in types
    assert "supervisor.checkpoint" in types
    fail_data = next(d for t, d in emitted if t == "job.failed")
    assert fail_data["code"] == "runtime_lost"
    assert "container log line" in fail_data["logs_tail"]
    sup.backend.stop.assert_awaited_once()
    sup.backend.remove.assert_awaited_once()


@pytest.mark.asyncio
async def test_checkpoint_emits_state():
    sup = _make_supervisor()

    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))

    sup.client.emit = fake_emit
    sup.event_cursor = "2026-01-01T00:00:00|abc"
    sup._pending["p1"] = SpawnedJob("p1", "e", "t", "r", "/r", "m", None)

    await sup._checkpoint()

    cp = next(d for t, d in emitted if t == "supervisor.checkpoint")
    assert cp["cursor"] == "2026-01-01T00:00:00|abc"
    assert cp["pending_jobs"] == ["p1"]


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


@pytest.mark.asyncio
async def test_replay_enqueues_not_launched():
    sup = _make_supervisor()

    async def fake_fetch_all(type_, source=None):
        if type_ == "supervisor.job.enqueued":
            return [_evt("q-1", "supervisor.job.enqueued", {
                "job_id": "job-A",
                "task_id": "t-1",
                "source_event_id": "sub-1",
                "task": "do X",
                "role": "default",
                "repo": "/r",
                "init_branch": "main",
                "evo_sha": "",
                "llm": {},
                "workspace": {},
                "publication": {},
                "parent_job_id": "",
                "condition": None,
                "job_context": {},
                "queue_state": "ready",
            })]
        if type_ == "task.created":
            return [_evt("c-1", "task.created", {"task_id": "t-1", "goal": "do X", "source_trigger_id": "sub-1"})]
        return []

    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup._ready_queue.qsize() == 1


@pytest.mark.asyncio
async def test_replay_skips_completed():
    sup = _make_supervisor()

    async def fake_fetch_all(type_, source=None):
        if type_ == "supervisor.job.enqueued":
            return [_evt("q-1", "supervisor.job.enqueued", {
                "job_id": "job-A",
                "task_id": "t-1",
                "source_event_id": "sub-1",
                "task": "do X",
                "role": "default",
                "repo": "/r",
                "init_branch": "main",
                "evo_sha": "",
                "llm": {},
                "workspace": {},
                "publication": {},
                "parent_job_id": "",
                "condition": None,
                "job_context": {},
                "queue_state": "ready",
            })]
        if type_ == "task.created":
            return [_evt("c-1", "task.created", {"task_id": "t-1", "goal": "do X", "source_trigger_id": "sub-1"})]
        if type_ == "supervisor.job.launched":
            return [_evt("l1", "supervisor.job.launched", {
                "source_event_id": "sub-1", "job_id": "job-A",
                "container_id": "ctr-A", "container_name": "yoitsu-job-job-A",
                "task_id": "t-1", "task": "do X", "role": "default", "repo": "/r"
            })]
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
async def test_replay_reenqueues_missing_container():
    sup = _make_supervisor()
    sup._inspect_replay_state = AsyncMock(return_value=ContainerState(exists=False))

    async def fake_fetch_all(type_, source=None):
        if type_ == "supervisor.job.enqueued":
            return [_evt("q-1", "supervisor.job.enqueued", {
                "job_id": "job-A",
                "task_id": "t-1",
                "source_event_id": "sub-1",
                "task": "do X",
                "role": "default",
                "repo": "/r",
                "init_branch": "main",
                "evo_sha": "",
                "llm": {},
                "workspace": {},
                "publication": {},
                "parent_job_id": "",
                "condition": None,
                "job_context": {},
                "queue_state": "ready",
            })]
        if type_ == "task.created":
            return [_evt("c-1", "task.created", {"task_id": "t-1", "goal": "do X", "source_trigger_id": "sub-1"})]
        if type_ == "supervisor.job.launched":
            return [_evt("l1", "supervisor.job.launched", {
                "source_event_id": "sub-1",
                "job_id": "job-A",
                "container_id": "ctr-A",
                "container_name": "yoitsu-job-job-A",
                "task_id": "t-1", "task": "do X", "role": "default", "repo": "/r"
            })]
        return []

    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup._ready_queue.qsize() == 1


@pytest.mark.asyncio
async def test_replay_reattaches_running_container():
    sup = _make_supervisor()
    sup._inspect_replay_state = AsyncMock(return_value=ContainerState(
        exists=True, status="running", running=True, exit_code=None,
    ))

    async def fake_fetch_all(type_, source=None):
        if type_ == "supervisor.job.enqueued":
            return [_evt("q-1", "supervisor.job.enqueued", {
                "job_id": "job-A",
                "task_id": "t-1",
                "source_event_id": "sub-1",
                "task": "do X",
                "role": "default",
                "repo": "/r",
                "init_branch": "main",
                "evo_sha": "",
                "llm": {},
                "workspace": {},
                "publication": {},
                "parent_job_id": "",
                "condition": None,
                "job_context": {},
                "queue_state": "ready",
            })]
        if type_ == "task.created":
            return [_evt("c-1", "task.created", {"task_id": "t-1", "goal": "do X", "source_trigger_id": "sub-1"})]
        if type_ == "supervisor.job.launched":
            return [_evt("l1", "supervisor.job.launched", {
                "source_event_id": "sub-1",
                "job_id": "job-A",
                "container_id": "ctr-A",
                "container_name": "yoitsu-job-job-A",
                "task_id": "t-1", "task": "do X", "role": "default", "repo": "/r"
            })]
        return []

    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup._ready_queue.qsize() == 0
    assert "job-A" in sup.jobs
    assert sup.jobs["job-A"].container_id == "ctr-A"


@pytest.mark.asyncio
async def test_replay_uses_checkpoint_cursor():
    sup = _make_supervisor()

    async def fake_fetch_all(type_, source=None):
        if type_ == "supervisor.checkpoint":
            return [_evt("cp1", "supervisor.checkpoint",
                         {"cursor": "2026-01-01T00:00:00|saved-cursor"})]
        if type_ == "supervisor.job.enqueued" or type_ == "task.created":
            return []
        return []

    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup.event_cursor == "2026-01-01T00:00:00|saved-cursor"


@pytest.mark.asyncio
async def test_start_calls_runtime_ready_replay_and_drain():
    sup = _make_supervisor()

    replay_called = False

    async def fake_replay():
        nonlocal replay_called
        replay_called = True

    sup._replay_unfinished_tasks = fake_replay

    with patch.object(sup.backend, "ensure_ready", new_callable=AsyncMock) as mock_ready, \
         patch.object(sup, "_try_register_webhook", new_callable=AsyncMock), \
         patch.object(sup, "_run_loop", new_callable=AsyncMock) as mock_run_loop, \
         patch.object(sup.client, "register_source", new_callable=AsyncMock), \
         patch.object(sup.client, "close", new_callable=AsyncMock), \
         patch.object(sup.backend, "close", new_callable=AsyncMock), \
         patch("asyncio.create_task") as mock_create_task:

        class AwaitableTask:
            def __init__(self) -> None:
                self.cancel = MagicMock()

            def __await__(self):
                async def _noop():
                    return None

                return _noop().__await__()

        mock_task = AwaitableTask()

        def fake_create_task(coro):
            coro.close()
            return mock_task

        mock_create_task.side_effect = fake_create_task

        await sup.start()

    assert replay_called
    mock_ready.assert_called_once()
    mock_run_loop.assert_called_once()
    mock_create_task.assert_called_once()
    mock_task.cancel.assert_called_once()


def test_status_reflects_unified_state():
    sup = _make_supervisor()
    sup._pending["p1"] = SpawnedJob("p1", "e", "t", "r", "/r", "m", None)
    st = sup.status
    assert st["pending_jobs"] == 1
    assert "ready_queue_size" in st
    assert st["runtime_kind"] == "podman"
    assert "fork_joins_active" not in st
