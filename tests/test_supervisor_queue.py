"""Tests for Trenni supervisor: queueing, Podman launch, checkpoint, and replay.

Per Bundle MVP: bundle and role are required fields in trigger events.
"""
import asyncio
import contextlib
import re
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trenni.config import BundleConfig, TrenniConfig
from trenni.pasloe_client import Event
from trenni.runtime_types import ContainerState, JobHandle, JobRuntimeSpec
from trenni.state import TaskRecord
from trenni.supervisor import SpawnDefaults, SpawnedJob, Supervisor
from trenni.workspace_manager import PreparedWorkspaces
from yoitsu_contracts.config import BundleSource, TargetSource


def _evt(id_: str, type_: str, data: dict | None = None) -> Event:
    return Event(id=id_, source_id="test", type=type_,
                 ts=datetime.utcnow(), data=data or {})


def _make_supervisor(**overrides) -> Supervisor:
    overrides.setdefault(
        "bundles",
        {
            "factorio": BundleConfig.from_dict(
                {"source": {"url": "https://github.com/guan-spicy-wolf/factorio-bundle.git"}}
            )
        },
    )
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
    event = _evt("evt-abc", "trigger.external.received",
                 {"goal": "do X", "budget": 0.5, "role": "default", "bundle": "factorio", "repo": "/r", "init_branch": "main"})

    await sup._handle_trigger(event)

    assert sup._ready_queue.qsize() == 1
    job = sup._ready_queue.get_nowait()
    assert job.source_event_id == "evt-abc"
    assert job.goal == "do X"
    assert job.bundle == "factorio"
    assert job.depends_on == frozenset()


@pytest.mark.asyncio
async def test_handle_trigger_budget_propagates_to_root_job():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    event = _evt(
        "evt-root-budget",
        "trigger.external.received",
        {"goal": "do X", "budget": 0.75, "role": "default", "bundle": "factorio", "repo": "/r", "init_branch": "main"},
    )

    await sup._handle_trigger(event)

    job = sup._ready_queue.get_nowait()
    # Per ADR-0007: budget is SpawnedJob.budget field
    assert job.budget == 0.75


@pytest.mark.asyncio
async def test_handle_trigger_deduplicates():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup._launched_event_ids.add("evt-dup")
    event = _evt("evt-dup", "trigger.external.received",
                 {"goal": "do X", "role": "default", "bundle": "factorio"})

    await sup._handle_trigger(event)
    assert sup._ready_queue.qsize() == 0


@pytest.mark.asyncio
async def test_handle_trigger_ignores_empty_goal():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    event = _evt("evt-empty", "trigger.external.received", {"goal": "", "role": "default", "bundle": "factorio"})

    await sup._handle_trigger(event)
    assert sup._ready_queue.qsize() == 0
    assert "evt-empty" not in sup._launched_event_ids


@pytest.mark.asyncio
async def test_handle_trigger_missing_bundle_rejected():
    """Per Bundle MVP: trigger without bundle is rejected."""
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    event = _evt("evt-no-bundle", "trigger.external.received",
                 {"goal": "do X", "role": "default"})

    await sup._handle_trigger(event)

    assert sup._ready_queue.qsize() == 0


@pytest.mark.asyncio
async def test_handle_trigger_missing_role_rejected():
    """Per Bundle MVP: trigger without role is rejected."""
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    event = _evt("evt-no-role", "trigger.external.received",
                 {"goal": "do X", "bundle": "factorio"})

    await sup._handle_trigger(event)

    assert sup._ready_queue.qsize() == 0


@pytest.mark.asyncio
async def test_handle_trigger_unknown_bundle_fails_task():
    sup = _make_supervisor()
    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))

    sup.client.emit = fake_emit
    event = _evt(
        "evt-unknown-bundle",
        "trigger.external.received",
        {"goal": "do X", "role": "default", "bundle": "missing"},
    )

    await sup._handle_trigger(event)

    assert sup._ready_queue.qsize() == 0
    assert len(sup.state.jobs_by_id) == 0
    assert len(sup.state.tasks) == 1
    assert [type_ for type_, _ in emitted] == [
        "supervisor.task.created",
        "supervisor.task.failed",
    ]
    assert emitted[-1][1]["reason"] == "Unknown bundle 'missing'"


@pytest.mark.asyncio
async def test_handle_trigger_bundle_without_source_fails_task():
    sup = _make_supervisor(bundles={"factorio": BundleConfig()})
    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))

    sup.client.emit = fake_emit
    event = _evt(
        "evt-no-bundle-source",
        "trigger.external.received",
        {"goal": "do X", "role": "default", "bundle": "factorio"},
    )

    await sup._handle_trigger(event)

    assert sup._ready_queue.qsize() == 0
    assert [type_ for type_, _ in emitted] == [
        "supervisor.task.created",
        "supervisor.task.failed",
    ]
    assert emitted[-1][1]["reason"] == "Bundle 'factorio' has no configured source"


@pytest.mark.asyncio
async def test_handle_spawn_creates_children_no_continuation():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup._spawn_defaults_by_job["parent-1"] = SpawnDefaults(
        repo="/parent-repo",
        init_branch="main",
        role="default",
        bundle_sha="parent-sha",
        task_id="task-1",
    )
    event = _evt("spawn-1", "agent.job.spawn_request", {
        "job_id": "parent-1",
        "tasks": [
            {"goal": "child A", "role": "worker", "bundle": "factorio", "repo": "/r", "init_branch": "main"},
            {"goal": "child B", "role": "worker", "bundle": "factorio", "repo": "/r", "init_branch": "main"},
        ],
        "wait_for": "all_complete",
    })

    await sup._handle_spawn(event)

    assert sup._ready_queue.qsize() == 2
    c0 = sup._ready_queue.get_nowait()
    c1 = sup._ready_queue.get_nowait()
    assert c0.job_id.startswith("parent-1-c")
    assert c1.job_id.startswith("parent-1-c")
    assert c0.job_id != c1.job_id
    assert c0.goal == "child A"
    assert c0.role == "worker"
    assert c0.bundle == "factorio"
    assert c0.depends_on == frozenset()
    assert len(sup._pending) == 0


def test_spawn_handler_join_job_uses_continuation_instruction():
    sup = _make_supervisor()
    parent = SpawnedJob(
        "parent-1",
        "e1",
        "Parent goal",
        "planner",
        "/repo",
        "main",
        "sha1",
        role_params={"goal": "Parent goal"},
        task_id="root-task",
    )
    sup.state.jobs_by_id["parent-1"] = parent
    sup.state.spawn_defaults_by_job["parent-1"] = SpawnDefaults(
        repo="/repo",
        init_branch="main",
        role="planner",
        bundle_sha="sha1",
        role_params={"goal": "Parent goal"},
        task_id="root-task",
    )

    event = _evt("spawn-join", "agent.job.spawn_request", {
        "job_id": "parent-1",
        "task_id": "root-task",
        "tasks": [
            {"goal": "child A", "role": "implementer", "bundle": "factorio", "repo": "/repo", "init_branch": "main"},
        ],
    })

    plan = sup.spawn_handler.expand(event)
    join_jobs = [job for job in plan.jobs if job.job_context.join is not None]
    assert len(join_jobs) == 1
    join_job = join_jobs[0]
    assert "continuation planning step" in join_job.goal
    assert join_job.role == "planner"
    # Per ADR-0007: role_params only has mode="join"
    assert join_job.role_params["mode"] == "join"
    assert join_job.job_context.join.parent_summary == "Parent goal"


@pytest.mark.asyncio
async def test_supervisor_start_validates_before_loop():
    sup = _make_supervisor()
    sup.client.register_source = AsyncMock()
    sup._replay_unfinished_tasks = AsyncMock()
    sup._try_register_webhook = AsyncMock()
    sup._run_loop = AsyncMock()
    sup.client.close = AsyncMock()
    sup.backend.close = AsyncMock()

    await sup.start()

    sup._run_loop.assert_called_once()


@pytest.mark.asyncio
async def test_handle_spawn_empty_tasks_ignored():
    sup = _make_supervisor()
    event = _evt("spawn-empty", "agent.job.spawn_request", {
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
        bundle_sha="parent-sha",
        task_id="task-1",
    )
    event = _evt("spawn-dup", "agent.job.spawn_request", {
        "job_id": "parent-1",
        "tasks": [
            {"goal": "child A", "role": "worker", "bundle": "factorio", "repo": "/r", "init_branch": "main"},
            {"goal": "child B", "role": "worker", "bundle": "factorio", "repo": "/r", "init_branch": "main"},
        ],
    })

    await sup._handle_event(event, realtime=True)
    await sup._handle_event(event)

    assert sup._ready_queue.qsize() == 2


@pytest.mark.asyncio
async def test_handle_spawn_inherits_parent_defaults_for_missing_params_fields():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup._spawn_defaults_by_job["parent-1"] = SpawnDefaults(
        repo="/parent-repo",
        init_branch="parent-branch",
        role="parent-role",
        bundle_sha="parent-sha",
        task_id="task-1",
    )
    event = _evt("spawn-inherit", "agent.job.spawn_request", {
        "job_id": "parent-1",
        "tasks": [
            {"goal": "child task", "role": "worker", "bundle": "factorio"},
        ],
    })

    await sup._handle_event(event, realtime=True)

    child = sup._ready_queue.get_nowait()
    assert child.goal == "child task"
    assert child.role == "worker"
    assert child.repo == "/parent-repo"
    assert child.init_branch == "parent-branch"
    assert child.bundle_sha == "parent-sha"
    assert child.bundle == "factorio"
    # Per ADR-0007: no execution config overrides inherited
    assert child.budget == 0.0


@pytest.mark.asyncio
async def test_handle_event_realtime_advances_cursor_and_delays_poll():
    sup = _make_supervisor()
    sup._webhook_active = True
    sup.event_cursor = "2026-01-01T00:00:00|old"
    event = _evt("evt-new", "agent.job.started", {"job_id": "job-1"})

    await sup._handle_event(event, realtime=True)

    assert sup.event_cursor.endswith("|evt-new")
    assert sup._webhook_poll_not_before > time.monotonic()


@pytest.mark.asyncio
async def test_handle_event_marks_processed_only_after_success():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    event = _evt(
        "evt-retry",
        "trigger.external.received",
        {"goal": "do X", "budget": 0.5, "role": "default", "bundle": "factorio"},
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
    event_ok = Event(id="evt-1", source_id="test", type="agent.job.started", ts=t0, data={"job_id": "j1"})
    event_fail = Event(id="evt-2", source_id="test", type="agent.job.started", ts=t1, data={"job_id": "j2"})

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
    job = SpawnedJob("j1", "e1", "task", "default", "/repo", "main", None, task_id="t1", bundle="factorio")

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
    job = SpawnedJob("j1", "e1", "t", "r", "/r", "main", None, bundle="factorio")
    await sup._enqueue(job)
    assert sup._ready_queue.qsize() == 1
    assert "j1" not in sup._pending


@pytest.mark.asyncio
async def test_enqueue_with_deps_goes_to_pending():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    job = SpawnedJob("j1", "e1", "t", "r", "/r", "main", None,
                     depends_on=frozenset({"dep-1"}), bundle="factorio")
    await sup._enqueue(job)
    assert sup._ready_queue.qsize() == 0
    assert "j1" in sup._pending


@pytest.mark.asyncio
async def test_enqueue_with_satisfied_deps_goes_to_ready():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup._completed_jobs.add("dep-1")
    job = SpawnedJob("j1", "e1", "t", "r", "/r", "main", None,
                     depends_on=frozenset({"dep-1"}), bundle="factorio")
    await sup._enqueue(job)
    assert sup._ready_queue.qsize() == 1
    assert "j1" not in sup._pending


@pytest.mark.asyncio
async def test_handle_job_done_resolves_pending():
    sup = _make_supervisor()
    dep_job = SpawnedJob("p-join", "e1", "join task", "default", "/r", "main", None,
                         depends_on=frozenset({"p-c0", "p-c1"}), bundle="factorio")
    sup._pending["p-join"] = dep_job

    await sup._handle_job_done(_evt("d1", "agent.job.completed", {
        "job_id": "p-c0", "summary": "done A",
    }))
    assert "p-c0" in sup._completed_jobs
    assert sup._ready_queue.qsize() == 0
    assert "p-join" in sup._pending

    await sup._handle_job_done(_evt("d2", "agent.job.completed", {
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
                         depends_on=frozenset({"p-c0", "p-c1"}), bundle="factorio")
    sup._pending["p-join"] = dep_job

    await sup._handle_job_done(_evt("d1", "agent.job.failed", {
        "job_id": "p-c0", "error": "crash",
    }))

    assert "p-c0" in sup._completed_jobs
    assert "p-c0" in sup._failed_jobs
    assert sup._ready_queue.qsize() == 0
    assert "p-join" not in sup._pending
    assert "p-join" in sup.state.cancelled_jobs

    cancel_events = [(t, d) for t, d in emitted if t == "agent.job.cancelled"]
    assert len(cancel_events) == 1
    assert cancel_events[0][1]["job_id"] == "p-join"
    assert cancel_events[0][1]["code"] == "condition_unsatisfied"


@pytest.mark.asyncio
async def test_handle_job_done_caches_summary():
    sup = _make_supervisor()
    await sup._handle_job_done(_evt("d1", "agent.job.completed", {
        "job_id": "j1", "summary": "all good",
    }))
    assert sup._job_summaries["j1"] == "all good"


@pytest.mark.asyncio
async def test_handle_job_budget_exhausted_emits_task_partial():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup.state.tasks["t1"] = TaskRecord(task_id="t1", goal="goal", bundle="factorio")
    sup.state.jobs_by_id["j1"] = SpawnedJob("j1", "e1", "goal", "default", "/r", "main", None, task_id="t1", bundle="factorio")

    await sup._handle_job_done(_evt("d1", "agent.job.completed", {
        "job_id": "j1",
        "summary": "stopped at budget",
        "code": "budget_exhausted",
    }))

    emitted = [call.args for call in sup.client.emit.await_args_list]
    assert any(event_type == "supervisor.task.partial" for event_type, _ in emitted)
    payload = next(data for event_type, data in emitted if event_type == "supervisor.task.partial")
    assert payload["task_id"] == "t1"
    assert payload["result"]["structural"]["partial"] == 1


@pytest.mark.asyncio
async def test_handle_job_done_removes_from_jobs():
    sup = _make_supervisor()
    sup.client.emit = AsyncMock()
    sup.scheduler.record_job_terminal = AsyncMock(return_value=(None, []))
    sup.backend.stop = AsyncMock()
    sup.backend.remove = AsyncMock()
    sup.jobs["j1"] = JobHandle("j1", "ctr-1", "yoitsu-job-j1")

    await sup._handle_job_done(_evt("d1", "agent.job.completed", {"job_id": "j1"}))
    assert "j1" not in sup.jobs
    sup.backend.stop.assert_awaited_once()
    sup.backend.remove.assert_awaited_once()


@pytest.mark.asyncio
async def test_drain_queue_launches_when_capacity():
    sup = _make_supervisor(max_workers=2)
    job = SpawnedJob("job-1", "evt-1", "task", "default", "/repo", "main", None, bundle="factorio")
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
    job = SpawnedJob("job-2", "evt-2", "task", "default", "/repo", "main", None, bundle="factorio")
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
    sup.backend.ensure_ready = AsyncMock()
    sup.backend.prepare = AsyncMock(return_value=JobHandle("job-xyz", "ctr-9999", "yoitsu-job-job-xyz"))
    sup.backend.start = AsyncMock()
    sup.workspace_manager.prepare = MagicMock(return_value=PreparedWorkspaces(
        bundle_source=BundleSource(name="factorio", workspace="/tmp/bundle"),
        target_source=TargetSource(repo_uri="/repo", workspace="/tmp/target"),
        temp_dirs=[],
    ))

    await sup._launch("job-xyz", "test", "default", "/repo", "main", None,
                      source_event_id="evt-src-123", bundle="factorio")

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
    sup.backend.ensure_ready = AsyncMock()
    sup.backend.prepare = AsyncMock(return_value=JobHandle("job-xyz", "ctr-9999", "yoitsu-job-job-xyz"))
    sup.backend.start = AsyncMock(side_effect=RuntimeError("boom"))
    sup.backend.remove = AsyncMock()
    sup.workspace_manager.prepare = MagicMock(return_value=PreparedWorkspaces(
        bundle_source=BundleSource(name="factorio", workspace="/tmp/bundle"),
        target_source=TargetSource(repo_uri="/repo", workspace="/tmp/target"),
        temp_dirs=[],
    ))

    with pytest.raises(RuntimeError):
        await sup._launch("job-xyz", "test", "default", "/repo", "main", None,
                          source_event_id="evt-src-123", bundle="factorio")

    sup.backend.remove.assert_awaited_once()


@pytest.mark.asyncio
async def test_launch_calls_ensure_ready_before_prepare():
    """Verify that _launch calls ensure_ready(spec) before prepare(spec)."""
    sup = _make_supervisor()

    # Track call order
    call_order: list[str] = []
    captured_spec: JobRuntimeSpec | None = None

    spec = JobRuntimeSpec(
        job_id="job-xyz",
        source_event_id="evt-src-123",
        container_name="yoitsu-job-job-xyz",
        image="localhost/yoitsu-palimpsest-job:dev",
        pod_name="yoitsu-dev",
        extra_networks=("network-a",),
        labels={},
        env={"PALIMPSEST_JOB_CONFIG_B64": "abc"},
        command=("palimpsest", "container-entrypoint"),
        config_payload_b64="abc",
    )

    async def mock_ensure_ready(s: JobRuntimeSpec) -> None:
        call_order.append("ensure_ready")
        nonlocal captured_spec
        captured_spec = s

    async def mock_prepare(s: JobRuntimeSpec) -> JobHandle:
        call_order.append("prepare")
        return JobHandle("job-xyz", "ctr-9999", "yoitsu-job-job-xyz")

    async def mock_start(h: JobHandle) -> None:
        call_order.append("start")

    sup.runtime_builder.build = MagicMock(return_value=spec)
    sup.backend.ensure_ready = mock_ensure_ready
    sup.backend.prepare = mock_prepare
    sup.backend.start = mock_start
    sup.client.emit = AsyncMock()
    sup.workspace_manager.prepare = MagicMock(return_value=PreparedWorkspaces(
        bundle_source=BundleSource(name="factorio", workspace="/tmp/bundle"),
        target_source=TargetSource(repo_uri="/repo", workspace="/tmp/target"),
        temp_dirs=[],
    ))

    await sup._launch("job-xyz", "test", "default", "/repo", "main", None,
                      source_event_id="evt-src-123", bundle="factorio")

    # Verify call order: ensure_ready must be called before prepare
    assert call_order == ["ensure_ready", "prepare", "start"], \
        f"Expected ensure_ready before prepare, got: {call_order}"

    # Verify spec was passed correctly to ensure_ready
    assert captured_spec is not None
    assert captured_spec.job_id == "job-xyz"
    assert captured_spec.pod_name == "yoitsu-dev"
    assert captured_spec.extra_networks == ("network-a",)


@pytest.mark.asyncio
async def test_launch_propagates_ensure_ready_errors():
    """Verify that errors from ensure_ready prevent container creation."""
    sup = _make_supervisor()

    spec = JobRuntimeSpec(
        job_id="job-xyz",
        source_event_id="evt-src-123",
        container_name="yoitsu-job-job-xyz",
        image="localhost/missing-image:dev",
        pod_name="missing-pod",
        labels={},
        env={"PALIMPSEST_JOB_CONFIG_B64": "abc"},
        command=("palimpsest", "container-entrypoint"),
        config_payload_b64="abc",
    )

    async def mock_ensure_ready(s: JobRuntimeSpec) -> None:
        raise RuntimeError("Podman pod 'missing-pod' does not exist")

    prepare_called = False

    async def mock_prepare(s: JobRuntimeSpec) -> JobHandle:
        nonlocal prepare_called
        prepare_called = True
        return JobHandle("job-xyz", "ctr-9999", "yoitsu-job-job-xyz")

    sup.runtime_builder.build = MagicMock(return_value=spec)
    sup.backend.ensure_ready = mock_ensure_ready
    sup.backend.prepare = mock_prepare
    sup.client.emit = AsyncMock()
    sup.workspace_manager.prepare = MagicMock(return_value=PreparedWorkspaces(
        bundle_source=BundleSource(name="factorio", workspace="/tmp/bundle"),
        target_source=TargetSource(repo_uri="/repo", workspace="/tmp/target"),
        temp_dirs=[],
    ))

    with pytest.raises(RuntimeError, match="missing-pod"):
        await sup._launch("job-xyz", "test", "default", "/repo", "main", None,
                          source_event_id="evt-src-123", bundle="factorio")

    # Verify prepare was never called due to ensure_ready failure
    assert not prepare_called, "prepare should not be called when ensure_ready fails"


@pytest.mark.asyncio
async def test_launch_fails_job_when_bundle_workspace_prepare_fails():
    sup = _make_supervisor()
    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))
        return None

    sup.client.emit = fake_emit
    sup.runtime_builder.build = MagicMock()
    sup.backend.ensure_ready = AsyncMock()
    sup.backend.prepare = AsyncMock()
    sup.state.tasks["t1"] = TaskRecord(task_id="t1", goal="...", bundle="factorio")
    sup.state.jobs_by_id["j1"] = SpawnedJob(
        job_id="j1",
        source_event_id="e1",
        goal="test",
        role="default",
        repo="",
        init_branch="main",
        bundle_sha="sha1",
        task_id="t1",
        bundle="factorio",
    )
    sup.workspace_manager.prepare = MagicMock(
        return_value=SimpleNamespace(bundle_source=None, target_source=None, temp_dirs=[])
    )

    await sup._launch(
        "j1",
        "test",
        "default",
        "",
        "main",
        "sha1",
        source_event_id="e1",
        task_id="t1",
        bundle="factorio",
    )

    assert not sup.runtime_builder.build.called
    assert not sup.backend.ensure_ready.await_count
    assert not sup.backend.prepare.await_count
    assert any(type_ == "supervisor.job.failed" for type_, _ in emitted)
    fail_data = next(data for type_, data in emitted if type_ == "supervisor.job.failed")
    assert fail_data["code"] == "bundle_workspace_prepare_failed"
    assert "factorio" in fail_data["error"]
    assert any(type_ == "supervisor.task.failed" for type_, _ in emitted)


@pytest.mark.asyncio
async def test_drain_queue_fails_job_when_launch_raises():
    sup = _make_supervisor()
    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))
        return None

    sup.client.emit = fake_emit
    sup.scheduler.evaluate_job = MagicMock(return_value=True)
    sup.scheduler.has_bundle_capacity = MagicMock(return_value=True)
    sup.scheduler.has_capacity = MagicMock(return_value=True)
    sup.state.tasks["t1"] = TaskRecord(task_id="t1", goal="...", bundle="factorio")
    job = SpawnedJob(
        job_id="j1",
        source_event_id="e1",
        goal="test",
        role="default",
        repo="",
        init_branch="main",
        bundle_sha="sha1",
        task_id="t1",
        bundle="factorio",
    )
    sup.state.jobs_by_id["j1"] = job
    sup._ready_queue.put_nowait(job)
    sup._launch_from_spawned = AsyncMock(side_effect=RuntimeError("podman create failed"))

    drain_task = asyncio.create_task(sup._drain_queue())
    try:
        await asyncio.sleep(0.05)
    finally:
        drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain_task

    assert any(type_ == "supervisor.job.failed" for type_, _ in emitted)
    fail_data = next(data for type_, data in emitted if type_ == "supervisor.job.failed")
    assert fail_data["code"] == "runtime_launch_failed"
    assert "podman create failed" in fail_data["error"]
    assert any(type_ == "supervisor.task.failed" for type_, _ in emitted)


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
async def test_mark_exited_jobs_fails_pre_start_container_exit():
    sup = _make_supervisor()
    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))
        return "evt-runtime-failed"

    sup.client.emit = fake_emit
    sup.backend.logs = AsyncMock(return_value="container log line")
    sup.backend.stop = AsyncMock()
    sup.backend.remove = AsyncMock()
    sup.state.tasks["t1"] = TaskRecord(task_id="t1", goal="...", bundle="factorio")
    sup.state.jobs_by_id["j1"] = SpawnedJob(
        job_id="j1", source_event_id="e1", goal="t", role="default",
        repo="r", init_branch="b", bundle_sha="s", task_id="t1", bundle="factorio"
    )
    handle = JobHandle("j1", "ctr-1", "yoitsu-job-j1")
    sup.jobs["j1"] = handle
    sup.backend.inspect = AsyncMock(return_value=ContainerState(
        exists=True, status="exited", running=False, exit_code=42,
    ))

    await sup._mark_exited_jobs()

    assert "j1" not in sup.jobs
    types = [event_type for event_type, _ in emitted]
    assert "supervisor.job.failed" in types
    assert "supervisor.task.failed" in types
    fail_data = next(data for event_type, data in emitted if event_type == "supervisor.job.failed")
    assert fail_data["code"] == "runtime_exit_before_start"
    assert fail_data["exit_code"] == 42
    assert "container log line" in fail_data["logs_tail"]


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

    sup.state.tasks["t1"] = TaskRecord(task_id="t1", goal="...", bundle="factorio")
    sup.state.jobs_by_id["j1"] = SpawnedJob(
        job_id="j1", source_event_id="e1", goal="t", role="default",
        repo="r", init_branch="b", bundle_sha="s", task_id="t1", bundle="factorio"
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
    assert "supervisor.job.failed" in types
    assert "supervisor.task.failed" in types
    assert "supervisor.checkpoint" in types
    fail_data = next(d for t, d in emitted if t == "supervisor.job.failed")
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
    sup._pending["p1"] = SpawnedJob("p1", "e", "t", "r", "/r", "m", None, bundle="factorio")

    await sup._checkpoint()

    cp = next(d for t, d in emitted if t == "supervisor.checkpoint")
    assert cp["cursor"] == "2026-01-01T00:00:00|abc"
    assert cp["pending_jobs"] == ["p1"]


@pytest.mark.asyncio
async def test_fetch_all_paginates():
    sup = _make_supervisor()
    page1 = ([_evt("e1", "agent.job.started", {"job_id": "e1"}),
              _evt("e2", "agent.job.started", {"job_id": "e2"})], "cursor-2")
    page2 = ([_evt("e3", "agent.job.started", {"job_id": "e3"})], None)
    pages = [page1, page2]
    call_count = 0

    async def fake_poll(cursor=None, source=None, type_=None, limit=100):
        nonlocal call_count
        result = pages[call_count]
        call_count += 1
        return result

    sup.client.poll = fake_poll
    events = await sup._fetch_all("agent.job.started")
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
                "goal": "do X",
                "role": "default",
                "bundle": "factorio",
                "repo": "/r",
                "init_branch": "main",
                "bundle_sha": "",
                "budget": 1.0,
                "parent_job_id": "",
                "condition": None,
                "job_context": {},
                "queue_state": "ready",
            })]
        if type_ == "supervisor.task.created":
            return [_evt("c-1", "supervisor.task.created", {"task_id": "t-1", "goal": "do X", "source_trigger_id": "sub-1"})]
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
                "goal": "do X",
                "role": "default",
                "bundle": "factorio",
                "repo": "/r",
                "init_branch": "main",
                "bundle_sha": "",
                "budget": 1.0,
                "parent_job_id": "",
                "condition": None,
                "job_context": {},
                "queue_state": "ready",
            })]
        if type_ == "supervisor.task.created":
            return [_evt("c-1", "supervisor.task.created", {"task_id": "t-1", "goal": "do X", "source_trigger_id": "sub-1"})]
        if type_ == "supervisor.job.launched":
            return [_evt("l1", "supervisor.job.launched", {
                "source_event_id": "sub-1", "job_id": "job-A",
                "container_id": "ctr-A", "container_name": "yoitsu-job-job-A",
                "task_id": "t-1", "goal": "do X", "role": "default", "repo": "/r"
            })]
        if type_ == "agent.job.started":
            return [_evt("s1", "agent.job.started", {"job_id": "job-A"})]
        if type_ == "agent.job.completed":
            return [_evt("c1", "agent.job.completed", {"job_id": "job-A", "summary": "ok"})]
        return []

    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup._ready_queue.qsize() == 0
    assert "sub-1" in sup._launched_event_ids
    assert "job-A" in sup._completed_jobs


@pytest.mark.asyncio
async def test_replay_fails_missing_launched_container():
    sup = _make_supervisor()
    sup._inspect_replay_state = AsyncMock(return_value=ContainerState(exists=False))
    emitted = []

    async def fake_emit(type_, data, **kwargs):
        emitted.append((type_, data))
        return "evt-replay-runtime-failed"

    sup.client.emit = fake_emit

    async def fake_fetch_all(type_, source=None):
        if type_ == "supervisor.job.enqueued":
            return [_evt("q-1", "supervisor.job.enqueued", {
                "job_id": "job-A",
                "task_id": "t-1",
                "source_event_id": "sub-1",
                "goal": "do X",
                "role": "default",
                "bundle": "factorio",
                "repo": "/r",
                "init_branch": "main",
                "bundle_sha": "",
                "budget": 1.0,
                "parent_job_id": "",
                "condition": None,
                "job_context": {},
                "queue_state": "ready",
            })]
        if type_ == "supervisor.task.created":
            return [_evt("c-1", "supervisor.task.created", {"task_id": "t-1", "goal": "do X", "source_trigger_id": "sub-1"})]
        if type_ == "supervisor.job.launched":
            return [_evt("l1", "supervisor.job.launched", {
                "source_event_id": "sub-1",
                "job_id": "job-A",
                "container_id": "ctr-A",
                "container_name": "yoitsu-job-job-A",
                "task_id": "t-1", "goal": "do X", "role": "default", "repo": "/r"
            })]
        return []

    sup._fetch_all = fake_fetch_all

    await sup._replay_unfinished_tasks()
    assert sup._ready_queue.qsize() == 0
    assert "job-A" in sup._failed_jobs
    assert any(type_ == "supervisor.job.failed" for type_, _ in emitted)


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
                "goal": "do X",
                "role": "default",
                "bundle": "factorio",
                "repo": "/r",
                "init_branch": "main",
                "bundle_sha": "",
                "budget": 1.0,
                "parent_job_id": "",
                "condition": None,
                "job_context": {},
                "queue_state": "ready",
            })]
        if type_ == "supervisor.task.created":
            return [_evt("c-1", "supervisor.task.created", {"task_id": "t-1", "goal": "do X", "source_trigger_id": "sub-1"})]
        if type_ == "supervisor.job.launched":
            return [_evt("l1", "supervisor.job.launched", {
                "source_event_id": "sub-1",
                "job_id": "job-A",
                "container_id": "ctr-A",
                "container_name": "yoitsu-job-job-A",
                "task_id": "t-1", "goal": "do X", "role": "default", "repo": "/r"
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
        if type_ == "supervisor.job.enqueued" or type_ == "supervisor.task.created":
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

    with patch.object(sup, "_try_register_webhook", new_callable=AsyncMock), \
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
    mock_run_loop.assert_called_once()
    mock_create_task.assert_called_once()
    mock_task.cancel.assert_called_once()


def test_status_reflects_unified_state():
    sup = _make_supervisor()
    sup._pending["p1"] = SpawnedJob("p1", "e", "t", "r", "/r", "m", None, bundle="factorio")
    st = sup.status
    assert st["pending_jobs"] == 1
    assert "ready_queue_size" in st
    assert st["runtime_kind"] == "podman"
    assert "fork_joins_active" not in st
