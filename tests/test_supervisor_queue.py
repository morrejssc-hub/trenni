"""Tests for Trenni supervisor queue and replay logic."""
import re
import pytest


def test_generate_job_id_is_uuid_v7():
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig
    sup = Supervisor(TrenniConfig())
    job_id = sup._generate_job_id()
    # UUID v7: xxxxxxxx-xxxx-7xxx-xxxx-xxxxxxxxxxxx
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        job_id,
    ), f"Not a UUID v7: {job_id}"


def test_task_item_fields():
    from trenni.supervisor import TaskItem
    item = TaskItem(
        source_event_id="evt-1",
        job_id="job-1",
        task="do something",
        role="default",
        repo="/repo",
        branch="main",
        evo_sha=None,
    )
    assert item.source_event_id == "evt-1"
    assert item.evo_sha is None


def test_supervisor_has_queue_and_dedup_set():
    import asyncio
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig
    sup = Supervisor(TrenniConfig())
    assert isinstance(sup._task_queue, asyncio.Queue)
    assert isinstance(sup._launched_event_ids, set)


import asyncio
import pytest


@pytest.mark.asyncio
async def test_handle_task_submit_enqueues():
    from unittest.mock import AsyncMock, patch
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig
    from trenni.pasloe_client import Event
    from datetime import datetime

    sup = Supervisor(TrenniConfig())
    event = Event(
        id="evt-abc",
        source_id="test",
        type="task.submit",
        ts=datetime.utcnow(),
        data={"task": "do X", "role": "default", "repo": "/r", "branch": "main"},
    )
    with patch.object(sup, "_launch", new_callable=AsyncMock) as mock_launch:
        await sup._handle_task_submit(event)
        mock_launch.assert_not_called()   # should not launch directly

    assert sup._task_queue.qsize() == 1
    item = sup._task_queue.get_nowait()
    assert item.source_event_id == "evt-abc"
    assert item.task == "do X"


@pytest.mark.asyncio
async def test_handle_task_submit_deduplicates():
    from unittest.mock import AsyncMock, patch
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig
    from trenni.pasloe_client import Event
    from datetime import datetime

    sup = Supervisor(TrenniConfig())
    sup._launched_event_ids.add("evt-dup")
    event = Event(
        id="evt-dup",
        source_id="test",
        type="task.submit",
        ts=datetime.utcnow(),
        data={"task": "do X", "role": "default", "repo": "/r", "branch": "main"},
    )
    await sup._handle_task_submit(event)
    assert sup._task_queue.qsize() == 0   # deduped, not enqueued


@pytest.mark.asyncio
async def test_drain_queue_launches_when_capacity():
    from unittest.mock import AsyncMock
    from trenni.supervisor import Supervisor, TaskItem
    from trenni.config import TrenniConfig

    sup = Supervisor(TrenniConfig(max_workers=2))
    item = TaskItem("evt-1", "job-1", "task", "default", "/repo", "main", None)
    await sup._task_queue.put(item)

    launched = []

    async def fake_launch_from_item(i):
        launched.append(i.job_id)

    sup._launch_from_item = fake_launch_from_item

    # Run _drain_queue briefly then cancel
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
    from unittest.mock import AsyncMock
    from trenni.supervisor import Supervisor, TaskItem
    from trenni.config import TrenniConfig
    from trenni.isolation import JobProcess

    sup = Supervisor(TrenniConfig(max_workers=1))

    # Fake a running job to fill capacity
    fake_proc = AsyncMock()
    fake_proc.returncode = None
    from pathlib import Path
    sup.jobs["existing-job"] = JobProcess(
        job_id="existing-job", proc=fake_proc,
        work_dir=Path("/tmp"), config_path=Path("/tmp/cfg.yaml")
    )

    item = TaskItem("evt-2", "job-2", "task", "default", "/repo", "main", None)
    await sup._task_queue.put(item)

    launched = []
    async def fake_launch_from_item(i):
        launched.append(i.job_id)
    sup._launch_from_item = fake_launch_from_item

    task = asyncio.create_task(sup._drain_queue())
    await asyncio.sleep(0.15)  # give drain loop time to run
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert launched == []   # still waiting for capacity


@pytest.mark.asyncio
async def test_launch_emits_source_event_id():
    from unittest.mock import AsyncMock, patch, MagicMock
    from trenni.supervisor import Supervisor, TaskItem
    from trenni.config import TrenniConfig
    from pathlib import Path

    sup = Supervisor(TrenniConfig())

    emitted = []
    async def fake_emit(type_, data):
        emitted.append((type_, data))
        return "evt-out"
    sup.client.emit = fake_emit

    # Mock the isolation backend to avoid real subprocess
    fake_proc = AsyncMock()
    fake_proc.returncode = None
    fake_proc.pid = 9999
    fake_jp_mock = MagicMock()
    fake_jp_mock.proc = fake_proc
    with patch("trenni.supervisor.launch_job", new_callable=AsyncMock, return_value=fake_jp_mock):
        await sup._launch(
            job_id="job-xyz",
            task="test",
            role="default",
            repo="/repo",
            branch="main",
            evo_sha=None,
            source_event_id="evt-src-123",
        )

    assert emitted, "No events emitted"
    type_, data = emitted[0]
    assert type_ == "supervisor.job.launched"
    assert data["source_event_id"] == "evt-src-123"
    assert data["job_id"] == "job-xyz"


@pytest.mark.asyncio
async def test_fetch_all_paginates_until_done():
    from unittest.mock import AsyncMock
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig
    from trenni.pasloe_client import Event
    from datetime import datetime

    sup = Supervisor(TrenniConfig())

    def make_event(id_):
        return Event(id=id_, source_id="s", type="job.started",
                     ts=datetime.utcnow(), data={"job_id": id_})

    page1 = ([make_event("e1"), make_event("e2")], "cursor-page2")
    page2 = ([make_event("e3")], None)

    poll_results = [page1, page2]
    call_count = 0

    async def fake_poll(cursor=None, source=None, type_=None, limit=100):
        nonlocal call_count
        result = poll_results[call_count]
        call_count += 1
        return result

    sup.client.poll = fake_poll
    events = await sup._fetch_all("job.started")
    assert [e.id for e in events] == ["e1", "e2", "e3"]
    assert call_count == 2


@pytest.mark.asyncio
async def test_replay_enqueues_not_launched():
    """task.submit with no supervisor.job.launched → re-enqueue."""
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig
    from trenni.pasloe_client import Event
    from datetime import datetime

    sup = Supervisor(TrenniConfig())

    def make_event(id_, type_, data=None):
        return Event(id=id_, source_id="s", type=type_,
                     ts=datetime.utcnow(), data=data or {})

    async def fake_fetch_all(type_, source=None):
        if type_ == "task.submit":
            return [make_event("sub-1", "task.submit",
                               {"task": "do X", "role": "default", "repo": "/r", "branch": "main"})]
        return []

    sup._fetch_all = fake_fetch_all
    await sup._replay_unfinished_tasks()
    assert sup._task_queue.qsize() == 1


@pytest.mark.asyncio
async def test_replay_skips_completed():
    """task.submit with launched + started + completed → skip."""
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig
    from trenni.pasloe_client import Event
    from datetime import datetime

    sup = Supervisor(TrenniConfig())

    def make_event(id_, type_, data=None):
        return Event(id=id_, source_id="s", type=type_,
                     ts=datetime.utcnow(), data=data or {})

    async def fake_fetch_all(type_, source=None):
        if type_ == "task.submit":
            return [make_event("sub-1", "task.submit",
                               {"task": "do X", "role": "default", "repo": "/r", "branch": "main"})]
        if type_ == "supervisor.job.launched":
            return [make_event("launched-1", "supervisor.job.launched",
                               {"source_event_id": "sub-1", "job_id": "job-A"})]
        if type_ == "job.started":
            return [make_event("started-1", "job.started", {"job_id": "job-A"})]
        if type_ == "job.completed":
            return [make_event("done-1", "job.completed", {"job_id": "job-A"})]
        return []

    sup._fetch_all = fake_fetch_all
    await sup._replay_unfinished_tasks()
    assert sup._task_queue.qsize() == 0
    assert "sub-1" in sup._launched_event_ids


@pytest.mark.asyncio
async def test_replay_reenqueues_launched_not_started():
    """launched + no job.started + no job end + not in self.jobs → re-enqueue."""
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig
    from trenni.pasloe_client import Event
    from datetime import datetime

    sup = Supervisor(TrenniConfig())

    def make_event(id_, type_, data=None):
        return Event(id=id_, source_id="s", type=type_,
                     ts=datetime.utcnow(), data=data or {})

    async def fake_fetch_all(type_, source=None):
        if type_ == "task.submit":
            return [make_event("sub-1", "task.submit",
                               {"task": "do X", "role": "default", "repo": "/r", "branch": "main"})]
        if type_ == "supervisor.job.launched":
            return [make_event("launched-1", "supervisor.job.launched",
                               {"source_event_id": "sub-1", "job_id": "job-A"})]
        return []  # no job.started, no job.completed, no job.failed

    sup._fetch_all = fake_fetch_all
    await sup._replay_unfinished_tasks()
    assert sup._task_queue.qsize() == 1


@pytest.mark.asyncio
async def test_start_calls_replay_and_drain():
    from unittest.mock import AsyncMock, patch
    from trenni.supervisor import Supervisor
    from trenni.config import TrenniConfig

    sup = Supervisor(TrenniConfig())

    replay_called = False

    async def fake_replay():
        nonlocal replay_called
        replay_called = True

    sup._replay_unfinished_tasks = fake_replay

    # Patch _run_loop to return immediately so start() doesn't loop forever
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
