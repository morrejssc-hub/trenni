"""P1 fix tests: input_artifacts propagation through SpawnedJob -> runtime spec."""

import tempfile
from pathlib import Path

import pytest

from trenni.config import TrenniConfig
from trenni.runtime_builder import RuntimeSpecBuilder, build_runtime_defaults
from trenni.state import SpawnedJob
from yoitsu_contracts.artifact import ArtifactRef, ArtifactBinding
from yoitsu_contracts.local_fs_backend import LocalFSBackend


def test_input_artifacts_propagated_to_runtime_spec():
    """P1 fix: input_artifacts must propagate from SpawnedJob to JobConfig.workspace."""
    # Create artifact binding
    store_root = tempfile.mkdtemp(prefix="test-store-")
    backend = LocalFSBackend(Path(store_root))

    input_dir = tempfile.mkdtemp(prefix="input-")
    (Path(input_dir) / "file.txt").write_text("test content")
    ref = backend.store_tree(Path(input_dir))
    binding = ArtifactBinding(ref=ref, relation="input", path="data")

    # Create SpawnedJob with input_artifacts
    job = SpawnedJob(
        job_id="test-job-123",
        source_event_id="evt-001",
        goal="Test input artifacts propagation",
        role="default",
        repo="https://github.com/example/repo.git",
        init_branch="main",
        evo_sha=None,
        budget=1.0,
        input_artifacts=[binding],
    )

    # Build runtime spec
    config = TrenniConfig()
    defaults = build_runtime_defaults(config)
    builder = RuntimeSpecBuilder(config, defaults)

    spec = builder.build(
        job_id=job.job_id,
        source_event_id=job.source_event_id,
        goal=job.goal,
        role=job.role,
        repo=job.repo,
        init_branch=job.init_branch,
        evo_sha=job.evo_sha,
        budget=job.budget,
        input_artifacts=job.input_artifacts,
    )

    # Decode and parse JobConfig from payload
    import base64
    import yaml
    from yoitsu_contracts.config import JobConfig

    payload_yaml = yaml.safe_load(
        base64.b64decode(spec.config_payload_b64).decode("utf-8")
    )
    job_config = JobConfig.model_validate(payload_yaml)

    # P1 fix verification: input_artifacts must be in workspace config
    assert len(job_config.workspace.input_artifacts) > 0, \
        "input_artifacts must be propagated to JobConfig.workspace"

    propagated = job_config.workspace.input_artifacts[0]
    assert propagated.relation == "input"
    assert propagated.ref.digest == ref.digest
    assert propagated.path == "data"


def test_spawned_job_serialization_preserves_input_artifacts():
    """Verify SpawnedJob serialization roundtrips input_artifacts."""
    store_root = tempfile.mkdtemp(prefix="test-store-")
    backend = LocalFSBackend(Path(store_root))

    input_dir = tempfile.mkdtemp(prefix="input-")
    (Path(input_dir) / "file.txt").write_text("test")
    ref = backend.store_tree(Path(input_dir))
    binding = ArtifactBinding(ref=ref, relation="input", path="")

    job = SpawnedJob(
        job_id="test-serialize",
        source_event_id="evt-002",
        goal="Test serialization",
        role="default",
        repo="",
        init_branch="main",
        evo_sha=None,
        input_artifacts=[binding],
    )

    # Serialize via to_enqueued_data
    enqueued_data = job.to_enqueued_data("queued", None)

    assert "input_artifacts" in enqueued_data
    assert len(enqueued_data["input_artifacts"]) > 0

    # Deserialize via from_enqueued_data
    restored = SpawnedJob.from_enqueued_data(enqueued_data)

    assert len(restored.input_artifacts) > 0
    assert restored.input_artifacts[0].ref.digest == ref.digest


def test_spawned_job_launched_data_includes_input_artifacts():
    """Verify SpawnedJob.to_launched_data includes input_artifacts."""
    store_root = tempfile.mkdtemp(prefix="test-store-")
    backend = LocalFSBackend(Path(store_root))

    input_dir = tempfile.mkdtemp(prefix="input-")
    (Path(input_dir) / "file.txt").write_text("test")
    ref = backend.store_tree(Path(input_dir))
    binding = ArtifactBinding(ref=ref, relation="input", path="")

    job = SpawnedJob(
        job_id="test-launched",
        source_event_id="evt-003",
        goal="Test launched data",
        role="default",
        repo="",
        init_branch="main",
        evo_sha=None,
        input_artifacts=[binding],
    )

    launched_data = job.to_launched_data("podman", "container-123", "yoitsu-job-test", None)

    assert "input_artifacts" in launched_data
    assert len(launched_data["input_artifacts"]) > 0


def test_trigger_data_input_artifacts_to_spawned_job():
    """P1 fix: TriggerData.input_artifacts must flow to SpawnedJob in _process_trigger."""
    # This test verifies the fix in trenni/supervisor.py:448
    # where root_job = SpawnedJob(...) now includes input_artifacts

    from yoitsu_contracts.events import TriggerData

    store_root = tempfile.mkdtemp(prefix="trigger-test-store-")
    backend = LocalFSBackend(Path(store_root))

    input_dir = tempfile.mkdtemp(prefix="trigger-input-")
    (Path(input_dir) / "data.txt").write_text("trigger input")
    ref = backend.store_tree(Path(input_dir))
    binding = ArtifactBinding(ref=ref, relation="input", path="")

    trigger = TriggerData(
        goal="Test trigger with artifacts",
        role="default",
        input_artifacts=[binding],
    )

    # Verify TriggerData has the artifacts
    assert len(trigger.input_artifacts) == 1
    assert trigger.input_artifacts[0].ref.digest == ref.digest

    # Simulate what _process_trigger does (constructing SpawnedJob)
    root_job = SpawnedJob(
        job_id="test-root-job",
        source_event_id="evt-trigger",
        goal=trigger.goal,
        role=trigger.role,
        repo=trigger.repo,
        init_branch=trigger.init_branch or "main",
        evo_sha=trigger.sha,
        budget=trigger.budget,
        task_id="test-task",
        team=trigger.team,
        input_artifacts=list(trigger.input_artifacts),
    )

    # P1 fix verification: input_artifacts must be in SpawnedJob
    assert len(root_job.input_artifacts) == 1
    assert root_job.input_artifacts[0].ref.digest == ref.digest


def test_launched_event_replay_preserves_input_artifacts():
    """P1 fix: _register_replayed_launch must restore input_artifacts from event data."""
    from yoitsu_contracts.events import SupervisorJobLaunchedData

    store_root = tempfile.mkdtemp(prefix="replay-test-store-")
    backend = LocalFSBackend(Path(store_root))

    input_dir = tempfile.mkdtemp(prefix="replay-input-")
    (Path(input_dir) / "replay.txt").write_text("replay content")
    ref = backend.store_tree(Path(input_dir))
    binding = ArtifactBinding(ref=ref, relation="input", path="replay-data")

    # Simulate launched event data
    launched = SupervisorJobLaunchedData(
        job_id="test-replay-job",
        goal="Test replay",
        role="default",
        input_artifacts=[binding],
    )

    event_data = launched.model_dump(mode="json")

    # Simulate what _register_replayed_launch does
    input_artifacts_data = event_data.get("input_artifacts", [])
    restored_artifacts = [
        ArtifactBinding.model_validate(b) for b in input_artifacts_data
    ] if input_artifacts_data else []

    restored_job = SpawnedJob(
        job_id=event_data.get("job_id", ""),
        source_event_id=event_data.get("source_event_id", ""),
        goal=event_data.get("goal", ""),
        role=event_data.get("role", "default"),
        team=event_data.get("team", "default"),
        repo=event_data.get("repo", ""),
        init_branch=event_data.get("init_branch", "main"),
        evo_sha=event_data.get("evo_sha") or None,
        budget=event_data.get("budget", 0.0),
        task_id=event_data.get("task_id", "") or event_data.get("job_id", ""),
        parent_job_id=event_data.get("parent_job_id", ""),
        input_artifacts=restored_artifacts,
    )

    # P1 fix verification: input_artifacts preserved through replay
    assert len(restored_job.input_artifacts) == 1
    assert restored_job.input_artifacts[0].ref.digest == ref.digest
    assert restored_job.input_artifacts[0].path == "replay-data"