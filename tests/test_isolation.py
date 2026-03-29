import base64
import json
from pathlib import Path

import httpx
import pytest
import yaml

from trenni.config import RuntimeConfig, TrenniConfig
from trenni.podman_backend import PodmanBackend
from trenni.runtime_builder import (
    RuntimeSpecBuilder,
    build_git_credential_env,
    build_runtime_defaults,
)
from trenni.runtime_types import JobHandle, JobRuntimeSpec, RuntimeDefaults


def test_build_git_credential_env(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "test-token-123")
    result = build_git_credential_env("MY_TOKEN")

    assert result["GIT_CONFIG_COUNT"] == "2"
    assert "AUTHORIZATION" in result["GIT_CONFIG_VALUE_1"]


def test_build_git_credential_env_empty_when_no_token(monkeypatch):
    monkeypatch.delenv("MY_TOKEN", raising=False)
    result = build_git_credential_env("MY_TOKEN")
    assert result == {}


def test_build_runtime_defaults_from_runtime_block():
    config = TrenniConfig(
        runtime=RuntimeConfig.from_dict({
            "kind": "podman",
            "podman": {
                "socket_uri": "unix:///tmp/podman.sock",
                "pod_name": "yoitsu-dev",
                "image": "localhost/yoitsu-palimpsest-job:dev",
                "pull_policy": "missing",
                "git_token_env": "GITHUB_TOKEN",
                "env_allowlist": ["OPENAI_API_KEY"],
                "labels": {"io.yoitsu.env": "test"},
            },
        })
    )

    defaults = build_runtime_defaults(config)

    assert defaults.kind == "podman"
    assert defaults.socket_uri == "unix:///tmp/podman.sock"
    assert defaults.pull_policy == "missing"
    assert defaults.labels["io.yoitsu.managed-by"] == "trenni"
    assert defaults.labels["io.yoitsu.env"] == "test"


def test_build_runtime_defaults_rejects_non_podman():
    config = TrenniConfig(runtime=RuntimeConfig(kind="subprocess"))
    with pytest.raises(ValueError):
        build_runtime_defaults(config)


def test_runtime_spec_builder_serializes_job_config(monkeypatch):
    monkeypatch.setenv("PASLOE_API_KEY", "pasloe-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")

    config = TrenniConfig(
        pasloe_api_key_env="PASLOE_API_KEY",
        default_llm={"api_key_env": "OPENAI_API_KEY", "model": "kimi"},
        default_workspace={"depth": 1},
        default_publication={"branch_prefix": "palimpsest/job"},
        runtime=RuntimeConfig.from_dict({
            "podman": {
                "env_allowlist": ["OPENAI_API_KEY"],
                "git_token_env": "GITHUB_TOKEN",
            },
        }),
    )
    defaults = build_runtime_defaults(config)
    builder = RuntimeSpecBuilder(config, defaults)

    # Per ADR-0007: budget is passed directly, execution config from defaults
    spec = builder.build(
        job_id="job-1",
        task_id="task-1",
        source_event_id="evt-1",
        task="do the thing",
        role="default",
        repo="git@example.com/repo.git",
        init_branch="main",
        evo_sha="abc123",
        budget=0.75,  # single-channel budget per ADR-0007
    )

    assert spec.container_name == "yoitsu-job-job-1"
    assert spec.env["OPENAI_API_KEY"] == "openai-secret"
    assert spec.env["PASLOE_API_KEY"] == "pasloe-secret"
    assert spec.env["GIT_CONFIG_COUNT"] == "2"
    assert spec.labels["io.yoitsu.job-id"] == "job-1"
    assert spec.command == ("palimpsest", "container-entrypoint")

    payload = yaml.safe_load(base64.b64decode(spec.config_payload_b64))
    assert payload["job_id"] == "job-1"
    assert payload["workspace"]["repo"] == "git@example.com/repo.git"
    # budget maps to llm.max_total_cost (single channel)
    assert payload["llm"]["max_total_cost"] == 0.75


def test_config_from_yaml_rejects_legacy_runtime_fields(tmp_path: Path):
    cfg = tmp_path / "trenni.yaml"
    cfg.write_text("palimpsest_command: palimpsest\n")

    with pytest.raises(ValueError):
        TrenniConfig.from_yaml(cfg)


@pytest.mark.asyncio
async def test_podman_backend_ensure_ready_pulls_missing_image():
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/libpod/pods/yoitsu-dev/exists"):
            return httpx.Response(204)
        if request.url.path.endswith("/exists") and "/images/" in request.url.path:
            return httpx.Response(404)
        if request.url.path.endswith("/libpod/images/pull"):
            return httpx.Response(200, json={"images": ["localhost/yoitsu-palimpsest-job:dev"]})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    defaults = RuntimeDefaults(
        kind="podman",
        socket_uri="unix:///tmp/podman.sock",
        pod_name="yoitsu-dev",
        image="localhost/yoitsu-palimpsest-job:dev",
        pull_policy="missing",
        stop_grace_seconds=10,
        cleanup_timeout_seconds=120,
        retain_on_failure=False,
        labels={},
        env_allowlist=(),
        git_token_env="GITHUB_TOKEN",
    )
    backend = PodmanBackend(defaults, transport=httpx.MockTransport(handler))

    await backend.ensure_ready()
    await backend.close()

    assert ("POST", "/v1.0.0/libpod/images/pull") in calls


@pytest.mark.asyncio
async def test_podman_backend_create_start_inspect_remove():
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/libpod/containers/create"):
            seen_payloads.append(json.loads(request.content.decode()))
            return httpx.Response(201, json={"Id": "ctr-123"})
        if request.url.path.endswith("/libpod/containers/ctr-123/start"):
            return httpx.Response(204)
        if request.url.path.endswith("/libpod/containers/ctr-123/json"):
            return httpx.Response(200, json={"State": {"Status": "running", "Running": True, "ExitCode": 0}})
        if request.url.path.endswith("/libpod/containers/ctr-123/logs"):
            return httpx.Response(200, text="hello\n")
        if request.url.path.endswith("/libpod/containers/ctr-123"):
            return httpx.Response(204)
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    defaults = RuntimeDefaults(
        kind="podman",
        socket_uri="unix:///tmp/podman.sock",
        pod_name="yoitsu-dev",
        image="localhost/yoitsu-palimpsest-job:dev",
        pull_policy="never",
        stop_grace_seconds=10,
        cleanup_timeout_seconds=120,
        retain_on_failure=False,
        labels={},
        env_allowlist=(),
        git_token_env="GITHUB_TOKEN",
    )
    backend = PodmanBackend(defaults, transport=httpx.MockTransport(handler))
    spec = JobRuntimeSpec(
        job_id="job-1",
        source_event_id="evt-1",
        container_name="yoitsu-job-job-1",
        image="localhost/yoitsu-palimpsest-job:dev",
        pod_name="yoitsu-dev",
        labels={"io.yoitsu.job-id": "job-1"},
        env={"PALIMPSEST_JOB_CONFIG_B64": "abc"},
        command=("palimpsest", "container-entrypoint"),
        config_payload_b64="abc",
    )

    handle = await backend.create(spec)
    await backend.start(handle)
    state = await backend.inspect(handle)
    logs = await backend.logs(handle)
    await backend.remove(handle)
    await backend.close()

    assert handle.container_id == "ctr-123"
    assert state.running is True
    assert logs == "hello\n"
    assert seen_payloads[0]["pod"] == "yoitsu-dev"
    assert seen_payloads[0]["command"] == ["palimpsest", "container-entrypoint"]


@pytest.mark.asyncio
async def test_podman_backend_inspect_missing_container():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/libpod/containers/ctr-404/json"):
            return httpx.Response(404)
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    defaults = RuntimeDefaults(
        kind="podman",
        socket_uri="unix:///tmp/podman.sock",
        pod_name="yoitsu-dev",
        image="localhost/yoitsu-palimpsest-job:dev",
        pull_policy="never",
        stop_grace_seconds=10,
        cleanup_timeout_seconds=120,
        retain_on_failure=False,
        labels={},
        env_allowlist=(),
        git_token_env="GITHUB_TOKEN",
    )
    backend = PodmanBackend(defaults, transport=httpx.MockTransport(handler))

    state = await backend.inspect(JobHandle(job_id="job-1", container_id="ctr-404", container_name="yoitsu-job-job-1"))
    await backend.close()

    assert state.exists is False
