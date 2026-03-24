from __future__ import annotations

import base64
import os

import yaml
from yoitsu_contracts.config import EventStoreConfig, JobConfig, JobContextConfig
from yoitsu_contracts.env import build_git_auth_env

from .config import TrenniConfig
from .runtime_types import JobRuntimeSpec, RuntimeDefaults

build_git_credential_env = build_git_auth_env


def build_runtime_defaults(config: TrenniConfig) -> RuntimeDefaults:
    if config.runtime.kind != "podman":
        raise ValueError(f"Unsupported runtime kind {config.runtime.kind!r}")

    podman = config.runtime.podman
    socket_uri = podman.socket_uri or os.environ.get("PODMAN_HOST", "") or "unix:///run/podman/podman.sock"
    labels = {
        "io.yoitsu.managed-by": "trenni",
        "io.yoitsu.stack": "yoitsu",
        **podman.labels,
    }
    return RuntimeDefaults(
        kind="podman",
        socket_uri=socket_uri,
        pod_name=podman.pod_name,
        image=podman.image,
        pull_policy=podman.pull_policy,
        stop_grace_seconds=podman.stop_grace_seconds,
        cleanup_timeout_seconds=podman.cleanup_timeout_seconds,
        retain_on_failure=podman.retain_on_failure,
        labels=labels,
        env_allowlist=tuple(podman.env_allowlist),
        git_token_env=podman.git_token_env,
    )


class RuntimeSpecBuilder:
    def __init__(self, config: TrenniConfig, defaults: RuntimeDefaults) -> None:
        self.config = config
        self.defaults = defaults

    def build(
        self,
        *,
        job_id: str,
        task_id: str | None = None,
        source_event_id: str,
        task: str,
        role: str,
        repo: str,
        init_branch: str,
        evo_sha: str | None,
        llm_overrides: dict | None = None,
        workspace_overrides: dict | None = None,
        publication_overrides: dict | None = None,
        job_context: JobContextConfig | None = None,
    ) -> JobRuntimeSpec:
        merged_workspace = {
            **self.config.default_workspace,
            **(workspace_overrides or {}),
            "repo": repo,
            "init_branch": init_branch,
        }
        job_config = JobConfig.model_validate(
            {
                "job_id": job_id,
                "task_id": task_id or job_id,
                "task": task,
                "role": role,
                "workspace": merged_workspace,
                "llm": {
                    **self.config.default_llm,
                    **(llm_overrides or {}),
                },
                "publication": {
                    **self.config.default_publication,
                    **(publication_overrides or {}),
                },
                "eventstore": EventStoreConfig(
                    url=self.config.eventstore_url,
                    api_key_env=self.config.pasloe_api_key_env,
                    source_id=self.config.default_eventstore_source,
                ).model_dump(mode="json"),
                "context": (job_context or JobContextConfig()).model_dump(mode="json", exclude_none=True),
            }
        )

        payload_text = yaml.safe_dump(
            job_config.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
        )
        payload_b64 = base64.b64encode(payload_text.encode("utf-8")).decode("utf-8")

        env: dict[str, str] = {
            "PALIMPSEST_JOB_CONFIG_B64": payload_b64,
        }
        for key in self.defaults.env_allowlist:
            value = os.environ.get(key)
            if value:
                env[key] = value

        eventstore_key = os.environ.get(self.config.pasloe_api_key_env, "")
        if eventstore_key:
            env[self.config.pasloe_api_key_env] = eventstore_key

        env.update(build_git_auth_env(self.defaults.git_token_env))

        labels = {
            **self.defaults.labels,
            "io.yoitsu.job-id": job_id,
            "io.yoitsu.source-event-id": source_event_id,
            "io.yoitsu.runtime": self.defaults.kind,
            "io.yoitsu.evo-sha": evo_sha or "",
        }

        return JobRuntimeSpec(
            job_id=job_id,
            source_event_id=source_event_id,
            container_name=f"yoitsu-job-{job_id}",
            image=self.defaults.image,
            pod_name=self.defaults.pod_name,
            labels=labels,
            env=env,
            command=("palimpsest", "container-entrypoint"),
            config_payload_b64=payload_b64,
        )
