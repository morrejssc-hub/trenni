from __future__ import annotations

import base64
import os

import yaml
from yoitsu_contracts.artifact import ArtifactBinding
from yoitsu_contracts.config import EventStoreConfig, JobConfig, JobContextConfig
from yoitsu_contracts.env import build_git_auth_env

from .config import TeamRuntimeConfig, TrenniConfig, _UNSET
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

    def _get_team_runtime(self, team: str) -> TeamRuntimeConfig | None:
        """Get runtime config for a team, or None if team not found."""
        team_config = self.config.teams.get(team)
        if team_config is None:
            return None
        return team_config.runtime

    def build(
        self,
        *,
        job_id: str,
        task_id: str | None = None,
        source_event_id: str,
        goal: str,
        role: str,
        role_params: dict | None = None,
        team: str = "default",
        repo: str,
        init_branch: str,
        evo_sha: str | None,
        budget: float | None = None,
        job_context: JobContextConfig | None = None,
        input_artifacts: list[ArtifactBinding] | None = None,  # ADR-0013
    ) -> JobRuntimeSpec:
        """Build JobRuntimeSpec from task semantics and role-derived defaults."""
        # Workspace: use defaults + repo/init_branch/input_artifacts from spawn
        merged_workspace = {
            **self.config.default_workspace,
            "repo": repo,
            "init_branch": init_branch,
        }
        # ADR-0013: propagate input_artifacts from SpawnedJob
        if input_artifacts:
            merged_workspace["input_artifacts"] = [
                b.model_dump(mode="json") for b in input_artifacts
            ]

        # LLM: use defaults + budget (single channel)
        llm_config = dict(self.config.default_llm)
        if budget is not None and budget > 0:
            llm_config["max_total_cost"] = budget
        # else: use default from TrenniConfig

        # Publication: use defaults only (no overrides)
        publication_config = dict(self.config.default_publication)

        job_config = JobConfig.model_validate(
            {
                "job_id": job_id,
                "task_id": task_id or job_id,
                "goal": goal,
                "evo_sha": evo_sha or "",
                "role": role,
                "role_params": dict(role_params or {}),
                "team": team,
                "workspace": merged_workspace,
                "llm": llm_config,
                "publication": publication_config,
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

        labels = {
            **self.defaults.labels,
            "io.yoitsu.job-id": job_id,
            "io.yoitsu.source-event-id": source_event_id,
            "io.yoitsu.runtime": self.defaults.kind,
            "io.yoitsu.evo-sha": evo_sha or "",
        }

        # Get team runtime config and merge with defaults per ADR-0011 D4
        team_runtime = self._get_team_runtime(team)

        # Merge semantics per ADR-0011 D4:
        # - image: team value overrides default if set (None = use default)
        # - pod_name: team value overrides default if set (None = no pod); unset inherits default
        # - env_allowlist: team value replaces default (not merged)
        # - extra_networks: team value used (default is empty)
        if team_runtime is not None:
            image = team_runtime.image if team_runtime.image is not None else self.defaults.image
            # pod_name: _UNSET = inherit default; None = explicit no pod; string = use that value
            if team_runtime.pod_name is _UNSET:
                pod_name = self.defaults.pod_name
            else:
                pod_name = team_runtime.pod_name  # Could be None (explicit no pod) or string
            env_allowlist = tuple(team_runtime.env_allowlist) if team_runtime.env_allowlist else self.defaults.env_allowlist
            extra_networks = tuple(team_runtime.extra_networks)
        else:
            # Team not found, use all defaults
            image = self.defaults.image
            pod_name = self.defaults.pod_name
            env_allowlist = self.defaults.env_allowlist
            extra_networks = ()

        # Rebuild env with team's env_allowlist
        env: dict[str, str] = {
            "PALIMPSEST_JOB_CONFIG_B64": payload_b64,
        }
        for key in env_allowlist:
            value = os.environ.get(key)
            if value:
                env[key] = value

        eventstore_key = os.environ.get(self.config.pasloe_api_key_env, "")
        if eventstore_key:
            env[self.config.pasloe_api_key_env] = eventstore_key

        env.update(build_git_auth_env(self.defaults.git_token_env))

        return JobRuntimeSpec(
            job_id=job_id,
            source_event_id=source_event_id,
            # Child task IDs use '/' as hierarchy separator (e.g. "abc123/fv7o-eval"),
            # which is invalid in Podman container names. Replace with '-'.
            container_name=f"yoitsu-job-{job_id.replace('/', '-')}",
            image=image,
            pod_name=pod_name,
            labels=labels,
            env=env,
            command=("palimpsest", "container-entrypoint"),
            config_payload_b64=payload_b64,
            extra_networks=extra_networks,
        )
