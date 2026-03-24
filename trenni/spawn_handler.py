from __future__ import annotations

from dataclasses import dataclass

from yoitsu_contracts.conditions import (
    AllCondition,
    AnyCondition,
    NotCondition,
    TaskIsCondition,
)
from yoitsu_contracts.config import JobContextConfig, JoinContextConfig
from yoitsu_contracts.events import SpawnRequestData

from .state import SpawnDefaults, SpawnedJob, TaskRecord, SupervisorState


@dataclass
class SpawnPlan:
    child_tasks: list[TaskRecord]
    jobs: list[SpawnedJob]


class SpawnHandler:
    def __init__(self, state: SupervisorState) -> None:
        self.state = state

    def expand(self, event) -> SpawnPlan:
        payload = SpawnRequestData.model_validate(event.data)
        parent_job_id = payload.job_id
        parent_job = self.state.jobs_by_id.get(parent_job_id)
        parent_defaults = self.state.spawn_defaults_by_job.get(parent_job_id)

        if parent_job is None and parent_defaults is None:
            raise ValueError(f"Unknown parent job {parent_job_id!r}")

        wait_for, on_fail = self._normalize_strategy(payload.wait_for, payload.on_fail)
        child_defs: list[tuple[str, str, str, str, str, str, str | None, dict, dict, dict]] = []

        for index, child in enumerate(payload.tasks):
            prompt = child.prompt.strip()
            if not prompt:
                continue

            task_id = f"{parent_job_id}:task:{index}"
            job_id = f"{parent_job_id}-c{index}"
            spec = child.job_spec

            role = spec.role or self._inherit("role", parent_job, parent_defaults, "default")
            repo = spec.repo or self._inherit("repo", parent_job, parent_defaults, "")
            init_branch = spec.init_branch or self._inherit("init_branch", parent_job, parent_defaults, "main")
            evo_sha = spec.evo_sha or self._inherit("evo_sha", parent_job, parent_defaults, None)

            llm = dict(self._inherit("llm_overrides", parent_job, parent_defaults, {}))
            llm.update(dict(spec.llm))

            workspace = dict(self._inherit("workspace_overrides", parent_job, parent_defaults, {}))
            workspace.update(dict(spec.workspace))

            publication = dict(self._inherit("publication_overrides", parent_job, parent_defaults, {}))
            publication.update(dict(spec.publication))

            child_defs.append(
                (
                    task_id,
                    job_id,
                    prompt,
                    role,
                    repo,
                    init_branch,
                    evo_sha,
                    llm,
                    workspace,
                    publication,
                )
            )

        child_task_ids = [task_id for task_id, *_ in child_defs]
        child_tasks = [
            TaskRecord(
                task_id=task_id,
                task=prompt,
                state="submitted",
                source_event_id=event.id,
                role=role,
                repo=repo,
                init_branch=init_branch,
                evo_sha=evo_sha,
            )
            for task_id, _, prompt, role, repo, init_branch, evo_sha, *_ in child_defs
        ]

        jobs: list[SpawnedJob] = []
        for task_id, job_id, prompt, role, repo, init_branch, evo_sha, llm, workspace, publication in child_defs:
            sibling_ids = [candidate for candidate in child_task_ids if candidate != task_id]
            guard_conditions = []

            if on_fail == "cancel_siblings" and sibling_ids:
                guard_conditions.append(
                    NotCondition(
                        condition=AnyCondition(
                            conditions=[
                                TaskIsCondition(task_id=sibling_id, state="failure")
                                for sibling_id in sibling_ids
                            ]
                        )
                    )
                )

            if wait_for == "any_success" and sibling_ids:
                guard_conditions.append(
                    NotCondition(
                        condition=AnyCondition(
                            conditions=[
                                TaskIsCondition(task_id=sibling_id, state="success")
                                for sibling_id in sibling_ids
                            ]
                        )
                    )
                )

            jobs.append(
                SpawnedJob(
                    job_id=job_id,
                    source_event_id=event.id,
                    task=prompt,
                    role=role,
                    repo=repo,
                    init_branch=init_branch,
                    evo_sha=evo_sha,
                    llm_overrides=llm,
                    workspace_overrides=workspace,
                    publication_overrides=publication,
                    task_id=task_id,
                    condition=self._combine_conditions(guard_conditions),
                    parent_job_id=parent_job_id,
                )
            )

        if parent_job is not None and child_task_ids:
            jobs.append(
                SpawnedJob(
                    job_id=f"{parent_job_id}-join",
                    source_event_id=event.id,
                    task=parent_job.task,
                    role=parent_job.role,
                    repo=parent_job.repo,
                    init_branch=parent_job.init_branch,
                    evo_sha=parent_job.evo_sha,
                    llm_overrides=dict(parent_job.llm_overrides),
                    workspace_overrides=dict(parent_job.workspace_overrides),
                    publication_overrides=dict(parent_job.publication_overrides),
                    task_id=parent_job.task_id or payload.task_id or parent_job.job_id,
                    condition=self._join_condition(child_task_ids, wait_for),
                    job_context=JobContextConfig(
                        join=JoinContextConfig(
                            parent_job_id=parent_job_id,
                            parent_task_id=parent_job.task_id or payload.task_id or parent_job.job_id,
                            parent_summary=parent_job.task,
                            child_task_ids=child_task_ids,
                        )
                    ),
                    parent_job_id=parent_job_id,
                )
            )

        return SpawnPlan(child_tasks=child_tasks, jobs=jobs)

    @staticmethod
    def _normalize_strategy(wait_for: str, on_fail: str) -> tuple[str, str]:
        normalized_wait_for = wait_for or "all_complete"
        normalized_on_fail = on_fail or "continue"

        if normalized_wait_for == "any_failed":
            normalized_wait_for = "all_complete"
            normalized_on_fail = "cancel_siblings"

        return normalized_wait_for, normalized_on_fail

    @staticmethod
    def _combine_conditions(conditions: list):
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return AllCondition(conditions=conditions)

    @staticmethod
    def _join_condition(child_task_ids: list[str], wait_for: str):
        if wait_for == "any_success":
            return AnyCondition(
                conditions=[TaskIsCondition(task_id=task_id, state="success") for task_id in child_task_ids]
            )
        return AllCondition(
            conditions=[TaskIsCondition(task_id=task_id, state="terminal") for task_id in child_task_ids]
        )

    @staticmethod
    def _inherit(name: str, parent_job: SpawnedJob | None, defaults: SpawnDefaults | None, fallback):
        if parent_job is not None and hasattr(parent_job, name):
            return getattr(parent_job, name)
        if defaults is not None and hasattr(defaults, name):
            return getattr(defaults, name)
        return fallback
