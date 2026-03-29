from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

from yoitsu_contracts.conditions import (
    AllCondition,
    AnyCondition,
    NotCondition,
    TaskIsCondition,
)
from yoitsu_contracts.config import JobContextConfig, JoinContextConfig
from yoitsu_contracts.events import EvalSpec, SpawnRequestData

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

        parent_task_id = parent_job.task_id if parent_job and parent_job.task_id else parent_job_id
        wait_for, on_fail = self._normalize_strategy(payload.wait_for, payload.on_fail)
        child_defs: list[
            tuple[str, str, str, str, dict, str, str, str | None, str, EvalSpec | None, float]
        ] = []

        for index, child in enumerate(payload.tasks):
            goal = (child.goal or child.prompt).strip()
            if not goal:
                continue

            token = self._id_hash(f"{parent_task_id}:{event.id}:{index}")
            task_id = f"{parent_task_id}/{token}"
            job_id = f"{parent_job_id}-c{token}"
            role = child.role or self._inherit("role", parent_job, parent_defaults, "default")
            
            # Per ADR-0007: role_params contains only role-internal flags (not goal/budget)
            role_params = dict(child.params or {})
            # goal and budget are NOT written to role_params
            
            # repo is a first-class field; fall back to params for backward compat
            repo = child.repo or child.params.get("repo") or self._inherit("repo", parent_job, parent_defaults, "")
            init_branch = child.params.get("branch") or child.params.get("init_branch") or self._inherit("init_branch", parent_job, parent_defaults, "main")
            evo_sha = child.sha or self._inherit("evo_sha", parent_job, parent_defaults, None)
            
            # Per ADR-0007: llm/workspace/publication overrides removed
            # budget goes to single channel (handled by runtime_builder)
            budget = child.budget
            
            team = self._inherit("team", parent_job, parent_defaults, "default")

            child_defs.append(
                (
                    task_id,
                    job_id,
                    goal,
                    role,
                    role_params,
                    repo,
                    init_branch,
                    evo_sha,
                    team,
                    child.eval_spec,
                    budget,
                )
            )

        child_task_ids = [task_id for task_id, *_ in child_defs]
        child_tasks = [
            TaskRecord(
                task_id=task_id,
                goal=goal,
                source_event_id=event.id,
                spec={
                    "role": role,
                    "role_params": role_params,
                    "team": team,
                    "repo": repo,
                    "init_branch": init_branch,
                    "evo_sha": evo_sha,
                },
                team=team,
                eval_spec=eval_spec,
            )
            for task_id, _, goal, role, role_params, repo, init_branch, evo_sha, team, eval_spec, _ in child_defs
        ]

        jobs: list[SpawnedJob] = []
        for task_id, job_id, goal, role, role_params, repo, init_branch, evo_sha, team, _, budget in child_defs:
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
                    task=goal,
                    role=role,
                    role_params=role_params,
                    repo=repo,
                    init_branch=init_branch,
                    evo_sha=evo_sha,
                    budget=budget or 0.0,  # task semantics field
                    task_id=task_id,
                    condition=self._combine_conditions(guard_conditions),
                    parent_job_id=parent_job_id,
                    team=team,
                )
            )

        if parent_job is not None and child_task_ids:
            join_token = self._id_hash(f"{parent_task_id}:{event.id}:join")
            join_task = self._join_task_instruction(parent_job.task)
            
            # Per ADR-0007: join role_params contains only mode="join"
            # parent_goal goes to JobContextConfig.join.parent_summary
            join_role_params = {"mode": "join"}
            
            jobs.append(
                SpawnedJob(
                    job_id=f"{parent_job_id}-j{join_token}",
                    source_event_id=event.id,
                    task=join_task,
                    role=parent_job.role,
                    role_params=join_role_params,
                    repo=parent_job.repo,
                    init_branch=parent_job.init_branch,
                    evo_sha=parent_job.evo_sha,
                    budget=parent_job.budget,  # inherit budget from parent
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
                    team=parent_job.team,
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
    def _join_task_instruction(parent_goal: str) -> str:
        return (
            "Review the completed child tasks for the parent goal below.\n\n"
            "This is a continuation planning step, not a fresh planning pass.\n"
            "Your job is to use join_context to decide whether the parent goal is already satisfied.\n"
            "If the goal is satisfied, stop and summarize the completion.\n"
            "If there is a clear gap, spawn only the minimal follow-up tasks needed to close it.\n"
            "Do not recreate work that child tasks have already completed.\n\n"
            f"Parent goal:\n{parent_goal}"
        )

    @staticmethod
    def _inherit(name: str, parent_job: SpawnedJob | None, defaults: SpawnDefaults | None, fallback):
        if parent_job is not None and hasattr(parent_job, name):
            return getattr(parent_job, name)
        if defaults is not None and hasattr(defaults, name):
            return getattr(defaults, name)
        return fallback

    @staticmethod
    def _id_hash(seed: str, *, length: int = 4) -> str:
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        token = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
        return token[:length]
