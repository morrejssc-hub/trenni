"""Tests for pod_name sentinel semantics in TeamRuntimeConfig.

Issue: pod_name "unset" vs "explicit null" semantics

- `TeamRuntimeConfig.from_dict()` uses `payload.get("pod_name")` which returns `None`
  for both missing key and explicit `null`
- `RuntimeSpecBuilder` interprets `None` as "no pod"
- A team that only wants to override `image` will accidentally脱离 default pod

Fix: Use a sentinel value to distinguish "unset" from "explicit null".
"""

import pytest

from trenni.config import PodmanRuntimeConfig, RuntimeConfig, TeamRuntimeConfig, TeamConfig, TrenniConfig, _UNSET
from trenni.runtime_builder import RuntimeSpecBuilder, build_runtime_defaults


class TestTeamRuntimeConfigSentinel:
    """Tests for TeamRuntimeConfig sentinel handling."""

    def test_pod_name_unset_when_missing(self):
        """When pod_name key is missing, it should be _UNSET (not None)."""
        config = TeamRuntimeConfig.from_dict({"image": "my-image"})
        # pod_name should be _UNSET sentinel, not None
        assert config.pod_name is _UNSET or config.pod_name == _UNSET

    def test_pod_name_explicit_null(self):
        """When pod_name is explicitly null, it should be None."""
        config = TeamRuntimeConfig.from_dict({"pod_name": None})
        assert config.pod_name is None

    def test_pod_name_explicit_string(self):
        """When pod_name is explicitly set to a string, use that value."""
        config = TeamRuntimeConfig.from_dict({"pod_name": "custom-pod"})
        assert config.pod_name == "custom-pod"

    def test_pod_name_unset_from_empty_dict(self):
        """Empty dict should leave pod_name unset."""
        config = TeamRuntimeConfig.from_dict({})
        assert config.pod_name is _UNSET or config.pod_name == _UNSET

    def test_pod_name_unset_from_none(self):
        """None input should leave pod_name unset."""
        config = TeamRuntimeConfig.from_dict(None)
        assert config.pod_name is _UNSET or config.pod_name == _UNSET


class TestRuntimeSpecBuilderPodNameInheritance:
    """Tests for RuntimeSpecBuilder pod_name inheritance behavior."""

    @pytest.fixture
    def base_config(self):
        """Create a base TrenniConfig with default runtime settings."""
        return TrenniConfig(
            runtime=RuntimeConfig(
                kind="podman",
                podman=PodmanRuntimeConfig(
                    pod_name="default-pod",
                    image="default-image",
                ),
            ),
            teams={},
        )

    def test_unset_pod_name_inherits_default(self, base_config):
        """When pod_name is unset, should inherit from defaults."""
        # Add team with only image override (no pod_name specified)
        base_config.teams["my-team"] = TeamConfig.from_dict({
            "runtime": {"image": "team-image"}
        })

        defaults = build_runtime_defaults(base_config)
        builder = RuntimeSpecBuilder(base_config, defaults)

        spec = builder.build(
            job_id="test-job",
            source_event_id="event-123",
            task="test-task",
            role="worker",
            repo="test/repo",
            init_branch="main",
            evo_sha=None,
            team="my-team",
        )

        # Should inherit default pod_name since it was unset
        assert spec.pod_name == "default-pod"

    def test_explicit_null_pod_name_means_no_pod(self, base_config):
        """When pod_name is explicitly null, should have no pod (None)."""
        # Add team with explicit null pod_name
        base_config.teams["my-team"] = TeamConfig.from_dict({
            "runtime": {"pod_name": None, "image": "team-image"}
        })

        defaults = build_runtime_defaults(base_config)
        builder = RuntimeSpecBuilder(base_config, defaults)

        spec = builder.build(
            job_id="test-job",
            source_event_id="event-123",
            task="test-task",
            role="worker",
            repo="test/repo",
            init_branch="main",
            evo_sha=None,
            team="my-team",
        )

        # Explicit null means no pod
        assert spec.pod_name is None

    def test_explicit_pod_name_overrides_default(self, base_config):
        """When pod_name is explicitly set, should override default."""
        # Add team with explicit pod_name
        base_config.teams["my-team"] = TeamConfig.from_dict({
            "runtime": {"pod_name": "custom-pod", "image": "team-image"}
        })

        defaults = build_runtime_defaults(base_config)
        builder = RuntimeSpecBuilder(base_config, defaults)

        spec = builder.build(
            job_id="test-job",
            source_event_id="event-123",
            task="test-task",
            role="worker",
            repo="test/repo",
            init_branch="main",
            evo_sha=None,
            team="my-team",
        )

        # Should use team's explicit pod_name
        assert spec.pod_name == "custom-pod"

    def test_unknown_team_uses_defaults(self, base_config):
        """Unknown team should use all defaults."""
        defaults = build_runtime_defaults(base_config)
        builder = RuntimeSpecBuilder(base_config, defaults)

        spec = builder.build(
            job_id="test-job",
            source_event_id="event-123",
            task="test-task",
            role="worker",
            repo="test/repo",
            init_branch="main",
            evo_sha=None,
            team="unknown-team",
        )

        # Unknown team uses defaults
        assert spec.pod_name == "default-pod"

    def test_image_override_preserves_pod_name(self, base_config):
        """Overriding only image should preserve default pod_name.

        This is the key bug fix scenario.
        """
        # Team wants custom image but default pod
        base_config.teams["my-team"] = TeamConfig.from_dict({
            "runtime": {"image": "custom-image"}
        })

        defaults = build_runtime_defaults(base_config)
        builder = RuntimeSpecBuilder(base_config, defaults)

        spec = builder.build(
            job_id="test-job",
            source_event_id="event-123",
            task="test-task",
            role="worker",
            repo="test/repo",
            init_branch="main",
            evo_sha=None,
            team="my-team",
        )

        # Image should be overridden
        assert spec.image == "custom-image"
        # Pod name should be inherited from defaults (not accidentally None!)
        assert spec.pod_name == "default-pod"