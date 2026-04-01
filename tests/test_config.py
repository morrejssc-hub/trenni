"""Tests for Trenni config dataclasses."""

import pytest

from trenni.config import (
    TeamRuntimeConfig,
    TeamSchedulingConfig,
    TeamConfig,
    TrenniConfig,
    _UNSET,
)


class TestTeamRuntimeConfig:
    """Tests for TeamRuntimeConfig dataclass."""

    def test_default_values(self):
        """TeamRuntimeConfig should have sensible defaults."""
        config = TeamRuntimeConfig()
        assert config.image is None
        assert config.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.env_allowlist == []
        assert config.extra_networks == []

    def test_from_dict_with_all_fields(self):
        """TeamRuntimeConfig.from_dict should parse all fields."""
        data = {
            "image": "my-image:latest",
            "pod_name": "my-pod",
            "env_allowlist": ["FOO", "BAR"],
            "extra_networks": ["network1", "network2"],
        }
        config = TeamRuntimeConfig.from_dict(data)
        assert config.image == "my-image:latest"
        assert config.pod_name == "my-pod"
        assert config.env_allowlist == ["FOO", "BAR"]
        assert config.extra_networks == ["network1", "network2"]

    def test_from_dict_with_none(self):
        """TeamRuntimeConfig.from_dict should handle None input."""
        config = TeamRuntimeConfig.from_dict(None)
        assert config.image is None
        assert config.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.env_allowlist == []
        assert config.extra_networks == []

    def test_from_dict_with_empty_dict(self):
        """TeamRuntimeConfig.from_dict should handle empty dict."""
        config = TeamRuntimeConfig.from_dict({})
        assert config.image is None
        assert config.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.env_allowlist == []
        assert config.extra_networks == []

    def test_from_dict_with_partial_fields(self):
        """TeamRuntimeConfig.from_dict should handle partial data."""
        data = {"image": "partial-image"}
        config = TeamRuntimeConfig.from_dict(data)
        assert config.image == "partial-image"
        assert config.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.env_allowlist == []
        assert config.extra_networks == []

    def test_from_dict_with_explicit_null_pod_name(self):
        """TeamRuntimeConfig.from_dict should distinguish explicit null from unset."""
        data = {"pod_name": None}
        config = TeamRuntimeConfig.from_dict(data)
        assert config.pod_name is None  # Explicit null means "no pod"

    def test_from_dict_unset_vs_null_semantics(self):
        """TeamRuntimeConfig should distinguish unset from explicit null for pod_name."""
        # Unset pod_name (missing key)
        config_unset = TeamRuntimeConfig.from_dict({"image": "test"})
        assert config_unset.pod_name is _UNSET

        # Explicit null pod_name
        config_null = TeamRuntimeConfig.from_dict({"image": "test", "pod_name": None})
        assert config_null.pod_name is None

        # Explicit string pod_name
        config_string = TeamRuntimeConfig.from_dict({"image": "test", "pod_name": "my-pod"})
        assert config_string.pod_name == "my-pod"


class TestTeamSchedulingConfig:
    """Tests for TeamSchedulingConfig dataclass."""

    def test_default_values(self):
        """TeamSchedulingConfig should have sensible defaults."""
        config = TeamSchedulingConfig()
        assert config.max_concurrent_jobs == 0

    def test_from_dict_with_value(self):
        """TeamSchedulingConfig.from_dict should parse max_concurrent_jobs."""
        data = {"max_concurrent_jobs": 5}
        config = TeamSchedulingConfig.from_dict(data)
        assert config.max_concurrent_jobs == 5

    def test_from_dict_with_none(self):
        """TeamSchedulingConfig.from_dict should handle None input."""
        config = TeamSchedulingConfig.from_dict(None)
        assert config.max_concurrent_jobs == 0

    def test_from_dict_with_empty_dict(self):
        """TeamSchedulingConfig.from_dict should handle empty dict."""
        config = TeamSchedulingConfig.from_dict({})
        assert config.max_concurrent_jobs == 0

    def test_from_dict_converts_to_int(self):
        """TeamSchedulingConfig.from_dict should convert value to int."""
        data = {"max_concurrent_jobs": "10"}
        config = TeamSchedulingConfig.from_dict(data)
        assert config.max_concurrent_jobs == 10
        assert isinstance(config.max_concurrent_jobs, int)


class TestTeamConfig:
    """Tests for TeamConfig dataclass."""

    def test_default_values(self):
        """TeamConfig should have default runtime and scheduling."""
        config = TeamConfig()
        assert isinstance(config.runtime, TeamRuntimeConfig)
        assert isinstance(config.scheduling, TeamSchedulingConfig)

    def test_from_dict_with_all_sections(self):
        """TeamConfig.from_dict should parse runtime and scheduling."""
        data = {
            "runtime": {
                "image": "team-image:latest",
                "pod_name": "team-pod",
                "env_allowlist": ["TEAM_VAR"],
                "extra_networks": ["team-net"],
            },
            "scheduling": {"max_concurrent_jobs": 3},
        }
        config = TeamConfig.from_dict(data)
        assert config.runtime.image == "team-image:latest"
        assert config.runtime.pod_name == "team-pod"
        assert config.runtime.env_allowlist == ["TEAM_VAR"]
        assert config.runtime.extra_networks == ["team-net"]
        assert config.scheduling.max_concurrent_jobs == 3

    def test_from_dict_with_none(self):
        """TeamConfig.from_dict should handle None input."""
        config = TeamConfig.from_dict(None)
        assert isinstance(config.runtime, TeamRuntimeConfig)
        assert isinstance(config.scheduling, TeamSchedulingConfig)
        assert config.runtime.image is None
        assert config.scheduling.max_concurrent_jobs == 0

    def test_from_dict_with_empty_dict(self):
        """TeamConfig.from_dict should handle empty dict."""
        config = TeamConfig.from_dict({})
        assert isinstance(config.runtime, TeamRuntimeConfig)
        assert isinstance(config.scheduling, TeamSchedulingConfig)

    def test_from_dict_with_partial_runtime(self):
        """TeamConfig.from_dict should handle partial runtime config."""
        data = {"runtime": {"image": "partial-image"}}
        config = TeamConfig.from_dict(data)
        assert config.runtime.image == "partial-image"
        assert config.runtime.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.scheduling.max_concurrent_jobs == 0

    def test_from_dict_with_only_scheduling(self):
        """TeamConfig.from_dict should handle only scheduling config."""
        data = {"scheduling": {"max_concurrent_jobs": 7}}
        config = TeamConfig.from_dict(data)
        assert config.runtime.image is None
        assert config.scheduling.max_concurrent_jobs == 7


class TestTrenniConfigTeams:
    """Tests for teams field in TrenniConfig."""

    def test_default_teams_empty(self):
        """TrenniConfig should have empty teams dict by default."""
        config = TrenniConfig()
        assert config.teams == {}

    def test_from_yaml_with_teams(self, tmp_path):
        """TrenniConfig.from_yaml should parse teams section."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
pasloe_url: http://localhost:9000
teams:
  team-a:
    runtime:
      image: team-a-image:v1
      env_allowlist:
        - VAR_A
    scheduling:
      max_concurrent_jobs: 2
  team-b:
    runtime:
      pod_name: team-b-pod
    scheduling:
      max_concurrent_jobs: 5
""")
        config = TrenniConfig.from_yaml(config_yaml)

        assert "team-a" in config.teams
        assert "team-b" in config.teams
        assert len(config.teams) == 2

        # Verify team-a
        team_a = config.teams["team-a"]
        assert team_a.runtime.image == "team-a-image:v1"
        assert team_a.runtime.env_allowlist == ["VAR_A"]
        assert team_a.scheduling.max_concurrent_jobs == 2

        # Verify team-b
        team_b = config.teams["team-b"]
        assert team_b.runtime.pod_name == "team-b-pod"
        assert team_b.scheduling.max_concurrent_jobs == 5

    def test_from_yaml_with_empty_teams(self, tmp_path):
        """TrenniConfig.from_yaml should handle empty teams dict."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
pasloe_url: http://localhost:9000
teams: {}
""")
        config = TrenniConfig.from_yaml(config_yaml)
        assert config.teams == {}

    def test_from_yaml_without_teams(self, tmp_path):
        """TrenniConfig.from_yaml should work without teams section."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
pasloe_url: http://localhost:9000
""")
        config = TrenniConfig.from_yaml(config_yaml)
        assert config.teams == {}

    def test_from_yaml_team_with_minimal_config(self, tmp_path):
        """TrenniConfig.from_yaml should handle team with minimal config."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
pasloe_url: http://localhost:9000
teams:
  minimal-team: {}
""")
        config = TrenniConfig.from_yaml(config_yaml)
        assert "minimal-team" in config.teams
        team = config.teams["minimal-team"]
        assert isinstance(team, TeamConfig)
        assert team.runtime.image is None
        assert team.scheduling.max_concurrent_jobs == 0