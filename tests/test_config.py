"""Tests for Trenni config dataclasses (Bundle MVP)."""

import pytest

from trenni.config import (
    BundleRuntimeConfig,
    BundleSchedulingConfig,
    BundleConfig,
    TrenniConfig,
    _UNSET,
)


class TestBundleRuntimeConfig:
    """Tests for BundleRuntimeConfig dataclass."""

    def test_default_values(self):
        """BundleRuntimeConfig should have sensible defaults."""
        config = BundleRuntimeConfig()
        assert config.image is None
        assert config.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.env_allowlist == []
        assert config.extra_networks == []

    def test_from_dict_with_all_fields(self):
        """BundleRuntimeConfig.from_dict should parse all fields."""
        data = {
            "image": "my-image:latest",
            "pod_name": "my-pod",
            "env_allowlist": ["FOO", "BAR"],
            "extra_networks": ["network1", "network2"],
        }
        config = BundleRuntimeConfig.from_dict(data)
        assert config.image == "my-image:latest"
        assert config.pod_name == "my-pod"
        assert config.env_allowlist == ["FOO", "BAR"]
        assert config.extra_networks == ["network1", "network2"]

    def test_from_dict_with_none(self):
        """BundleRuntimeConfig.from_dict should handle None input."""
        config = BundleRuntimeConfig.from_dict(None)
        assert config.image is None
        assert config.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.env_allowlist == []
        assert config.extra_networks == []

    def test_from_dict_with_empty_dict(self):
        """BundleRuntimeConfig.from_dict should handle empty dict."""
        config = BundleRuntimeConfig.from_dict({})
        assert config.image is None
        assert config.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.env_allowlist == []
        assert config.extra_networks == []

    def test_from_dict_with_partial_fields(self):
        """BundleRuntimeConfig.from_dict should handle partial data."""
        data = {"image": "partial-image"}
        config = BundleRuntimeConfig.from_dict(data)
        assert config.image == "partial-image"
        assert config.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.env_allowlist == []
        assert config.extra_networks == []

    def test_from_dict_with_explicit_null_pod_name(self):
        """BundleRuntimeConfig.from_dict should distinguish explicit null from unset."""
        data = {"pod_name": None}
        config = BundleRuntimeConfig.from_dict(data)
        assert config.pod_name is None  # Explicit null means "no pod"

    def test_from_dict_unset_vs_null_semantics(self):
        """BundleRuntimeConfig should distinguish unset from explicit null for pod_name."""
        # Unset pod_name (missing key)
        config_unset = BundleRuntimeConfig.from_dict({"image": "test"})
        assert config_unset.pod_name is _UNSET

        # Explicit null pod_name
        config_null = BundleRuntimeConfig.from_dict({"image": "test", "pod_name": None})
        assert config_null.pod_name is None

        # Explicit string pod_name
        config_string = BundleRuntimeConfig.from_dict({"image": "test", "pod_name": "my-pod"})
        assert config_string.pod_name == "my-pod"


class TestBundleSchedulingConfig:
    """Tests for BundleSchedulingConfig dataclass."""

    def test_default_values(self):
        """BundleSchedulingConfig should have sensible defaults."""
        config = BundleSchedulingConfig()
        assert config.max_concurrent_jobs == 0

    def test_from_dict_with_value(self):
        """BundleSchedulingConfig.from_dict should parse max_concurrent_jobs."""
        data = {"max_concurrent_jobs": 5}
        config = BundleSchedulingConfig.from_dict(data)
        assert config.max_concurrent_jobs == 5

    def test_from_dict_with_none(self):
        """BundleSchedulingConfig.from_dict should handle None input."""
        config = BundleSchedulingConfig.from_dict(None)
        assert config.max_concurrent_jobs == 0

    def test_from_dict_with_empty_dict(self):
        """BundleSchedulingConfig.from_dict should handle empty dict."""
        config = BundleSchedulingConfig.from_dict({})
        assert config.max_concurrent_jobs == 0

    def test_from_dict_converts_to_int(self):
        """BundleSchedulingConfig.from_dict should convert value to int."""
        data = {"max_concurrent_jobs": "10"}
        config = BundleSchedulingConfig.from_dict(data)
        assert config.max_concurrent_jobs == 10
        assert isinstance(config.max_concurrent_jobs, int)


class TestBundleConfig:
    """Tests for BundleConfig dataclass."""

    def test_default_values(self):
        """BundleConfig should have default runtime and scheduling."""
        config = BundleConfig()
        assert isinstance(config.runtime, BundleRuntimeConfig)
        assert isinstance(config.scheduling, BundleSchedulingConfig)
        assert config.default_role == ""

    def test_from_dict_with_all_sections(self):
        """BundleConfig.from_dict should parse runtime, scheduling, and default_role."""
        data = {
            "runtime": {
                "image": "bundle-image:latest",
                "pod_name": "bundle-pod",
                "env_allowlist": ["BUNDLE_VAR"],
                "extra_networks": ["bundle-net"],
            },
            "scheduling": {"max_concurrent_jobs": 3},
            "default_role": "planner",
        }
        config = BundleConfig.from_dict(data)
        assert config.runtime.image == "bundle-image:latest"
        assert config.runtime.pod_name == "bundle-pod"
        assert config.runtime.env_allowlist == ["BUNDLE_VAR"]
        assert config.runtime.extra_networks == ["bundle-net"]
        assert config.scheduling.max_concurrent_jobs == 3
        assert config.default_role == "planner"

    def test_from_dict_with_none(self):
        """BundleConfig.from_dict should handle None input."""
        config = BundleConfig.from_dict(None)
        assert isinstance(config.runtime, BundleRuntimeConfig)
        assert isinstance(config.scheduling, BundleSchedulingConfig)
        assert config.runtime.image is None
        assert config.scheduling.max_concurrent_jobs == 0

    def test_from_dict_with_empty_dict(self):
        """BundleConfig.from_dict should handle empty dict."""
        config = BundleConfig.from_dict({})
        assert isinstance(config.runtime, BundleRuntimeConfig)
        assert isinstance(config.scheduling, BundleSchedulingConfig)

    def test_from_dict_with_partial_runtime(self):
        """BundleConfig.from_dict should handle partial runtime config."""
        data = {"runtime": {"image": "partial-image"}}
        config = BundleConfig.from_dict(data)
        assert config.runtime.image == "partial-image"
        assert config.runtime.pod_name is _UNSET  # Sentinel value for "not set"
        assert config.scheduling.max_concurrent_jobs == 0

    def test_from_dict_with_only_scheduling(self):
        """BundleConfig.from_dict should handle only scheduling config."""
        data = {"scheduling": {"max_concurrent_jobs": 7}}
        config = BundleConfig.from_dict(data)
        assert config.runtime.image is None
        assert config.scheduling.max_concurrent_jobs == 7


class TestTrenniConfigBundles:
    """Tests for bundles field in TrenniConfig."""

    def test_default_bundles_empty(self):
        """TrenniConfig should have empty bundles dict by default."""
        config = TrenniConfig()
        assert config.bundles == {}

    def test_from_yaml_with_bundles(self, tmp_path):
        """TrenniConfig.from_yaml should parse bundles section."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
pasloe_url: http://localhost:9000
bundles:
  factorio:
    runtime:
      image: factorio-image:v1
      env_allowlist:
        - VAR_A
    scheduling:
      max_concurrent_jobs: 2
  backend:
    runtime:
      pod_name: backend-pod
    scheduling:
      max_concurrent_jobs: 5
""")
        config = TrenniConfig.from_yaml(config_yaml)

        assert "factorio" in config.bundles
        assert "backend" in config.bundles
        assert len(config.bundles) == 2

        # Verify factorio
        factorio = config.bundles["factorio"]
        assert factorio.runtime.image == "factorio-image:v1"
        assert factorio.runtime.env_allowlist == ["VAR_A"]
        assert factorio.scheduling.max_concurrent_jobs == 2

        # Verify backend
        backend = config.bundles["backend"]
        assert backend.runtime.pod_name == "backend-pod"
        assert backend.scheduling.max_concurrent_jobs == 5

    def test_from_yaml_with_empty_bundles(self, tmp_path):
        """TrenniConfig.from_yaml should handle empty bundles dict."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
pasloe_url: http://localhost:9000
bundles: {}
""")
        config = TrenniConfig.from_yaml(config_yaml)
        assert config.bundles == {}

    def test_from_yaml_without_bundles(self, tmp_path):
        """TrenniConfig.from_yaml should work without bundles section."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
pasloe_url: http://localhost:9000
workspace_root: /var/tmp/yoitsu-workspaces
""")
        config = TrenniConfig.from_yaml(config_yaml)
        assert config.bundles == {}
        assert config.workspace_root == "/var/tmp/yoitsu-workspaces"

    def test_from_yaml_bundle_with_minimal_config(self, tmp_path):
        """TrenniConfig.from_yaml should handle bundle with minimal config."""
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("""
pasloe_url: http://localhost:9000
bundles:
  minimal-bundle: {}
""")
        config = TrenniConfig.from_yaml(config_yaml)
        assert "minimal-bundle" in config.bundles
        bundle = config.bundles["minimal-bundle"]
        assert isinstance(bundle, BundleConfig)
        assert bundle.runtime.image is None
        assert bundle.scheduling.max_concurrent_jobs == 0
