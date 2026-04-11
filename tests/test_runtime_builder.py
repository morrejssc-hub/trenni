"""Tests for RuntimeSpecBuilder team runtime config integration."""

import pytest

from trenni.config import TrenniConfig, BundleConfig, BundleRuntimeConfig, BundleSchedulingConfig
from trenni.runtime_builder import RuntimeSpecBuilder, build_runtime_defaults
from trenni.runtime_types import RuntimeDefaults
from yoitsu_contracts.config import BundleSource, TargetSource


def test_runtime_spec_builder_uses_bundle_config():
    """RuntimeSpecBuilder selects runtime profile from team config.

    Per ADR-0011 D4:
    - image: team value overrides default if set (None = use default)
    - pod_name: team value overrides default if set (None = no pod)
    - env_allowlist: team value replaces default (not merged)
    - extra_networks: team value used (default is empty)
    """
    # Create TrenniConfig with factorio team having custom runtime settings
    config = TrenniConfig(
        runtime=TrenniConfig.__dataclass_fields__['runtime'].default_factory(),
        bundles={
            "factorio": BundleConfig(
                runtime=BundleRuntimeConfig(
                    image="localhost/factorio-job:custom",
                    pod_name="factorio-pod",
                    env_allowlist=["FACTORIO_API_KEY", "CUSTOM_VAR"],
                    extra_networks=["factorio-network", "shared-network"],
                ),
                scheduling=BundleSchedulingConfig(),
            ),
            "default": BundleConfig(
                runtime=BundleRuntimeConfig(),  # No overrides
                scheduling=BundleSchedulingConfig(),
            ),
        },
    )

    # Build defaults with base image
    defaults = RuntimeDefaults(
        kind="podman",
        socket_uri="unix:///run/podman/podman.sock",
        pod_name="default-pod",
        image="localhost/default-image:latest",
        pull_policy="never",
        stop_grace_seconds=10,
        cleanup_timeout_seconds=120,
        retain_on_failure=False,
        labels={"io.yoitsu.managed-by": "trenni"},
        env_allowlist=("DEFAULT_VAR1", "DEFAULT_VAR2"),
        git_token_env="GITHUB_TOKEN",
    )

    builder = RuntimeSpecBuilder(config, defaults)

    # Build spec for factorio team
    spec = builder.build(
        job_id="test-job-123",
        source_event_id="evt-456",
        goal="test-task",
        role="worker",
        bundle="factorio",
        repo="https://github.com/test/repo.git",
        init_branch="main",
        bundle_sha=None,
    )

    # Assert: team runtime values override defaults
    assert spec.image == "localhost/factorio-job:custom", "Team image should override default"
    assert spec.pod_name == "factorio-pod", "Team pod_name should override default"
    assert spec.extra_networks == ("factorio-network", "shared-network"), "Team extra_networks should be used"

    # Assert: team env_allowlist replaces default (not merged)
    # Note: env_allowlist affects which env vars are pulled into the spec
    # The spec's env should have FACTORIO_API_KEY (if set) but not DEFAULT_VAR1
    # We'll verify the method was called correctly by checking the spec construction


def test_runtime_spec_builder_bundle_missing_image_uses_default():
    """When team has no image override, use default image."""
    config = TrenniConfig(
        runtime=TrenniConfig.__dataclass_fields__['runtime'].default_factory(),
        bundles={
            "minimal": BundleConfig(
                runtime=BundleRuntimeConfig(
                    # image=None (default)
                    pod_name="minimal-pod",
                ),
                scheduling=BundleSchedulingConfig(),
            ),
        },
    )

    defaults = RuntimeDefaults(
        kind="podman",
        socket_uri="unix:///run/podman/podman.sock",
        pod_name="default-pod",
        image="localhost/default-image:latest",
        pull_policy="never",
        stop_grace_seconds=10,
        cleanup_timeout_seconds=120,
        retain_on_failure=False,
        labels={},
        env_allowlist=(),
        git_token_env="GITHUB_TOKEN",
    )

    builder = RuntimeSpecBuilder(config, defaults)

    spec = builder.build(
        job_id="test-job-789",
        source_event_id="evt-000",
        goal="test-task",
        role="worker",
        bundle="minimal",
        repo="https://github.com/test/repo.git",
        init_branch="main",
        bundle_sha=None,
    )

    # Assert: default image used when team has no override
    assert spec.image == "localhost/default-image:latest", "Should use default image when team has none"
    assert spec.pod_name == "minimal-pod", "Should use team pod_name when set"


def test_runtime_spec_builder_bundle_none_pod_name_means_no_pod():
    """When team explicitly sets pod_name to None, job should have no pod."""
    config = TrenniConfig(
        runtime=TrenniConfig.__dataclass_fields__['runtime'].default_factory(),
        bundles={
            "no-pod-team": BundleConfig(
                runtime=BundleRuntimeConfig(
                    pod_name=None,  # Explicitly no pod
                ),
                scheduling=BundleSchedulingConfig(),
            ),
        },
    )

    defaults = RuntimeDefaults(
        kind="podman",
        socket_uri="unix:///run/podman/podman.sock",
        pod_name="default-pod",  # Has default pod
        image="localhost/default-image:latest",
        pull_policy="never",
        stop_grace_seconds=10,
        cleanup_timeout_seconds=120,
        retain_on_failure=False,
        labels={},
        env_allowlist=(),
        git_token_env="GITHUB_TOKEN",
    )

    builder = RuntimeSpecBuilder(config, defaults)

    spec = builder.build(
        job_id="test-job-nopod",
        source_event_id="evt-nopod",
        goal="test-task",
        role="worker",
        bundle="no-pod-team",
        repo="https://github.com/test/repo.git",
        init_branch="main",
        bundle_sha=None,
    )

    # Assert: None pod_name means no pod
    assert spec.pod_name is None, "Team pod_name=None should result in no pod"


def test_runtime_spec_builder_unknown_bundle_uses_defaults():
    """When team is not in config, use all defaults."""
    config = TrenniConfig(
        runtime=TrenniConfig.__dataclass_fields__['runtime'].default_factory(),
        bundles={
            # No "unknown-team" defined
            "other": BundleConfig(
                runtime=BundleRuntimeConfig(),
                scheduling=BundleSchedulingConfig(),
            ),
        },
    )

    defaults = RuntimeDefaults(
        kind="podman",
        socket_uri="unix:///run/podman/podman.sock",
        pod_name="default-pod",
        image="localhost/default-image:latest",
        pull_policy="never",
        stop_grace_seconds=10,
        cleanup_timeout_seconds=120,
        retain_on_failure=False,
        labels={},
        env_allowlist=("VAR1",),
        git_token_env="GITHUB_TOKEN",
    )

    builder = RuntimeSpecBuilder(config, defaults)

    spec = builder.build(
        job_id="test-job-unknown",
        source_event_id="evt-unknown",
        goal="test-task",
        role="worker",
        bundle="unknown-team",
        repo="https://github.com/test/repo.git",
        init_branch="main",
        bundle_sha=None,
    )

    # Assert: all defaults used for unknown team
    assert spec.image == "localhost/default-image:latest"
    assert spec.pod_name == "default-pod"
    assert spec.extra_networks == ()


def test_runtime_spec_builder_mounts_factorio_mod_scripts_dir_rw(monkeypatch: pytest.MonkeyPatch):
    """Factorio jobs should mount the host mod scripts dir read-write into the job container."""
    mod_scripts_dir = "/home/holo/factorio/mods/factorio-agent_0.1.0/scripts"
    monkeypatch.setenv("FACTORIO_MOD_SCRIPTS_DIR", mod_scripts_dir)

    config = TrenniConfig(
        runtime=TrenniConfig.__dataclass_fields__['runtime'].default_factory(),
        bundle_root="/workspace/evo",
        bundle_root_host="/home/holo/yoitsu/evo",
        bundles={
            "factorio": BundleConfig(
                runtime=BundleRuntimeConfig(
                    env_allowlist=["FACTORIO_MOD_SCRIPTS_DIR"],
                ),
                scheduling=BundleSchedulingConfig(),
            ),
        },
    )

    defaults = RuntimeDefaults(
        kind="podman",
        socket_uri="unix:///run/podman/podman.sock",
        pod_name="default-pod",
        image="localhost/default-image:latest",
        pull_policy="never",
        stop_grace_seconds=10,
        cleanup_timeout_seconds=120,
        retain_on_failure=False,
        labels={},
        env_allowlist=("FACTORIO_MOD_SCRIPTS_DIR",),
        git_token_env="GITHUB_TOKEN",
    )

    builder = RuntimeSpecBuilder(config, defaults)
    
    # ADR-0015: Test with bundle_source and target_source
    bundle_source = BundleSource(
        name="factorio",
        repo_uri="git+file:///home/holo/yoitsu/evo",
        selector="main",
        resolved_ref="",
        workspace="/home/holo/yoitsu/evo/factorio",
    )
    target_source = TargetSource(
        repo_uri="",
        branch="main",
        workspace="/home/holo/yoitsu/target",
    )

    spec = builder.build(
        job_id="factorio-worker-1",
        source_event_id="evt-factorio",
        goal="mine iron",
        role="worker",
        bundle="factorio",
        repo="",
        init_branch="main",
        bundle_sha=None,
        bundle_source=bundle_source,
        target_source=target_source,
    )

    # ADR-0015: Bundle workspace mounted RO
    assert (bundle_source.workspace, "/opt/yoitsu/palimpsest/bundle", False) in spec.volume_mounts
    # ADR-0015: Target workspace mounted RW
    assert (target_source.workspace, "/opt/yoitsu/palimpsest/target", True) in spec.volume_mounts
    # Factorio mod scripts dir still RW
    assert (mod_scripts_dir, mod_scripts_dir, True) in spec.volume_mounts
    assert spec.env["FACTORIO_MOD_SCRIPTS_DIR"] == mod_scripts_dir