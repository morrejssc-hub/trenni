"""Tests for PodmanBackend pod_name=None and extra_networks handling."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx

from trenni.podman_backend import PodmanBackend
from trenni.runtime_types import JobRuntimeSpec, RuntimeDefaults


@pytest.fixture
def defaults() -> RuntimeDefaults:
    """Create default RuntimeDefaults for testing."""
    return RuntimeDefaults(
        kind="podman",
        socket_uri="unix:///run/podman/podman.sock",
        pod_name="test-pod",
        image="localhost/test:latest",
        pull_policy="missing",
        stop_grace_seconds=30,
        cleanup_timeout_seconds=60,
        retain_on_failure=False,
        labels={},
        env_allowlist=(),
        git_token_env="GIT_TOKEN",
    )


@pytest.fixture
def mock_transport() -> MagicMock:
    """Create a mock transport for httpx."""
    transport = MagicMock(spec=httpx.AsyncBaseTransport)
    transport.handle_async_request = AsyncMock()
    return transport


def create_response(status_code: int, json_data: dict | None = None) -> httpx.Response:
    """Create a mock httpx Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json = MagicMock(return_value=json_data or {})
    response.text = ""
    response.raise_for_status = MagicMock()
    return response


class TestEnsureReadyPodNameNone:
    """Tests for ensure_ready() handling pod_name=None."""

    @pytest.mark.asyncio
    async def test_skips_pod_check_when_pod_name_is_none(
        self, defaults: RuntimeDefaults, mock_transport: MagicMock
    ) -> None:
        """ensure_ready should skip pod existence check when pod_name is None."""
        # Arrange
        backend = PodmanBackend(defaults, transport=mock_transport)
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="test-container",
            image="localhost/test:latest",
            pod_name=None,  # No pod
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
        )

        # Track which endpoints were called
        called_endpoints: list[str] = []

        async def mock_request(method: str, path: str, **kwargs) -> httpx.Response:
            called_endpoints.append(f"{method} {path}")
            if "pods" in path:
                # Should NOT reach here
                raise RuntimeError("Pod endpoint should not be called")
            if "images" in path:
                return create_response(204)  # Image exists
            return create_response(200)

        # Patch the _request method
        backend._request = mock_request  # type: ignore

        # Act & Assert - should not raise
        await backend.ensure_ready(spec)

        # Verify pod endpoint was NOT called
        assert not any("pods" in ep for ep in called_endpoints), \
            f"Pod endpoint should not be called, but got: {called_endpoints}"

    @pytest.mark.asyncio
    async def test_checks_pod_when_pod_name_is_string(
        self, defaults: RuntimeDefaults, mock_transport: MagicMock
    ) -> None:
        """ensure_ready should check pod existence when pod_name is a string."""
        # Arrange
        backend = PodmanBackend(defaults, transport=mock_transport)
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="test-container",
            image="localhost/test:latest",
            pod_name="test-pod",  # Pod specified
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
        )

        called_endpoints: list[str] = []

        async def mock_request(method: str, path: str, **kwargs) -> httpx.Response:
            called_endpoints.append(f"{method} {path}")
            if "pods" in path:
                return create_response(204)  # Pod exists
            if "images" in path:
                return create_response(204)  # Image exists
            return create_response(200)

        backend._request = mock_request  # type: ignore

        # Act
        await backend.ensure_ready(spec)

        # Assert - pod endpoint SHOULD be called
        assert any("pods" in ep for ep in called_endpoints), \
            f"Pod endpoint should be called, got: {called_endpoints}"


class TestEnsureReadyExtraNetworks:
    """Tests for ensure_ready() validating extra_networks."""

    @pytest.mark.asyncio
    async def test_validates_extra_networks_exist(
        self, defaults: RuntimeDefaults, mock_transport: MagicMock
    ) -> None:
        """ensure_ready should validate that each extra network exists."""
        # Arrange
        backend = PodmanBackend(defaults, transport=mock_transport)
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="test-container",
            image="localhost/test:latest",
            pod_name="test-pod",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
            extra_networks=("network-a", "network-b"),
        )

        called_endpoints: list[str] = []

        async def mock_request(method: str, path: str, **kwargs) -> httpx.Response:
            called_endpoints.append(f"{method} {path}")
            if "pods" in path:
                return create_response(204)
            if "images" in path:
                return create_response(204)
            if "networks" in path:
                return create_response(204)  # Network exists
            return create_response(200)

        backend._request = mock_request  # type: ignore

        # Act
        await backend.ensure_ready(spec)

        # Assert - both networks should be checked
        network_calls = [ep for ep in called_endpoints if "networks" in ep]
        assert len(network_calls) == 2, \
            f"Expected 2 network checks, got: {network_calls}"

    @pytest.mark.asyncio
    async def test_raises_when_extra_network_does_not_exist(
        self, defaults: RuntimeDefaults, mock_transport: MagicMock
    ) -> None:
        """ensure_ready should raise RuntimeError when network doesn't exist."""
        # Arrange
        backend = PodmanBackend(defaults, transport=mock_transport)
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="test-container",
            image="localhost/test:latest",
            pod_name="test-pod",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
            extra_networks=("missing-network",),
        )

        async def mock_request(method: str, path: str, **kwargs) -> httpx.Response:
            if "pods" in path:
                return create_response(204)
            if "images" in path:
                return create_response(204)
            if "networks" in path and "missing-network" in path:
                return create_response(404)  # Network doesn't exist
            return create_response(200)

        backend._request = mock_request  # type: ignore

        # Act & Assert
        with pytest.raises(RuntimeError, match="network.*missing-network.*does not exist"):
            await backend.ensure_ready(spec)

    @pytest.mark.asyncio
    async def test_no_network_validation_when_extra_networks_empty(
        self, defaults: RuntimeDefaults, mock_transport: MagicMock
    ) -> None:
        """ensure_ready should skip network validation when extra_networks is empty."""
        # Arrange
        backend = PodmanBackend(defaults, transport=mock_transport)
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="test-container",
            image="localhost/test:latest",
            pod_name="test-pod",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
            extra_networks=(),  # Empty
        )

        called_endpoints: list[str] = []

        async def mock_request(method: str, path: str, **kwargs) -> httpx.Response:
            called_endpoints.append(f"{method} {path}")
            if "pods" in path:
                return create_response(204)
            if "images" in path:
                return create_response(204)
            return create_response(200)

        backend._request = mock_request  # type: ignore

        # Act
        await backend.ensure_ready(spec)

        # Assert - no network calls
        assert not any("networks" in ep for ep in called_endpoints), \
            f"Network endpoints should not be called, got: {called_endpoints}"


class TestPreparePodNameNone:
    """Tests for prepare() handling pod_name=None."""

    @pytest.mark.asyncio
    async def test_omits_pod_field_when_pod_name_is_none(
        self, defaults: RuntimeDefaults, mock_transport: MagicMock
    ) -> None:
        """prepare should not include 'pod' field in payload when pod_name is None."""
        # Arrange
        backend = PodmanBackend(defaults, transport=mock_transport)
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="test-container",
            image="localhost/test:latest",
            pod_name=None,  # No pod
            labels={"app": "test"},
            env={"FOO": "bar"},
            command=("/bin/bash", "-c", "echo hello"),
            config_payload_b64="",
        )

        captured_payload: dict | None = None

        async def mock_request(method: str, path: str, **kwargs) -> httpx.Response:
            nonlocal captured_payload
            if "json" in kwargs:
                captured_payload = kwargs["json"]
            return create_response(200, {"Id": "container-123"})

        backend._request = mock_request  # type: ignore

        # Act
        handle = await backend.prepare(spec)

        # Assert
        assert handle.container_id == "container-123"
        assert captured_payload is not None
        assert "pod" not in captured_payload, \
            f"'pod' should not be in payload when pod_name is None, got: {captured_payload}"

    @pytest.mark.asyncio
    async def test_includes_pod_field_when_pod_name_is_string(
        self, defaults: RuntimeDefaults, mock_transport: MagicMock
    ) -> None:
        """prepare should include 'pod' field in payload when pod_name is a string."""
        # Arrange
        backend = PodmanBackend(defaults, transport=mock_transport)
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="test-container",
            image="localhost/test:latest",
            pod_name="test-pod",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
        )

        captured_payload: dict | None = None

        async def mock_request(method: str, path: str, **kwargs) -> httpx.Response:
            nonlocal captured_payload
            if "json" in kwargs:
                captured_payload = kwargs["json"]
            return create_response(200, {"Id": "container-123"})

        backend._request = mock_request  # type: ignore

        # Act
        await backend.prepare(spec)

        # Assert
        assert captured_payload is not None
        assert captured_payload.get("pod") == "test-pod", \
            f"'pod' should be 'test-pod', got: {captured_payload}"


class TestPrepareExtraNetworks:
    """Tests for prepare() handling extra_networks."""

    @pytest.mark.asyncio
    async def test_includes_networks_when_extra_networks_provided(
        self, defaults: RuntimeDefaults, mock_transport: MagicMock
    ) -> None:
        """prepare should include 'networks' field when extra_networks is non-empty."""
        # Arrange
        backend = PodmanBackend(defaults, transport=mock_transport)
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="test-container",
            image="localhost/test:latest",
            pod_name="test-pod",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
            extra_networks=("network-a", "network-b"),
        )

        captured_payload: dict | None = None

        async def mock_request(method: str, path: str, **kwargs) -> httpx.Response:
            nonlocal captured_payload
            if "json" in kwargs:
                captured_payload = kwargs["json"]
            return create_response(200, {"Id": "container-123"})

        backend._request = mock_request  # type: ignore

        # Act
        await backend.prepare(spec)

        # Assert
        assert captured_payload is not None
        assert captured_payload.get("networks") == ["network-a", "network-b"], \
            f"'networks' should be ['network-a', 'network-b'], got: {captured_payload}"

    @pytest.mark.asyncio
    async def test_omits_networks_when_extra_networks_empty(
        self, defaults: RuntimeDefaults, mock_transport: MagicMock
    ) -> None:
        """prepare should not include 'networks' field when extra_networks is empty."""
        # Arrange
        backend = PodmanBackend(defaults, transport=mock_transport)
        spec = JobRuntimeSpec(
            job_id="job-1",
            source_event_id="evt-1",
            container_name="test-container",
            image="localhost/test:latest",
            pod_name="test-pod",
            labels={},
            env={},
            command=("/bin/bash",),
            config_payload_b64="",
            extra_networks=(),  # Empty
        )

        captured_payload: dict | None = None

        async def mock_request(method: str, path: str, **kwargs) -> httpx.Response:
            nonlocal captured_payload
            if "json" in kwargs:
                captured_payload = kwargs["json"]
            return create_response(200, {"Id": "container-123"})

        backend._request = mock_request  # type: ignore

        # Act
        await backend.prepare(spec)

        # Assert
        assert captured_payload is not None
        assert "networks" not in captured_payload, \
            f"'networks' should not be in payload when empty, got: {captured_payload}"