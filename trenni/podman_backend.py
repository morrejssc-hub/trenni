from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import httpx

from .runtime_types import ContainerExit, ContainerState, JobHandle, JobRuntimeSpec, RuntimeDefaults


class PodmanBackend:
    def __init__(
        self,
        defaults: RuntimeDefaults,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str = "http://d/v1.0.0",
    ) -> None:
        self.defaults = defaults
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ensure_ready(self, spec: JobRuntimeSpec) -> None:
        # Check pod only if pod_name is not None
        if spec.pod_name is not None:
            await self._ensure_pod_exists(spec.pod_name)

        await self._ensure_image_available(spec.image, self.defaults.pull_policy)

        # Validate extra networks exist
        for network in spec.extra_networks:
            await self._ensure_network_exists(network)

    async def prepare(self, spec: JobRuntimeSpec) -> JobHandle:
        payload: dict[str, Any] = {
            "name": spec.container_name,
            "image": spec.image,
            "env": dict(spec.env),
            "labels": dict(spec.labels),
            "command": list(spec.command),
        }

        # Only include pod if pod_name is not None
        if spec.pod_name is not None:
            payload["pod"] = spec.pod_name

        # Attach extra networks
        if spec.extra_networks:
            payload["networks"] = list(spec.extra_networks)

        response = await self._request("POST", "/libpod/containers/create", json=payload)
        data = response.json()
        return JobHandle(
            job_id=spec.job_id,
            container_id=data["Id"],
            container_name=spec.container_name,
        )

    async def create(self, spec: JobRuntimeSpec) -> JobHandle:
        return await self.prepare(spec)

    async def start(self, handle: JobHandle) -> None:
        response = await self._request(
            "POST",
            f"/libpod/containers/{quote(handle.container_id, safe='')}/start",
            expected={204, 304},
        )
        if response.status_code not in {204, 304}:
            response.raise_for_status()

    async def inspect(self, handle: JobHandle) -> ContainerState:
        response = await self._request(
            "GET",
            f"/libpod/containers/{quote(self._container_ref(handle), safe='')}/json",
            expected={200, 404},
        )
        if response.status_code == 404:
            return ContainerState(exists=False)

        data = response.json()
        state = data.get("State", {})
        return ContainerState(
            exists=True,
            status=state.get("Status", ""),
            running=bool(state.get("Running")),
            exit_code=state.get("ExitCode"),
        )

    async def wait(self, handle: JobHandle) -> ContainerExit:
        response = await self._request(
            "POST",
            f"/libpod/containers/{quote(self._container_ref(handle), safe='')}/wait",
        )
        data = response.json()
        return ContainerExit(status_code=data.get("StatusCode"))

    async def logs(self, handle: JobHandle) -> str:
        response = await self._request(
            "GET",
            f"/libpod/containers/{quote(self._container_ref(handle), safe='')}/logs",
            params={
                "stdout": "true",
                "stderr": "true",
                "follow": "false",
                "timestamps": "false",
            },
            expected={200, 404},
        )
        if response.status_code == 404:
            return ""
        return response.text

    async def stop(self, handle: JobHandle, timeout_s: int) -> None:
        response = await self._request(
            "POST",
            f"/libpod/containers/{quote(self._container_ref(handle), safe='')}/stop",
            params={"t": str(timeout_s)},
            expected={204, 304, 404},
        )
        if response.status_code not in {204, 304, 404}:
            response.raise_for_status()

    async def remove(self, handle: JobHandle, *, force: bool = False) -> None:
        response = await self._request(
            "DELETE",
            f"/libpod/containers/{quote(self._container_ref(handle), safe='')}",
            params={"force": json.dumps(force)},
            expected={204, 404},
        )
        if response.status_code not in {204, 404}:
            response.raise_for_status()

    async def _ensure_pod_exists(self, pod_name: str) -> None:
        response = await self._request(
            "GET",
            f"/libpod/pods/{quote(pod_name, safe='')}/exists",
            expected={204, 404},
        )
        if response.status_code == 404:
            raise RuntimeError(f"Podman pod {pod_name!r} does not exist")

    async def _ensure_network_exists(self, network_name: str) -> None:
        """Validate that a network exists in Podman."""
        response = await self._request(
            "GET",
            f"/libpod/networks/{quote(network_name, safe='')}/exists",
            expected={204, 404},
        )
        if response.status_code == 404:
            raise RuntimeError(f"Podman network {network_name!r} does not exist")

    async def _ensure_image_available(self, image: str, pull_policy: str) -> None:
        image_exists = await self._image_exists(image)
        if pull_policy == "never":
            if not image_exists:
                raise RuntimeError(f"Podman image {image!r} is not available locally")
            return

        if pull_policy == "missing" and image_exists:
            return

        if pull_policy in {"always", "newer"} or not image_exists:
            response = await self._request(
                "POST",
                "/libpod/images/pull",
                params={"reference": image, "quiet": "true"},
            )
            response.raise_for_status()

    async def _image_exists(self, image: str) -> bool:
        response = await self._request(
            "GET",
            f"/libpod/images/{quote(image, safe='')}/exists",
            expected={204, 404},
        )
        return response.status_code == 204

    def _container_ref(self, handle: JobHandle) -> str:
        return handle.container_id or handle.container_name

    async def _request(
        self,
        method: str,
        path: str,
        *,
        expected: set[int] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        client = await self._get_client()
        response = await client.request(method, path, **kwargs)
        if expected is not None and response.status_code not in expected:
            response.raise_for_status()
        elif expected is None:
            response.raise_for_status()
        return response

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            transport = self._transport
            if transport is None:
                transport = httpx.AsyncHTTPTransport(uds=self._socket_path(self.defaults.socket_uri))
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                transport=transport,
                timeout=30.0,
            )
        return self._client

    @staticmethod
    def _socket_path(socket_uri: str) -> str:
        prefix = "unix://"
        if not socket_uri.startswith(prefix):
            raise ValueError(f"Unsupported Podman socket URI {socket_uri!r}")
        return socket_uri[len(prefix):]
