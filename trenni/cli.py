"""CLI entry point: start / stop / status."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

import click

from .config import TrenniConfig
from .supervisor import Supervisor


@click.group()
def main():
    """Trenni — minimal Palimpsest job supervisor."""
    pass


@main.command()
@click.option("--config", "-c", "config_path", default=None,
              type=click.Path(exists=True), help="Path to config YAML")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def start(config_path: str | None, verbose: bool):
    """Start the supervisor event loop and control API."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if config_path:
        config = TrenniConfig.from_yaml(config_path)
    else:
        config = TrenniConfig()

    supervisor = Supervisor(config)

    loop = asyncio.new_event_loop()

    def _shutdown(signum, frame):
        click.echo(f"Received signal {signum}, shutting down...")
        loop.create_task(supervisor.stop())

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    async def _run_all():
        import uvicorn
        from .control_api import build_control_app
        control_app = build_control_app(supervisor)
        uv_config = uvicorn.Config(
            control_app,
            host=config.api_host,
            port=config.api_port,
            log_level="warning",
            loop="none",
        )
        uv_server = uvicorn.Server(uv_config)
        server_task = asyncio.create_task(uv_server.serve())
        try:
            await supervisor.start()
        finally:
            uv_server.should_exit = True
            await server_task

    try:
        loop.run_until_complete(_run_all())
    finally:
        loop.close()


@main.command()
@click.option("--config", "-c", "config_path", default=None,
              type=click.Path(exists=True))
def status(config_path: str | None):
    """Show supervisor status (requires running supervisor API)."""
    import httpx

    if config_path:
        config = TrenniConfig.from_yaml(config_path)
    else:
        config = TrenniConfig()

    url = f"http://{config.api_host}:{config.api_port}/status"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        click.echo(f"Running:      {data['running']}")
        click.echo(f"Active jobs:  {data['running_jobs']}/{data['max_workers']}")
        click.echo(f"Fork-joins:   {data['fork_joins_active']}")
    except httpx.ConnectError:
        click.echo("Supervisor is not running (connection refused)")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)


@main.command()
@click.option("--config", "-c", "config_path", default=None,
              type=click.Path(exists=True))
def pause(config_path: str | None):
    """Pause job dispatch (running jobs continue)."""
    import httpx
    config = TrenniConfig.from_yaml(config_path) if config_path else TrenniConfig()
    url = f"http://{config.api_host}:{config.api_port}/control/pause"
    try:
        resp = httpx.post(url, timeout=5.0)
        resp.raise_for_status()
        click.echo("Supervisor paused")
    except httpx.ConnectError:
        click.echo("Supervisor is not running (connection refused)")
        raise SystemExit(1)


@main.command()
@click.option("--config", "-c", "config_path", default=None,
              type=click.Path(exists=True))
def resume(config_path: str | None):
    """Resume job dispatch."""
    import httpx
    config = TrenniConfig.from_yaml(config_path) if config_path else TrenniConfig()
    url = f"http://{config.api_host}:{config.api_port}/control/resume"
    try:
        resp = httpx.post(url, timeout=5.0)
        resp.raise_for_status()
        click.echo("Supervisor resumed")
    except httpx.ConnectError:
        click.echo("Supervisor is not running (connection refused)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
