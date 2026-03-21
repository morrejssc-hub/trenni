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
    """Start the supervisor event loop."""
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

    try:
        loop.run_until_complete(supervisor.start())
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


if __name__ == "__main__":
    main()
