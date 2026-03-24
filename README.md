# Trenni

Trenni is the scheduler and isolation control plane for the [Yoitsu](https://github.com/guan-spicy-wolf/yoitsu) stack.

It owns:

- task progression
- condition evaluation for spawned jobs
- queue drain and capacity control
- replay from Pasloe
- periodic checkpoint and reap
- isolation backend lifecycle

It does not execute agent logic itself.

## Internal Modules

- `state.py`: typed mutable supervisor state
- `scheduler.py`: queue admission and task-state driven condition evaluation
- `spawn_handler.py`: expand one spawn request into child jobs plus a join job
- `replay.py`: rebuild scheduler state from Pasloe plus runtime inspection
- `checkpoint.py`: container reap and checkpoint payload generation
- `isolation.py`: backend protocol
- `podman_backend.py`: current isolation implementation

## Control API

Trenni exposes an HTTP control plane, by default on `http://127.0.0.1:8100`.

| Endpoint | Method | Description |
|---|---|---|
| `/control/status` | `GET` | scheduler status, queue sizes, task-state snapshot |
| `/control/pause` | `POST` | stop dispatching new jobs |
| `/control/resume` | `POST` | resume dispatch |
| `/control/stop` | `POST` | graceful shutdown |
| `/hooks/events` | `POST` | webhook intake from Pasloe |

## Webhooks And Polling

Trenni registers a webhook with Pasloe on startup when possible. Webhooks are the fast path. Polling remains as the fallback path and as a resilience sweep even when webhooks are enabled.

Webhook deliveries can be HMAC-signed with `webhook_secret`.

## Configuration

```yaml
pasloe_url: "http://127.0.0.1:8000"
pasloe_api_key_env: "PASLOE_API_KEY"
source_id: "trenni-supervisor"

runtime:
  kind: "podman"
  podman:
    socket_uri: "unix:///run/podman/podman.sock"
    pod_name: "yoitsu-dev"
    image: "localhost/yoitsu-palimpsest-job:dev"
    pull_policy: "never"
    git_token_env: "GITHUB_TOKEN"
    env_allowlist:
      - "OPENAI_API_KEY"

max_workers: 4
poll_interval: 2.0
api_host: "127.0.0.1"
api_port: 8100
webhook_secret: ""
trenni_public_url: ""
webhook_poll_interval: 30.0
```

## Running

```bash
uv run trenni start --config trenni-config.yaml
```
