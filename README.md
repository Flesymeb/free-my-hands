# free-my-hands

Feishu bot for automating model deployment tasks. It watches Feishu group/task
messages, extracts model weight paths from subtasks, selects reusable workers
from a deployed-models document, starts or replaces vLLM services through
tmux/SSH, updates Feishu cards/docs, and notifies follow-up owners or bots.

## Features

- Polling mode, so the bot can run on a private 24h machine without a public URL.
- Reusable worker selection for idle rows and rows whose required tests are done.
- Safety checks for running markers, fresh untested deployments, and same-worker conflicts.
- Bounded parallel deployment across different workers.
- Manual group trigger through @/refresh-style messages.
- Compact Feishu cards, retry/cancel handoff, and post-deploy notification.

## Quick Start

```bash
python -m pip install -e ".[dev]"
cp config.example.toml config.local.toml
```

Fill `config.local.toml` with your Feishu app credentials, chat IDs, document
tokens, SSH host, worker path prefixes, and owner/bot mentions. Do not commit
that file.

Start the long-running tmux services:

```bash
scripts/start_fmh_tmux.sh
tmux attach -t fmh
```

The start script forwards standard proxy variables such as `HTTP_PROXY`,
`HTTPS_PROXY`, and `NO_PROXY` into tmux. Keep those set if Feishu OpenAPI is
slow or unreachable from the server directly.

The main windows are:

- `poll`: reads Feishu messages/tasks and creates deployment reviews.
- `review-auditor`: approves or hands off staged actions, then runs deployment.
- `api`: optional local HTTP API.

## Useful Commands

```bash
fmh --config config.local.toml list-chats
fmh --config config.local.toml poll --once --lookback-sec 600
fmh --config config.local.toml workers
fmh --config config.local.toml reviews
fmh --config config.local.toml send-test --chat-id oc_xxx --card --text "connected"
pytest -q
```

In a Feishu group, send `@bot 检测任务` or `@bot 刷新任务` to force a recent
task scan. If nothing new is found, the bot replies `目前无新任务`.

## Config Notes

- Keep `[runner].mode = "dry-run"` until templates and Feishu permissions are verified.
- Use `[runner].mode = "tmux"` for real deployment.
- Tune `[reusable_workers].max_parallel_deployments` to control concurrent deployments.
- Configure human fallback mentions under `[approval]`; do not hard-code names in code.
- Configure follow-up bot notification under `[post_deploy_notify]`.
- Feishu setup references: [permissions](feishu/permissions.json) and
  [event callbacks](feishu/event_callbacks.json).

## Sharing

Local secrets, runtime state, logs, archives, and caches are ignored by git.
Use the example config as the public template:

```bash
cp config.example.toml config.local.toml
```

For a shareable tarball:

```bash
scripts/make_share_archive.sh
```

See [docs/workflows.md](docs/workflows.md) for the detailed workflow contract.
