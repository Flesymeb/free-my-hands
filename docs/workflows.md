# Workflow Design

## Workflow 1: Auto Deploy vLLM From Feishu Request

### Goal

When a user updates a Feishu document or sends a group message containing a model weight path, the bot automatically provisions GPU resources, starts a vLLM service, writes the result back to the document, and privately notifies the requester.

### Trigger Sources

- Feishu document update event.
- Feishu group message event.

Both trigger sources should normalize into the same internal request object.

```yaml
request_id: stable id from source + message/doc block id
source_type: doc | group_message
source_ref: Feishu document token, message id, or block id
requester:
  user_id: Feishu user id
  display_name: optional
weight_path: /path/to/checkpoint
model_name: optional logical model name
gpu:
  count: optional
  type: optional
runtime:
  image: optional
  env_name: optional
  extra_args: optional vLLM args
status: pending | launching | configuring | serving | failed | cancelled
```

### Required Parser Rules

The parser should be strict enough to avoid accidental GPU launches.

- Required: an absolute or known-storage weight path.
- Optional: GPU count, GPU type, port, model alias, vLLM args.
- Reject ambiguous messages and reply with the missing fields instead of launching.
- Store the normalized request before any side effect.

Recommended initial message format:

```text
deploy_vllm
weight_path: /mnt/path/to/checkpoint
model_name: qwen-test
gpu_count: 1
gpu_type: A100
extra_args: --max-model-len 32768
```

### State Machine

```text
pending
  -> launching_resource
  -> resource_ready
  -> configuring_env
  -> starting_vllm
  -> health_checking
  -> serving
```

Failure states:

```text
failed_parse
failed_resource
failed_env
failed_vllm
failed_health_check
cancelled
```

Each transition should append an event log entry:

```yaml
timestamp: ISO-8601
request_id: string
state_from: string
state_to: string
summary: short human-readable message
raw_output_ref: optional path to command log
```

### Execution Contract

The executor should run commands through a controlled job runner, not directly inside the Feishu webhook handler.

Recommended execution layers:

- `feishu_adapter`: receives events, sends doc updates and private messages.
- `request_parser`: extracts deployment requests from documents and messages.
- `state_store`: persists requests, transitions, job metadata, and logs.
- `orchestrator`: owns the state machine and idempotency checks.
- `runner`: starts tmux sessions and executes shell commands.
- `resource_provider`: wraps `rlaunch` and GPU allocation details.
- `vllm_deployer`: builds the final vLLM launch command and health checks the endpoint.

### Idempotency Rules

- Same `request_id` must not launch twice.
- If a tmux session already exists for the request, reuse or inspect it before starting a new one.
- If a vLLM endpoint already passes health check, mark the request as `serving`.
- Retried Feishu events should only resume from the last persisted state.

### tmux Convention

Session name:

```text
fmh-{request_id_short}
```

Windows:

- `rlaunch`: resource allocation and job bootstrap.
- `env`: environment setup.
- `vllm`: vLLM server process.
- `logs`: tail of structured logs.

### rlaunch Contract

The first implementation should keep `rlaunch` configurable because cluster syntax often changes by environment.

```yaml
rlaunch:
  command_template: "rlaunch --gpu {gpu_count} --type {gpu_type} -- ..."
  workdir: "/path/to/workdir"
  setup_commands:
    - "source ~/.bashrc"
    - "conda activate vllm"
```

### vLLM Command Contract

```text
python -m vllm.entrypoints.openai.api_server \
  --model {weight_path} \
  --served-model-name {model_name} \
  --host 0.0.0.0 \
  --port {port} \
  {extra_args}
```

### Feishu Updates

Document update should include:

- Current status.
- tmux session name.
- GPU allocation summary.
- vLLM endpoint.
- Health check result.
- Last error if failed.

Private message should include:

- Request summary.
- Final endpoint if serving.
- Failure reason and log pointer if failed.

### Minimum Viable Implementation Order

1. Define config schema for Feishu, rlaunch, tmux, vLLM, and storage.
2. Implement parser and normalized request model.
3. Implement local dry-run runner that prints commands without allocating GPU.
4. Add persistent state store, initially SQLite or JSONL.
5. Implement tmux runner.
6. Implement `rlaunch` provider.
7. Implement vLLM deployment and health check.
8. Wire Feishu webhook receive, document update, and private message send.

### Open Decisions

- Whether the production service should be Python FastAPI, Node.js, or another stack.
- Exact Feishu document format to parse.
- Exact `rlaunch` syntax and cluster environment setup commands.
- Where to persist state: SQLite, Postgres, Redis, or filesystem JSONL.
- How to expose vLLM endpoint to users: raw host/port, reverse proxy, or gateway.

## Reusable Worker Workflow

The current production direction is to reuse long-lived workers from the Feishu "已部署模型" table instead of allocating a new worker for every request.

### Deterministic Guardrails

- Read the deployed-models document through `docs/v1/content` Markdown export.
- Parse the first HTML table.
- A row is reusable only when the "已经测试完的任务" column contains both `tau2` and `vita`, and does not contain `(running)`.
- Normalize task model paths:
  - source path: `/mnt/shared-models/...`
  - worker path: `/mnt/worker-models/...`
  - table path: suffix after `/mnt/worker-models/`
- Derive `model_id` from the final path segment.
- Infer tmux session from worker IP and GPU count, for example `192.0.2.156` with 4 GPUs becomes `ssh_4_gpu_2_156`.
- Reconnect SSH commands must include `-o ServerAliveInterval=60 -o ServerAliveCountMax=3`.

### Human/Agent Boundary

The bot may generate a plan automatically, but stopping a running vLLM process and rewriting the table should remain gated until the plan is validated. A Codex/operator layer can inspect tmux history, recent pane output, health checks, and document state, then decide whether to proceed.

### Review Packets

Every risky stage should create an operator review packet:

- task parsed
- reusable row selected
- before reconnect
- before stopping vLLM
- before starting vLLM
- after starting vLLM
- before document write
- new worker required
- any exception or failed check

The packet contains deterministic context, proposed commands, document fields, checks, risks, next actions, and a Codex prompt. The reviewer must return one of:

- `APPROVE`
- `BLOCK`
- `RETRY`
- `REQUEST_INFO`

No destructive action should run unless the current stage has an approved packet or an explicit operator override.
