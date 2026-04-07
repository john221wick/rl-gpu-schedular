---
title: GPU Scheduler for ML Workloads
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - scheduling
  - gpu
---

# GPU Scheduler for ML Workloads

This repo is a reinforcement learning environment for GPU scheduling.

Current scope:

- single-node scheduling
- mixed training and inference workloads
- OpenEnv-compatible environment
- CPU-only simulation of GPU behaviour

Main goal:

- keep GPUs busy
- reduce fragmentation
- prioritise urgent inference
- avoid bad placement and unnecessary model churn

## What Problem It Solves

- poor GPU utilization
- urgent inference blocked by training jobs
- low-priority training using memory needed by important requests
- memory fragmentation
- bad placement across multiple GPUs
- unnecessary model load and unload cycles
- weak checkpoint and preemption decisions

Typical trade-offs:

- assign now or wait for a better GPU
- keep training running or preempt it
- reuse a warm model or free memory
- quantize a model to create space or avoid quality loss

## Current Scope

- one machine with multiple GPUs
- training and inference together
- topology-aware placement
- warm-model reuse
- checkpoint preemption
- quantization decisions

This is not a full multi-node scheduler yet.

## Future Scope

- scheduling across many machines
- distributed training placement
- node-level routing
- network cost between nodes
- rack or zone awareness
- failover and rebalancing

## What The Agent Controls

At each step, the agent sees the system state and picks one action.

Actions:

- `assign_training`
- `assign_inference`
- `queue`
- `checkpoint_preempt`
- `preempt_kill`
- `wait`
- `unload_model`
- `quantize_model`

Observation includes:

- current step and max steps
- training queue and inference queue
- per-GPU memory, fragmentation, and loaded models
- pending arrivals
- aggregate metrics

## Tasks

- `easy_001`: training-heavy scheduling
- `medium_001`: mixed training and inference
- `hard_001`: bursty inference and harder placement

Each episode returns a score in `[0.0, 1.0]`.

## How It Works

- generate jobs and requests
- expose state and valid actions
- choose one action per step
- return a normalized final score

## Repository layout

```text
gpu_scheduler/tasks.py       task definitions and workload generation
gpu_scheduler/simulator.py   cluster state transitions
gpu_scheduler/env.py         environment reset/step/state logic
gpu_scheduler/rewards.py     step reward shaping
gpu_scheduler/graders.py     end-of-episode score calculation
gpu_scheduler/baseline.py    heuristic policies used for fallback and testing
server/app.py                FastAPI wrapper and CLI entrypoint
inference.py                 submission runner with required stdout logging
openenv.yaml                 OpenEnv metadata
```

## Local setup

Use Python `3.10` to `3.12`. The checked-in version file uses `3.12`.

```bash
uv sync
make test
make api
```

For Hugging Face CLI, use:

```bash
uv run hf auth login
uv run hf auth whoami
```

Compatibility alias:

```bash
uv run huggingface-cli login
```

Run the API:

```bash
uv run server --host 0.0.0.0 --port 7860
```

Quick API check:

```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{}'
curl http://127.0.0.1:7860/state
curl http://127.0.0.1:7860/valid-actions
```

## Submission runner

`inference.py` is the root-level submission script.

- uses the injected `API_BASE_URL` and `API_KEY`
- requires `API_BASE_URL` and `API_KEY` in the environment
- makes a small bounded proxy warm-up call, then limits model usage to a few fast decisions
- runs all three tasks
- prints `[START]`, `[STEP]`, and `[END]`
- falls back to the built-in heuristic if the LLM call fails

Required:

- `API_BASE_URL`
- `API_KEY`

Default is set for:

- `MODEL_NAME`

Optional:

- `LOCAL_IMAGE_NAME`
- `LLM_REQUEST_TIMEOUT_SECONDS`
- `MAX_LLM_RETRIES`
- `MAX_LLM_CALLS`
- `LLM_TOTAL_BUDGET_SECONDS`
- `PROXY_PING_MAX_TOKENS`

Example:

```bash
cp .env.example .env
uv run python inference.py
```

Pre-submission checks:

```bash
make test
docker build -t gpu-scheduler-ml .
openenv validate
```

## Hugging Face Space deployment

This repo is configured as a Docker Space on port `7860`.

Set these Space variables or secrets:

- `API_KEY`
- `MODEL_NAME`
- `API_BASE_URL`

## API

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `GET /state`
- `POST /state`
- `GET /valid-actions`
- `POST /step`

## Notes for judges and reviewers

- `POST /reset` accepts `{}` and resets to `easy_001` by default.
- scoring stays in the `[0.0, 1.0]` range
- Docker uses Python `3.12` and installs through `uv`
