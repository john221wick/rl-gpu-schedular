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

This project is about a practical scheduling problem in ML systems(single node for now).

In real teams, one machine or one server may have multiple GPUs, and many workloads want to use them at the same time. Some jobs are long training jobs. Some jobs are short inference requests. If scheduling is poor, GPUs stay idle, memory gets fragmented, urgent requests get delayed, and important jobs may fail even when total GPU memory looks enough on paper.

This project builds a simulated OpenEnv-compatible GPU scheduler for that problem. The environment tries to decide which job should go to which GPU, when to wait, when to queue, when to reuse a warm model, and when to preempt or quantize to handle pressure better.

One important real-world issue is priority conflict. A training job may be low priority and can wait or be checkpointed, but an inference burst may be very urgent because it is user-facing and has SLA pressure. In that case, the scheduler should not behave as if all jobs are equally important. It should understand when inference must go first, when a low-importance training job can be delayed, and when waiting for a better GPU is smarter than assigning immediately.

## What Problem It Solves

This project is trying to solve these common issues:

- poor GPU utilization
- slow inference because urgent requests are blocked by training jobs
- low-priority training taking GPU space when high-priority inference needs immediate service
- memory fragmentation
- bad placement decisions in multi-GPU machines
- unnecessary model loading and unloading
- weak handling of preemption and checkpoint trade-offs

Instead of hardcoding one fixed rule, the environment is designed so a policy can learn better scheduling decisions over time.

Some typical decisions in this problem are:

- should a low-priority training job keep running, or should it be checkpointed for urgent inference?
- should the scheduler place a job now, or wait a few seconds for a better GPU to become free?
- should inference stay on a warm model, or should memory be freed for another important job?
- should a model be quantized to create enough space for incoming requests?

## Current Scope

Right now this is a single-node problem, not a full multi-node cluster scheduler.

The simulator models one machine with multiple GPUs and local topology between those GPUs. It already includes important single-node concerns like:

- training and inference together
- GPU memory pressure
- fragmentation
- warm-model reuse
- checkpoint preemption
- quantization decisions
- topology-aware placement

So the current focus is: make scheduling better inside one GPU server.

## Future Scope

In future, this can be extended to a multi-node scheduler. That version can include:

- scheduling across many machines
- node-level placement and routing
- network cost between nodes
- distributed training placement
- rack or zone awareness
- failover and rebalancing

The current single-node version is a good base for that, because the local GPU-level scheduling logic is already separated in the simulator and environment.

## What The Agent Controls

At each step the agent can choose one of eight actions:

- `assign_training`
- `assign_inference`
- `queue`
- `checkpoint_preempt`
- `preempt_kill`
- `wait`
- `unload_model`
- `quantize_model`

Observations include the current step, queue contents, per-GPU state, pending arrivals, and aggregate metrics.

## Tasks

The repository has three tasks:

| Task | Difficulty | Description |
| --- | --- | --- |
| `easy_001` | Easy | Two 24 GB GPUs, training only, fragmentation-aware placement |
| `medium_001` | Medium | Four GPUs, mixed training and inference, SLA pressure, warm models, quantization |
| `hard_001` | Hard | Bursty inference, multi-GPU training, topology constraints, all edge cases |

Each episode returns a normalized score in `[0.0, 1.0]`.

## How It Works

The simulator generates a workload of training jobs and inference requests. The environment exposes the system state and a list of valid actions. A policy or LLM can then take one action per step. At the end of the episode, the grader gives a final score based on things like utilization, throughput, SLA behaviour, queueing, and efficiency.

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

This repo is aligned for Python `3.10` to `3.12`. The checked-in local version file uses Python `3.12`.

```bash
uv sync
make test
make api
```

Run the API locally:

```bash
uv run server --host 0.0.0.0 --port 7860
```

Smoke test the endpoints:

```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{}'
curl http://127.0.0.1:7860/state
curl http://127.0.0.1:7860/valid-actions
```

## Submission runner

`inference.py` is the root-level submission script. It:

- uses the OpenAI client when `HF_TOKEN` is available
- runs all three tasks
- prints the required `[START]`, `[STEP]`, and `[END]` lines
- falls back to the built-in smart heuristic if an LLM call fails during local testing

Required environment variables for submission:

- `HF_TOKEN`
- `MODEL_NAME`
- `API_BASE_URL`

Defaults are only set in `inference.py` for:

- `API_BASE_URL`
- `MODEL_NAME`

`HF_TOKEN` has no default.

Optional:

- `LOCAL_IMAGE_NAME`
  This project does not use it because `inference.py` runs the local environment directly.

Example:

```bash
cp .env.example .env
uv run python inference.py
```

Recommended pre-submission checks:

```bash
make test
docker build -t gpu-scheduler-ml .
openenv validate
```

## Hugging Face Space deployment

The repo is configured as a Docker Space. The container serves the FastAPI app on port `7860`.

Set these Space secrets or variables:

- `HF_TOKEN`
- `MODEL_NAME`
- `API_BASE_URL`

## API

Available endpoints:

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `GET /state`
- `POST /state`
- `GET /valid-actions`
- `POST /step`

## Notes for judges and reviewers

- `POST /reset` accepts `{}` and resets to `easy_001` by default.
- The environment keeps task scoring in the `[0.0, 1.0]` range.
- The Docker image uses Python `3.12` and installs dependencies through `uv`.
