from __future__ import annotations

from copy import deepcopy
from typing import Any
import random

from .simulator import InferenceRequest, TrainingJob

TASKS: dict[str, dict[str, Any]] = {
    "easy_001": {
        "id": "easy_001",
        "difficulty": "easy",
        "description": "2 GPUs, training only, learn bin packing",
        "cluster": [
            {"id": 0, "memory": 24, "nvlink_peers": [1]},
            {"id": 1, "memory": 24, "nvlink_peers": [0]},
        ],
        "num_training": 10,
        "num_inference": 0,
        "max_steps": 100,
        "pattern": "steady",
        "seed": 42,
    },
    "medium_001": {
        "id": "medium_001",
        "difficulty": "medium",
        "description": "4 GPUs, mixed training+inference, SLA, quantization",
        "cluster": [
            {"id": 0, "memory": 24, "nvlink_peers": [1]},
            {"id": 1, "memory": 24, "nvlink_peers": [0]},
            {"id": 2, "memory": 16, "nvlink_peers": [3]},
            {"id": 3, "memory": 16, "nvlink_peers": [2]},
        ],
        "num_training": 8,
        "num_inference": 30,
        "max_steps": 200,
        "pattern": "mixed",
        "seed": 123,
    },
    "hard_001": {
        "id": "hard_001",
        "difficulty": "hard",
        "description": "Full: bursty inference, topology, multi-GPU, all edge cases",
        "cluster": [
            {"id": 0, "memory": 24, "nvlink_peers": [1]},
            {"id": 1, "memory": 24, "nvlink_peers": [0]},
            {"id": 2, "memory": 16, "nvlink_peers": [3]},
            {"id": 3, "memory": 16, "nvlink_peers": [2]},
        ],
        "num_training": 10,
        "num_inference": 60,
        "max_steps": 300,
        "pattern": "bursty",
        "seed": 456,
    },
}

MODEL_LIBRARY: list[dict[str, Any]] = [
    {"name": "bert-large", "memory_gb": 4.0, "precision": "fp16"},
    {"name": "phi-3-mini", "memory_gb": 6.0, "precision": "fp16"},
    {"name": "mistral-7b", "memory_gb": 12.0, "precision": "fp16"},
    {"name": "llama-7b", "memory_gb": 14.0, "precision": "fp16"},
]


def list_tasks() -> list[dict[str, Any]]:
    return [deepcopy(task) for task in TASKS.values()]


def get_task(task_id: str, seed: int | None = None) -> dict[str, Any]:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    task = deepcopy(TASKS[task_id])
    if seed is not None:
        task["seed"] = seed
    return task


def generate_job_sequence(task: dict[str, Any]) -> list[tuple[int, TrainingJob | InferenceRequest]]:
    rng = random.Random(task["seed"])
    max_steps = task["max_steps"]
    training_steps = _arrival_steps(task["num_training"], max_steps, task["pattern"], rng, kind="training")
    inference_steps = _arrival_steps(task["num_inference"], max_steps, task["pattern"], rng, kind="inference")
    sequence: list[tuple[int, TrainingJob | InferenceRequest]] = []

    for idx, step in enumerate(training_steps):
        sequence.append((step, _make_training_job(idx, task, rng)))
    for idx, step in enumerate(inference_steps):
        sequence.append((step, _make_inference_request(idx, task, rng)))

    sequence.sort(key=lambda item: (item[0], 0 if isinstance(item[1], TrainingJob) else 1))
    return sequence


def _arrival_steps(
    count: int,
    max_steps: int,
    pattern: str,
    rng: random.Random,
    *,
    kind: str,
) -> list[int]:
    if count <= 0:
        return []
    if pattern == "steady":
        horizon = max(1, int(max_steps * 0.75))
        return _spread_steps(count, horizon, rng, jitter=2)
    if pattern == "mixed":
        if kind == "training":
            horizon = max(1, int(max_steps * 0.8))
            return _spread_steps(count, horizon, rng, jitter=4)
        anchors = [int(max_steps * 0.2), int(max_steps * 0.45), int(max_steps * 0.7)]
        return _burst_steps(count, max_steps, rng, anchors, spread=max(2, max_steps // 20))
    if kind == "training":
        anchors = [0, int(max_steps * 0.25), int(max_steps * 0.55)]
        return _burst_steps(count, max_steps, rng, anchors, spread=max(2, max_steps // 18))
    anchors = [int(max_steps * 0.18), int(max_steps * 0.38), int(max_steps * 0.62), int(max_steps * 0.82)]
    return _burst_steps(count, max_steps, rng, anchors, spread=max(3, max_steps // 16))


def _spread_steps(count: int, horizon: int, rng: random.Random, *, jitter: int) -> list[int]:
    gap = max(1, horizon // max(count, 1))
    steps: list[int] = []
    for idx in range(count):
        base = idx * gap
        step = base + rng.randint(0, min(jitter, gap))
        steps.append(min(step, max(horizon - 1, 0)))
    return steps


def _burst_steps(
    count: int,
    max_steps: int,
    rng: random.Random,
    anchors: list[int],
    *,
    spread: int,
) -> list[int]:
    steps: list[int] = []
    for idx in range(count):
        anchor = anchors[idx % len(anchors)] if anchors else max_steps // 2
        offset = int(rng.gauss(0, spread))
        steps.append(max(0, min(max_steps - 1, anchor + offset)))
    steps.sort()
    return steps


def _make_training_job(idx: int, task: dict[str, Any], rng: random.Random) -> TrainingJob:
    task_id = task["id"]
    if task_id == "easy_001":
        memory = rng.choice([6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
        num_gpus = 1
        priority = rng.choices([0, 1, 2], weights=[3, 5, 2], k=1)[0]
        duration = rng.randint(12, 28)
    elif task_id == "medium_001":
        memory = rng.choice([8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        num_gpus = 1 if rng.random() < 0.8 else 2
        priority = rng.choices([0, 1, 2, 3], weights=[2, 4, 3, 1], k=1)[0]
        duration = rng.randint(18, 42)
    else:
        multi_gpu = idx == 0 or rng.random() < 0.35
        num_gpus = 2 if multi_gpu else 1
        memory = rng.choice([12.0, 14.0, 16.0, 18.0, 20.0, 22.0]) if multi_gpu else rng.choice(
            [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        )
        priority = 3 if idx == 0 else rng.choices([0, 1, 2, 3], weights=[2, 3, 3, 2], k=1)[0]
        duration = rng.randint(24, 60)

    est_duration = max(5, duration + rng.randint(-4, 4))
    actual_duration = max(4, duration + rng.randint(-5, 5))
    return TrainingJob(
        job_id=f"train_{idx:03d}",
        memory_gb=memory,
        estimated_duration=est_duration,
        actual_duration=actual_duration,
        priority=priority,
        num_gpus=num_gpus,
        preemptible=True,
        checkpoint_cost=2,
        auto_checkpoint_interval=30,
    )


def _make_inference_request(idx: int, task: dict[str, Any], rng: random.Random) -> InferenceRequest:
    model = deepcopy(rng.choice(MODEL_LIBRARY))
    if task["id"] == "hard_001" and idx % 6 == 0:
        model = deepcopy(MODEL_LIBRARY[-1])
    base_kv = round(rng.uniform(0.1, 0.6), 2)
    kv_growth = round(rng.uniform(0.15, 0.45), 2)
    max_kv = round(base_kv + rng.uniform(1.0, 4.0), 2)
    actual_duration = rng.randint(2, 8)
    est_duration = max(1, actual_duration + rng.randint(-1, 2))
    if task["id"] == "medium_001":
        sla_seconds = rng.choice([120, 180, 240, 300])
    else:
        sla_seconds = rng.choice([60, 120, 180, 240])
    return InferenceRequest(
        request_id=f"inf_{idx:03d}",
        model_name=model["name"],
        model_memory_gb=model["memory_gb"],
        model_precision=model["precision"],
        initial_kv_gb=base_kv,
        kv_growth_rate=kv_growth,
        max_kv_gb=max_kv,
        estimated_duration=est_duration,
        actual_duration=actual_duration,
        sla_seconds=sla_seconds,
        priority=2,
    )
