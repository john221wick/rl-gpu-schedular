from __future__ import annotations

from typing import Any, Literal

from .compat import BaseModel, ConfigDict


class GPUState(BaseModel):
    gpu_id: int
    total_memory_gb: float
    raw_free_gb: float
    effective_free_gb: float
    fragmentation_score: float
    utilization: float
    next_free_in_steps: int
    next_free_memory_gb: float
    training_jobs: list[dict[str, Any]]
    inference_requests: list[dict[str, Any]]
    loaded_models: dict[str, dict[str, Any]]
    warm_models: dict[str, int]
    quantizing: dict[str, dict[str, Any]]
    nvlink_peers: list[int]


class Observation(BaseModel):
    step: int
    max_steps: int
    gpus: list[GPUState]
    training_queue: list[dict[str, Any]]
    inference_queue: list[dict[str, Any]]
    pending: list[dict[str, Any]]
    metrics: dict[str, Any]


class Action(BaseModel):
    action_type: Literal[
        "assign_training",
        "assign_inference",
        "queue",
        "checkpoint_preempt",
        "preempt_kill",
        "wait",
        "unload_model",
        "quantize_model",
    ]
    item_id: str
    target_gpu: int | None = None
    preempt_job_id: str | None = None
    model_to_unload: str | None = None
    quantize_target: Literal["fp8", "int8", "int4"] | None = None


class Reward(BaseModel):
    value: float


class EnvState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step: int
    task_id: str
    history: list[dict[str, Any]]
    done: bool
