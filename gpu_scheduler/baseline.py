from __future__ import annotations

from collections.abc import Callable
from typing import Any
import random

from .env import GPUSchedulerEnv
from .graders import grade_episode
from .models import Action
from .simulator import GPU, InferenceRequest, PRECISION_BYTES, TrainingJob

Policy = Callable[[GPUSchedulerEnv], Action | None]


def random_policy(env: GPUSchedulerEnv, rng: random.Random | None = None) -> Action | None:
    rng = rng or random.Random()
    actions = env.get_valid_actions()
    if not actions:
        return None
    return Action.model_validate(rng.choice(actions))


def first_fit_policy(env: GPUSchedulerEnv) -> Action | None:
    if env.sim.inference_queue:
        request = env.sim.inference_queue[0]
        for gpu_id in sorted(env.sim.gpus):
            can_assign, _ = env.sim.can_assign_inference(request, gpu_id)
            if can_assign:
                return Action(action_type="assign_inference", item_id=request.request_id, target_gpu=gpu_id)
        return Action(action_type="queue", item_id=request.request_id)

    if env.sim.training_queue:
        job = env.sim.training_queue[0]
        for gpu_id in sorted(env.sim.gpus):
            can_assign, _ = env.sim.can_assign_training(job, gpu_id)
            if can_assign:
                return Action(action_type="assign_training", item_id=job.job_id, target_gpu=gpu_id)
        return Action(action_type="queue", item_id=job.job_id)

    return None


def inference_priority_policy(env: GPUSchedulerEnv) -> Action | None:
    request = _most_urgent_request(env)
    if request is not None:
        assign = _best_inference_assignment(env, request)
        if assign is not None:
            return assign
        quantize = _quantize_candidate(env, request)
        if quantize is not None:
            return quantize
        if request.sla_budget_steps <= 2:
            preempt = _checkpoint_candidate(env, request)
            if preempt is not None:
                return preempt
        future = _future_wait_for_inference(env, request)
        if future is not None:
            return future
        return Action(action_type="queue", item_id=request.request_id)

    job = _highest_priority_training(env)
    if job is not None:
        assign = _best_training_assignment(env, job)
        if assign is not None:
            return assign
        return Action(action_type="queue", item_id=job.job_id)

    return None


def smart_heuristic_policy(env: GPUSchedulerEnv) -> Action | None:
    request = _most_urgent_request(env)
    if request is not None:
        assign = _best_inference_assignment(env, request)
        future = _future_wait_for_inference(env, request)
        if assign is not None and future is not None and request.sla_budget_steps > 2:
            target_gpu = env.sim.gpus[assign.target_gpu]
            future_gpu = env.sim.gpus[future.target_gpu]
            if (
                future_gpu.next_free_in_steps <= request.sla_budget_steps - 1
                and future_gpu.raw_free_gb + future_gpu.next_free_memory_gb > target_gpu.effective_free_gb + 2
            ):
                return future
        if assign is not None:
            return assign
        quantize = _quantize_candidate(env, request)
        if quantize is not None:
            return quantize
        preempt = _checkpoint_candidate(env, request)
        if preempt is not None and request.sla_budget_steps <= 3:
            return preempt
        if future is not None:
            return future
        return Action(action_type="queue", item_id=request.request_id)

    job = _highest_priority_training(env)
    if job is None:
        unload = _unload_candidate(env)
        return unload

    assign = _best_training_assignment(env, job)
    if assign is not None:
        return assign

    unload = _unload_candidate(env)
    if unload is not None:
        return unload

    future = _future_wait_for_training(env, job)
    if future is not None and job.wait_time < 4:
        return future
    return Action(action_type="queue", item_id=job.job_id)


def run_baseline(
    name: str,
    task_id: str,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    env = GPUSchedulerEnv()
    observation = env.reset(task_id=task_id, seed=seed)
    policy = _resolve_policy(name)
    rng = random.Random(seed)
    steps = 0
    done = False
    total_reward = 0.0
    last_result: dict[str, Any] | None = None

    while not done and (max_steps is None or steps < max_steps):
        if name == "random":
            action = random_policy(env, rng=rng)
        else:
            action = policy(env)
        last_result = env.step(action)
        observation = last_result["observation"]
        total_reward += last_result["reward"]
        done = last_result["done"]
        steps += 1

    return {
        "baseline": name,
        "task_id": task_id,
        "steps": steps,
        "done": done,
        "total_reward": round(total_reward, 3),
        "final_metrics": observation["metrics"],
        "final_grade": (
            last_result["info"]["grade"]
            if last_result and last_result["info"]["grade"] is not None
            else grade_episode(observation["metrics"], env.task)
        ),
    }


def _resolve_policy(name: str) -> Policy:
    policies: dict[str, Policy] = {
        "random": lambda env: random_policy(env),
        "first_fit": first_fit_policy,
        "inference_priority": inference_priority_policy,
        "smart": smart_heuristic_policy,
    }
    if name not in policies:
        raise KeyError(f"Unknown baseline: {name}")
    return policies[name]


def _most_urgent_request(env: GPUSchedulerEnv) -> InferenceRequest | None:
    if not env.sim.inference_queue:
        return None
    return min(
        env.sim.inference_queue,
        key=lambda request: (request.sla_budget_steps, -request.wait_time, request.request_id),
    )


def _highest_priority_training(env: GPUSchedulerEnv) -> TrainingJob | None:
    if not env.sim.training_queue:
        return None
    return max(
        env.sim.training_queue,
        key=lambda job: (job.priority, job.wait_time, -job.estimated_duration, job.job_id),
    )


def _best_training_assignment(env: GPUSchedulerEnv, job: TrainingJob) -> Action | None:
    candidates: list[tuple[float, float, int]] = []
    for gpu_id in sorted(env.sim.gpus):
        can_assign, gpu_ids = env.sim.can_assign_training(job, gpu_id)
        if not can_assign:
            continue
        waste = sum(env.sim.gpus[candidate_id].effective_free_gb - job.memory_gb for candidate_id in gpu_ids)
        frag = sum(env.sim.gpus[candidate_id].fragmentation_score for candidate_id in gpu_ids)
        candidates.append((waste, frag, gpu_id))
    if not candidates:
        return None
    _, _, gpu_id = min(candidates)
    return Action(action_type="assign_training", item_id=job.job_id, target_gpu=gpu_id)


def _best_inference_assignment(env: GPUSchedulerEnv, request: InferenceRequest) -> Action | None:
    candidates: list[tuple[int, float, float, int]] = []
    for gpu_id, gpu in env.sim.gpus.items():
        can_assign, info = env.sim.can_assign_inference(request, gpu_id)
        if not can_assign:
            continue
        required = request.current_kv_gb + (0.0 if request.model_name in gpu.loaded_models else request.model_memory_gb)
        warm_bias = 0 if info.get("warm_model_reused") else 1
        waste = gpu.effective_free_gb - required
        candidates.append((warm_bias, waste, gpu.fragmentation_score, gpu_id))
    if not candidates:
        return None
    _, _, _, gpu_id = min(candidates)
    return Action(action_type="assign_inference", item_id=request.request_id, target_gpu=gpu_id)


def _future_wait_for_inference(env: GPUSchedulerEnv, request: InferenceRequest) -> Action | None:
    required = request.current_kv_gb + request.model_memory_gb
    candidates: list[tuple[int, float, int]] = []
    for gpu_id, gpu in env.sim.gpus.items():
        if gpu.next_free_in_steps <= 0 or gpu.next_free_in_steps >= request.sla_budget_steps:
            continue
        projected = gpu.raw_free_gb + gpu.next_free_memory_gb
        if projected >= required:
            candidates.append((gpu.next_free_in_steps, -projected, gpu_id))
    if not candidates:
        return None
    _, _, gpu_id = min(candidates)
    return Action(action_type="wait", item_id=request.request_id, target_gpu=gpu_id)


def _future_wait_for_training(env: GPUSchedulerEnv, job: TrainingJob) -> Action | None:
    candidates: list[tuple[int, float, int]] = []
    for gpu_id, gpu in env.sim.gpus.items():
        if gpu.next_free_in_steps <= 0 or gpu.next_free_in_steps > 3:
            continue
        projected = gpu.raw_free_gb + gpu.next_free_memory_gb
        if projected >= job.memory_gb:
            candidates.append((gpu.next_free_in_steps, -projected, gpu_id))
    if not candidates:
        return None
    _, _, gpu_id = min(candidates)
    return Action(action_type="wait", item_id=job.job_id, target_gpu=gpu_id)


def _checkpoint_candidate(env: GPUSchedulerEnv, request: InferenceRequest) -> Action | None:
    candidates: list[tuple[int, int, str]] = []
    for job in env.sim._unique_running_training_jobs():
        job_gpu = env.sim.gpus[job.assigned_gpus[0]] if job.assigned_gpus else None
        if job_gpu is None:
            continue
        projected_free = job_gpu.raw_free_gb + job.memory_gb
        if projected_free >= request.current_kv_gb + request.model_memory_gb:
            candidates.append((job.priority, -job.time_remaining, job.job_id))
    if not candidates:
        return None
    _, _, job_id = min(candidates)
    return Action(action_type="checkpoint_preempt", item_id=job_id, preempt_job_id=job_id)


def _quantize_candidate(env: GPUSchedulerEnv, request: InferenceRequest) -> Action | None:
    if not env.sim.inference_queue:
        return None
    candidates: list[tuple[float, int, str, str]] = []
    for gpu_id, gpu in env.sim.gpus.items():
        if request.model_name not in gpu.loaded_models or request.model_name in gpu.quantizing:
            continue
        current = gpu.loaded_models[request.model_name]
        for target in ("fp8", "int8", "int4"):
            if PRECISION_BYTES[target] >= PRECISION_BYTES.get(current["precision"], 2.0):
                continue
            ratio = PRECISION_BYTES[target] / PRECISION_BYTES[current["precision"]]
            new_mem = current["mem"] * ratio
            saved = current["mem"] - new_mem
            projected_free = gpu.raw_free_gb + saved
            if projected_free >= request.current_kv_gb:
                candidates.append((-saved, gpu_id, request.model_name, target))
                break
    if not candidates:
        return None
    _, gpu_id, model_name, target = min(candidates)
    return Action(
        action_type="quantize_model",
        item_id=f"model:{model_name}",
        target_gpu=gpu_id,
        model_to_unload=model_name,
        quantize_target=target,
    )


def _unload_candidate(env: GPUSchedulerEnv) -> Action | None:
    if env.sim.inference_queue:
        return None
    candidates: list[tuple[float, int, str]] = []
    for gpu_id, gpu in env.sim.gpus.items():
        for model_name, model in gpu.loaded_models.items():
            if any(request.model_name == model_name for request in gpu.active_requests):
                continue
            candidates.append((-model["mem"], gpu_id, model_name))
    if not candidates:
        return None
    _, gpu_id, model_name = min(candidates)
    return Action(
        action_type="unload_model",
        item_id=f"model:{model_name}",
        target_gpu=gpu_id,
        model_to_unload=model_name,
    )
