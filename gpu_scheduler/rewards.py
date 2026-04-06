from __future__ import annotations

from typing import Any


def compute_step_reward(
    prev: dict[str, Any],
    curr: dict[str, Any],
    action_result: dict[str, Any],
) -> float:
    reward = 0.0

    reward += (curr["metrics"]["avg_utilization"] - prev["metrics"]["avg_utilization"]) * 1.5

    if action_result.get("action_type") == "assign_inference":
        reward += 0.08
        if action_result.get("sla_budget_steps", 99) < 3:
            reward += 0.10

    new_violations = curr["metrics"]["sla_violations"] - prev["metrics"]["sla_violations"]
    reward -= new_violations * 0.30

    if action_result.get("action_type") == "assign_training":
        reward += 0.06
        if action_result.get("priority", 0) >= 3:
            reward += 0.10

    if action_result.get("smart_wait_payoff"):
        reward += 0.12
    if action_result.get("action_type") == "wait":
        reward -= 0.01

    if action_result.get("action_type") == "checkpoint_preempt":
        reward -= 0.03
    if action_result.get("action_type") == "preempt_kill":
        reward -= 0.15

    if action_result.get("action_type") == "quantize_model":
        memory_saved = action_result.get("memory_saved_gb", 0.0)
        reward += min(memory_saved * 0.03, 0.15)

    for gpu in curr["gpus"]:
        if gpu["raw_free_gb"] < 1.0 and gpu.get("inference_requests"):
            reward -= 0.05

    if action_result.get("placed_on_fragmented"):
        reward -= 0.03

    idle = sum(1 for gpu in curr["gpus"] if gpu["utilization"] < 0.05)
    waiting = curr["metrics"]["inference_queued"] + curr["metrics"]["training_queued"]
    if idle > 0 and waiting > 0:
        reward -= 0.05 * idle

    if action_result.get("warm_model_reused"):
        reward += 0.05

    for gpu in curr["gpus"]:
        if gpu["total_memory_gb"] >= 24 and gpu["utilization"] < 0.15:
            reward -= 0.02

    if action_result.get("action_type") == "quantize_model" and curr["metrics"]["inference_queued"] == 0:
        reward -= 0.05

    return round(max(-1.0, min(1.0, reward)), 4)
