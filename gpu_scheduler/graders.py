from __future__ import annotations

from typing import Any


def grade_episode(metrics: dict[str, Any], task: dict[str, Any]) -> dict[str, float]:
    total_inf = max(task["num_inference"], 1)
    total_train = max(task["num_training"], 1)
    util = min(metrics["avg_utilization"] / 0.75, 1.0)
    throughput = metrics.get("training_completed", 0) / total_train
    sla = max(0.0, 1 - metrics["sla_violations"] / total_inf)
    latency = max(0.0, 1 - metrics.get("avg_inference_wait", 0) / 5.0)
    efficiency = max(
        0.0,
        1 - metrics.get("oom_count", 0) * 0.15 - metrics.get("kill_preempts", 0) * 0.1,
    )
    wait_quality = min(
        metrics.get("smart_waits", 0) / max(metrics.get("wait_decisions", 0), 1),
        1.0,
    )
    quant_bonus = min(metrics.get("quantize_memory_saved_gb", 0) / 20.0, 1.0)
    score = (
        util * 0.15
        + throughput * 0.15
        + sla * 0.25
        + latency * 0.13
        + efficiency * 0.10
        + wait_quality * 0.10
        + quant_bonus * 0.07
        + metrics.get("completion_rate", 0) * 0.05
    )
    return {"score": round(min(max(score, 0.0), 1.0), 3)}
