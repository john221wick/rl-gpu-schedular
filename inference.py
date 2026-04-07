from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from gpu_scheduler.baseline import smart_heuristic_policy
from gpu_scheduler.env import GPUSchedulerEnv
from gpu_scheduler.graders import grade_episode
from gpu_scheduler.models import Action
from gpu_scheduler.tasks import list_tasks

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

# Required by the hackathon submission format.
API_BASE_URL = _require_env("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = _require_env("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional when using from_docker_image().
BENCHMARK = "gpu-scheduler-ml"
TEMPERATURE = 0.0
MAX_TOKENS = 220
LLM_REQUEST_TIMEOUT_SECONDS = max(0.1, float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "3.0")))
MAX_LLM_RETRIES = max(1, int(os.getenv("MAX_LLM_RETRIES", "1")))
MAX_LLM_CALLS = max(1, int(os.getenv("MAX_LLM_CALLS", "3")))
LLM_TOTAL_BUDGET_SECONDS = max(0.0, float(os.getenv("LLM_TOTAL_BUDGET_SECONDS", "8.0")))
PROXY_PING_MAX_TOKENS = max(1, int(os.getenv("PROXY_PING_MAX_TOKENS", "8")))
LLM_DISABLED = False
LLM_CALL_COUNT = 0
LLM_TOTAL_LATENCY_SECONDS = 0.0
PROXY_PRIMED = False

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are operating a GPU scheduler environment.
    Choose exactly one valid action from the provided list.

    Return strict JSON only:
    {"action_index": 0, "reason": "short justification"}

    Rules:
    - Pick only from the valid_actions list.
    - Prefer urgent inference requests that are near SLA breach.
    - Prefer warm-model reuse for inference when possible.
    - Avoid fragmentation-heavy placements when a better wait option exists.
    - Use checkpoint_preempt before preempt_kill when possible.
    - Use quantize_model only when it clearly unlocks blocked inference.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = _sanitize(error) if error else "null"
    action_value = _sanitize(action)
    print(
        f"[STEP] step={step} action={action_value} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_value = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_value}",
        flush=True,
    )


def build_client() -> OpenAI | None:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


def prime_proxy(client: OpenAI | None, model_name: str) -> None:
    global LLM_DISABLED, LLM_TOTAL_LATENCY_SECONDS, PROXY_PRIMED

    if client is None or LLM_DISABLED or PROXY_PRIMED:
        return

    started = time.monotonic()
    try:
        client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=PROXY_PING_MAX_TOKENS,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": '{"ping":"proxy","reply":"ok"}'},
            ],
        )
        PROXY_PRIMED = True
    except Exception as exc:  # pragma: no cover - network/runtime fallback
        _disable_llm(f"Proxy warm-up failed, using heuristic policy: {exc}")
    finally:
        LLM_TOTAL_LATENCY_SECONDS += time.monotonic() - started


def action_to_str(action: Action | None) -> str:
    if action is None:
        return "noop()"

    parts = [f"item_id={action.item_id}"]
    if action.target_gpu is not None:
        parts.append(f"target_gpu={action.target_gpu}")
    if action.preempt_job_id is not None:
        parts.append(f"preempt_job_id={action.preempt_job_id}")
    if action.model_to_unload is not None:
        parts.append(f"model_to_unload={action.model_to_unload}")
    if action.quantize_target is not None:
        parts.append(f"quantize_target={action.quantize_target}")
    return f"{action.action_type}({','.join(parts)})"


def choose_action(
    env: GPUSchedulerEnv,
    client: OpenAI | None,
    model_name: str,
    task: dict[str, Any],
    observation: dict[str, Any],
) -> Action | None:
    global LLM_CALL_COUNT, LLM_DISABLED, LLM_TOTAL_LATENCY_SECONDS

    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return None

    heuristic_action = smart_heuristic_policy(env)
    if client is None or LLM_DISABLED:
        return heuristic_action or Action.model_validate(valid_actions[0])

    if (
        LLM_CALL_COUNT >= MAX_LLM_CALLS
        or LLM_TOTAL_LATENCY_SECONDS >= LLM_TOTAL_BUDGET_SECONDS
    ):
        _disable_llm("LLM budget exhausted, using heuristic policy for remaining steps.")
        return heuristic_action or Action.model_validate(valid_actions[0])

    prompt = {
        "task": {
            "id": task["id"],
            "difficulty": task["difficulty"],
            "description": task["description"],
        },
        "observation_summary": summarize_observation(observation),
        "valid_actions": [
            {"index": index, "action": action_to_str(Action.model_validate(action))}
            for index, action in enumerate(valid_actions)
        ],
        "recommended_action": action_to_str(heuristic_action) if heuristic_action is not None else None,
        "model": model_name,
    }

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        started = time.monotonic()
        try:
            LLM_CALL_COUNT += 1
            response = client.chat.completions.create(
                model=model_name,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = extract_json(content)
            action_index = parsed.get("action_index")
            if isinstance(action_index, int) and 0 <= action_index < len(valid_actions):
                return Action.model_validate(valid_actions[action_index])
            _disable_llm("Model returned an invalid action, using heuristic policy for remaining steps.")
            break
        except Exception as exc:  # pragma: no cover - network/runtime fallback
            status_code = getattr(exc, "status_code", None)
            if status_code in {429, 500, 502, 503, 504} and attempt < MAX_LLM_RETRIES:
                continue
            _disable_llm(f"Model action selection failed, disabling LLM path: {exc}")
            break
        finally:
            LLM_TOTAL_LATENCY_SECONDS += time.monotonic() - started

    return heuristic_action or Action.model_validate(valid_actions[0])


def extract_json(content: str) -> dict[str, Any]:
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return {}


def summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": observation["step"],
        "max_steps": observation["max_steps"],
        "training_queue_len": len(observation["training_queue"]),
        "inference_queue_len": len(observation["inference_queue"]),
        "pending_len": len(observation["pending"]),
        "metrics": {
            "completion_rate": observation["metrics"].get("completion_rate"),
            "avg_utilization": observation["metrics"].get("avg_utilization"),
            "sla_violations": observation["metrics"].get("sla_violations"),
            "oom_count": observation["metrics"].get("oom_count"),
        },
        "gpus": [
            {
                "gpu_id": gpu["gpu_id"],
                "effective_free_gb": gpu["effective_free_gb"],
                "fragmentation_score": gpu["fragmentation_score"],
                "next_free_in_steps": gpu["next_free_in_steps"],
                "training_jobs": [job["job_id"] for job in gpu["training_jobs"]],
                "inference_requests": [request["request_id"] for request in gpu["inference_requests"]],
                "loaded_models": sorted(gpu["loaded_models"].keys()),
            }
            for gpu in observation["gpus"]
        ],
        "training_queue": [
            {
                "job_id": job["job_id"],
                "memory_gb": job["memory_gb"],
                "priority": job["priority"],
                "num_gpus": job["num_gpus"],
                "wait_time": job["wait_time"],
            }
            for job in observation["training_queue"][:4]
        ],
        "inference_queue": [
            {
                "request_id": request["request_id"],
                "model_name": request["model_name"],
                "current_kv_gb": request.get("current_kv_gb", 0.0),
                "sla_budget_steps": request["sla_budget_steps"],
                "wait_time": request["wait_time"],
            }
            for request in observation["inference_queue"][:6]
        ],
    }


def run_task(task_id: str, client: OpenAI | None) -> bool:
    env = GPUSchedulerEnv()
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    model_name = MODEL_NAME

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    try:
        observation = env.reset(task_id=task_id)
        task = env.task or next(task for task in list_tasks() if task["id"] == task_id)
        done = False
        final_result: dict[str, Any] | None = None

        while not done and steps_taken < task["max_steps"]:
            action = choose_action(env, client, model_name, task, observation)
            final_result = env.step(action)
            observation = final_result["observation"]
            reward = float(final_result["reward"])
            done = bool(final_result["done"])
            error = final_result["info"].get("last_action_error")

            steps_taken += 1
            rewards.append(reward)
            log_step(
                step=steps_taken,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=error,
            )

        grade = (
            final_result["info"]["grade"]
            if final_result is not None and final_result["info"]["grade"] is not None
            else grade_episode(observation["metrics"], task)
        )
        score = min(max(float(grade["score"]), 0.0), 1.0)
        success = done
    except Exception as exc:  # pragma: no cover - submission entrypoint behavior
        print(f"Task {task_id} failed: {exc}", file=sys.stderr)
    finally:
        try:
            env.close()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            print(f"Close failed for {task_id}: {exc}", file=sys.stderr)
            success = False
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success


def _sanitize(value: str | None) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip() or "null"


def _disable_llm(message: str) -> None:
    global LLM_DISABLED
    if not LLM_DISABLED:
        print(message, file=sys.stderr)
    LLM_DISABLED = True


def main() -> int:
    client = build_client()
    prime_proxy(client, MODEL_NAME)
    task_ids = [task["id"] for task in list_tasks()]
    all_ok = True

    for task_id in task_ids:
        if not run_task(task_id, client):
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
