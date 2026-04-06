from __future__ import annotations

from typing import Any

from .graders import grade_episode
from .models import Action, EnvState, Observation, Reward
from .rewards import compute_step_reward
from .simulator import ClusterSimulator, InferenceRequest, TrainingJob, PRECISION_BYTES
from .tasks import generate_job_sequence, get_task


class GPUSchedulerEnv:
    def __init__(self, tasks: dict[str, dict[str, Any]] | None = None) -> None:
        self.tasks = tasks
        self.task: dict[str, Any] | None = None
        self.sim: ClusterSimulator | None = None
        self.env_state: EnvState | None = None
        self.last_action_error: str | None = None
        self.last_wait_item_id: str | None = None
        self.last_wait_target: int | None = None
        self.last_wait_step: int | None = None

    def reset(self, task_id: str = "easy_001", seed: int | None = None) -> dict[str, Any]:
        self.task = get_task(task_id, seed=seed)
        job_sequence = generate_job_sequence(self.task)
        self.sim = ClusterSimulator(
            gpu_configs=self.task["cluster"],
            job_sequence=job_sequence,
            max_steps=self.task["max_steps"],
        )
        self.env_state = EnvState(step=0, task_id=task_id, history=[], done=False)
        self.last_action_error = None
        self.last_wait_item_id = None
        self.last_wait_target = None
        self.last_wait_step = None
        return self.current_observation()

    def current_observation(self) -> dict[str, Any]:
        self._require_reset()
        observation = Observation.model_validate(self.sim.get_snapshot())
        return observation.model_dump()

    def state(self) -> dict[str, Any]:
        self._require_reset()
        return self.current_observation()

    def step(self, action: Action | dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_reset()
        prev = self.current_observation()
        action_model = Action.model_validate(action) if action is not None else None
        action_result = self._execute(action_model)
        self.last_action_error = None if action_result.get("success", True) else "action_failed"
        self.sim.advance_time()
        curr = self.current_observation()
        reward = compute_step_reward(prev, curr, action_result)
        reward_model = Reward(value=reward)
        done = self.sim.is_done
        grade = grade_episode(curr["metrics"], self.task) if done else None
        self.env_state.step = self.sim.current_step
        self.env_state.done = done
        self.env_state.history.append(
            {
                "step": self.sim.current_step,
                "action": action_model.model_dump() if action_model is not None else None,
                "reward": reward,
                "metrics": curr["metrics"],
            }
        )
        return {
            "observation": curr,
            "reward": reward,
            "done": done,
            "info": {
                "action_result": action_result,
                "grade": grade,
                "last_action_error": self.last_action_error,
                "reward_model": reward_model.model_dump(),
            },
        }

    def close(self) -> None:
        self.sim = None
        self.env_state = None
        self.task = None
        self.last_action_error = None
        self.last_wait_item_id = None
        self.last_wait_target = None
        self.last_wait_step = None

    def get_valid_actions(self) -> list[dict[str, Any]]:
        self._require_reset()
        actions: list[Action] = []
        seen: set[tuple[Any, ...]] = set()

        for request in self.sim.inference_queue:
            for gpu_id in sorted(self.sim.gpus):
                can_assign, _ = self.sim.can_assign_inference(request, gpu_id)
                if can_assign:
                    self._append_action(
                        actions,
                        seen,
                        Action(
                            action_type="assign_inference",
                            item_id=request.request_id,
                            target_gpu=gpu_id,
                        ),
                    )
            future_gpu = self._best_future_gpu_for_inference(request)
            if future_gpu is not None:
                self._append_action(
                    actions,
                    seen,
                    Action(action_type="wait", item_id=request.request_id, target_gpu=future_gpu),
                )
            self._append_action(actions, seen, Action(action_type="queue", item_id=request.request_id))

        for job in self.sim.training_queue:
            for gpu_id in sorted(self.sim.gpus):
                can_assign, _ = self.sim.can_assign_training(job, gpu_id)
                if can_assign:
                    self._append_action(
                        actions,
                        seen,
                        Action(
                            action_type="assign_training",
                            item_id=job.job_id,
                            target_gpu=gpu_id,
                        ),
                    )
            future_gpu = self._best_future_gpu_for_training(job)
            if future_gpu is not None:
                self._append_action(
                    actions,
                    seen,
                    Action(action_type="wait", item_id=job.job_id, target_gpu=future_gpu),
                )
            self._append_action(actions, seen, Action(action_type="queue", item_id=job.job_id))

        for job in self.sim._unique_running_training_jobs():
            if job.preemptible:
                self._append_action(
                    actions,
                    seen,
                    Action(
                        action_type="checkpoint_preempt",
                        item_id=job.job_id,
                        preempt_job_id=job.job_id,
                    ),
                )
            self._append_action(
                actions,
                seen,
                Action(action_type="preempt_kill", item_id=job.job_id, preempt_job_id=job.job_id),
            )

        for gpu_id, gpu in self.sim.gpus.items():
            for model_name, model in gpu.loaded_models.items():
                if not any(request.model_name == model_name for request in gpu.active_requests):
                    self._append_action(
                        actions,
                        seen,
                        Action(
                            action_type="unload_model",
                            item_id=f"model:{model_name}",
                            target_gpu=gpu_id,
                            model_to_unload=model_name,
                        ),
                    )
                for target in self._lower_precisions(model["precision"]):
                    self._append_action(
                        actions,
                        seen,
                        Action(
                            action_type="quantize_model",
                            item_id=f"model:{model_name}",
                            target_gpu=gpu_id,
                            model_to_unload=model_name,
                            quantize_target=target,
                        ),
                    )

        return [action.model_dump() for action in actions]

    def _execute(self, action: Action | None) -> dict[str, Any]:
        if action is None:
            return {"action_type": "noop", "success": True}

        result: dict[str, Any] = {
            "action_type": action.action_type,
            "item_id": action.item_id,
            "success": False,
        }

        if action.action_type == "assign_training":
            gpu = self.sim.gpus.get(action.target_gpu) if action.target_gpu is not None else None
            result.update(self.sim.assign_training(action.item_id, action.target_gpu))
            if result["success"] and gpu is not None:
                result["placed_on_fragmented"] = gpu.fragmentation_score > 0.05
                self._maybe_mark_smart_wait(action, result)

        elif action.action_type == "assign_inference":
            gpu = self.sim.gpus.get(action.target_gpu) if action.target_gpu is not None else None
            result.update(self.sim.assign_inference(action.item_id, action.target_gpu))
            if result["success"] and gpu is not None:
                result["placed_on_fragmented"] = gpu.fragmentation_score > 0.05
                self._maybe_mark_smart_wait(action, result)

        elif action.action_type == "queue":
            item = self._find_item(action.item_id)
            if item is not None:
                self._ensure_item_in_queue(item)
                result["success"] = True

        elif action.action_type == "checkpoint_preempt":
            job_id = action.preempt_job_id or action.item_id
            result.update(self.sim.checkpoint_preempt(job_id))

        elif action.action_type == "preempt_kill":
            job_id = action.preempt_job_id or action.item_id
            result.update(self.sim.preempt_kill(job_id))

        elif action.action_type == "wait":
            item = self._find_item(action.item_id)
            if item is not None:
                self._ensure_item_in_queue(item)
                result["success"] = True
                self.sim.wait_decisions += 1
                self.last_wait_item_id = action.item_id
                self.last_wait_target = action.target_gpu
                self.last_wait_step = self.sim.current_step

        elif action.action_type == "unload_model":
            if action.target_gpu is not None and action.model_to_unload:
                result["success"] = self.sim.force_unload_model(action.target_gpu, action.model_to_unload)

        elif action.action_type == "quantize_model":
            if action.target_gpu is not None and action.quantize_target:
                gpu = self.sim.gpus.get(action.target_gpu)
                model_name = action.model_to_unload
                if gpu is not None and model_name and model_name in gpu.loaded_models:
                    old_mem = gpu.loaded_models[model_name]["mem"]
                    success = self.sim.request_quantize(action.target_gpu, model_name, action.quantize_target)
                    if success:
                        result["success"] = True
                        new_mem = gpu.quantizing[model_name]["new_mem"]
                        result["memory_saved_gb"] = round(old_mem - new_mem, 3)

        return result

    def _find_item(self, item_id: str) -> TrainingJob | InferenceRequest | None:
        for job in self.sim.training_queue:
            if job.job_id == item_id:
                return job
        for request in self.sim.inference_queue:
            if request.request_id == item_id:
                return request
        for item in self.sim.pending:
            if isinstance(item, TrainingJob) and item.job_id == item_id:
                return item
            if isinstance(item, InferenceRequest) and item.request_id == item_id:
                return item
        for job in self.sim._unique_running_training_jobs():
            if job.job_id == item_id:
                return job
        for gpu in self.sim.gpus.values():
            for request in gpu.active_requests:
                if request.request_id == item_id:
                    return request
        return None

    def _ensure_item_in_queue(self, item: TrainingJob | InferenceRequest) -> None:
        if isinstance(item, TrainingJob):
            item.status = "waiting"
            if item not in self.sim.training_queue and not item.assigned_gpus:
                self.sim.training_queue.append(item)
        else:
            item.status = "waiting"
            if item not in self.sim.inference_queue and item.assigned_gpu is None:
                self.sim.inference_queue.append(item)

    def _maybe_mark_smart_wait(self, action: Action, result: dict[str, Any]) -> None:
        if (
            self.last_wait_item_id == action.item_id
            and self.last_wait_target == action.target_gpu
            and self.last_wait_step is not None
            and self.sim.current_step > self.last_wait_step
        ):
            result["smart_wait_payoff"] = True
            self.sim.smart_waits += 1
        if self.last_wait_item_id == action.item_id:
            self.last_wait_item_id = None
            self.last_wait_target = None
            self.last_wait_step = None

    def _best_future_gpu_for_inference(self, request: InferenceRequest) -> int | None:
        required = request.current_kv_gb + request.model_memory_gb
        candidates: list[tuple[int, float, int]] = []
        for gpu_id, gpu in self.sim.gpus.items():
            if gpu.next_free_in_steps <= 0:
                continue
            projected = gpu.raw_free_gb + gpu.next_free_memory_gb
            if projected >= required:
                candidates.append((gpu.next_free_in_steps, -projected, gpu_id))
        return min(candidates)[2] if candidates else None

    def _best_future_gpu_for_training(self, job: TrainingJob) -> int | None:
        candidates: list[tuple[int, float, int]] = []
        for gpu_id, gpu in self.sim.gpus.items():
            if gpu.next_free_in_steps <= 0:
                continue
            projected = gpu.raw_free_gb + gpu.next_free_memory_gb
            if projected >= job.memory_gb:
                candidates.append((gpu.next_free_in_steps, -projected, gpu_id))
        return min(candidates)[2] if candidates else None

    def _lower_precisions(self, current_precision: str) -> list[str]:
        candidates = ["fp8", "int8", "int4"]
        current = PRECISION_BYTES.get(current_precision, 2.0)
        return [precision for precision in candidates if PRECISION_BYTES[precision] < current]

    def _append_action(self, actions: list[Action], seen: set[tuple[Any, ...]], action: Action) -> None:
        key = (
            action.action_type,
            action.item_id,
            action.target_gpu,
            action.preempt_job_id,
            action.model_to_unload,
            action.quantize_target,
        )
        if key not in seen:
            seen.add(key)
            actions.append(action)

    def _require_reset(self) -> None:
        if self.sim is None or self.env_state is None or self.task is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
