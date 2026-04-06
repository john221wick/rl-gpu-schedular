from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PRECISION_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass
class TrainingJob:
    job_id: str
    memory_gb: float
    estimated_duration: int
    actual_duration: int
    priority: int
    num_gpus: int = 1
    preemptible: bool = True
    has_checkpoint: bool = False
    checkpoint_age: int = 0
    checkpoint_cost: int = 2
    auto_checkpoint_interval: int = 30
    time_remaining: int = 0
    time_elapsed: int = 0
    assigned_gpus: list[int] = field(default_factory=list)
    wait_time: int = 0
    status: str = "waiting"

    def __post_init__(self) -> None:
        self.time_remaining = self.actual_duration


@dataclass
class InferenceRequest:
    request_id: str
    model_name: str
    model_memory_gb: float
    model_precision: str
    initial_kv_gb: float
    kv_growth_rate: float
    max_kv_gb: float
    estimated_duration: int
    actual_duration: int
    sla_seconds: int
    priority: int = 2
    current_kv_gb: float = 0.0
    time_remaining: int = 0
    wait_time: int = 0
    assigned_gpu: int | None = None
    load_steps_remaining: int = 0
    status: str = "waiting"

    def __post_init__(self) -> None:
        self.time_remaining = self.actual_duration
        self.current_kv_gb = self.initial_kv_gb

    @property
    def sla_budget_steps(self) -> float:
        return max(0.0, self.sla_seconds / 60 - self.wait_time)


@dataclass
class GPU:
    gpu_id: int
    total_memory_gb: float
    nvlink_peers: list[int] = field(default_factory=list)
    training_jobs: list[TrainingJob] = field(default_factory=list)
    loaded_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_requests: list[InferenceRequest] = field(default_factory=list)
    fragmentation_score: float = 0.0
    warm_models: dict[str, int] = field(default_factory=dict)
    model_keep_warm: int = 5
    quantizing: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def raw_free_gb(self) -> float:
        train = sum(job.memory_gb for job in self.training_jobs)
        models = sum(model["mem"] for model in self.loaded_models.values())
        kv = sum(request.current_kv_gb for request in self.active_requests)
        return self.total_memory_gb - train - models - kv

    @property
    def effective_free_gb(self) -> float:
        return self.raw_free_gb * (1.0 - self.fragmentation_score)

    @property
    def utilization(self) -> float:
        used = min(self.total_memory_gb, max(0.0, self.total_memory_gb - self.raw_free_gb))
        return round(used / self.total_memory_gb, 4) if self.total_memory_gb > 0 else 0.0

    @property
    def next_free_in_steps(self) -> int:
        times = [job.time_remaining for job in self.training_jobs]
        times.extend(
            request.load_steps_remaining + request.time_remaining for request in self.active_requests
        )
        return min(times) if times else -1

    @property
    def next_free_memory_gb(self) -> float:
        if not self.training_jobs and not self.active_requests:
            return 0.0
        soonest_time = float("inf")
        soonest_mem = 0.0
        for job in self.training_jobs:
            if job.time_remaining < soonest_time:
                soonest_time = job.time_remaining
                soonest_mem = job.memory_gb
        for request in self.active_requests:
            request_time = request.load_steps_remaining + request.time_remaining
            if request_time < soonest_time:
                soonest_time = request_time
                soonest_mem = request.current_kv_gb
        return soonest_mem

    def can_fit(self, memory_gb: float, *, use_effective: bool = True) -> bool:
        free = self.effective_free_gb if use_effective else self.raw_free_gb
        return free + 1e-9 >= memory_gb

    def update_fragmentation(self) -> None:
        if self.fragmentation_score > 0.0:
            self.fragmentation_score *= 0.6
            if self.fragmentation_score < 0.02:
                self.fragmentation_score = 0.0

    def on_job_freed(self, memory_gb: float) -> None:
        frag = min(0.20, memory_gb / self.total_memory_gb * 0.3)
        self.fragmentation_score = min(1.0, self.fragmentation_score + frag)

    def model_load_steps(self, memory_gb: float) -> int:
        if memory_gb <= 6:
            return 1
        if memory_gb <= 12:
            return 2
        return 3

    def quantize_model(self, model_name: str, target_precision: str) -> bool:
        if model_name not in self.loaded_models:
            return False
        current = self.loaded_models[model_name]
        current_bpw = PRECISION_BYTES.get(current["precision"], 2.0)
        target_bpw = PRECISION_BYTES.get(target_precision, 1.0)
        if target_bpw >= current_bpw:
            return False
        ratio = target_bpw / current_bpw
        new_mem = current["mem"] * ratio
        self.quantizing[model_name] = {
            "target": target_precision,
            "new_mem": new_mem,
            "old_mem": current["mem"],
            "steps_left": 1,
        }
        return True


class ClusterSimulator:
    def __init__(
        self,
        gpu_configs: list[dict[str, Any]],
        job_sequence: list[tuple[int, TrainingJob | InferenceRequest]],
        max_steps: int = 200,
    ) -> None:
        self.max_steps = max_steps
        self.current_step = 0
        self.gpus = {
            config["id"]: GPU(
                gpu_id=config["id"],
                total_memory_gb=config["memory"],
                nvlink_peers=config.get("nvlink_peers", []),
            )
            for config in gpu_configs
        }
        self.job_sequence = sorted(job_sequence, key=lambda item: item[0])
        self.arrival_cursor = 0
        self.training_queue: list[TrainingJob] = []
        self.inference_queue: list[InferenceRequest] = []
        self.completed: list[TrainingJob | InferenceRequest] = []
        self.pending: list[TrainingJob | InferenceRequest] = []
        self.sla_violations = 0
        self.total_util = 0.0
        self.wait_decisions = 0
        self.smart_waits = 0
        self.quantize_count = 0
        self.quantize_memory_saved = 0.0
        self.oom_count = 0
        self.kill_preempts = 0
        self.checkpoint_preempts = 0
        self.training_completed = 0
        self.inference_completed = 0
        self.total_inference_wait = 0.0
        self.total_work_items = len(job_sequence)
        self.pending = self._release_arrivals_for_step(0)

    def _release_arrivals_for_step(self, step: int) -> list[TrainingJob | InferenceRequest]:
        pending: list[TrainingJob | InferenceRequest] = []
        while self.arrival_cursor < len(self.job_sequence):
            arrival_step, item = self.job_sequence[self.arrival_cursor]
            if arrival_step != step:
                break
            item.status = "waiting"
            pending.append(item)
            if isinstance(item, TrainingJob):
                self.training_queue.append(item)
            else:
                self.inference_queue.append(item)
            self.arrival_cursor += 1
        return pending

    def _unique_running_training_jobs(self) -> list[TrainingJob]:
        jobs: list[TrainingJob] = []
        seen: set[str] = set()
        for gpu in self.gpus.values():
            for job in gpu.training_jobs:
                if job.job_id not in seen:
                    seen.add(job.job_id)
                    jobs.append(job)
        return jobs

    def _locate_running_training_job(self, job_id: str) -> tuple[TrainingJob | None, list[GPU]]:
        gpus: list[GPU] = []
        found: TrainingJob | None = None
        for gpu in self.gpus.values():
            for job in gpu.training_jobs:
                if job.job_id == job_id:
                    found = job
                    gpus.append(gpu)
                    break
        return found, gpus

    def can_assign_training(self, job: TrainingJob, target_gpu: int) -> tuple[bool, list[int]]:
        gpu = self.gpus.get(target_gpu)
        if gpu is None:
            return False, []
        selected = [target_gpu]
        if job.num_gpus > 1:
            for peer_id in gpu.nvlink_peers:
                if peer_id not in selected and peer_id in self.gpus:
                    selected.append(peer_id)
                if len(selected) == job.num_gpus:
                    break
            if len(selected) < job.num_gpus:
                return False, []
        if any(not self.gpus[gpu_id].can_fit(job.memory_gb) for gpu_id in selected):
            return False, []
        return True, selected

    def can_assign_inference(self, request: InferenceRequest, target_gpu: int) -> tuple[bool, dict[str, Any]]:
        gpu = self.gpus.get(target_gpu)
        if gpu is None:
            return False, {}
        if request.model_name in gpu.quantizing:
            return False, {}
        warm = request.model_name in gpu.loaded_models
        required = request.current_kv_gb + (0.0 if warm else request.model_memory_gb)
        if not gpu.can_fit(required):
            return False, {}
        return True, {
            "warm_model_reused": request.model_name in gpu.warm_models,
            "cold_load": not warm,
            "load_steps": 0 if warm else gpu.model_load_steps(request.model_memory_gb),
        }

    def advance_time(self) -> None:
        self.current_step += 1

        completed_training: list[TrainingJob] = []
        for job in self._unique_running_training_jobs():
            job.time_remaining -= 1
            job.time_elapsed += 1
            job.checkpoint_age += 1
            if job.checkpoint_age >= job.auto_checkpoint_interval:
                job.has_checkpoint = True
                job.checkpoint_age = 0
            if job.time_remaining <= 0:
                completed_training.append(job)
        for job in completed_training:
            job.status = "completed"
            for gpu_id in list(job.assigned_gpus):
                gpu = self.gpus[gpu_id]
                if job in gpu.training_jobs:
                    gpu.training_jobs.remove(job)
                    gpu.on_job_freed(job.memory_gb)
            job.assigned_gpus.clear()
            self.completed.append(job)
            self.training_completed += 1

        for gpu in self.gpus.values():
            completed_requests: list[InferenceRequest] = []
            for request in list(gpu.active_requests):
                if request.load_steps_remaining > 0:
                    request.load_steps_remaining -= 1
                    if request.load_steps_remaining == 0:
                        request.status = "running"
                    continue
                request.time_remaining -= 1
                request.current_kv_gb = min(
                    request.current_kv_gb + request.kv_growth_rate,
                    request.max_kv_gb,
                )
                if request.time_remaining <= 0:
                    completed_requests.append(request)
            for request in completed_requests:
                request.status = "completed"
                gpu.active_requests.remove(request)
                self.completed.append(request)
                self.inference_completed += 1
                self.total_inference_wait += request.wait_time

            for model_name in list(gpu.loaded_models.keys()):
                active = model_name in gpu.quantizing or any(
                    request.model_name == model_name for request in gpu.active_requests
                )
                if not active:
                    gpu.warm_models[model_name] = gpu.warm_models.get(model_name, 0) + 1
                    if gpu.warm_models[model_name] > gpu.model_keep_warm:
                        memory = gpu.loaded_models.pop(model_name, {}).get("mem", 0.0)
                        gpu.warm_models.pop(model_name, None)
                        if memory > 0:
                            gpu.on_job_freed(memory)
                else:
                    gpu.warm_models.pop(model_name, None)

            for model_name in list(gpu.quantizing.keys()):
                quantization = gpu.quantizing[model_name]
                quantization["steps_left"] -= 1
                if quantization["steps_left"] <= 0:
                    saved = quantization["old_mem"] - quantization["new_mem"]
                    gpu.loaded_models[model_name] = {
                        "mem": quantization["new_mem"],
                        "precision": quantization["target"],
                    }
                    self.quantize_count += 1
                    self.quantize_memory_saved += saved
                    gpu.quantizing.pop(model_name)

        for gpu in self.gpus.values():
            while gpu.raw_free_gb < -0.1 and gpu.active_requests:
                victim = gpu.active_requests[-1]
                victim.status = "oom_killed"
                gpu.active_requests.remove(victim)
                self.sla_violations += 1
                self.oom_count += 1

        for gpu in self.gpus.values():
            gpu.update_fragmentation()

        expired_requests: list[InferenceRequest] = []
        for request in self.inference_queue:
            request.wait_time += 1
            if request.sla_budget_steps <= 0:
                request.status = "sla_violated"
                expired_requests.append(request)
                self.sla_violations += 1
        for request in expired_requests:
            if request in self.inference_queue:
                self.inference_queue.remove(request)

        for job in self.training_queue:
            job.wait_time += 1

        self.pending = self._release_arrivals_for_step(self.current_step)

        total_used = 0.0
        total_capacity = 0.0
        for gpu in self.gpus.values():
            used = min(gpu.total_memory_gb, max(0.0, gpu.total_memory_gb - gpu.raw_free_gb))
            total_used += used
            total_capacity += gpu.total_memory_gb
        self.total_util += total_used / total_capacity if total_capacity > 0 else 0.0

    def assign_training(self, job_id: str, target_gpu: int | None) -> dict[str, Any]:
        if target_gpu is None:
            return {"success": False}
        job = next((candidate for candidate in self.training_queue if candidate.job_id == job_id), None)
        if job is None:
            return {"success": False}
        can_assign, gpu_ids = self.can_assign_training(job, target_gpu)
        if not can_assign:
            return {"success": False}
        for gpu_id in gpu_ids:
            self.gpus[gpu_id].training_jobs.append(job)
        job.assigned_gpus = gpu_ids
        job.status = "running"
        if job in self.training_queue:
            self.training_queue.remove(job)
        if job in self.pending:
            self.pending.remove(job)
        return {
            "success": True,
            "assigned_gpus": gpu_ids,
            "priority": job.priority,
            "num_gpus": job.num_gpus,
        }

    def assign_inference(self, request_id: str, target_gpu: int | None) -> dict[str, Any]:
        if target_gpu is None:
            return {"success": False}
        request = next((candidate for candidate in self.inference_queue if candidate.request_id == request_id), None)
        if request is None:
            return {"success": False}
        can_assign, info = self.can_assign_inference(request, target_gpu)
        if not can_assign:
            return {"success": False}
        gpu = self.gpus[target_gpu]
        if request.model_name not in gpu.loaded_models:
            gpu.loaded_models[request.model_name] = {
                "mem": request.model_memory_gb,
                "precision": request.model_precision,
            }
            request.load_steps_remaining = info.get("load_steps", 0)
            request.status = "loading" if request.load_steps_remaining > 0 else "running"
        else:
            loaded = gpu.loaded_models[request.model_name]
            request.model_memory_gb = loaded["mem"]
            request.model_precision = loaded["precision"]
            request.status = "running"
        request.assigned_gpu = target_gpu
        gpu.active_requests.append(request)
        gpu.warm_models.pop(request.model_name, None)
        if request in self.inference_queue:
            self.inference_queue.remove(request)
        if request in self.pending:
            self.pending.remove(request)
        return {
            "success": True,
            "sla_budget_steps": request.sla_budget_steps,
            "warm_model_reused": info.get("warm_model_reused", False),
            "cold_load": info.get("cold_load", False),
            "load_steps": info.get("load_steps", 0),
        }

    def checkpoint_preempt(self, job_id: str) -> dict[str, Any]:
        job, gpus = self._locate_running_training_job(job_id)
        if job is None or not job.preemptible:
            return {"success": False}
        freed_gpus = []
        for gpu in gpus:
            if job in gpu.training_jobs:
                gpu.training_jobs.remove(job)
                gpu.on_job_freed(job.memory_gb)
                freed_gpus.append(gpu.gpu_id)
        job.assigned_gpus.clear()
        job.status = "waiting"
        job.has_checkpoint = True
        job.checkpoint_age = 0
        job.time_remaining += job.checkpoint_cost
        if job not in self.training_queue:
            self.training_queue.append(job)
        self.checkpoint_preempts += 1
        return {"success": True, "freed_gpus": freed_gpus, "priority": job.priority}

    def preempt_kill(self, job_id: str) -> dict[str, Any]:
        job, gpus = self._locate_running_training_job(job_id)
        if job is None:
            return {"success": False}
        lost_progress = job.checkpoint_age if job.has_checkpoint else job.time_elapsed
        job.time_remaining += lost_progress
        job.time_elapsed = max(0, job.time_elapsed - lost_progress)
        job.checkpoint_age = 0
        job.status = "waiting"
        job.assigned_gpus.clear()
        for gpu in gpus:
            if job in gpu.training_jobs:
                gpu.training_jobs.remove(job)
                gpu.on_job_freed(job.memory_gb)
        if job not in self.training_queue:
            self.training_queue.append(job)
        self.kill_preempts += 1
        return {"success": True, "lost_progress": lost_progress, "priority": job.priority}

    def force_unload_model(self, gpu_id: int, model_name: str) -> bool:
        gpu = self.gpus.get(gpu_id)
        if gpu is None:
            return False
        if any(request.model_name == model_name for request in gpu.active_requests):
            return False
        if model_name in gpu.quantizing:
            return False
        memory = gpu.loaded_models.pop(model_name, {}).get("mem", 0.0)
        gpu.warm_models.pop(model_name, None)
        if memory <= 0:
            return False
        gpu.on_job_freed(memory)
        return True

    def request_quantize(self, gpu_id: int, model_name: str, target_precision: str) -> bool:
        gpu = self.gpus.get(gpu_id)
        if gpu is None:
            return False
        return gpu.quantize_model(model_name, target_precision)

    def get_snapshot(self) -> dict[str, Any]:
        avg_utilization = round(self.total_util / max(self.current_step, 1), 3)
        avg_inference_wait = round(self.total_inference_wait / max(self.inference_completed, 1), 3)
        completion_rate = round(len(self.completed) / max(self.total_work_items, 1), 3)
        return {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "gpus": [
                {
                    "gpu_id": gpu.gpu_id,
                    "total_memory_gb": gpu.total_memory_gb,
                    "raw_free_gb": round(gpu.raw_free_gb, 1),
                    "effective_free_gb": round(gpu.effective_free_gb, 1),
                    "fragmentation_score": round(gpu.fragmentation_score, 3),
                    "utilization": round(gpu.utilization, 2),
                    "next_free_in_steps": gpu.next_free_in_steps,
                    "next_free_memory_gb": round(gpu.next_free_memory_gb, 1),
                    "training_jobs": [
                        {
                            "job_id": job.job_id,
                            "memory_gb": job.memory_gb,
                            "time_remaining": job.time_remaining,
                            "priority": job.priority,
                            "has_checkpoint": job.has_checkpoint,
                            "checkpoint_age": job.checkpoint_age,
                            "preemptible": job.preemptible,
                        }
                        for job in gpu.training_jobs
                    ],
                    "inference_requests": [
                        {
                            "request_id": request.request_id,
                            "model_name": request.model_name,
                            "current_kv_gb": round(request.current_kv_gb, 2),
                            "max_kv_gb": request.max_kv_gb,
                            "time_remaining": request.time_remaining,
                            "load_steps_remaining": request.load_steps_remaining,
                        }
                        for request in gpu.active_requests
                    ],
                    "loaded_models": {
                        name: {
                            "mem": round(model["mem"], 1),
                            "precision": model["precision"],
                        }
                        for name, model in gpu.loaded_models.items()
                    },
                    "warm_models": dict(gpu.warm_models),
                    "quantizing": {
                        name: {
                            "target": state["target"],
                            "steps_left": state["steps_left"],
                        }
                        for name, state in gpu.quantizing.items()
                    },
                    "nvlink_peers": list(gpu.nvlink_peers),
                }
                for gpu in self.gpus.values()
            ],
            "training_queue": [
                {
                    "job_id": job.job_id,
                    "memory_gb": job.memory_gb,
                    "estimated_duration": job.estimated_duration,
                    "priority": job.priority,
                    "num_gpus": job.num_gpus,
                    "wait_time": job.wait_time,
                    "has_checkpoint": job.has_checkpoint,
                }
                for job in self.training_queue
            ],
            "inference_queue": [
                {
                    "request_id": request.request_id,
                    "model_name": request.model_name,
                    "model_memory_gb": request.model_memory_gb,
                    "model_precision": request.model_precision,
                    "max_kv_gb": request.max_kv_gb,
                    "estimated_duration": request.estimated_duration,
                    "sla_seconds": request.sla_seconds,
                    "wait_time": request.wait_time,
                    "sla_budget_steps": round(request.sla_budget_steps, 1),
                }
                for request in self.inference_queue
            ],
            "pending": [
                {
                    "type": "training" if isinstance(item, TrainingJob) else "inference",
                    "id": item.job_id if isinstance(item, TrainingJob) else item.request_id,
                    "memory_gb": item.memory_gb if isinstance(item, TrainingJob) else item.model_memory_gb,
                    "priority": item.priority,
                }
                for item in self.pending
            ],
            "metrics": {
                "avg_utilization": avg_utilization,
                "sla_violations": self.sla_violations,
                "completed": len(self.completed),
                "training_completed": self.training_completed,
                "inference_completed": self.inference_completed,
                "avg_inference_wait": avg_inference_wait,
                "oom_count": self.oom_count,
                "kill_preempts": self.kill_preempts,
                "checkpoint_preempts": self.checkpoint_preempts,
                "training_queued": len(self.training_queue),
                "inference_queued": len(self.inference_queue),
                "wait_decisions": self.wait_decisions,
                "smart_waits": self.smart_waits,
                "quantize_count": self.quantize_count,
                "quantize_memory_saved_gb": round(self.quantize_memory_saved, 1),
                "completion_rate": completion_rate,
            },
        }

    @property
    def is_done(self) -> bool:
        future = any(step > self.current_step for step, _ in self.job_sequence[self.arrival_cursor :])
        active = bool(self.training_queue or self.inference_queue)
        running = any(gpu.training_jobs or gpu.active_requests for gpu in self.gpus.values())
        return self.current_step >= self.max_steps or (not future and not active and not running)
