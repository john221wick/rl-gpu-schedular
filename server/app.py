from __future__ import annotations

import argparse
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from gpu_scheduler.baseline import run_baseline
from gpu_scheduler.compat import BaseModel
from gpu_scheduler.env import GPUSchedulerEnv
from gpu_scheduler.models import Action
from gpu_scheduler.tasks import list_tasks


class ResetRequest(BaseModel):
    task_id: str = "easy_001"
    seed: int | None = None


class StepRequest(BaseModel):
    action: Action | None = None


class BaselineRequest(BaseModel):
    task_id: str = "medium_001"
    seed: int | None = None
    max_steps: int | None = None


app = FastAPI(
    title="GPU Job Scheduler for ML Workloads",
    version="1.0.0",
    description="CPU-only RL environment for scheduling training jobs and inference requests across a simulated GPU cluster.",
)
ENV = GPUSchedulerEnv()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "gpu-scheduler-ml",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots() -> str:
    return "User-agent: *\nAllow: /\n"


@app.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    return list_tasks()


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict[str, Any]:
    payload = request or ResetRequest()
    return ENV.reset(task_id=payload.task_id, seed=payload.seed)


@app.get("/state")
def state() -> dict[str, Any]:
    try:
        return ENV.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/state")
def post_state() -> dict[str, Any]:
    return state()


@app.get("/valid-actions")
def valid_actions() -> list[dict[str, Any]]:
    try:
        return ENV.get_valid_actions()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(request: StepRequest | None = None) -> dict[str, Any]:
    payload = request or StepRequest()
    try:
        return ENV.step(payload.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/baseline/{name}")
def baseline(name: str, request: BaselineRequest | None = None) -> dict[str, Any]:
    payload = request or BaselineRequest()
    try:
        return run_baseline(name, task_id=payload.task_id, seed=payload.seed, max_steps=payload.max_steps)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GPU scheduler OpenEnv server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
