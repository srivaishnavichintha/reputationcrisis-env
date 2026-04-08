"""
FastAPI server for Digital Reputation Crisis Manager OpenEnv.

Endpoints:
  POST /reset          - Reset environment, returns initial Observation
  POST /step           - Take action, returns (Observation, Reward, done, info)
  GET  /state          - Get full internal state
  GET  /tasks          - List all tasks
  POST /run_task       - Run a complete episode for a task
  GET  /health         - Health check
"""

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.env.environment import ReputationCrisisEnv
from backend.env.models import Observation, Action, Reward, EnvironmentState
from backend.tasks.tasks import TASKS, run_episode, EpisodeResult
from backend.agents.baseline_agent import create_agent

app = FastAPI(
    title="Digital Reputation Crisis Manager",
    description="OpenEnv-compliant environment for AI-powered PR crisis simulation",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Global Environment Registry (per session)
# ─────────────────────────────────────────────────────────────

_envs: Dict[str, ReputationCrisisEnv] = {}


def _get_env(session_id: str = "default") -> ReputationCrisisEnv:
    if session_id not in _envs:
        env = ReputationCrisisEnv()
        env.reset()  # always initialize so /step is safe without a prior /reset
        _envs[session_id] = env
    return _envs[session_id]


# ─────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    session_id: str = "default"
    task_name: Optional[str] = None
    noise_seed: Optional[int] = None


class StepRequest(BaseModel):
    session_id: str = "default"
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class RunTaskRequest(BaseModel):
    task_name: str
    agent_mode: str = "rule_based"
    noise_seed: Optional[int] = 42


class TaskInfo(BaseModel):
    name: str
    description: str
    difficulty: str
    max_steps: int
    success_criteria: Dict[str, Any]


class EpisodeResultResponse(BaseModel):
    task_name: str
    steps: int
    total_reward: float
    grader_score: float
    grader_breakdown: Dict[str, float]
    success: bool
    misinformation_neutralized: bool
    action_history: list
    final_observation: Observation


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest):
    """
    Reset the environment.
    Optionally specify a task_name to use that task's scenario config.
    """
    if request.task_name and request.task_name in TASKS:
        task = TASKS[request.task_name]
        env = ReputationCrisisEnv(
            max_steps=task.max_steps,
            noise_seed=request.noise_seed,
            scenario_config=task.scenario_config,
        )
    else:
        env = ReputationCrisisEnv(
            noise_seed=request.noise_seed,
        )

    _envs[request.session_id] = env
    obs = env.reset()
    return obs


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """Take an action in the environment."""
    env = _get_env(request.session_id)

    try:
        obs, reward, done, info = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=EnvironmentState)
async def get_state(session_id: str = "default"):
    """Get full internal environment state."""
    env = _get_env(session_id)
    return env.state()


@app.get("/tasks", response_model=Dict[str, TaskInfo])
async def list_tasks():
    """List all available tasks."""
    return {
        name: TaskInfo(
            name=task.name,
            description=task.description,
            difficulty=task.difficulty,
            max_steps=task.max_steps,
            success_criteria=task.success_criteria,
        )
        for name, task in TASKS.items()
    }


@app.post("/run_task", response_model=EpisodeResultResponse)
async def run_task(request: RunTaskRequest):
    """Run a complete episode for a given task with the baseline agent."""
    if request.task_name not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{request.task_name}' not found. Available: {list(TASKS.keys())}"
        )

    agent = create_agent(request.agent_mode)
    result: EpisodeResult = run_episode(
        task_name=request.task_name,
        agent_fn=agent,
        noise_seed=request.noise_seed,
    )

    return EpisodeResultResponse(
        task_name=result.task_name,
        steps=result.steps,
        total_reward=round(result.total_reward, 4),
        grader_score=result.grader_score,
        grader_breakdown=result.grader_breakdown,
        success=result.success,
        misinformation_neutralized=result.misinformation_neutralized,
        action_history=result.action_history,
        final_observation=result.final_observation,
    )


@app.get("/")
async def root():
    return {
        "name": "Digital Reputation Crisis Manager",
        "description": "OpenEnv-compliant PR crisis simulation environment",
        "docs": "/docs",
        "tasks": list(TASKS.keys()),
    }
