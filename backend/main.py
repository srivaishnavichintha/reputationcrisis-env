"""
FastAPI server for Digital Reputation Crisis Manager OpenEnv.

Endpoints (all registered at BOTH bare path AND /api/ prefix):
  POST /reset          - Reset environment, returns initial Observation
  POST /api/reset      - Same (OpenEnv validator calls this path)  # FIXED
  POST /step           - Take action, returns (Observation, Reward, done, info)
  POST /api/step       - Same (OpenEnv validator calls this path)  # FIXED
  GET  /state          - Get full internal state
  GET  /tasks          - List all tasks
  POST /run_task       - Run a complete episode for a task
  GET  /health         - Health check
"""

import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.env.environment import ReputationCrisisEnv
from backend.env.models import Observation, Action, Reward, EnvironmentState
from backend.tasks.tasks import TASKS, run_episode, EpisodeResult, safe_score, grade_episode
from backend.agents.baseline_agent import create_agent

app = FastAPI(
    title="Digital Reputation Crisis Manager",
    description="OpenEnv-compliant environment for AI-powered PR crisis simulation",
    version="1.0.0",
)

ASSETS_DIR = "frontend/dist/assets"
FRONTEND_INDEX = "frontend/dist/index.html"

if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

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
_session_tasks: Dict[str, Optional[str]] = {}   # session_id → task_name


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
    agent_mode: str = "llm_with_fallback"
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
# Shared handler logic (reused by both /reset and /api/reset)
# ─────────────────────────────────────────────────────────────

async def _handle_reset(request: Request):
    """
    Core reset logic — called by both /reset and /api/reset.
    # FIXED: extracted into shared function so /api/reset is not a stub.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = body.get("session_id", "default")
    task_name  = body.get("task_name")
    noise_seed = body.get("noise_seed")

    try:
        if task_name and task_name in TASKS:
            task = TASKS[task_name]
            env = ReputationCrisisEnv(
                max_steps=task.max_steps,
                noise_seed=noise_seed,
                scenario_config=task.scenario_config,
            )
        else:
            env = ReputationCrisisEnv(noise_seed=noise_seed)

        _envs[session_id] = env
        _session_tasks[session_id] = task_name   # remember which task this session is running
        obs = env.reset()

        return {
            "sentiment_score": float(obs.sentiment_score),
            "crisis_level":    str(obs.crisis_level),
            "trending_topics": list(obs.trending_topics),
            "public_trust":    float(obs.public_trust),
            "virality_index":  float(obs.virality_index),
            "time_step":       int(obs.time_step),
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "traceback": traceback.format_exc()
        })


async def _handle_step(request: StepRequest):
    """
    Core step logic — called by both /step and /api/step.
    # FIXED: extracted into shared function so /api/step is not a stub.
    """
    env = _get_env(request.session_id)

    try:
        obs, reward, done, info = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Inject graded score into info so the HF validator can read it.
    # FIXED: also injects score for /api/step path.
    if done:
        task_name = _session_tasks.get(request.session_id)
        if task_name and task_name in TASKS:
            full_state = env.state()
            score_val, breakdown = grade_episode(TASKS[task_name], full_state, env)
            info["score"]            = safe_score(score_val)
            info["grader_score"]     = safe_score(score_val)
            info["grader_breakdown"] = breakdown
        else:
            info["score"] = safe_score(info.get("cumulative_reward", 0.5))

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


# ─────────────────────────────────────────────────────────────
# Endpoints — bare paths (legacy / openenv.yaml)
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset")
async def reset(request: Request):
    return await _handle_reset(request)


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    return await _handle_step(request)


# ─────────────────────────────────────────────────────────────
# Endpoints — /api/ prefix  (OpenEnv validator calls these)
# FIXED: these were missing entirely, causing 405 Method Not Allowed
# ─────────────────────────────────────────────────────────────

@app.post("/api/reset")
async def api_reset(request: Request):
    """POST /api/reset — identical to POST /reset. # FIXED: was missing."""
    return await _handle_reset(request)


@app.post("/api/step", response_model=StepResponse)
async def api_step(request: StepRequest):
    """POST /api/step — identical to POST /step. # FIXED: was missing."""
    return await _handle_step(request)


@app.get("/api/health")
async def api_health():
    """GET /api/health — mirrors /health. # FIXED: added for completeness."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/tasks")
async def api_list_tasks():
    """GET /api/tasks — mirrors /tasks. # FIXED: added for completeness."""
    return {
        name: {
            "name": task.name,
            "description": task.description,
            "difficulty": task.difficulty,
            "max_steps": task.max_steps,
            "success_criteria": task.success_criteria,
        }
        for name, task in TASKS.items()
    }


@app.get("/api/state")
async def api_get_state(session_id: str = "default"):
    """GET /api/state — mirrors /state. # FIXED: added for completeness."""
    env = _get_env(session_id)
    return env.state()


# ─────────────────────────────────────────────────────────────
# Remaining original endpoints
# ─────────────────────────────────────────────────────────────

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

    api_key = os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    effective_mode = "llm_with_fallback" if api_key else request.agent_mode
    agent = create_agent(effective_mode)
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


@app.get("/debug")
async def debug():
    import sys, os
    return {
        "python": sys.version,
        "cwd": os.getcwd(),
        "files": os.listdir("."),
        "frontend_dist_exists": os.path.exists("frontend/dist"),
        "frontend_index_exists": os.path.exists("frontend/dist/index.html"),
    }


@app.get("/")
def serve_frontend():
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {"status": "API running", "tasks": list(TASKS.keys())}


@app.get("/{full_path:path}")
def serve_all(full_path: str):
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {"status": "API running, frontend not built"}
