"""
inference.py - Digital Reputation Crisis Manager
OpenEnv Inference Script

Runs all 3 tasks using the baseline (rule-based + LLM hybrid) agent.
Logs in STRICT OpenEnv format.

Usage:
    python inference.py
    python inference.py --agent rule_based
    python inference.py --agent llm
    python inference.py --seed 42
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI

from backend.env.environment import ReputationCrisisEnv
from backend.env.models import Action, ActionType, CrisisLevel
from backend.tasks.tasks import TASKS, run_episode, EpisodeResult
from backend.agents.baseline_agent import create_agent, RuleBasedAgent, LLMAgent


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ─────────────────────────────────────────────────────────────
# Verbose Episode Runner (for inference logging)
# ─────────────────────────────────────────────────────────────

def run_task_verbose(task_name: str, agent_fn, noise_seed: int = 42) -> EpisodeResult:
    """Run a task and emit strict OpenEnv step logs."""
    from backend.tasks.tasks import grade_episode

    task = TASKS[task_name]
    env = ReputationCrisisEnv(
        max_steps=task.max_steps,
        noise_seed=noise_seed,
        scenario_config=task.scenario_config,
    )

    print(f"\n[START]")
    print(f"task: {task_name}")
    print(f"difficulty: {task.difficulty}")
    print(f"description: {task.description}")

    obs = env.reset()
    info = {}
    done = False
    step_logs = []

    while not done:
        action = agent_fn(obs, info)
        obs, reward, done, info = env.step(action)

        step_log = {
            "step": obs.time_step,
            "action": action.type if isinstance(action.type, str) else action.type.value,
            "reward": reward.value,
            "state": {
                "sentiment_score": obs.sentiment_score,
                "crisis_level": obs.crisis_level,
                "public_trust": obs.public_trust,
                "virality_index": obs.virality_index,
                "time_step": obs.time_step,
            },
            "reward_reason": reward.reason,
        }
        step_logs.append(step_log)

        print(f"\n[STEP]")
        print(f"action: {step_log['action']}")
        print(f"reward: {step_log['reward']}")
        print(f"state: {json.dumps(step_log['state'], indent=2)}")

    full_state = env.state()
    score, breakdown = grade_episode(task, full_state, env)

    from backend.tasks.tasks import _check_success
    success = _check_success(task, obs, full_state)

    print(f"\n[END]")
    print(f"final_score: {score}")
    print(f"success: {success}")
    print(f"grader_breakdown: {json.dumps(breakdown, indent=2)}")
    print(f"total_steps: {full_state.episode_step}")
    print(f"cumulative_reward: {round(full_state.cumulative_reward, 4)}")
    print("-" * 60)

    return EpisodeResult(
        task_name=task_name,
        steps=full_state.episode_step,
        final_observation=obs,
        total_reward=full_state.cumulative_reward,
        action_history=full_state.action_history,
        trust_history=full_state.trust_history,
        sentiment_history=full_state.sentiment_history,
        virality_history=full_state.virality_history,
        events_log=full_state.events_log,
        grader_score=score,
        grader_breakdown=breakdown,
        success=success,
        misinformation_neutralized=not full_state.misinformation_active,
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digital Reputation Crisis Manager - Inference")
    parser.add_argument("--agent", default="rule_based", choices=["rule_based", "llm", "llm_with_fallback"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific tasks to run")
    args = parser.parse_args()

    print("=" * 60)
    print("  DIGITAL REPUTATION CRISIS MANAGER")
    print("  OpenEnv Inference Script")
    print(f"  Agent: {args.agent}")
    print(f"  Seed: {args.seed}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Configure LLM if needed
    if args.agent in ("llm", "llm_with_fallback"):
        if not OPENAI_API_KEY:
            print("[WARNING] OPENAI_API_KEY not set. Falling back to rule-based.")
            args.agent = "rule_based"
        else:
            print(f"[INFO] Using LLM: {MODEL_NAME} at {API_BASE_URL}")

    task_names = args.tasks or list(TASKS.keys())
    results = []

    for task_name in task_names:
        if task_name not in TASKS:
            print(f"[WARNING] Unknown task: {task_name}. Skipping.")
            continue

        # Create fresh agent per task (reset state)
        agent = create_agent(args.agent)
        result = run_task_verbose(task_name, agent, noise_seed=args.seed)
        results.append(result)

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)

    total_score = 0.0
    for result in results:
        task_def = TASKS[result.task_name]
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"\n  {status} | {result.task_name}")
        print(f"         Difficulty: {task_def.difficulty}")
        print(f"         Score:      {result.grader_score:.4f}")
        print(f"         Steps:      {result.steps}")
        print(f"         Misinfo:    {'NEUTRALIZED' if result.misinformation_neutralized else 'ACTIVE'}")
        total_score += result.grader_score

    avg_score = total_score / len(results) if results else 0.0
    print(f"\n  OVERALL AVERAGE SCORE: {avg_score:.4f}")
    print("=" * 60)

    return avg_score


if __name__ == "__main__":
    main()
