"""
Task definitions and deterministic graders for the
Digital Reputation Crisis Manager OpenEnv environment.

TASK 1 (EASY):   Sentiment Stabilization
TASK 2 (MEDIUM): Viral Outrage Control
TASK 3 (HARD):   Misinformation Crisis
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from backend.env.environment import ReputationCrisisEnv
from backend.env.models import Observation, Action, ActionType


# ─────────────────────────────────────────────────────────────
# ONE clamping function used everywhere.
# Guarantees result is STRICTLY inside (0, 1):
#   never 0.0, never 1.0, no matter what value goes in.
# ─────────────────────────────────────────────────────────────

_LO: float = 1e-4   # 0.0001
_HI: float = 0.9999

def _c(v: float) -> float:
    """Clamp to strict open interval (0, 1)."""
    return float(max(_LO, min(_HI, v)))

# Legacy aliases so nothing breaks if external code imports them
_clamp_sub   = _c
_clamp_score = _c


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class TaskDefinition:
    name: str
    description: str
    difficulty: str
    scenario_config: Dict[str, Any]
    max_steps: int
    success_criteria: Dict[str, Any]
    grader_weights: Dict[str, float]


@dataclass
class EpisodeResult:
    task_name: str
    steps: int
    final_observation: Observation
    total_reward: float
    action_history: list
    trust_history: list
    sentiment_history: list
    virality_history: list
    events_log: list
    grader_score: float
    grader_breakdown: Dict[str, float]
    success: bool
    misinformation_neutralized: bool


# ─────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────

TASKS: Dict[str, TaskDefinition] = {

    "task_1_sentiment_stabilization": TaskDefinition(
        name="task_1_sentiment_stabilization",
        description=(
            "A wave of negative tweets has hit your brand after a minor product issue. "
            "Sentiment is mildly negative, virality is low. "
            "Goal: Stabilize public sentiment above 0.0 while maintaining trust."
        ),
        difficulty="EASY",
        scenario_config={
            "initial_sentiment": -0.35,
            "initial_trust": 0.70,
            "initial_virality": 0.20,
            "misinformation_active": False,
            "influencer_credibility": 0.65,
        },
        max_steps=15,
        success_criteria={
            "min_final_sentiment": 0.0,
            "min_final_trust": 0.55,
        },
        grader_weights={
            "avg_sentiment": 0.45,
            "trust_recovery": 0.35,
            "response_efficiency": 0.20,
        },
    ),

    "task_2_viral_outrage_control": TaskDefinition(
        name="task_2_viral_outrage_control",
        description=(
            "Viral outrage has erupted over a controversial business decision. "
            "Mixed sentiment, high virality index. "
            "Goal: Reduce virality below 0.3 AND recover trust above 0.5."
        ),
        difficulty="MEDIUM",
        scenario_config={
            "initial_sentiment": -0.55,
            "initial_trust": 0.55,
            "initial_virality": 0.65,
            "misinformation_active": False,
            "influencer_credibility": 0.60,
        },
        max_steps=20,
        success_criteria={
            "max_final_virality": 0.30,
            "min_final_trust": 0.50,
        },
        grader_weights={
            "virality_reduction": 0.40,
            "trust_recovery": 0.35,
            "response_time": 0.25,
        },
    ),

    "task_3_misinformation_crisis": TaskDefinition(
        name="task_3_misinformation_crisis",
        description=(
            "A coordinated misinformation campaign with influencer amplification "
            "is causing rapid trust collapse. "
            "Goal: Prevent trust falling below 0.3 and neutralize misinformation."
        ),
        difficulty="HARD",
        scenario_config={
            "initial_sentiment": -0.70,
            "initial_trust": 0.50,
            "initial_virality": 0.75,
            "misinformation_active": True,
            "influencer_credibility": 0.45,
        },
        max_steps=20,
        success_criteria={
            "min_final_trust": 0.30,
            "misinformation_neutralized": True,
        },
        grader_weights={
            "misinformation_control": 0.35,
            "trust_stability": 0.40,
            "decision_quality": 0.25,
        },
    ),
}


# ─────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────

def run_episode(
    task_name: str,
    agent_fn,
    noise_seed: Optional[int] = 42,
) -> EpisodeResult:
    task = TASKS[task_name]
    env = ReputationCrisisEnv(
        max_steps=task.max_steps,
        noise_seed=noise_seed,
        scenario_config=task.scenario_config,
    )

    obs = env.reset()
    info: Dict = {}
    done = False

    while not done:
        action = agent_fn(obs, info)
        obs, reward, done, info = env.step(action)

    full_state = env.state()
    score, breakdown = grade_episode(task, full_state, env)

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
        success=_check_success(task, obs, full_state),
        misinformation_neutralized=not full_state.misinformation_active,
    )


# ─────────────────────────────────────────────────────────────
# Grade dispatcher
# ─────────────────────────────────────────────────────────────

def grade_episode(
    task: TaskDefinition,
    state,
    env: ReputationCrisisEnv,
) -> Tuple[float, Dict[str, float]]:
    graders = {
        "task_1_sentiment_stabilization": _grade_task1,
        "task_2_viral_outrage_control":   _grade_task2,
        "task_3_misinformation_crisis":   _grade_task3,
    }
    grader = graders.get(task.name)
    if not grader:
        raise ValueError(f"No grader found for task: {task.name}")
    return grader(task, state, env)


# ─────────────────────────────────────────────────────────────
# Task 1 grader
# ─────────────────────────────────────────────────────────────

def _grade_task1(
    task: TaskDefinition, state, env
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    # avg sentiment score
    sentiments     = state.sentiment_history or [-1.0]
    avg_sent       = sum(sentiments) / len(sentiments)
    avg_sent_score = (avg_sent + 1.0) / 2.0        # [-1,1] -> [0,1]
    if sentiments[-1] > 0:
        avg_sent_score += 0.10
    avg_sent_score = _c(avg_sent_score)
    breakdown["avg_sentiment_score"] = round(avg_sent_score, 4)

    # trust recovery score
    final_trust    = float(state.trust_history[-1]) if state.trust_history else 0.0
    initial_trust  = float(task.scenario_config["initial_trust"])
    trust_recovery = 0.5 + (final_trust - initial_trust)
    if final_trust < task.success_criteria["min_final_trust"]:
        trust_recovery *= 0.5
    trust_recovery = _c(trust_recovery)
    breakdown["trust_recovery_score"] = round(trust_recovery, 4)

    # response efficiency score
    ignore_count     = state.action_history.count("ignore")
    total_actions    = len(state.action_history) or 1
    ignore_ratio     = ignore_count / total_actions
    first_non_ignore = next(
        (i for i, a in enumerate(state.action_history) if a != "ignore"),
        total_actions,
    )
    delay_penalty       = min(0.3, first_non_ignore * 0.05)
    response_efficiency = _c(1.0 - ignore_ratio * 2.0 - delay_penalty)
    breakdown["response_efficiency_score"] = round(response_efficiency, 4)

    # weighted sum
    w     = task.grader_weights
    score = _c(
        w["avg_sentiment"]         * avg_sent_score
        + w["trust_recovery"]      * trust_recovery
        + w["response_efficiency"] * response_efficiency
    )
    return round(score, 4), breakdown


# ─────────────────────────────────────────────────────────────
# Task 2 grader
# ─────────────────────────────────────────────────────────────

def _grade_task2(
    task: TaskDefinition, state, env
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    # virality reduction score
    initial_virality = float(task.scenario_config["initial_virality"])
    final_virality   = float(state.virality_history[-1]) if state.virality_history else 1.0
    virality_score   = (initial_virality - final_virality) / initial_virality  # raw

    if final_virality <= task.success_criteria["max_final_virality"]:
        virality_score += 0.15
    else:
        overshoot       = final_virality - task.success_criteria["max_final_virality"]
        virality_score -= overshoot * 1.5
    virality_score = _c(virality_score)
    breakdown["virality_reduction_score"] = round(virality_score, 4)

    # trust recovery score
    # KEY FIX: raw final_trust can be exactly 0.0 from the environment.
    # In the original code trust_score entered the weighted sum unclamped,
    # pushing the final score to exactly 0.0 which the validator rejects.
    final_trust = float(state.trust_history[-1]) if state.trust_history else 0.0
    trust_score = final_trust
    if final_trust < task.success_criteria["min_final_trust"]:
        trust_score *= 0.6
    trust_score = _c(trust_score)          # ← this line was missing
    breakdown["trust_recovery_score"] = round(trust_score, 4)

    # response time score
    first_real = next(
        (i for i, a in enumerate(state.action_history) if a != "ignore"),
        len(state.action_history),
    )
    if first_real <= 1:
        rt = 0.95
    elif first_real <= 3:
        rt = 0.75
    elif first_real <= 5:
        rt = 0.50
    else:
        rt = max(0.10, 0.50 - (first_real - 5) * 0.05)

    ignore_count        = state.action_history.count("ignore")
    response_time_score = _c(rt - ignore_count * 0.03)
    breakdown["response_time_score"] = round(response_time_score, 4)

    # weighted sum
    w     = task.grader_weights
    score = _c(
        w["virality_reduction"] * virality_score
        + w["trust_recovery"]  * trust_score
        + w["response_time"]   * response_time_score
    )
    return round(score, 4), breakdown


# ─────────────────────────────────────────────────────────────
# Task 3 grader
# ─────────────────────────────────────────────────────────────

def _grade_task3(
    task: TaskDefinition, state, env
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    # misinformation control score
    misinfo_neutralized = not state.misinformation_active
    clarification_steps = [
        i for i, a in enumerate(state.action_history) if a == "clarification"
    ]
    if misinfo_neutralized:
        first_clarify = (
            clarification_steps[0] if clarification_steps
            else len(state.action_history)
        )
        total       = len(state.action_history) or 1
        speed_bonus = _c(1.0 - first_clarify / total)
        misinfo_raw = 0.70 + 0.30 * speed_bonus    # can approach 1.0
    else:
        clarify_attempts = len(clarification_steps)
        misinfo_raw      = min(0.15, 0.05 + clarify_attempts * 0.03)

    misinfo_score = _c(misinfo_raw)    # clamp in BOTH branches
    breakdown["misinformation_control_score"] = round(misinfo_score, 4)

    # trust stability score
    trust_history = state.trust_history or [0.0]
    min_trust     = float(min(trust_history))
    final_trust   = float(trust_history[-1])

    if min_trust >= 0.30:
        ts = min_trust * 0.6 + final_trust * 0.4
    elif min_trust >= 0.20:
        ts = (min_trust - 0.10) * 0.5
    else:
        ts = _LO    # worst case: use _LO constant, never literal 0 or 0.001

    if len(trust_history) > 2:
        mean_t   = sum(trust_history) / len(trust_history)
        variance = sum((t - mean_t) ** 2 for t in trust_history) / len(trust_history)
        ts      -= variance * 2.0

    trust_stability_score = _c(ts)
    breakdown["trust_stability_score"] = round(trust_stability_score, 4)

    # decision quality score
    actions          = state.action_history
    total            = len(actions) or 1
    good_actions     = {"clarification", "apology", "pr_campaign"}
    bad_actions_here = {"ignore", "legal_action"}
    good_count = sum(1 for a in actions if a in good_actions)
    bad_count  = sum(1 for a in actions if a in bad_actions_here)
    dq = good_count / total - bad_count / total * 0.5
    if not clarification_steps:
        dq *= 0.6
    decision_score = _c(dq)
    breakdown["decision_quality_score"] = round(decision_score, 4)

    # weighted sum
    w     = task.grader_weights
    score = _c(
        w["misinformation_control"] * misinfo_score
        + w["trust_stability"]      * trust_stability_score
        + w["decision_quality"]     * decision_score
    )
    return round(score, 4), breakdown


# ─────────────────────────────────────────────────────────────
# Success check
# ─────────────────────────────────────────────────────────────

def _check_success(task: TaskDefinition, obs: Observation, state) -> bool:
    criteria = task.success_criteria
    if "min_final_sentiment" in criteria:
        if obs.sentiment_score < criteria["min_final_sentiment"]:
            return False
    if "min_final_trust" in criteria:
        if obs.public_trust < criteria["min_final_trust"]:
            return False
    if "max_final_virality" in criteria:
        if obs.virality_index > criteria["max_final_virality"]:
            return False
    if criteria.get("misinformation_neutralized") and state.misinformation_active:
        return False
    return True
