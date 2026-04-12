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
# CLAMPING — used at EVERY score return point.
#
# FIXED: eps=1e-4 (NOT 1e-6).
#   round(1e-6,  4) = 0.0      ← INVALID (fails validator)
#   round(1e-4,  4) = 0.0001   ← valid ✓
#   round(0.9999,4) = 0.9999   ← valid ✓
#   round(1.0-1e-6,4) = 1.0    ← INVALID (fails validator)
#   round(1.0-1e-4,4) = 0.9999 ← valid ✓
# ─────────────────────────────────────────────────────────────

_LO: float = 1e-4        # FIXED: was 1e-6 — round(1e-6,4)=0.0 (invalid)
_HI: float = 1.0 - 1e-4  # FIXED: was 1-1e-6 — round(0.999999,4)=1.0 (invalid)


def safe_score(score) -> float:
    """
    Clamp to strictly-open interval (0, 1).
    Handles None, NaN, ±inf, bool, non-numeric → returns 0.5.

    FIXED: eps=1e-4 so round(result, 4) also stays strictly in (0, 1).
           Previous eps=1e-6 caused round(1e-6,4)=0.0 which the validator rejects.
    """
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.5
    if score != score:                                   # NaN
        return 0.5
    if score == float("inf") or score == float("-inf"):
        return 0.5
    eps = _LO  # FIXED: 1e-4
    return max(eps, min(1.0 - eps, score))


def _c(v: float) -> float:
    """Clamp any sub-score — always delegates to safe_score."""
    try:
        v = float(v)
    except Exception:
        return 0.5
    return safe_score(v)


# Legacy aliases — keep so nothing breaks if external code imports them
_safe_score  = safe_score
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
    score, breakdown = grader(task, state, env)
    # FIXED: triple-safety clamp at dispatcher level — catches any grader edge case
    score = safe_score(score)
    return score, breakdown


# ─────────────────────────────────────────────────────────────
# Task 1 grader — Sentiment Stabilization
# ─────────────────────────────────────────────────────────────

def _grade_task1(
    task: TaskDefinition, state, env
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    # --- avg sentiment score ---
    sentiments     = state.sentiment_history or [-1.0]
    avg_sent       = sum(sentiments) / max(len(sentiments), 1)  # FIXED: guard /0
    avg_sent_score = (avg_sent + 1.0) / 2.0           # [-1,1] -> [0,1]
    if sentiments[-1] > 0:
        avg_sent_score += 0.10
    avg_sent_score = _c(avg_sent_score)                # FIXED: clamp sub-score
    breakdown["avg_sentiment_score"] = round(avg_sent_score, 4)

    # --- trust recovery score ---
    final_trust    = float(state.trust_history[-1]) if state.trust_history else _LO
    initial_trust  = float(task.scenario_config["initial_trust"])
    trust_recovery = 0.5 + (final_trust - initial_trust)
    if final_trust < task.success_criteria["min_final_trust"]:
        trust_recovery *= 0.5
    trust_recovery = _c(trust_recovery)                # FIXED: clamp sub-score
    breakdown["trust_recovery_score"] = round(trust_recovery, 4)

    # --- response efficiency score ---
    action_history   = state.action_history or []
    ignore_count     = action_history.count("ignore")
    total_actions    = max(len(action_history), 1)     # FIXED: guard /0
    ignore_ratio     = ignore_count / total_actions
    first_non_ignore = next(
        (i for i, a in enumerate(action_history) if a != "ignore"),
        total_actions,
    )
    delay_penalty       = min(0.3, first_non_ignore * 0.05)
    response_efficiency = _c(1.0 - ignore_ratio * 2.0 - delay_penalty)  # FIXED: clamp
    breakdown["response_efficiency_score"] = round(response_efficiency, 4)

    # --- weighted sum ---
    w = task.grader_weights
    raw_score = (
        w["avg_sentiment"]         * avg_sent_score
        + w["trust_recovery"]      * trust_recovery
        + w["response_efficiency"] * response_efficiency
    )
    # FIXED: safe_score at FINAL RETURN — bulletproof regardless of inputs
    score = safe_score(raw_score)
    return score, breakdown


# ─────────────────────────────────────────────────────────────
# Task 2 grader — Viral Outrage Control
# ─────────────────────────────────────────────────────────────

def _grade_task2(
    task: TaskDefinition, state, env
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    # --- virality reduction score ---
    initial_virality = float(task.scenario_config["initial_virality"])
    # FIXED: guard against division by zero if initial_virality is somehow 0
    if initial_virality <= 0:
        initial_virality = _LO
    final_virality = float(state.virality_history[-1]) if state.virality_history else 1.0
    virality_score = (initial_virality - final_virality) / initial_virality

    if final_virality <= task.success_criteria["max_final_virality"]:
        virality_score += 0.15
    else:
        overshoot       = final_virality - task.success_criteria["max_final_virality"]
        virality_score -= overshoot * 1.5
    virality_score = max(-1.0, min(1.0, virality_score))
    virality_score = _c(virality_score)                # FIXED: clamp sub-score
    breakdown["virality_reduction_score"] = round(virality_score, 4)

    # --- trust recovery score ---
    # FIXED: raw trust from env can be exactly 0.0; clamp before multiplying
    final_trust = _c(float(state.trust_history[-1])) if state.trust_history else _LO
    trust_score = final_trust
    if final_trust < task.success_criteria["min_final_trust"]:
        trust_score *= 0.6
    trust_score = _c(trust_score)                      # FIXED: clamp after penalty
    breakdown["trust_recovery_score"] = round(trust_score, 4)

    # --- response time score ---
    action_history = state.action_history or []
    first_real = next(
        (i for i, a in enumerate(action_history) if a != "ignore"),
        len(action_history),
    )
    if first_real <= 1:
        rt = 0.95
    elif first_real <= 3:
        rt = 0.75
    elif first_real <= 5:
        rt = 0.50
    else:
        rt = max(0.10, 0.50 - (first_real - 5) * 0.05)

    ignore_count        = action_history.count("ignore")
    response_time_score = _c(rt - ignore_count * 0.03)  # FIXED: clamp sub-score
    breakdown["response_time_score"] = round(response_time_score, 4)

    # --- weighted sum ---
    w = task.grader_weights
    raw_score = (
        w["virality_reduction"] * virality_score
        + w["trust_recovery"]  * trust_score
        + w["response_time"]   * response_time_score
    )
    # FIXED: safe_score at FINAL RETURN — bulletproof regardless of inputs
    score = safe_score(raw_score)
    return score, breakdown


# ─────────────────────────────────────────────────────────────
# Task 3 grader — Misinformation Crisis
# ─────────────────────────────────────────────────────────────

def _grade_task3(
    task: TaskDefinition, state, env
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, float] = {}

    # --- misinformation control score ---
    misinfo_neutralized = not state.misinformation_active
    action_history      = state.action_history or []
    clarification_steps = [
        i for i, a in enumerate(action_history) if a == "clarification"
    ]
    if misinfo_neutralized:
        first_clarify = (
            clarification_steps[0] if clarification_steps
            else len(action_history)
        )
        total = max(len(action_history), 1)            # FIXED: guard /0
        # safe_score ensures speed_bonus is strictly < 1.0
        speed_bonus = safe_score(1.0 - first_clarify / total)
        # 0.70 + 0.30 * speed_bonus < 0.70 + 0.30 * 1.0 = 1.0 always
        misinfo_raw = 0.70 + 0.30 * speed_bonus
    else:
        clarify_attempts = len(clarification_steps)
        misinfo_raw      = min(0.15, 0.05 + clarify_attempts * 0.03)
        # FIXED: ensure misinfo_raw never hits 0.0 when no attempts
        misinfo_raw = max(_LO, misinfo_raw)

    misinfo_score = _c(misinfo_raw)                    # FIXED: clamp sub-score
    breakdown["misinformation_control_score"] = round(misinfo_score, 4)

    # --- trust stability score ---
    trust_history = state.trust_history or [_LO]
    min_trust     = float(min(trust_history))
    final_trust   = float(trust_history[-1])

    if min_trust >= 0.30:
        ts = min_trust * 0.6 + final_trust * 0.4
    elif min_trust >= 0.20:
        ts = (min_trust - 0.10) * 0.5
    else:
        ts = _LO                                       # FIXED: never literal 0

    if len(trust_history) > 2:
        mean_t   = sum(trust_history) / len(trust_history)
        variance = sum((t - mean_t) ** 2 for t in trust_history) / len(trust_history)
        ts      -= variance * 2.0

    trust_stability_score = _c(ts)                     # FIXED: clamp sub-score
    breakdown["trust_stability_score"] = round(trust_stability_score, 4)

    # --- decision quality score ---
    total            = max(len(action_history), 1)     # FIXED: guard /0
    good_actions     = {"clarification", "apology", "pr_campaign"}
    bad_actions_here = {"ignore", "legal_action"}
    good_count = sum(1 for a in action_history if a in good_actions)
    bad_count  = sum(1 for a in action_history if a in bad_actions_here)
    dq = good_count / total - bad_count / total * 0.5
    if not clarification_steps:
        dq *= 0.6
    decision_score = _c(dq)                            # FIXED: clamp sub-score
    breakdown["decision_quality_score"] = round(decision_score, 4)

    # --- weighted sum ---
    w = task.grader_weights
    raw_score = (
        w["misinformation_control"] * misinfo_score
        + w["trust_stability"]      * trust_stability_score
        + w["decision_quality"]     * decision_score
    )
    # FIXED: safe_score at FINAL RETURN — bulletproof regardless of inputs
    score = safe_score(raw_score)
    return score, breakdown


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
