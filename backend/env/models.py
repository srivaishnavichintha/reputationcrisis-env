"""
Pydantic models for the Digital Reputation Crisis Manager OpenEnv environment.
Strict OpenEnv spec compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class CrisisLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ActionType(str, Enum):
    IGNORE = "ignore"
    APOLOGY = "apology"
    CLARIFICATION = "clarification"
    PR_CAMPAIGN = "pr_campaign"
    INFLUENCER_ENGAGEMENT = "influencer_engagement"
    LEGAL_ACTION = "legal_action"


class Observation(BaseModel):
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Public sentiment (-1=very negative, +1=very positive)")
    crisis_level: CrisisLevel = Field(..., description="Current crisis severity")
    trending_topics: list[str] = Field(..., description="Currently trending topics related to crisis")
    public_trust: float = Field(..., ge=0.0, le=1.0, description="Public trust in the brand (0=none, 1=full)")
    virality_index: float = Field(..., ge=0.0, le=1.0, description="How viral the crisis is (0=none, 1=fully viral)")
    time_step: int = Field(..., ge=0, description="Current simulation time step")

    class Config:
        use_enum_values = True


class Action(BaseModel):
    type: ActionType = Field(..., description="The PR action to take")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional action parameters")

    class Config:
        use_enum_values = True


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Reward value for this step")
    reason: str = Field(..., description="Human-readable explanation of reward")
    breakdown: Optional[Dict[str, float]] = Field(default_factory=dict, description="Component breakdown of reward")


class EnvironmentState(BaseModel):
    """Full internal state for debugging and transparency."""
    observation: Observation
    cumulative_reward: float
    episode_step: int
    total_steps: int
    done: bool
    misinformation_active: bool
    influencer_credibility: float
    last_action: Optional[str]
    action_history: list[str]
    trust_history: list[float]
    sentiment_history: list[float]
    virality_history: list[float]
    events_log: list[str]
