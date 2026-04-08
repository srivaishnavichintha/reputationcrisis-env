"""
Baseline Inference Agent for Digital Reputation Crisis Manager.

Implements a rule-based + LLM hybrid strategy:
- HIGH crisis  → apology + PR campaign
- MEDIUM       → clarification + influencer engagement
- LOW          → monitor (clarification if trending)

Also supports OpenAI API integration for LLM-guided decisions.
"""

import os
import json
from typing import Optional, Tuple
from backend.env.models import Observation, Action, ActionType, CrisisLevel


# ─────────────────────────────────────────────────────────────
# RULE-BASED AGENT
# ─────────────────────────────────────────────────────────────

class RuleBasedAgent:
    """
    Deterministic rule-based agent that follows a priority decision tree.
    
    Decision logic:
    1. If misinformation active → clarification (top priority)
    2. If HIGH crisis + high virality → apology
    3. If HIGH crisis → PR campaign
    4. If MEDIUM crisis + high virality → PR campaign
    5. If MEDIUM crisis → clarification
    6. If LOW crisis + negative sentiment → clarification
    7. Otherwise → ignore (monitor)
    """

    def __init__(self):
        self.step_count = 0
        self.last_action: Optional[str] = None
        self.consecutive_same: int = 0

    def __call__(self, obs: Observation, info: dict) -> Action:
        self.step_count += 1
        action_type = self._decide(obs, info)

        # Avoid repeating same action more than 3 times
        if action_type == self.last_action:
            self.consecutive_same += 1
            if self.consecutive_same >= 3:
                action_type = self._fallback_action(obs, action_type)
                self.consecutive_same = 0
        else:
            self.consecutive_same = 0

        self.last_action = action_type
        return Action(type=action_type)

    def _decide(self, obs: Observation, info: dict) -> str:
        crisis = obs.crisis_level
        virality = obs.virality_index
        sentiment = obs.sentiment_score
        trust = obs.public_trust
        misinfo = info.get("misinformation_active", False)

        # Priority 1: Neutralize misinformation immediately
        if misinfo:
            return ActionType.CLARIFICATION.value

        # Priority 2: HIGH crisis response
        if crisis == CrisisLevel.HIGH.value or crisis == "HIGH":
            if virality > 0.6:
                return ActionType.APOLOGY.value
            elif virality > 0.4:
                return ActionType.PR_CAMPAIGN.value
            else:
                return ActionType.APOLOGY.value

        # Priority 3: MEDIUM crisis response
        elif crisis == CrisisLevel.MEDIUM.value or crisis == "MEDIUM":
            if virality > 0.5:
                return ActionType.PR_CAMPAIGN.value
            elif trust < 0.5:
                return ActionType.INFLUENCER_ENGAGEMENT.value
            else:
                return ActionType.CLARIFICATION.value

        # Priority 4: LOW crisis - light touch
        else:
            if sentiment < -0.2:
                return ActionType.CLARIFICATION.value
            elif virality > 0.2:
                return ActionType.CLARIFICATION.value
            else:
                return ActionType.IGNORE.value

    def _fallback_action(self, obs: Observation, current: str) -> str:
        """Cycle to next appropriate action if stuck in a loop."""
        fallback_map = {
            ActionType.APOLOGY.value: ActionType.PR_CAMPAIGN.value,
            ActionType.PR_CAMPAIGN.value: ActionType.INFLUENCER_ENGAGEMENT.value,
            ActionType.CLARIFICATION.value: ActionType.APOLOGY.value,
            ActionType.INFLUENCER_ENGAGEMENT.value: ActionType.CLARIFICATION.value,
            ActionType.IGNORE.value: ActionType.CLARIFICATION.value,
            ActionType.LEGAL_ACTION.value: ActionType.PR_CAMPAIGN.value,
        }
        return fallback_map.get(current, ActionType.CLARIFICATION.value)


# ─────────────────────────────────────────────────────────────
# LLM AGENT (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────

class LLMAgent:
    """
    LLM-powered agent that uses OpenAI API for contextual decision-making.
    Falls back to rule-based agent if API call fails.
    """

    VALID_ACTIONS = [a.value for a in ActionType]
    SYSTEM_PROMPT = """You are an expert PR crisis manager AI assistant.

Your job is to choose the best action for managing a digital reputation crisis.

Available actions:
- ignore: Take no action (risky if crisis is active)
- apology: Issue a public apology (good for high crisis, can backfire for low)
- clarification: Clarify facts and address misinformation (essential for misinfo)
- pr_campaign: Launch a PR campaign (reduces virality, some cost)
- influencer_engagement: Partner with influencers (high risk/reward)
- legal_action: Take legal action (suppresses spread, damages trust)

Rules:
1. If misinformation is active, prioritize clarification
2. HIGH crisis → strong response (apology, PR campaign)
3. MEDIUM crisis → clarification or PR campaign
4. LOW crisis → light touch, avoid overreacting
5. Never ignore if virality > 0.5 or trust < 0.4
6. Legal action is a last resort and damages trust

Respond with ONLY the action name, nothing else."""

    def __init__(self, fallback: Optional[RuleBasedAgent] = None):
        self.fallback = fallback or RuleBasedAgent()
        self.api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base_url,
                )
            except Exception:
                self._client = None
        return self._client

    def __call__(self, obs: Observation, info: dict) -> Action:
        if not self.api_key:
            return self.fallback(obs, info)

        try:
            action_str = self._llm_decide(obs, info)
            if action_str in self.VALID_ACTIONS:
                return Action(type=action_str)
        except Exception as e:
            print(f"[LLM Agent] API error, falling back to rule-based: {e}")

        return self.fallback(obs, info)

    def _llm_decide(self, obs: Observation, info: dict) -> str:
        client = self._get_client()
        if client is None:
            raise RuntimeError("OpenAI client not available")

        user_msg = f"""Current crisis state:
- Sentiment: {obs.sentiment_score:.3f} (range: -1 to +1)
- Crisis Level: {obs.crisis_level}
- Public Trust: {obs.public_trust:.3f} (range: 0 to 1)
- Virality Index: {obs.virality_index:.3f} (range: 0 to 1)
- Trending Topics: {', '.join(obs.trending_topics)}
- Misinformation Active: {info.get('misinformation_active', False)}
- Influencer Credibility: {info.get('influencer_credibility', 0.7):.2f}
- Time Step: {obs.time_step}

Choose the single best action."""

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=20,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip().lower().replace("-", "_")


# ─────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────

def create_agent(mode: str = "rule_based") -> callable:
    """
    Create an agent instance.

    Args:
        mode: "rule_based" | "llm" | "llm_with_fallback"
    """
    if mode == "rule_based":
        return RuleBasedAgent()
    elif mode == "llm":
        return LLMAgent()
    elif mode == "llm_with_fallback":
        return LLMAgent(fallback=RuleBasedAgent())
    else:
        raise ValueError(f"Unknown agent mode: {mode}")
