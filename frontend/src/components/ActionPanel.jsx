import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const ACTIONS = [
  {
    id: "clarification",
    label: "CLARIFY",
    icon: "◎",
    key: "C",
    desc: "Address misinformation directly. Highly effective when false narratives are spreading.",
    effect: "↑ Sentiment  ↓ Virality  ✓ Kills misinfo",
    risk: "LOW",
  },
  {
    id: "apology",
    label: "APOLOGIZE",
    icon: "◇",
    key: "A",
    desc: "Issue a public apology. Strong effect in HIGH crisis but can backfire in LOW.",
    effect: "↑ Trust  ↑ Sentiment  ↓ Virality",
    risk: "MEDIUM",
  },
  {
    id: "pr_campaign",
    label: "PR CAMPAIGN",
    icon: "◈",
    key: "P",
    desc: "Launch a coordinated PR campaign. Expensive but reduces virality significantly.",
    effect: "↓↓ Virality  ↑ Sentiment  (cost: -trust)",
    risk: "LOW",
  },
  {
    id: "influencer_engagement",
    label: "INFLUENCER",
    icon: "✦",
    key: "I",
    desc: "Engage influencers. High risk/reward — depends on credibility score.",
    effect: "If credible: ↑↑ Sentiment  |  If not: ↓↓ Trust",
    risk: "HIGH",
  },
  {
    id: "legal_action",
    label: "LEGAL",
    icon: "⚖",
    key: "L",
    desc: "Take legal action. Suppresses spread but severely damages public perception.",
    effect: "↓↓ Virality  ↓↓ Trust  ↓ Sentiment",
    risk: "CRITICAL",
  },
  {
    id: "ignore",
    label: "MONITOR",
    icon: "◉",
    key: "M",
    desc: "Take no action. Only safe in LOW crisis. Repeated ignoring causes exponential decay.",
    effect: "No change  (penalty if repeated)",
    risk: "VARIABLE",
  },
];

const RISK_COLORS = {
  LOW: "#7cfc00",
  MEDIUM: "#ffd700",
  HIGH: "#ff8c00",
  CRITICAL: "#ff1a1a",
  VARIABLE: "#aaa",
};

const CRISIS_RECOMMENDATIONS = {
  LOW: ["clarification", "ignore"],
  MEDIUM: ["clarification", "pr_campaign", "influencer_engagement"],
  HIGH: ["apology", "pr_campaign", "clarification"],
};

export default function ActionPanel({ onAction, disabled, crisisLevel, lastReward }) {
  const [hoveredAction, setHoveredAction] = useState(null);
  const [lastTaken, setLastTaken] = useState(null);

  const recommended = CRISIS_RECOMMENDATIONS[crisisLevel] || [];

  const handleAction = (actionId) => {
    if (disabled) return;
    setLastTaken(actionId);
    onAction(actionId);
    setTimeout(() => setLastTaken(null), 800);
  };

  return (
    <div className="action-panel">
      {/* Last reward flash */}
      <AnimatePresence>
        {lastReward && (
          <motion.div
            className={`reward-flash ${lastReward.value > 0.5 ? "reward-good" : "reward-bad"}`}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            {lastReward.value > 0.5 ? "+" : ""}{(lastReward.value * 100).toFixed(0)} pts — {lastReward.reason}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Crisis level indicator */}
      <div className={`crisis-badge crisis-${crisisLevel?.toLowerCase()}`}>
        <span className="crisis-dot" />
        {crisisLevel} CRISIS
      </div>

      {/* Action grid */}
      <div className="action-grid">
        {ACTIONS.map((action) => {
          const isRecommended = recommended.includes(action.id);
          const isActive = lastTaken === action.id;
          const isHovered = hoveredAction === action.id;

          return (
            <motion.button
              key={action.id}
              className={`action-btn ${isRecommended ? "recommended" : ""} ${isActive ? "active" : ""} ${disabled ? "disabled" : ""}`}
              onClick={() => handleAction(action.id)}
              onMouseEnter={() => setHoveredAction(action.id)}
              onMouseLeave={() => setHoveredAction(null)}
              whileHover={!disabled ? { scale: 1.02 } : {}}
              whileTap={!disabled ? { scale: 0.97 } : {}}
              animate={isActive ? { scale: [1, 1.08, 1] } : {}}
              transition={{ duration: 0.2 }}
              disabled={disabled}
            >
              <span className="action-icon">{action.icon}</span>
              <span className="action-label">{action.label}</span>
              <span className="action-key">[{action.key}]</span>
              {isRecommended && <span className="recommended-dot" />}
              <div
                className="risk-bar"
                style={{ background: RISK_COLORS[action.risk] }}
                title={`Risk: ${action.risk}`}
              />
            </motion.button>
          );
        })}
      </div>

      {/* Tooltip */}
      <AnimatePresence>
        {hoveredAction && (
          <motion.div
            className="action-tooltip"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            {(() => {
              const a = ACTIONS.find(x => x.id === hoveredAction);
              return (
                <>
                  <div className="tooltip-title">{a.label}</div>
                  <div className="tooltip-desc">{a.desc}</div>
                  <div className="tooltip-effect">{a.effect}</div>
                  <div className="tooltip-risk" style={{ color: RISK_COLORS[a.risk] }}>
                    Risk Level: {a.risk}
                  </div>
                </>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
