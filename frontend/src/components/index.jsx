import { useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ─────────────────────────────────────────────────────────────
// EVENT LOG
// ─────────────────────────────────────────────────────────────

export function EventLog({ events = [] }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  const getEventStyle = (event) => {
    if (event.includes("[REWARD]")) return "event-reward";
    if (event.includes("[EVENT]")) return "event-alert";
    if (event.includes("[STEP]")) return "event-step";
    if (event.includes("[RESET]") || event.includes("[INIT]")) return "event-system";
    return "event-default";
  };

  return (
    <div className="event-log">
      <AnimatePresence initial={false}>
        {events.map((evt, i) => (
          <motion.div
            key={`${evt}-${i}`}
            className={`event-item ${getEventStyle(evt)}`}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <span className="event-bullet">▸</span>
            <span className="event-text">{evt}</span>
          </motion.div>
        ))}
      </AnimatePresence>
      <div ref={bottomRef} />
    </div>
  );
}


// ─────────────────────────────────────────────────────────────
// TASK SELECTOR
// ─────────────────────────────────────────────────────────────

const TASK_META = {
  "task_1_sentiment_stabilization": { label: "Task 1 — Easy", short: "T1", color: "#7cfc00" },
  "task_2_viral_outrage_control": { label: "Task 2 — Medium", short: "T2", color: "#ffd700" },
  "task_3_misinformation_crisis": { label: "Task 3 — Hard", short: "T3", color: "#ff4444" },
};

export function TaskSelector({ currentTask, onSelect, disabled }) {
  return (
    <div className="task-selector">
      {Object.entries(TASK_META).map(([key, meta]) => (
        <button
          key={key}
          className={`task-btn ${currentTask === key ? "task-active" : ""}`}
          onClick={() => !disabled && onSelect(key)}
          disabled={disabled}
          style={{ "--task-color": meta.color }}
        >
          <span className="task-dot" />
          {meta.short}
        </button>
      ))}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────
// METRICS HEADER
// ─────────────────────────────────────────────────────────────

export function MetricsHeader({ observation, isDone, finalScore }) {
  if (!observation) return <div className="metrics-header" />;

  const metrics = [
    {
      label: "SENTIMENT",
      value: `${observation.sentiment_score >= 0 ? "+" : ""}${(observation.sentiment_score * 100).toFixed(0)}`,
      unit: "%",
      color: observation.sentiment_score > 0 ? "#7cfc00" : observation.sentiment_score > -0.3 ? "#ffd700" : "#ff4444",
    },
    {
      label: "TRUST",
      value: (observation.public_trust * 100).toFixed(0),
      unit: "%",
      color: observation.public_trust > 0.6 ? "#7cfc00" : observation.public_trust > 0.35 ? "#ffd700" : "#ff4444",
    },
    {
      label: "VIRALITY",
      value: (observation.virality_index * 100).toFixed(0),
      unit: "%",
      color: observation.virality_index < 0.3 ? "#7cfc00" : observation.virality_index < 0.6 ? "#ff8c00" : "#ff4444",
    },
    {
      label: "STEP",
      value: observation.time_step,
      unit: "",
      color: "#888",
    },
  ];

  return (
    <div className="metrics-header">
      {metrics.map(m => (
        <motion.div
          key={m.label}
          className="metric-chip"
          animate={{ borderColor: m.color + "44" }}
        >
          <span className="metric-label">{m.label}</span>
          <span className="metric-value" style={{ color: m.color }}>
            {m.value}<span className="metric-unit">{m.unit}</span>
          </span>
        </motion.div>
      ))}
      {isDone && finalScore !== null && (
        <motion.div
          className="metric-chip score-chip"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring" }}
        >
          <span className="metric-label">SCORE</span>
          <span className="metric-value" style={{ color: finalScore > 0.6 ? "#7cfc00" : "#ff8c00" }}>
            {(finalScore * 100).toFixed(1)}<span className="metric-unit">%</span>
          </span>
        </motion.div>
      )}
    </div>
  );
}
