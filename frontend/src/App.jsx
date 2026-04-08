import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import CrisisGlobe from "./components/CrisisGlobe";
import SentimentChart from "./components/SentimentChart";
import TrustMeter from "./components/TrustMeter";
import ActionPanel from "./components/ActionPanel";
import EventLog from "./components/EventLog";
import TaskSelector from "./components/TaskSelector";
import MetricsHeader from "./components/MetricsHeader";
import { useEnvironment } from "./hooks/useEnvironment";

export default function App() {
  const {
    observation,
    reward,
    state,
    history,
    events,
    isRunning,
    isDone,
    currentTask,
    finalScore,
    selectTask,
    takeAction,
    resetEnvironment,
    autoPlay,
    setAutoPlay,
  } = useEnvironment();

  return (
    <div className="app-shell">
      {/* Ambient background */}
      <div className="bg-ambient" />

      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <motion.div
            className="brand-icon"
            animate={{ rotate: isDone ? 0 : [0, 360] }}
            transition={{ duration: 20, repeat: isDone ? 0 : Infinity, ease: "linear" }}
          >
            ◈
          </motion.div>
          <div>
            <h1>REPUTATION OS</h1>
            <span className="brand-sub">Digital Crisis Intelligence Platform</span>
          </div>
        </div>

        <MetricsHeader observation={observation} isDone={isDone} finalScore={finalScore} />

        <div className="header-controls">
          <TaskSelector
            currentTask={currentTask}
            onSelect={selectTask}
            disabled={isRunning}
          />
          <button
            className={`btn-auto ${autoPlay ? "active" : ""}`}
            onClick={() => setAutoPlay(!autoPlay)}
            disabled={isDone}
          >
            {autoPlay ? "⏸ AUTO" : "▶ AUTO"}
          </button>
          <button className="btn-reset" onClick={resetEnvironment}>
            ↺ RESET
          </button>
        </div>
      </header>

      {/* Main grid */}
      <main className="main-grid">

        {/* Left column: Globe */}
        <section className="panel panel-globe">
          <div className="panel-label">CRISIS HOTSPOTS</div>
          <CrisisGlobe
            viralityIndex={observation?.virality_index ?? 0}
            crisisLevel={observation?.crisis_level ?? "LOW"}
            sentimentScore={observation?.sentiment_score ?? 0}
          />
          <div className="globe-topics">
            {(observation?.trending_topics ?? []).map((t, i) => (
              <motion.span
                key={t + i}
                className="topic-tag"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
              >
                {t}
              </motion.span>
            ))}
          </div>
        </section>

        {/* Center column: Charts */}
        <section className="panel panel-center">
          <div className="chart-row">
            <div className="panel panel-chart">
              <div className="panel-label">SENTIMENT TIMELINE</div>
              <SentimentChart history={history} />
            </div>
          </div>
          <div className="chart-row">
            <div className="panel panel-trust">
              <div className="panel-label">PUBLIC TRUST</div>
              <TrustMeter trust={observation?.public_trust ?? 0.5} />
            </div>
            <div className="panel panel-virality">
              <div className="panel-label">VIRALITY INDEX</div>
              <ViralityMeter virality={observation?.virality_index ?? 0} />
            </div>
          </div>
        </section>

        {/* Right column: Actions + Log */}
        <section className="panel panel-right">
          <div className="panel panel-actions">
            <div className="panel-label">ACTION CONSOLE</div>
            <ActionPanel
              onAction={takeAction}
              disabled={isDone || autoPlay}
              crisisLevel={observation?.crisis_level ?? "LOW"}
              lastReward={reward}
            />
          </div>
          <div className="panel panel-log">
            <div className="panel-label">INTELLIGENCE FEED</div>
            <EventLog events={events} />
          </div>
        </section>
      </main>

      {/* Done overlay */}
      <AnimatePresence>
        {isDone && (
          <motion.div
            className="done-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="done-card"
              initial={{ scale: 0.8, y: 40 }}
              animate={{ scale: 1, y: 0 }}
              transition={{ type: "spring", damping: 20 }}
            >
              <div className="done-icon">
                {(finalScore ?? 0) >= 0.7 ? "◈" : (finalScore ?? 0) >= 0.4 ? "◇" : "✕"}
              </div>
              <h2>{(finalScore ?? 0) >= 0.7 ? "CRISIS MANAGED" : (finalScore ?? 0) >= 0.4 ? "PARTIAL SUCCESS" : "CRISIS UNRESOLVED"}</h2>
              <div className="done-score">
                <span className="score-label">FINAL SCORE</span>
                <span className="score-value">{((finalScore ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <button className="btn-reset done-reset" onClick={resetEnvironment}>
                ↺ NEW EPISODE
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ViralityMeter({ virality }) {
  const level = virality > 0.7 ? "critical" : virality > 0.4 ? "high" : virality > 0.2 ? "medium" : "low";
  return (
    <div className="virality-meter">
      <div className="virality-bar-track">
        <motion.div
          className={`virality-bar-fill virality-${level}`}
          animate={{ width: `${virality * 100}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
      <div className="virality-labels">
        <span className="virality-value">{(virality * 100).toFixed(1)}%</span>
        <span className={`virality-level level-${level}`}>{level.toUpperCase()}</span>
      </div>
      {virality > 0.6 && (
        <motion.div
          className="virality-alert"
          animate={{ opacity: [1, 0.4, 1] }}
          transition={{ duration: 0.8, repeat: Infinity }}
        >
          ⚡ VIRAL SPREAD DETECTED
        </motion.div>
      )}
    </div>
  );
}
