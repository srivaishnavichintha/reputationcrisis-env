import { useState, useEffect, useRef, useCallback } from "react";

// Use Vite proxy when running dev server (/api → localhost:8000)
// Fall back to direct URL for production builds
const API_BASE = import.meta.env.VITE_API_URL || "";
const SESSION_ID = `session_${Date.now()}`;

function apiUrl(path) {
  // In dev: "" + "/api" + "/reset" → uses Vite proxy → localhost:8000/reset
  // With VITE_API_URL set: "http://localhost:8000" + "/reset"
  if (API_BASE) return `${API_BASE}${path}`;
  return `/api${path}`;
}

const DEBUG = true;
function log(...args) {
  if (DEBUG) console.log("[ReputationOS]", ...args);
}

const TASK_NAMES = [
  "task_1_sentiment_stabilization",
  "task_2_viral_outrage_control",
  "task_3_misinformation_crisis",
];

const RULE_BASED_LOGIC = (obs, info) => {
  const { crisis_level, virality_index, sentiment_score, public_trust } = obs;
  const misinfo = info?.misinformation_active;

  if (misinfo) return "clarification";
  if (crisis_level === "HIGH") {
    if (virality_index > 0.6) return "apology";
    if (virality_index > 0.4) return "pr_campaign";
    return "apology";
  }
  if (crisis_level === "MEDIUM") {
    if (virality_index > 0.5) return "pr_campaign";
    if (public_trust < 0.5) return "influencer_engagement";
    return "clarification";
  }
  if (sentiment_score < -0.2 || virality_index > 0.2) return "clarification";
  return "ignore";
};

export function useEnvironment() {
  const [observation, setObservation] = useState(null);
  const [reward, setReward] = useState(null);
  const [state, setState] = useState(null);
  const [history, setHistory] = useState([]);
  const [events, setEvents] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [currentTask, setCurrentTask] = useState(TASK_NAMES[0]);
  const [finalScore, setFinalScore] = useState(null);
  const [autoPlay, setAutoPlay] = useState(false);
  const autoPlayRef = useRef(false);
  const autoPlayTimer = useRef(null);

  useEffect(() => {
    autoPlayRef.current = autoPlay;
  }, [autoPlay]);

  const resetEnvironment = useCallback(async (taskName) => {
    const task = taskName || currentTask;
    setIsRunning(true);
    setIsDone(false);
    setFinalScore(null);
    setHistory([]);
    setEvents([`[INIT] Starting ${task}...`]);

    const url = apiUrl("/reset");
    const body = JSON.stringify({ session_id: SESSION_ID, task_name: task });
    log("POST", url, body);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const obs = await res.json();
      log("Reset response:", obs);

      setObservation(obs);
      // Seed history with TWO identical points so chart renders immediately (needs length >= 2)
      const seed = {
        step: 0,
        sentiment: obs.sentiment_score,
        trust: obs.public_trust,
        virality: obs.virality_index,
      };
      setHistory([seed, { ...seed, step: 0.001 }]); // tiny offset so chart draws
      setEvents([
        `[STATE] Crisis: ${obs.crisis_level} | Trust: ${(obs.public_trust * 100).toFixed(0)}% | Virality: ${(obs.virality_index * 100).toFixed(0)}%`,
        `[RESET] ${task} initialized — session ${SESSION_ID}`,
      ]);
    } catch (e) {
      console.error("[ReputationOS] Reset failed, using offline demo mode:", e);
      // Offline mode: use mock data
      const mockObs = getMockObservation(task, 0);
      setObservation(mockObs);
      const seed = { step: 0, sentiment: mockObs.sentiment_score, trust: mockObs.public_trust, virality: mockObs.virality_index };
      setHistory([seed, { ...seed, step: 0.001 }]);
      setEvents([
        `[STATE] Crisis: ${mockObs.crisis_level} | Trust: ${(mockObs.public_trust * 100).toFixed(0)}%`,
        `[DEMO] Running in offline demo mode — backend unreachable`,
      ]);
    } finally {
      setIsRunning(false);
    }
  }, [currentTask]);

  const takeAction = useCallback(async (actionType) => {
    if (isDone || isRunning) {
      log("takeAction blocked — isDone:", isDone, "isRunning:", isRunning);
      return;
    }
    setIsRunning(true);

    const url = apiUrl("/step");
    const body = JSON.stringify({
      session_id: SESSION_ID,
      action: { type: actionType },
    });
    log("POST", url, "action:", actionType);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      log("Step response:", data);

      const { observation: obs, reward: rew, done, info } = data;

      setObservation(obs);
      setReward(rew);
      setIsDone(done);
      setHistory(prev => [...prev, {
        step: obs.time_step,
        sentiment: obs.sentiment_score,
        trust: obs.public_trust,
        virality: obs.virality_index,
        action: actionType,
        reward: rew.value,
      }]);
      setEvents(prev => [
        `[STEP ${obs.time_step}] → ${actionType.toUpperCase()}`,
        `[REWARD] ${rew.value.toFixed(3)} — ${rew.reason}`,
        ...(info?.events || []),
        ...prev.slice(0, 20),
      ]);

      if (done) {
        log("Episode done, fetching full state...");
        try {
          const stateRes = await fetch(apiUrl(`/state?session_id=${SESSION_ID}`));
          const fullState = await stateRes.json();
          setState(fullState);
          const episodeLen = obs.time_step || 1;
          setFinalScore(Math.min(1, fullState.cumulative_reward / episodeLen));
        } catch (stateErr) {
          console.error("[ReputationOS] Failed to fetch final state:", stateErr);
          setFinalScore(rew.value);
        }
        setAutoPlay(false);
      }
    } catch (e) {
      console.error("[ReputationOS] Step failed, using offline simulation:", e);
      // Offline: simulate step
      const step = (observation?.time_step ?? 0) + 1;
      const mockObs = getMockObservation(currentTask, step, actionType, observation);
      const mockReward = { value: Math.random() * 0.4 + 0.3, reason: "demo simulation" };
      const done = step >= 10;

      setObservation(mockObs);
      setReward(mockReward);
      setIsDone(done);
      setHistory(prev => [...prev, {
        step,
        sentiment: mockObs.sentiment_score,
        trust: mockObs.public_trust,
        virality: mockObs.virality_index,
        action: actionType,
      }]);
      setEvents(prev => [
        `[STEP ${step}] → ${actionType.toUpperCase()} [DEMO]`,
        `[REWARD] ${mockReward.value.toFixed(3)} — demo simulation`,
        ...prev.slice(0, 20),
      ]);
      if (done) {
        setFinalScore(0.65);
        setAutoPlay(false);
      }
    } finally {
      setIsRunning(false);
    }
  }, [isDone, isRunning, observation, currentTask]);

  // Auto-play loop
  useEffect(() => {
    if (autoPlay && !isDone && observation) {
      autoPlayTimer.current = setTimeout(() => {
        if (autoPlayRef.current && !isDone) {
          const action = RULE_BASED_LOGIC(observation, {});
          takeAction(action);
        }
      }, 1200);
    }
    return () => clearTimeout(autoPlayTimer.current);
  }, [autoPlay, observation, isDone, takeAction]);

  const selectTask = useCallback((taskName) => {
    setCurrentTask(taskName);
    resetEnvironment(taskName);
  }, [resetEnvironment]);

  // Initialize on mount — use ref trick to call the latest resetEnvironment
  const resetRef = useRef(resetEnvironment);
  useEffect(() => { resetRef.current = resetEnvironment; }, [resetEnvironment]);

  useEffect(() => {
    log("Initializing environment...");
    resetRef.current(currentTask);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // intentionally empty — runs once on mount

  return {
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
    resetEnvironment: () => resetEnvironment(currentTask),
    autoPlay,
    setAutoPlay,
  };
}

// Mock observation for offline demo
function getMockObservation(task, step, lastAction, prevObs) {
  const base = prevObs || {
    sentiment_score: task.includes("3") ? -0.7 : task.includes("2") ? -0.55 : -0.35,
    public_trust: task.includes("3") ? 0.5 : task.includes("2") ? 0.55 : 0.7,
    virality_index: task.includes("3") ? 0.75 : task.includes("2") ? 0.65 : 0.2,
  };

  const actionEffect = {
    clarification: { sentiment: 0.1, trust: 0.04, virality: -0.08 },
    apology: { sentiment: 0.08, trust: 0.05, virality: -0.06 },
    pr_campaign: { sentiment: 0.06, trust: 0.04, virality: -0.12 },
    influencer_engagement: { sentiment: 0.07, trust: 0.03, virality: -0.07 },
    legal_action: { sentiment: -0.05, trust: -0.07, virality: -0.15 },
    ignore: { sentiment: -0.04, trust: -0.03, virality: 0.05 },
  }[lastAction] || { sentiment: 0, trust: 0, virality: 0 };

  const noise = () => (Math.random() - 0.5) * 0.06;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const sentiment = clamp(base.sentiment_score + actionEffect.sentiment + noise(), -1, 1);
  const trust = clamp(base.public_trust + actionEffect.trust + noise() * 0.5, 0, 1);
  const virality = clamp(base.virality_index + actionEffect.virality + noise() * 0.5, 0, 1);

  const crisisLevel = sentiment < -0.5 || virality > 0.7 ? "HIGH"
    : sentiment < -0.1 || virality > 0.3 ? "MEDIUM" : "LOW";

  return {
    sentiment_score: Math.round(sentiment * 1000) / 1000,
    public_trust: Math.round(trust * 1000) / 1000,
    virality_index: Math.round(virality * 1000) / 1000,
    crisis_level: crisisLevel,
    trending_topics: ["#BrandCrisis", "#Trending", "#PR", "#Statement"].slice(0, 3),
    time_step: step,
  };
}
