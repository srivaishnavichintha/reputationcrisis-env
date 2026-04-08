# 🌐 Digital Reputation Crisis Manager

**OpenEnv-compliant AI environment for simulating real-world PR crisis management**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-00d4ff)](https://openenv.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://docker.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB)](https://python.org)

---

## 🎯 Motivation

Real-world PR crises destroy billions in market value annually. From United Airlines losing $1B after a passenger removal video went viral, to Samsung's Galaxy Note 7 recall — the speed and quality of crisis response is the difference between recovery and collapse.

This environment gives AI agents a realistic simulation of online reputation crisis management, modeling:

- **Viral outrage propagation** with exponential growth dynamics
- **Misinformation spread** that poisons sentiment if not neutralized
- **Influencer amplification** — credibility-dependent high-risk/reward actions
- **Trust asymmetry** — trust decays 2× faster than it recovers
- **Delayed response penalties** — every ignored step compounds the crisis

Unlike toy environments, this models the real trade-offs PR teams face: aggressive legal action suppresses virality but destroys public trust; premature apologies for minor issues backfire; influencer engagement fails catastrophically with low-credibility partners.

---

## 🧠 OpenEnv Specification

### Observation Space

| Field | Type | Range | Description |
|-------|------|--------|-------------|
| `sentiment_score` | float | -1.0 to +1.0 | Public sentiment (-1=outrage, +1=praise) |
| `crisis_level` | str | LOW/MEDIUM/HIGH | Derived crisis severity |
| `trending_topics` | list[str] | — | Active hashtags and topics |
| `public_trust` | float | 0.0 to 1.0 | Brand trust index |
| `virality_index` | float | 0.0 to 1.0 | Spread velocity |
| `time_step` | int | 0+ | Current episode step |

### Action Space

| Action | Immediate Effect | Risk | Best For |
|--------|-----------------|------|---------|
| `ignore` | No effect | HIGH if repeated | Never sustained |
| `apology` | ↑ trust, ↑ sentiment | MEDIUM (backfires in LOW) | HIGH crisis |
| `clarification` | ↑ sentiment, kills misinfo | LOW | Misinformation |
| `pr_campaign` | ↓↓ virality, ↑ trust | LOW (cost) | MEDIUM-HIGH |
| `influencer_engagement` | High impact if credible | HIGH | Credibility > 0.6 |
| `legal_action` | ↓↓ virality, ↓↓ trust | CRITICAL | Last resort |

### Reward Function (Dense)

```
reward = 0.30 × sentiment_component
       + 0.35 × trust_component
       + 0.25 × virality_control_component
       - action_cost
       - inaction_penalty
       - collapse_penalty
       - extreme_sentiment_penalty
       - misinformation_penalty
```

---

## 🏆 Tasks

### Task 1 — Sentiment Stabilization (EASY)
- **Scenario**: Minor product issue, negative tweet wave
- **Initial state**: sentiment=-0.35, trust=0.70, virality=0.20
- **Goal**: Stabilize sentiment above 0.0
- **Grader**: `0.45×avg_sentiment + 0.35×trust_recovery + 0.20×response_efficiency`
- **Baseline score**: ~0.62

### Task 2 — Viral Outrage Control (MEDIUM)
- **Scenario**: Controversial business decision goes viral
- **Initial state**: sentiment=-0.55, trust=0.55, virality=0.65
- **Goal**: Reduce virality below 0.30 AND recover trust above 0.50
- **Grader**: `0.40×virality_reduction + 0.35×trust_recovery + 0.25×response_time`
- **Baseline score**: ~0.51

### Task 3 — Misinformation Crisis (HARD)
- **Scenario**: Coordinated misinfo campaign + low-credibility influencers
- **Initial state**: sentiment=-0.70, trust=0.50, virality=0.75, misinfo=ACTIVE
- **Goal**: Prevent trust collapse (<0.30), neutralize misinformation
- **Grader**: `0.35×misinfo_control + 0.40×trust_stability + 0.25×decision_quality`
- **Baseline score**: ~0.41

---

## 🚀 Setup

### Quick Start (Local)

```bash
# Clone and install
git clone <repo_url>
cd reputation-crisis-env

# Backend
pip install -r requirements.txt

# Start API server
uvicorn backend.main:app --reload --port 8000

# Run inference (new terminal)
python inference.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Docker

```bash
# Build
docker build -t reputation-env .

# Run
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e MODEL_NAME=gpt-4o-mini \
  reputation-env

# Verify
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test"}'
```

---

## 🤖 Inference

```bash
# Rule-based baseline (default)
python inference.py

# LLM agent (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
python inference.py --agent llm

# Specific tasks only
python inference.py --tasks task_1_sentiment_stabilization task_3_misinformation_crisis

# Fixed seed for reproducibility
python inference.py --seed 123
```

### Log Format (Strict OpenEnv)

```
[START]
task: task_1_sentiment_stabilization
difficulty: EASY

[STEP]
action: clarification
reward: 0.6234
state: {
  "sentiment_score": -0.15,
  "crisis_level": "MEDIUM",
  "public_trust": 0.68,
  "virality_index": 0.18,
  "time_step": 1
}

[END]
final_score: 0.7123
success: True
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/step` | POST | Take action |
| `/state` | GET | Full internal state |
| `/tasks` | GET | List all tasks |
| `/run_task` | POST | Run complete episode |
| `/docs` | GET | Interactive API docs |

---

## 🚀 Hugging Face Deployment

```bash
# 1. Create HF Space (Docker SDK)
# Go to https://huggingface.co/new-space
# SDK: Docker, hardware: CPU Basic

# 2. Upload via git
git init
git remote add origin https://huggingface.co/spaces/<username>/reputation-crisis-env
git add .
git commit -m "Initial deployment"
git push origin main

# 3. Set environment variables in Space Settings:
#    API_BASE_URL = https://api.openai.com/v1
#    MODEL_NAME   = gpt-4o-mini
#    HF_TOKEN     = <your_token>
#    OPENAI_API_KEY = <your_key>
```

Add `README.md` with tag:
```yaml
tags:
  - openenv
  - reinforcement-learning
  - simulation
```

---

## 📁 Project Structure

```
reputation-crisis-env/
├── backend/
│   ├── env/
│   │   ├── models.py          # Pydantic Observation/Action/Reward
│   │   └── environment.py     # Core OpenEnv environment
│   ├── tasks/
│   │   └── tasks.py           # 3 tasks + deterministic graders
│   ├── agents/
│   │   └── baseline_agent.py  # Rule-based + LLM hybrid agent
│   └── main.py                # FastAPI server
├── frontend/
│   ├── src/
│   │   ├── components/        # React UI components
│   │   ├── hooks/             # useEnvironment state hook
│   │   ├── App.jsx            # Main app layout
│   │   └── index.css          # Premium black/silver theme
│   ├── package.json
│   └── vite.config.js
├── inference.py               # OpenEnv inference script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Baseline Scores

| Task | Difficulty | Rule-Based | LLM (GPT-4o-mini) |
|------|-----------|------------|-------------------|
| Sentiment Stabilization | EASY | ~0.62 | ~0.71 |
| Viral Outrage Control | MEDIUM | ~0.51 | ~0.64 |
| Misinformation Crisis | HARD | ~0.41 | ~0.56 |
| **Average** | | **~0.51** | **~0.64** |

---

## 🏅 Hackathon Notes

**Why this environment is competitive:**

1. **Real-world utility** — Models billion-dollar business problems
2. **Realistic dynamics** — Trust asymmetry, viral growth, credibility mechanics
3. **Dense rewards** — No sparse reward problem; smooth gradient signal
4. **Deterministic graders** — Reproducible, fair evaluation
5. **Three distinct difficulties** — Tests agent adaptability
6. **Visual demo** — 3D globe, live charts, premium UI in <2 min demo
7. **OpenEnv compliant** — `reset()`, `step()`, `state()` spec followed strictly
