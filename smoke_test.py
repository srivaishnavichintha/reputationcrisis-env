#!/usr/bin/env python3
"""
smoke_test.py — Quick backend endpoint verification.

Run this BEFORE starting the frontend to confirm the backend is wired correctly.

Usage:
    cd reputation-crisis-env
    python smoke_test.py
"""

import sys
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
SESSION = "smoke_test"
PASS = "✓"
FAIL = "✗"
errors = []


def req(method, path, body=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def check(label, status, data, expected_status=200, required_keys=()):
    ok = status == expected_status and all(k in data for k in required_keys)
    symbol = PASS if ok else FAIL
    print(f"  {symbol}  {label} [{status}]")
    if not ok:
        errors.append(label)
        if status != expected_status:
            print(f"       Expected status {expected_status}, got {status}")
        for k in required_keys:
            if k not in data:
                print(f"       Missing key: '{k}' — got: {list(data.keys())[:8]}")
    return ok


print("\n══════════════════════════════════════════")
print("  Reputation Crisis Manager — Smoke Test")
print("══════════════════════════════════════════\n")

# 1. Health
s, d = req("GET", "/health")
check("GET /health", s, d, required_keys=["status"])

# 2. Root
s, d = req("GET", "/")
check("GET /", s, d, required_keys=["tasks"])

# 3. Tasks list
s, d = req("GET", "/tasks")
check("GET /tasks", s, d, required_keys=["task_1_sentiment_stabilization"])

# 4. Reset (no task)
s, d = req("POST", "/reset", {"session_id": SESSION})
check("POST /reset (no task)", s, d,
      required_keys=["sentiment_score", "crisis_level", "public_trust", "virality_index", "time_step"])

# 5. Reset (with task)
s, d = req("POST", "/reset", {"session_id": SESSION, "task_name": "task_1_sentiment_stabilization"})
check("POST /reset (task_1)", s, d,
      required_keys=["sentiment_score", "crisis_level", "public_trust", "virality_index", "time_step"])
if "sentiment_score" in d:
    print(f"       sentiment={d['sentiment_score']:.3f}  trust={d['public_trust']:.3f}  virality={d['virality_index']:.3f}  crisis={d['crisis_level']}")

# 6. Step — clarification
s, d = req("POST", "/step", {"session_id": SESSION, "action": {"type": "clarification"}})
check("POST /step (clarification)", s, d,
      required_keys=["observation", "reward", "done", "info"])
if "observation" in d:
    obs = d["observation"]
    rew = d["reward"]
    print(f"       step={obs['time_step']}  reward={rew['value']:.4f}  done={d['done']}")
    print(f"       reason: {rew['reason']}")

# 7. Step — apology
s, d = req("POST", "/step", {"session_id": SESSION, "action": {"type": "apology"}})
check("POST /step (apology)", s, d, required_keys=["observation", "reward", "done"])

# 8. Step — pr_campaign
s, d = req("POST", "/step", {"session_id": SESSION, "action": {"type": "pr_campaign"}})
check("POST /step (pr_campaign)", s, d, required_keys=["observation", "reward", "done"])

# 9. Step — influencer_engagement
s, d = req("POST", "/step", {"session_id": SESSION, "action": {"type": "influencer_engagement"}})
check("POST /step (influencer_engagement)", s, d, required_keys=["observation", "reward", "done"])

# 10. Step — ignore
s, d = req("POST", "/step", {"session_id": SESSION, "action": {"type": "ignore"}})
check("POST /step (ignore)", s, d, required_keys=["observation", "reward", "done"])

# 11. GET /state
s, d = req("GET", f"/state?session_id={SESSION}")
check("GET /state", s, d,
      required_keys=["observation", "cumulative_reward", "episode_step", "action_history",
                     "trust_history", "sentiment_history", "misinformation_active"])
if "action_history" in d:
    print(f"       action_history: {d['action_history']}")
    print(f"       cumulative_reward: {d['cumulative_reward']:.4f}")

# 12. Reset task_2
s, d = req("POST", "/reset", {"session_id": SESSION, "task_name": "task_2_viral_outrage_control"})
check("POST /reset (task_2)", s, d, required_keys=["virality_index"])
if "virality_index" in d:
    print(f"       Initial virality: {d['virality_index']:.3f} (expected ~0.65)")

# 13. Reset task_3
s, d = req("POST", "/reset", {"session_id": SESSION, "task_name": "task_3_misinformation_crisis"})
check("POST /reset (task_3)", s, d, required_keys=["crisis_level"])
if "crisis_level" in d:
    print(f"       Crisis level: {d['crisis_level']} (expected HIGH)")

# 14. Invalid action → should return 422
s, d = req("POST", "/step", {"session_id": SESSION, "action": {"type": "invalid_action"}})
check("POST /step (invalid action → 422)", s, d, expected_status=422)

# 15. Step after done — reset first, run to done via legal_action spam
req("POST", "/reset", {"session_id": "done_test", "task_name": "task_1_sentiment_stabilization"})
for _ in range(16):  # task_1 maxSteps=15
    s, d = req("POST", "/step", {"session_id": "done_test", "action": {"type": "ignore"}})
    if d.get("done"):
        break
check("Episode termination (done=True)", s, d, required_keys=["done"])
if "done" in d:
    print(f"       done={d['done']} (expected True after maxSteps)")

# ── Summary ───────────────────────────────────────────────────
print("\n══════════════════════════════════════════")
if not errors:
    print(f"  {PASS} ALL CHECKS PASSED — backend is fully operational")
    print("  You can now start the frontend: cd frontend && npm run dev")
else:
    print(f"  {FAIL} {len(errors)} check(s) FAILED:")
    for e in errors:
        print(f"     • {e}")
    print("\n  Make sure the backend is running:")
    print("  uvicorn backend.main:app --host 0.0.0.0 --port 8000")
print("══════════════════════════════════════════\n")
sys.exit(1 if errors else 0)
