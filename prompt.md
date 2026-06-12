# NEXUS — Fix Residual Bugs from Re-Verification + Model Config Updates

A re-verification pass confirmed most earlier fixes landed correctly
(17/17 language tests pass, all video invariants intact). It found ONE
critical residual gap, TWO new bugs introduced by the fixes, two smaller
residuals, and a set of model-ID config updates. Fix ONLY the items below —
do not refactor anything else.

## CRITICAL INSTRUCTIONS

- Read every function COMPLETELY before editing it.
- When changing units, keys, or signatures, update ALL consumers in the same
  change (grep every usage first).
- Never modify Dockerfiles or requirements.txt. Never start/restart Docker
  containers. Never run INSERT/UPDATE/DELETE/DROP on any DB.
- Signal invariants: confidence cap 0.85 global / 0.55 deception-related.
- After finishing: run
  `pytest backend/services/language_agent/tests/ -v` (must stay 17/17),
  AST-parse every touched file, and append a dated entry to `task_log.txt`.

---

## BUG 1 (CRITICAL) — Interrogator exclusion still half-connected in fusion

**File:** `backend/agents/fusion_service.py`

Conversation signals now reach fusion (unpacked at ~line 395 into
`conversation_summary` / `conv_sigs`), but steps 2.6/2.7 run BEFORE that:

- The `interrogator_technique` scan at ~lines 326-338 loops over
  `all_fusion_signals` — that signal is produced only by the conversation
  agent and is never in `all_fusion_signals`. The loop is dead code in the
  live path.
- `all_source` at ~line 340 = voice + language + video dicts only, so the
  `FalseConfessionRiskAssessor` auto-detect
  (`interrogation_patterns.py:408-428`) also never sees conversation
  signals. Exclusion currently works only via the voice-agent
  `preceding_speaker` fallback.

**Fix:**
1. Move the extraction of the conversation summary + its `"signals"` list
   (currently at ~line 395: `(request.voice_summary or {}).get("conversation", {})`)
   to BEFORE step 2.6, storing `conv_sigs` once.
2. Include `conv_sigs` in `all_source` so the risk assessor and the
   interrogator scan both see them.
3. Scan for `interrogator_technique` (and its `metadata.all_interrogators`)
   over `conv_sigs + all_fusion_signals`.
4. Make sure the LATER use at ~line 395-440 (signal graph + `all_audio_signals`
   for the narrative) reuses the same variables instead of re-extracting —
   one extraction, used everywhere. Do NOT change the narrative input shape.
5. Verify end-to-end: with a conversation agent emitting
   `interrogator_technique` for Speaker_0, `FalseConfessionRiskAssessor.evaluate`
   must receive `interrogators={"Speaker_0", ...}` (explicit set, not None)
   in the live pipeline path (`backend/pipeline/analysis_pipeline.py` →
   `FusionAnalyseRequest.voice_summary["conversation"]["signals"]`).

---

## BUG 2 — Exception object stored in the narrative report

**File:** `backend/services/fusion_agent/narrative.py` (~lines 151-190)

The `asyncio.gather(..., return_exceptions=True)` fix handles the forensic
failure correctly, but:

1. Line ~185: `report["raw_response"] = main_raw` runs unconditionally. If
   the main (gpt-4o-mini) call failed, `main_raw` is an **Exception object**
   → the report dict fails JSON serialization when persisted
   (`analysis_pipeline.py` ~669, swallowed) → a SUCCESSFUL forensic report
   is silently discarded. Fix: only assign `raw_response` when
   `isinstance(main_raw, str)`.
2. Main-call failures are never logged (only deep failures are). Add a
   `logger.error` with the exception when `main_raw` is an Exception.
3. Double-failure regression: when BOTH calls fail, the old code fell into
   `_fallback_narrative`; the new code returns a near-empty report. Restore:
   if the main call failed AND (no forensic result or it failed too), call
   `_fallback_narrative` as before. If the main call failed but the forensic
   succeeded, build the report from the fallback narrative MERGED with the
   forensic fields (forensic data must not be lost).

Trace every branch: main ok + deep ok, main ok + deep fail, main fail +
deep ok, both fail. Each must produce a JSON-serializable report dict and a
log line for any failure.

---

## BUG 3 — Minutes leaked into millisecond fields (denial evolution)

**File:** `backend/services/language_agent/interrogation_rules.py`

The slope unit fix converted the regression timestamps to minutes
(~line 624: `timestamps = [t / 60_000 for t in timestamps_ms]`) but three
downstream lines were not updated:

1. ~Lines 680-681: `"window_start_ms": timestamps[0]` /
   `"window_end_ms": timestamps[-1]` now emit MINUTES (e.g. 5.0 instead of
   300000) — corrupts timeline placement of `denial_weakening` signals and
   fusion's session-duration math (`interrogation_patterns.py` ~436-438
   uses `window_*_ms`). Fix: use `timestamps_ms[0]` / `timestamps_ms[-1]`.
2. ~Line 668: `value = round(abs(slope) * 60_000, 4)` — slope is ALREADY
   per-minute, so value is inflated 60,000×. Drop the `* 60_000`.
3. ~Line 690: `"slope_per_min": round(slope * 60_000, 8)` — same 60,000×
   inflation. Drop the `* 60_000`.

Sanity check after fix: denial strength falling 1.0→0.0 over 30 min →
slope ≈ −0.0333/min → `value` ≈ 0.0333, `slope_per_min` ≈ −0.0333,
`window_*_ms` in real milliseconds. Confirm tests still pass and no test
asserts the inflated values.

---

## BUG 4 — Stale pause_ms lookup flattens INTERROG-VOICE-02

**File:** `backend/services/voiceAgent/interrogation_rules.py` (~line 183)

The gating filter at ~line 160 was fixed
(`s.get("metadata", {}).get("max_pause_ms", 0)`), but line ~183 still uses
the old dead path:

```python
pause_ms = ps.get("metadata", {}).get("evidence", {}).get("max_pause_ms", 3_000)
```

`metadata["evidence"]` never exists (VOICE-PAUSE-01 stores the evidence dict
DIRECTLY as metadata — see `rules.py` ~229-233 and ~1087-1090), so
`pause_ms` is always the 3000 default → emitted `value` is always 0.3 and
`confidence` always 0.25 regardless of the actual pause.

**Fix:** `pause_ms = ps.get("metadata", {}).get("max_pause_ms", 3_000)`.
Then verify the value/confidence formulas downstream (~lines 206-215) now
vary with real pause durations.

---

## BUG 5 — Lock release in `except`, not `finally` (still leaks on Redis outage)

**Files:**
- `backend/services/voiceAgent/main.py` (analyse path)
- `backend/services/language_agent/main.py`
- `backend/agents/language_service.py`
- `backend/agents/conversation_service.py`

Current shape in all four: `try: <body + release on success> except Exception:
set_agent_status(failed); event_store.append(agent_failed); release(); raise`.

Two failure modes remain:
1. If Redis is down (the most likely cause of failure), the handler's own
   `set_agent_status` raises BEFORE `release()` → lock leaks until TTL and
   the new exception masks the original.
2. If response-model construction raises AFTER the success-path
   "completed" status was written, the handler overwrites it with "failed"
   and emits both `agent_completed` and `agent_failed`.

**Fix (same pattern in all four files):**
```python
lock acquired (outside try)
completed = False
try:
    ... body ...
    set status completed / append agent_completed
    completed = True
    build response
    return response
except Exception:
    if not completed:
        try:
            set_agent_status(failed); append agent_failed
        except Exception:
            logger.warning("failed to write failed-status", exc_info=True)
    raise
finally:
    try:
        lock_manager.release(...)
    except Exception:
        logger.warning("lock release failed", exc_info=True)
```
- Exactly ONE release site (the `finally`); remove the success-path and
  except-path release calls.
- The failed-status write must be wrapped so its own failure can't prevent
  the `finally` release or mask the original exception.
- The `completed` flag prevents the completed→failed overwrite.
- Do NOT move the lock acquire or the 409-conflict raise into the `try`.
- Verify hunk-by-hunk that the body is a pure re-indent — no logic moved
  in or out of lock scope.

---

## PART 2 — Model config updates (string/config-level only)

### 6.1 Claude model ID retires June 15, 2026 (URGENT)

`backend/shared/utils/llm_client.py` (~line 51): default Anthropic model is
`claude-sonnet-4-20250514` — deprecated, retires 2026-06-15. Update to
`claude-sonnet-4-6`. Grep the whole repo for `claude-sonnet-4-20250514` and
update every occurrence (also check `.env.example`, docs are optional).

### 6.2 AssemblyAI multilingual fallback

`backend/shared/utils/assemblyai_client.py` (~line 102) and the duplicate
client in `backend/shared/utils/external_apis.py` (~line 772): change
`"speech_models": ["universal-3-pro"]` → `["universal-3-pro", "universal-2"]`
(Universal-3 Pro natively covers 6 languages; the array is a fallback
cascade so other languages degrade to Universal-2 instead of failing).
Update BOTH client copies.

### 6.3 Parakeet v2 → v3

`backend/shared/utils/parakeet_client.py` (~line 149): update model id
`parakeet-tdt-0.6b-v2` → `parakeet-tdt-0.6b-v3` (multilingual, 3-hour
local-attention support). If the id is sent to an external GPU server,
make it env-overridable (`PARAKEET_MODEL`, default v3) rather than
hardcoded, so the server can be upgraded independently.

### 6.4 Local Whisper fallback model

`backend/services/voiceAgent/transcriber.py` (~line 43): change the local
faster-whisper default from `medium` → `large-v3-turbo` (env
`WHISPER_MODEL` still overrides). Keep compute_type/int8 settings as-is.

### 6.5 LLM tier updates

1. **Main narrative report**: `backend/services/fusion_agent/narrative.py`
   (~lines 155 and 173): `gpt-4o-mini` → `gpt-5-mini`. NOTE: gpt-5-family
   models route through the `_uses_completion_tokens` branch in
   `llm_client.py` (max_completion_tokens, no temperature). Verify the call
   does not pass an explicit `temperature` kwarg that would now error, and
   that `json_response=True` still produces `response_format` on that
   branch. Keep `max_tokens=4096` → consider raising to 8192 since
   reasoning tokens share the budget on gpt-5-family.
2. **Chat / RAG**: `backend/api/chat.py` (~line 131) and
   `backend/api/sessions.py` (~line 1029): `gpt-4o` → `gpt-5`. Same
   temperature/response_format caveat as above — check what kwargs those
   call sites pass and adjust.
3. **Forensic report**: keep `gpt-5` (no change).
4. Do NOT change `text-embedding-3-small` (still current) and do NOT touch
   the embedding dimension/pgvector schema.

After each model-string change, grep for any code that branches on the old
model name (e.g. `startswith("gpt-4o")`, health endpoints reporting model
names) and update those too.

---

## OUT OF SCOPE (do not attempt)

- CONV-03 turn-merging (dedupe-by-response-turn already fixed the
  inflation; remaining behavior is conservative and acceptable).
- ReDimNet/ECAPA speaker-embedding migration (needs pgvector dim change).
- Local ModernBERT sentiment classifier (separate task).
- Anthropic async routing in `acomplete()` / Claude A/B for the forensic
  call (decide provider strategy first).
- SCRFD/EmotiEffLib/AdaFace CV model swaps (separate task).
- Security/multitenancy issues.

## VERIFICATION CHECKLIST

1. `pytest backend/services/language_agent/tests/ -v` → 17/17 pass.
2. AST-parse every touched Python file.
3. BUG 1: trace that `FalseConfessionRiskAssessor.evaluate` receives the
   explicit interrogator set in the live pipeline path (not None) when a
   conversation `interrogator_technique` signal exists.
4. BUG 2: all four gather outcome branches produce JSON-serializable
   reports (`json.dumps(report)` must not raise).
5. BUG 3: `window_start_ms`/`window_end_ms` are milliseconds; `value` and
   `slope_per_min` are per-minute magnitudes (no 60,000× factor).
6. BUG 5: exactly one `release()` per analyse path, located in `finally`.
7. Grep: zero remaining `claude-sonnet-4-20250514`, zero
   `parakeet-tdt-0.6b-v2`, both AssemblyAI clients send the two-model list.
8. Append a dated summary of every change to `task_log.txt`.
