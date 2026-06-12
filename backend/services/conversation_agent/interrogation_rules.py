# services/conversation_agent/interrogation_rules.py
"""
Interrogation-specific conversation rules (NEXUS INTERROGATION_IMPLEMENTATION.MD v2.0).

Rules implemented:
  INTERROG-CONV-01  Evidence Response Processing Delay    (Hartwig et al. 2014-2016, d=1.83)
  INTERROG-CONV-02  Interrogator Technique Classification (Meissner et al. 2014)  [InterrogatorTechniqueClassifier]
  INTERROG-CONV-03  Verbal Uncertainty Cluster            (CBCA Criterion 15, Steller & Köhnken 1989)

INTERROG-CONV-01 research: SUE (Strategic Use of Evidence) framework. Long response latency
after evidence disclosure is the highest-evidence interrogation signal (85% field accuracy),
but is equally present in innocent suspects confronted with fabricated or unexpected evidence.
Confidence scales from 0.40 → 0.85 with latency magnitude (§5 table: 0.85 all quality tiers
because signal is timestamp-based, not video/audio quality dependent).

INTERROG-CONV-03 research: CBCA Criterion 15 (Steller & Köhnken 1989) classifies "Admissions
of Lack of Memory" as a TRUTHFULNESS indicator — truth tellers admit memory gaps more often
than liars, who fabricate certainty. DePaulo et al. (2003) meta-analysis: uncertainty
expressions show small/null effect for deception detection. Signal indicates cognitive load,
NOT deception, and MUST NOT feed into FalseConfessionRiskAssessor.
"""
from __future__ import annotations

import re
import logging
from collections import defaultdict

logger = logging.getLogger("nexus.conversation.interrogation")


try:
    from shared.utils.interrogator_detection import detect_interrogators as _detect_interrogators
except ImportError:
    from backend.shared.utils.interrogator_detection import detect_interrogators as _detect_interrogators

# Evidence disclosure keywords (CONV-01) — originally forensic evidence only,
# extended to cover deposition/cross-examination challenges so CONV-01 fires
# in sworn-testimony contexts where the interrogator confronts facts rather
# than presenting physical evidence.
_EVIDENCE_RE = re.compile(
    r"\b("
    r"we found|forensics|DNA|fingerprint(s)?|"
    r"witness(es)? (says?|told|saw|report(ed)?|identif)|"
    r"camera (shows?|footage|captured|recorded)|"
    r"(phone|cell) (records?|data|logs?)|surveillance|"
    r"evidence (shows?|indicates?|proves?|links?)|"
    r"(test|lab|autopsy|tox(icology)?|ballistics) results?|"
    r"trace evidence|fiber(s)?|"
    r"we (have|got) (proof|evidence)|we know (you|that)|"
    r"records? show|data shows?|we can prove|"
    r"(cell tower|GPS|location data|timestamp|CCTV|security footage)|"
    r"I have (proof|evidence|a witness)|"
    r"the (report|analysis|findings?) (shows?|says?|indicates?)|"
    r"your (blood|DNA|prints?|hair|fibres?) (was|were|matched|found)|"
    r"is (this|that) the same (man|person|woman|individual)|"
    r"you (said|told|claimed|stated|testified) (that|to |you )|"
    r"how do you explain|isn'?t (it|that) (true|correct|right)|"
    r"why did you (call|go|leave|come|stay|meet|send|delete|hide|text)"
    r")\b",
    re.IGNORECASE,
)

# Latency threshold: >2000ms is "extended" per SUE framework spec
_LATENCY_THRESHOLD_MS = 2_000

# Do not associate a response if the gap to the next speaker exceeds this
# (interrogator asked multiple questions in between, or there is a long unrelated pause)
_MAX_ASSOCIATION_GAP_MS = 20_000

# ── CONV-03 patterns ──────────────────────────────────────────────────────────

# Verbal uncertainty markers — cognitive load expressed as words, not silence.
# CBCA Criterion 15 (Steller & Köhnken 1989): admissions of lack of memory are
# MORE common in truthful accounts. These mark cognitive load, NOT deception.
_UNCERTAINTY_RE = re.compile(
    r"\b("
    r"I don'?t know|I'?m not sure|I don'?t remember|I can'?t remember|"
    r"I don'?t recall|I can'?t recall|I'?m not certain|"
    r"I have no idea|have no idea|"          # "I have no idea" is a very common hedge
    r"Don'?t know|"                          # "Don't know" without leading "I"
    r"I think|I guess|I believe|I suppose|I imagine|"
    r"maybe|possibly|probably|perhaps|"
    r"I barely|I couldn'?t|I can'?t really|"
    r"not really|not exactly|sort of|kind of"
    r")\b",
    re.IGNORECASE,
)

# Challenging question patterns — covers depositions and cross-examinations
# in addition to police interrogations.
_CHALLENGE_QUESTION_RE = re.compile(
    r"\b("
    r"do you (remember|recall|know)|"
    r"would you (remember|recall)|"
    r"can you (explain|tell me|describe)|"
    r"is (this|that) (the same|true|correct|right)|"
    r"why did you|how did you|when did you|where did you|"
    r"did you (not|ever|actually)|"
    r"isn'?t (it|that) (true|correct|right)|"
    r"you (said|told|claimed|stated|testified|mentioned)|"
    r"how (do you|would you|can you) explain"
    r")\b",
    re.IGNORECASE,
)


class InterrogationConversationRules:
    """
    Stateless per-session conversation interrogation rules.
    Operates directly on raw diar_segments — no pre-extracted features needed.
    """

    def evaluate(
        self,
        segments: list[dict],
        session_id: str = "",
    ) -> list[dict]:
        """
        INTERROG-CONV-01: Evidence Response Processing Delay.

        For each turn containing evidence-disclosure language, measure the gap
        before the next different-speaker turn. Fires when gap > 2000 ms.

        Segments must have: speaker, start_ms (or start in seconds), end_ms (or end).
        """
        normed = self._normalise(segments)
        if not normed:
            return []

        # Detect interrogators BEFORE the loop so we only treat turns from
        # interrogator speakers as evidence-disclosure turns.  Without this gate
        # a suspect saying "you said that to me first" would generate an
        # evidence_response_processing_delay attributed to themselves.
        interrogators, _ = _detect_interrogators(normed)

        signals: list[dict] = []
        for i, turn in enumerate(normed[:-1]):
            # Gate: only interrogator turns can disclose evidence
            if interrogators and turn["speaker"] not in interrogators:
                continue
            if not _EVIDENCE_RE.search(turn["text"]):
                continue

            # Find the immediately next turn from a different speaker (the suspect).
            next_turn: dict | None = None
            for j in range(i + 1, len(normed)):
                if normed[j]["speaker"] != turn["speaker"]:
                    next_turn = normed[j]
                    break

            if next_turn is None:
                continue

            # Measure latency from the END of the LAST interrogator segment that
            # precedes the suspect's response turn — not just the evidence segment.
            # If the interrogator continued talking after the evidence disclosure,
            # their own speech would otherwise count as "processing delay".
            last_interrog_end = turn["end_ms"]
            for k in range(i + 1, len(normed)):
                if normed[k] is next_turn:
                    break
                if normed[k]["speaker"] == turn["speaker"]:
                    last_interrog_end = normed[k]["end_ms"]

            latency_ms = next_turn["start_ms"] - last_interrog_end
            if latency_ms <= 0 or latency_ms > _MAX_ASSOCIATION_GAP_MS:
                continue
            if latency_ms <= _LATENCY_THRESHOLD_MS:
                continue

            keywords = list({m.group(0).lower() for m in _EVIDENCE_RE.finditer(turn["text"])})[: 5]
            signals.append({
                "agent":           "conversation",
                "speaker_id":      next_turn["speaker"],
                "signal_type":     "evidence_response_processing_delay",
                # value: latency normalised to 0-1 (10 s = 1.0)
                "value":           round(min(latency_ms / 10_000, 1.0), 4),
                "value_text":      "extended_processing_delay",
                # Confidence scales 0.40→0.85 with latency; cap=0.85 per §5 (timestamp-based, quality-invariant)
                "confidence":      round(min(0.85, 0.40 + (latency_ms - _LATENCY_THRESHOLD_MS) / 20_000), 4),
                "window_start_ms": turn["start_ms"],
                "window_end_ms":   next_turn["end_ms"],
                "metadata": {
                    "rule_id":               "INTERROG-CONV-01",
                    "latency_ms":            latency_ms,
                    "latency_s":             round(latency_ms / 1000, 2),
                    "evidence_turn_speaker": turn["speaker"],
                    "evidence_keywords":     keywords,
                    "research":              "Hartwig et al. (2014-2016) SUE framework — d=1.83, 85% field accuracy",
                    "interpretations": [
                        "Cognitive processing of unexpected or complex information",
                        "Surprise — innocent suspect confronted with accusation they didn't anticipate",
                        "Fabrication time for deceptive response",
                        "False-evidence effect: innocent suspects show long latency when confronted with fabricated evidence (Reid manual; Cato 2024)",
                        "Emotional processing of traumatic or threatening information",
                    ],
                    "recommendation": (
                        "Cross-reference with statement-evidence consistency, contamination detection, "
                        "and whether the evidence presented could be fabricated or incorrect."
                    ),
                },
            })

        # INTERROG-CONV-03: Verbal Uncertainty Cluster
        interrogators, _ = _detect_interrogators(normed)
        if interrogators:
            signals.extend(self._verbal_uncertainty_cluster(normed, interrogators, session_id))

        return signals

    def _verbal_uncertainty_cluster(
        self,
        normed: list[dict],
        interrogators: set[str],
        session_id: str = "",
    ) -> list[dict]:
        """
        INTERROG-CONV-03: Verbal Uncertainty Cluster.

        Detects clusters of uncertainty expressions ("I don't know", "I'm not sure",
        "I can't remember") in suspect responses to challenging questions.
        Fires when ≥3 out of 5 consecutive challenge-response pairs contain
        at least one uncertainty marker.

        CRITICAL: This signal uses signal_type "verbal_uncertainty_cluster",
        NOT "evidence_response_processing_delay". It intentionally does NOT feed
        FalseConfessionRiskAssessor (which only uses _PROCESSING_TYPES =
        {"evidence_response_processing_delay"}).

        Research:
          CBCA Criterion 15 (Steller & Köhnken 1989): "Admissions of Lack of
          Memory" are a TRUTHFULNESS indicator — more common in truthful accounts.
          Porter & Yuille (1996): memory admissions significantly higher in truthful.
          DePaulo et al. (2003): uncertainty expressions show small/null effect.
          → Cognitive load indicator with multiple valid interpretations.

        DSA: O(S) single pass to build pairs, O(P) sliding window.
        Confidence cap: 0.40 — not a reliable deception cue.
        """
        if len(normed) < 5:
            return []

        # Build challenge-response pairs: interrogator question → suspect answer.
        # Dedupe by response-turn identity so consecutive interrogator segments that
        # all pair with the SAME response only produce ONE pair (prevents MIN_HITS=3
        # from being met by a single "I don't know" via duplicate pair inflation).
        pairs: list[dict] = []
        _seen_response_ids: set[int] = set()
        for i, turn in enumerate(normed):
            if turn["speaker"] not in interrogators:
                continue
            if not (("?" in turn["text"]) or _detect_interrogators([turn])[0]):
                # Use punctuation check first; the shared helper covers no-punctuation
                if "?" not in turn["text"]:
                    continue

            # Next non-interrogator turn is the response
            response: dict | None = None
            for j in range(i + 1, len(normed)):
                if normed[j]["speaker"] not in interrogators:
                    response = normed[j]
                    break

            if response is None:
                continue

            # Apply same 20s max-gap as CONV-01 to prevent minute-2 question
            # pairing with a minute-30 answer
            gap_ms = response["start_ms"] - turn["end_ms"]
            if gap_ms > _MAX_ASSOCIATION_GAP_MS:
                continue

            # Dedupe: skip if this response turn was already paired with a
            # previous question segment
            if id(response) in _seen_response_ids:
                continue
            _seen_response_ids.add(id(response))

            markers = _UNCERTAINTY_RE.findall(response["text"])
            # findall on a group-less regex returns strings; on a group-ful regex
            # returns tuples. Normalise to plain strings.
            marker_strs = [m if isinstance(m, str) else m[0] for m in markers]

            pairs.append({
                "challenge":          turn,
                "response":           response,
                "marker_count":       len(marker_strs),
                "markers":            [m.lower() for m in marker_strs],
                "is_memory_challenge": bool(_CHALLENGE_QUESTION_RE.search(turn["text"])),
            })

        WINDOW      = 5
        MIN_HITS    = 3

        # Need at least WINDOW pairs for the sliding window to have one iteration.
        # (range(len(pairs) - WINDOW + 1) is empty for len < WINDOW)
        if len(pairs) < WINDOW:
            return []
        signals: list[dict] = []

        for start in range(len(pairs) - WINDOW + 1):
            window = pairs[start : start + WINDOW]

            uncertain = [p for p in window if p["marker_count"] >= 1]
            if len(uncertain) < MIN_HITS:
                continue

            total_markers   = sum(p["marker_count"] for p in window)
            memory_chal     = sum(1 for p in window if p["is_memory_challenge"])
            first_resp      = window[0]["response"]
            last_resp       = window[-1]["response"]
            respondent      = first_resp["speaker"]

            all_markers: list[str] = []
            for p in uncertain:
                all_markers.extend(p["markers"])
            sample_markers = list(dict.fromkeys(all_markers))[:5]

            # Skip if a signal for the same speaker already covers this window
            overlap = any(
                e["speaker_id"] == respondent
                and e["window_start_ms"] <= last_resp["end_ms"]
                and e["window_end_ms"] >= first_resp["start_ms"]
                for e in signals
            )
            if overlap:
                continue

            # conf: 0.25 base + 0.05 per uncertain pair beyond MIN_HITS, cap 0.40
            conf = round(min(0.40, 0.25 + (len(uncertain) - MIN_HITS) * 0.05), 4)

            signals.append({
                "agent":           "conversation",
                "speaker_id":      respondent,
                "signal_type":     "verbal_uncertainty_cluster",
                "value":           round(min(1.0, total_markers / (WINDOW * 3)), 4),
                "value_text":      "uncertainty_cluster",
                "confidence":      conf,
                "window_start_ms": first_resp["start_ms"],
                "window_end_ms":   last_resp["end_ms"],
                "metadata": {
                    "rule_id":           "INTERROG-CONV-03",
                    "uncertain_pairs":   len(uncertain),
                    "total_markers":     total_markers,
                    "window_pairs":      WINDOW,
                    "memory_challenges": memory_chal,
                    "sample_markers":    sample_markers,
                    "interpretations": [
                        "Genuine memory difficulty — truthful speaker cannot recall details "
                        "(CBCA Criterion 15: admissions of lack of memory are MORE common "
                        "in truthful accounts, Steller & Köhnken 1989)",
                        "Stress response — anxiety-driven uncertainty regardless of truthfulness "
                        "(DePaulo et al. 2003: uncertainty is not a reliable deception discriminator)",
                        "Evasion — speaker knows the answer but avoids committing to a position",
                        "Cognitive overload — rapid or complex questioning causes response fatigue",
                    ],
                    "research": (
                        "CBCA Criterion 15 (Steller & Köhnken 1989): admissions of lack of memory "
                        "are a TRUTHFULNESS indicator — presence suggests genuine recall difficulty. "
                        "Porter & Yuille (1996): memory admissions significantly higher in truthful. "
                        "DePaulo et al. (2003): uncertainty expressions show small/null effect."
                    ),
                    "critical_note": (
                        "COGNITIVE LOAD indicator, NOT a deception indicator. "
                        "Research shows truthful speakers use more uncertainty expressions. "
                        "Does NOT feed FalseConfessionRiskAssessor."
                    ),
                },
            })

        if signals:
            logger.info(
                "[%s] INTERROG-CONV-03: %d verbal uncertainty cluster(s)",
                session_id, len(signals),
            )
        return signals

    @staticmethod
    def _normalise(segments: list[dict]) -> list[dict]:
        """Accept both start_ms/end_ms (int) and start/end (float seconds) formats."""
        out = []
        for seg in segments:
            if "start_ms" in seg:
                start = int(seg["start_ms"])
                end   = int(seg["end_ms"])
            else:
                start = int(float(seg.get("start", 0)) * 1000)
                end   = int(float(seg.get("end",   0)) * 1000)
            text    = str(seg.get("text", "")).strip()
            speaker = str(seg.get("speaker", "unknown"))
            if end > start and text:
                out.append({"speaker": speaker, "start_ms": start, "end_ms": end, "text": text})
        out.sort(key=lambda s: s["start_ms"])
        return out


# ── InterrogatorTechniqueClassifier ──────────────────────────────────────────

class InterrogatorTechniqueClassifier:
    """
    Classifies interrogator behavior as PEACE, Reid, or coercive (INTERROG-CONV-02).

    Frameworks:
      PEACE (UK/Canada/Australia): information-gathering; open-ended questions,
        free narrative, non-accusatory. Research: Williamson (1993).
      Reid Technique (US): accusation-based; direct confrontation, minimisation,
        alternative questions, theme development. Research: Inbau et al. (2001).
      Coercive: explicit threats, false evidence claims, conditional promises —
        legally and ethically problematic. Research: Kassin et al. (2010).

    Output: one session-level signal tagged to the interrogator speaker.
    Confidence cap: 0.55 — linguistic proxies are imperfect technique indicators.
    """

    # PEACE markers — open-ended, information-gathering language
    _PEACE_OPEN = re.compile(
        r"\b(tell me (about|what|how|when|where|why)|"
        r"can you (describe|explain|walk me through|help me understand)|"
        r"what (happened|did you|were you|can you tell)|"
        r"in your own words|take me through|describe (the|what|how)|"
        r"help me understand|I('?d like| want) to understand|"
        r"what (else|more) can you tell|is there anything else|"
        r"could you (expand|elaborate|tell me more)|"
        r"go on|please continue|what do you (remember|recall))\b",
        re.IGNORECASE,
    )

    # Reid — accusatory direct confrontation
    _REID_ACCUSATORY = re.compile(
        r"\b(I know you (did|were|lied)|we know (you|that)|"
        r"you (did|killed|were there|lied|took|stole|shot|hurt)|"
        r"the evidence (shows|proves|indicates) (you|that)|"
        r"you can'?t (deny|explain away|account for)|"
        r"you were (seen|identified|captured on)|"
        r"we (found|have) (your|evidence)|"
        r"your (DNA|prints?|blood|hair|fibres?) (was|were|matched|found)|"
        r"(witnesses?|cameras?|records?) (saw|show|confirm) you)\b",
        re.IGNORECASE,
    )

    # Reid — minimisation and theme development
    _REID_MINIMIZATION = re.compile(
        r"\b(maybe it was (an accident|a mistake|not planned)|"
        r"I (understand|can see) (why|how) (you|this)|"
        r"anyone (could have|might have|would have) (done|reacted)|"
        r"it'?s? (understandable|human|natural|normal)|"
        r"I'?m not (here to judge|judging you)|"
        r"between (you and me|us|you and I)|"
        r"the (judge|jury|prosecutor|court) will (understand|consider|take into account)|"
        r"(things|it) (will|could|might) go (easier|better|smoother) (if|when)|"
        r"this (kind of thing|happens|can happen)|"
        r"I'?ve (seen|heard) (this|worse) before)\b",
        re.IGNORECASE,
    )

    # Reid — alternative questions (forcing a binary choice between two bad options)
    _REID_ALTERNATIVE = re.compile(
        r"\b(did you (plan|premeditate|think about|intend) this or|"
        r"was this (planned|intentional|deliberate|premeditated) or (spontaneous|impulse|accident)|"
        r"did you (mean to|intend to) or (was it|did it just)|"
        r"(alone|by yourself) or (with|together with)|"
        r"(first|one more|only) time or (more than once|before))\b",
        re.IGNORECASE,
    )

    # Coercion — explicit threats about consequences
    _COERCION_THREAT = re.compile(
        r"\b(if you (don'?t|refuse|won'?t) (cooperate|talk|tell)|"
        r"things (will|could|might) get (worse|harder|more serious)|"
        r"(cooperate|tell the truth|confess) or (we|they|I)|"
        r"(additional|more|extra|serious) charges? (if|when|unless)|"
        r"(prison|jail|sentence) (will be|could be|might be) (longer|worse|harder)|"
        r"you'?re (only|just) making (this|things|it) (worse|harder))\b",
        re.IGNORECASE,
    )

    # Coercion — conditional promises (improper inducements)
    _COERCION_PROMISE = re.compile(
        r"\b(if you (tell|confess|admit|cooperate)|"
        r"I (can|will|could) (help|talk to|speak to|put in a word)|"
        r"the (prosecutor|DA|judge|court) (will|might|could) (consider|look|take into account)|"
        r"(deal|plea|arrangement|leniency) (if|when|after) you|"
        r"(cooperating|cooperation) (will|could|might) (help|benefit|matter)|"
        r"(go easier|better for you) (if|when) you)\b",
        re.IGNORECASE,
    )

    def evaluate(
        self,
        segments: list[dict],
        session_id: str = "",
    ) -> list[dict]:
        """
        Classify interrogator technique from full session transcript.
        Returns a list with at most one session-level signal.
        """
        normed = InterrogationConversationRules._normalise(segments)
        if not normed:
            return []

        # Identify ALL interrogators by question proportion (≥15% of all ?s)
        interrogators, interrogator = _detect_interrogators(normed)
        if not interrogators:
            return []

        interrog_segs = [s for s in normed if s["speaker"] == interrogator]

        if len(interrog_segs) < 3:
            return []

        full_text = " ".join(s["text"] for s in interrog_segs)

        peace_count   = len(self._PEACE_OPEN.findall(full_text))
        reid_acc      = len(self._REID_ACCUSATORY.findall(full_text))
        reid_min      = len(self._REID_MINIMIZATION.findall(full_text))
        reid_alt      = len(self._REID_ALTERNATIVE.findall(full_text))
        coerce_threat = len(self._COERCION_THREAT.findall(full_text))
        coerce_promise = len(self._COERCION_PROMISE.findall(full_text))

        reid_total     = reid_acc + reid_min + reid_alt
        coercion_total = coerce_threat + coerce_promise
        total_markers  = peace_count + reid_total

        # Require at least one match to avoid empty-text sessions
        if total_markers == 0 and coercion_total == 0:
            return []

        if coercion_total >= 3:
            technique = "coercive"
            conf = round(min(0.55, 0.25 + coercion_total * 0.05), 4)
        elif total_markers > 0 and reid_total > peace_count * 1.5:
            technique = "reid"
            conf = round(min(0.55, 0.25 + reid_total * 0.04), 4)
        elif total_markers > 0 and peace_count > reid_total * 1.5:
            technique = "peace"
            conf = round(min(0.55, 0.25 + peace_count * 0.04), 4)
        else:
            technique = "mixed"
            conf = round(min(0.40, 0.15 + total_markers * 0.02), 4)

        # value: 0.0 = pure PEACE, 1.0 = pure Reid/coercive
        reid_ratio = round(reid_total / max(total_markers, 1), 4)

        session_start = normed[0]["start_ms"]
        session_end   = normed[-1]["end_ms"]

        logger.info(
            "[%s] InterrogatorTechnique: %s (interrogator=%s PEACE=%d Reid=%d coercion=%d)",
            session_id, technique, interrogator, peace_count, reid_total, coercion_total,
        )
        return [{
            "agent":           "conversation",
            "speaker_id":      interrogator,
            "signal_type":     "interrogator_technique",
            "value":           reid_ratio,
            "value_text":      technique,
            "confidence":      conf,
            "window_start_ms": session_start,
            "window_end_ms":   session_end,
            "metadata": {
                "rule_id":               "INTERROG-CONV-02",
                "interrogator_id":       interrogator,
                "all_interrogators":     sorted(interrogators),
                "technique":             technique,
                "peace_open_count":      peace_count,
                "reid_accusatory_count": reid_acc,
                "reid_minimization_count": reid_min,
                "reid_alternative_count":  reid_alt,
                "coercion_threat_count":   coerce_threat,
                "coercion_promise_count":  coerce_promise,
                "reid_total":            reid_total,
                "coercion_total":        coercion_total,
                "interpretation": (
                    f"Interrogator speech classified as '{technique}'. "
                    f"PEACE open-question markers: {peace_count}. "
                    f"Reid markers (accusatory={reid_acc}, minimisation={reid_min}, "
                    f"alternative={reid_alt}). Coercion markers: {coercion_total}."
                ),
                "context": (
                    "PEACE (UK/Canada): information-gathering, open questions, non-accusatory. "
                    "Reid (US): accusation-based, minimisation, alternative questions. "
                    "Coercive: explicit threats/promises — legally problematic. "
                    "Reid and coercive techniques are associated with elevated false confession rates "
                    "(Meissner et al. 2014 meta-analysis; Kassin et al. 2010)."
                ),
                "research": "Meissner et al. (2014); Williamson (1993); Kassin et al. (2010)",
                "recommendation": (
                    "Cross-reference with false_confession_risk signal. "
                    "Coercive or high-Reid sessions warrant independent legal review "
                    "of any resulting admission or confession."
                ),
            },
        }]
