# services/conversation_agent/interrogation_rules.py
"""
Interrogation-specific conversation rules (NEXUS INTERROGATION_IMPLEMENTATION.MD v2.0).

Rules implemented:
  INTERROG-CONV-01  Evidence Response Processing Delay  (conf 0.65 — Hartwig et al. 2014-2016, d=1.83)

Research: SUE (Strategic Use of Evidence) framework. Long response latency after
evidence disclosure is the highest-evidence interrogation signal (85% field accuracy),
but is equally present in innocent suspects confronted with fabricated or unexpected evidence.
Confidence scales from 0.40 → 0.85 with latency magnitude (§5 table: 0.85 all quality tiers
because signal is timestamp-based, not video/audio quality dependent).
"""
from __future__ import annotations

import re
import logging
from collections import defaultdict

logger = logging.getLogger("nexus.conversation.interrogation")

# Evidence disclosure keywords — interrogator presents facts/evidence
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
    r"your (blood|DNA|prints?|hair|fibres?) (was|were|matched|found)"
    r")\b",
    re.IGNORECASE,
)

# Latency threshold: >2000ms is "extended" per SUE framework spec
_LATENCY_THRESHOLD_MS = 2_000

# Do not associate a response if the gap to the next speaker exceeds this
# (interrogator asked multiple questions in between, or there is a long unrelated pause)
_MAX_ASSOCIATION_GAP_MS = 20_000


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

        signals: list[dict] = []
        for i, turn in enumerate(normed[:-1]):
            if not _EVIDENCE_RE.search(turn["text"]):
                continue

            # Find the immediately next turn from a different speaker
            next_turn: dict | None = None
            for j in range(i + 1, len(normed)):
                if normed[j]["speaker"] != turn["speaker"]:
                    next_turn = normed[j]
                    break

            if next_turn is None:
                continue

            latency_ms = next_turn["start_ms"] - turn["end_ms"]
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

        # Identify interrogator: speaker with most question marks
        question_counts: dict[str, int] = defaultdict(int)
        for seg in normed:
            if "?" in seg["text"]:
                question_counts[seg["speaker"]] += seg["text"].count("?")

        if not question_counts:
            return []

        interrogator = max(question_counts, key=question_counts.get)
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
