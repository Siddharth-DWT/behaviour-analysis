# services/language_agent/tests/test_interrogation_rules.py
"""
Validation tests for INTERROG-LANG-03 (contamination) and INTERROG-LANG-04 (denial).

Run with:  pytest backend/services/language_agent/tests/test_interrogation_rules.py -v
"""
import sys
import os
import pytest

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from backend.services.language_agent.interrogation_rules import (
    ContaminationDetector,
    InterrogationLanguageRules,
    _classify_denial,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _seg(speaker: str, text: str, start_ms: int, end_ms: int) -> dict:
    return {"speaker": speaker, "text": text, "start_ms": start_ms, "end_ms": end_ms}


DETECTIVE = "detective"
SUSPECT   = "suspect"


# ── TestContaminationDetector ─────────────────────────────────────────────────

class TestContaminationDetector:

    def setup_method(self):
        self.det = ContaminationDetector()

    def test_filters_common_words(self):
        """False positives from confirmed session: yeah/guess/told must NOT fire."""
        segments = [
            _seg(DETECTIVE, "Yeah, we know what happened there.", 0, 2000),
            _seg(SUSPECT,   "Yeah, I told you I don't know anything.", 5000, 7000),
            _seg(DETECTIVE, "I guess you can explain the situation.", 8000, 10000),
            _seg(SUSPECT,   "I guess I was just there that night.", 12000, 14000),
        ]
        signals = self.det.detect(segments, DETECTIVE)
        signal_types = [s["signal_type"] for s in signals]
        assert "statement_contamination" not in signal_types, (
            "Common words (yeah/guess/told) must not trigger contamination"
        )

    def test_detects_case_specific_entity(self):
        """Proper nouns (PERSON name, location) should be flagged when adopted."""
        segments = [
            _seg(DETECTIVE, "We have witnesses near Riverside Mall at midnight.", 0, 3000),
            _seg(DETECTIVE, "Your friend Marcus was seen there.", 3000, 6000),
            _seg(SUSPECT,   "I don't know anything about Riverside Mall or Marcus.", 10000, 14000),
            _seg(SUSPECT,   "Marcus never called me that night.", 15000, 18000),
        ]
        signals = self.det.detect(segments, DETECTIVE, min_matches=2)
        contamination = [s for s in signals if s["signal_type"] == "statement_contamination"]
        assert len(contamination) >= 1, (
            "Case-specific proper nouns (Marcus, Riverside Mall) should trigger contamination"
        )

    def test_respects_temporal_ordering(self):
        """Suspect using a word BEFORE interrogator introduces it must NOT fire."""
        segments = [
            _seg(SUSPECT,   "I was driving my Toyota down Lincoln Avenue.", 0, 3000),
            _seg(DETECTIVE, "We found your Toyota near Lincoln Avenue.", 5000, 8000),
            _seg(SUSPECT,   "Like I said, I drove my Toyota on Lincoln Avenue.", 10000, 13000),
        ]
        # The suspect used "toyota" and "lincoln" before the detective — no contamination
        signals = self.det.detect(segments, DETECTIVE)
        contamination = [s for s in signals if s["signal_type"] == "statement_contamination"]
        assert len(contamination) == 0, (
            "Terms suspect used before interrogator must not count as contaminated"
        )

    def test_requires_minimum_matches(self):
        """Single contaminated term below min_matches threshold must not fire."""
        segments = [
            _seg(DETECTIVE, "We found the weapon near Oakwood Park.", 0, 3000),
            _seg(SUSPECT,   "I was nowhere near Oakwood Park.", 5000, 8000),
        ]
        signals = self.det.detect(segments, DETECTIVE, min_matches=2)
        contamination = [s for s in signals if s["signal_type"] == "statement_contamination"]
        assert len(contamination) == 0, (
            "Single match below min_matches=2 must not produce a signal"
        )

    def test_single_match_fires_at_min_1(self):
        """With min_matches=1, a single entity adoption should fire."""
        segments = [
            _seg(DETECTIVE, "The victim was identified as Jennifer Walsh.", 0, 3000),
            _seg(SUSPECT,   "I never met Jennifer Walsh in my life.", 5000, 8000),
        ]
        signals = self.det.detect(segments, DETECTIVE, min_matches=1)
        contamination = [s for s in signals if s["signal_type"] == "statement_contamination"]
        assert len(contamination) >= 1

    def test_graceful_degradation_no_crash(self):
        """Must not crash with empty segments or missing interrogator."""
        assert self.det.detect([], DETECTIVE) == []
        assert self.det.detect([], "") == []
        segments = [_seg(SUSPECT, "I didn't do anything.", 0, 2000)]
        result = self.det.detect(segments, DETECTIVE)
        assert isinstance(result, list)

    def test_signal_structure(self):
        """Output signals must have all required fields."""
        segments = [
            _seg(DETECTIVE, "Forensics found blood near Greenfield Road.", 0, 3000),
            _seg(DETECTIVE, "The victim Marcus Johnson was there.", 3000, 6000),
            _seg(SUSPECT,   "I never went to Greenfield Road or near Marcus Johnson.", 10000, 14000),
        ]
        signals = self.det.detect(segments, DETECTIVE, min_matches=1)
        for sig in signals:
            for field in ("agent", "speaker_id", "signal_type", "value", "confidence",
                          "window_start_ms", "window_end_ms", "metadata"):
                assert field in sig, f"Missing field: {field}"
            assert sig["confidence"] == 0.80
            assert sig["signal_type"] == "statement_contamination"
            assert "detection_method" in sig["metadata"]


# ── TestAdaptiveDenialThreshold ───────────────────────────────────────────────

class TestAdaptiveDenialThreshold:

    def setup_method(self):
        self.rules = InterrogationLanguageRules()

    def _make_denial_segments(self, labels_at_minutes: list[tuple[float, str]]) -> list[dict]:
        """Build suspect segments with denial labels at given minute offsets."""
        texts = {
            "categorical":   "I did not do it, I am innocent.",
            "strong":        "I didn't do anything, I wasn't there.",
            "weak":          "I don't think I did that, I don't remember.",
            "acquiescence":  "Maybe I did, I suppose it's possible.",
        }
        segs = []
        for minute, label in labels_at_minutes:
            ms = int(minute * 60_000)
            segs.append(_seg(SUSPECT, texts[label], ms, ms + 3000))
        return segs

    def test_does_not_fire_on_stable_denials(self):
        """Consistent categorical denials across 30 min must not fire."""
        segs = self._make_denial_segments([
            (0, "categorical"), (5, "categorical"), (10, "categorical"),
            (15, "categorical"), (20, "categorical"), (25, "categorical"),
        ])
        signals = self.rules._denial_evolution(segs, None)
        assert len(signals) == 0, "Stable denials must not fire denial_weakening"

    def test_fires_on_clear_weakening_long_session(self):
        """
        54-min session: categorical → strong → weak → acquiescence.
        Windowed trigger should fire (first-third mean >> last-third mean).
        """
        segs = self._make_denial_segments([
            (2,  "categorical"), (5,  "categorical"), (8,  "categorical"),
            (15, "strong"),      (20, "strong"),
            (30, "weak"),        (35, "weak"),
            (45, "acquiescence"),(50, "acquiescence"),
        ])
        signals = self.rules._denial_evolution(segs, None)
        denial_sigs = [s for s in signals if s["signal_type"] == "denial_weakening"]
        assert len(denial_sigs) >= 1, (
            "Clear categorical→acquiescence trajectory in 54-min session must fire"
        )
        sig = denial_sigs[0]
        assert sig["metadata"]["trigger"] in ("window", "slope+window")

    def test_windowed_comparison_fires_when_slope_borderline(self):
        """
        Gradual weakening over 60 min where slope alone is borderline
        but windowed drop > 0.10 (threshold for ≤90 min sessions).
        """
        segs = self._make_denial_segments([
            (0,  "categorical"), (5,  "categorical"),
            (20, "strong"),
            (40, "weak"),
            (55, "weak"),        (58, "acquiescence"),
        ])
        signals = self.rules._denial_evolution(segs, None)
        denial_sigs = [s for s in signals if s["signal_type"] == "denial_weakening"]
        assert len(denial_sigs) >= 1

    def test_short_session_requires_larger_drop(self):
        """
        12-min session: weak→acquiescence (drop=0.2) should trigger
        since min_windowed_drop=0.20 for ≤15 min sessions.
        """
        segs = self._make_denial_segments([
            (0, "categorical"), (3, "strong"), (6, "weak"), (10, "acquiescence"),
        ])
        signals = self.rules._denial_evolution(segs, None)
        denial_sigs = [s for s in signals if s["signal_type"] == "denial_weakening"]
        assert len(denial_sigs) >= 1

    def test_confidence_scales_with_duration(self):
        """Longer sessions with more denial events should produce higher confidence."""
        short_segs = self._make_denial_segments([
            (0, "categorical"), (3, "strong"), (7, "acquiescence"),
        ])
        long_segs = self._make_denial_segments([
            (2,  "categorical"), (8,  "categorical"),
            (20, "strong"),      (30, "strong"),
            (45, "weak"),        (55, "acquiescence"), (60, "acquiescence"),
        ])
        short_sigs = self.rules._denial_evolution(short_segs, None)
        long_sigs  = self.rules._denial_evolution(long_segs,  None)

        if short_sigs and long_sigs:
            short_conf = short_sigs[0]["confidence"]
            long_conf  = long_sigs[0]["confidence"]
            assert long_conf >= short_conf, (
                "Longer session with more denial events should have ≥ confidence"
            )

    def test_signal_has_trigger_field(self):
        """Output signal metadata must include trigger field."""
        segs = self._make_denial_segments([
            (0, "categorical"), (10, "strong"), (30, "weak"), (50, "acquiescence"),
        ])
        signals = self.rules._denial_evolution(segs, None)
        for sig in signals:
            assert "trigger" in sig["metadata"]
            assert sig["metadata"]["trigger"] in ("slope", "window", "slope+window")

    def test_skips_speakers_with_fewer_than_3_denials(self):
        """Must require ≥3 denial events before computing slope."""
        segs = [
            _seg(SUSPECT, "I did not do it.", 0, 2000),
            _seg(SUSPECT, "Maybe I did.",     60_000, 62_000),
        ]
        signals = self.rules._denial_evolution(segs, None)
        assert len(signals) == 0, "< 3 denial events must not produce a signal"


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:

    def setup_method(self):
        self.rules = InterrogationLanguageRules()

    def test_full_pipeline_returns_list(self):
        """evaluate_all must return a list without crashing on minimal input."""
        segments = [
            _seg(DETECTIVE, "Where were you on the night of the 12th?", 0, 3000),
            _seg(SUSPECT,   "I was home all night. I did not go out.", 4000, 7000),
            _seg(DETECTIVE, "We have GPS data showing your phone near the scene.", 8000, 12000),
            _seg(SUSPECT,   "I did not do it. I was home.", 13000, 16000),
        ]
        result = self.rules.evaluate_all(segments, [])
        assert isinstance(result, list)

    def test_no_false_positives_on_filler_words(self):
        """
        Realistic interrogation filler speech must not produce contamination signals.
        Validates the confirmed false positives from the 54-minute session.
        """
        segments = [
            _seg(DETECTIVE, "Yeah, okay so tell me what happened.", 0, 3000),
            _seg(SUSPECT,   "Yeah, I don't know what you're talking about.", 4000, 7000),
            _seg(DETECTIVE, "I guess you were driving around, right? Going back and forth?", 8000, 12000),
            _seg(SUSPECT,   "I was driving back home, I guess.", 13000, 16000),
            _seg(DETECTIVE, "And you told other people about this?", 17000, 20000),
            _seg(SUSPECT,   "I didn't tell anyone anything.", 21000, 24000),
        ]
        result = self.rules.evaluate_all(segments, [])
        contamination = [s for s in result if s["signal_type"] == "statement_contamination"]
        assert len(contamination) == 0, (
            f"Filler words should not produce contamination. Got: "
            f"{[s['metadata'].get('contaminated_terms') for s in contamination]}"
        )

    def test_realistic_54min_session_contamination_count(self):
        """
        Full realistic session: contamination should fire on genuine evidence terms,
        not on common words. Expects 1-4 hits (not 26 as in the unfixed version).
        """
        segments = [
            # Early interrogator introduces case facts
            _seg(DETECTIVE, "We have footage from the parking lot of Westgate Mall.", 60_000, 65_000),
            _seg(DETECTIVE, "Your car, a blue Honda, was captured on camera.", 65_000, 70_000),
            _seg(DETECTIVE, "A witness, Terry Hoffman, says he saw you there.", 70_000, 76_000),

            # Common filler back-and-forth (should NOT trigger)
            _seg(SUSPECT,   "I don't know what you're talking about.", 80_000, 84_000),
            _seg(DETECTIVE, "Yeah, okay, but the records show you were there.", 85_000, 89_000),
            _seg(SUSPECT,   "Yeah, well, I wasn't there that night.", 90_000, 94_000),
            _seg(DETECTIVE, "We also have your phone pinging nearby.", 95_000, 99_000),
            _seg(SUSPECT,   "I told you, I was back home all night.", 100_000, 104_000),

            # Later: suspect adopts case-specific terms (SHOULD trigger)
            _seg(SUSPECT,   "I was nowhere near Westgate Mall or any Honda.", 200_000, 205_000),
            _seg(SUSPECT,   "Terry Hoffman is lying, I never saw Terry Hoffman.", 210_000, 216_000),
        ]
        result = self.rules.evaluate_all(segments, [])
        contamination = [s for s in result if s["signal_type"] == "statement_contamination"]
        # Should be 1-4: "westgate mall", "honda", "terry hoffman" are genuine
        assert 1 <= len(contamination) <= 4, (
            f"Expected 1-4 genuine contamination signals, got {len(contamination)}: "
            f"{[s['metadata'].get('contaminated_terms') for s in contamination]}"
        )
