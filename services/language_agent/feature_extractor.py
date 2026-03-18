"""
NEXUS Language Agent - Feature Extractor
Extracts linguistic features from transcript segments for rule evaluation.

Produces per-utterance feature vectors containing:
  - Sentiment: polarity + magnitude via DistilBERT
  - Buying signal keywords: SPIN-derived pattern matching (Rackham 1988)
  - Objection markers: hedges, resistance patterns, negative framing
  - Power language: Lakoff/O'Barr powerless feature counting
  - Lexical statistics: word count, sentence length, question detection
"""
import re
import logging
from typing import Optional

logger = logging.getLogger("nexus.language.features")

# ── Lazy-loaded sentiment model (loaded once on first call) ──
_sentiment_pipeline = None
_sentiment_backend = "none"  # "distilbert" | "vader" | "none"

# ── VADER fallback ──
_vader_analyzer = None


def _get_vader_analyzer():
    """Get VADER sentiment analyzer (rule-based, no GPU needed)."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded.")
        except ImportError:
            logger.warning("vaderSentiment not installed — sentiment will be neutral.")
    return _vader_analyzer


def _get_sentiment_pipeline():
    """
    Lazy-load DistilBERT sentiment pipeline, with VADER fallback.
    Set SENTIMENT_BACKEND=vader to skip DistilBERT entirely.
    """
    import os
    global _sentiment_pipeline, _sentiment_backend
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    if _sentiment_backend == "vader":
        return None  # Already using VADER

    # Check if we should skip DistilBERT (env var or PyTorch issues)
    force_vader = os.environ.get("SENTIMENT_BACKEND", "auto").lower() == "vader"
    if force_vader:
        logger.info("SENTIMENT_BACKEND=vader — skipping DistilBERT.")
        _sentiment_backend = "vader"
        return None

    # Try DistilBERT
    try:
        from transformers import pipeline as tf_pipeline
        logger.info("Loading DistilBERT sentiment model...")
        _sentiment_pipeline = tf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,  # CPU; use 0 for GPU
            truncation=True,
            max_length=512,
        )
        _sentiment_backend = "distilbert"
        logger.info("Sentiment model loaded (DistilBERT).")
        return _sentiment_pipeline
    except Exception as e:
        logger.warning(f"DistilBERT failed ({e}), falling back to VADER...")
        _sentiment_pipeline = None
        _sentiment_backend = "vader"
        return None


# ═══════════════════════════════════════════════════════════════
# Buying Signal Patterns — Rackham 1988 (SPIN Selling, 35K calls)
# ═══════════════════════════════════════════════════════════════

BUYING_SIGNAL_PATTERNS = {
    "future_projection": [
        r"\bwhen\s+we\s+(implement|deploy|use|integrate|start|launch|adopt|roll\s*out)\b",
        r"\bonce\s+(this|it|that)\s+is\s+(set\s+up|running|live|deployed|integrated)\b",
        r"\bafter\s+we\s+(sign|go\s+live|onboard|get\s+started)\b",
        r"\bif\s+we\s+(were\s+to|decide\s+to|go\s+with|move\s+forward)\b",
    ],
    "usage_scenario": [
        r"\bhow\s+would\s+(this|it|that)\s+work\s+with\b",
        r"\bcould\s+we\s+use\s+(this|it|that)\s+for\b",
        r"\bwould\s+(this|it|that)\s+(handle|support|integrate\s+with|work\s+for)\b",
        r"\bcan\s+(this|it|that)\s+do\b",
        r"\bhow\s+does\s+(this|it|that)\s+(handle|deal\s+with|manage)\b",
    ],
    "specification_question": [
        r"\bwhat('s|\s+is)\s+the\s+(timeline|pricing|cost|price|lead\s+time)\b",
        r"\bhow\s+many\s+(users|seats|licenses|instances)\b",
        r"\bwhat\s+(are\s+the|kind\s+of)\s+(specs|features|capabilities|requirements)\b",
        r"\bwhat('s|\s+is)\s+included\b",
        r"\bwhat('s|\s+is)\s+the\s+(SLA|uptime|guarantee)\b",
    ],
    "social_proof": [
        r"\bwho\s+else\s+(uses|has\s+used|is\s+using)\b",
        r"\bcan\s+I\s+(talk|speak)\s+to\s+a\s+(reference|customer)\b",
        r"\bdo\s+you\s+have\s+(any\s+)?(case\s+studies|testimonials|references)\b",
        r"\bany\s+(other\s+)?(companies|clients|customers)\s+(in|like)\b",
    ],
    "price_terms": [
        r"\bwhat('s|\s+is)\s+the\s+(cost|price|pricing|fee|rate)\b",
        r"\bare\s+there\s+(any\s+)?(discounts|deals|promotions)\b",
        r"\bwhat\s+(are\s+the|about)\s+(terms|payment\s+terms|contract)\b",
        r"\bcan\s+we\s+(negotiate|get\s+a\s+discount|do\s+a\s+trial)\b",
        r"\bhow\s+much\s+(does|would|is)\b",
    ],
    "implementation_question": [
        r"\bhow\s+long\s+(to|does\s+it\s+take\s+to)\s+(deploy|implement|set\s+up|onboard)\b",
        r"\bwhat('s|\s+is)\s+the\s+(onboarding|implementation|setup)\s*(process|like)?\b",
        r"\bwhat\s+do\s+we\s+need\s+to\s+(get\s+started|prepare|do\s+on\s+our\s+end)\b",
        r"\bhow\s+(long|quickly|soon)\s+(can|could|would)\b",
    ],
    "positive_reframing": [
        r"\bso\s+(basically|essentially|what\s+you're\s+saying\s+is)\b",
        r"\bthat\s+(would|could)\s+(really|definitely|certainly)\s+help\b",
        r"\bthat('s|\s+is)\s+(exactly|precisely)\s+what\s+we\s+need\b",
        r"\bthat\s+makes\s+(a\s+lot\s+of\s+)?sense\b",
    ],
    # ── Conversational buying signals (cold call / informal sales) ──
    "specification_question_conversational": [
        # Prospect researching the seller's fit — "have you worked in education?"
        r"\bhave\s+you\s+(worked|dealt|done\s+work)\s+(with|in|for)\b",
        r"\bdo\s+you\s+(work|deal|speciali[sz]e)\s+(with|in)\b",
        r"\bhave\s+you\s+(ever|already)\s+(done|handled|managed)\b",
        r"\bwhat\s+kind\s+of\s+(clients|companies|businesses|work)\b",
        r"\bwhat\s+(industries|sectors|areas)\s+do\s+you\b",
        r"\bdo\s+you\s+have\s+experience\s+(with|in)\b",
    ],
    "next_step_acceptance": [
        # Agreeing to next steps — "send me the proposal", "let's hop on a call"
        r"\bsend\s+(me|us|it|that|the|your)\b",
        r"\blet('s|\s+us)\s+(schedule|set\s+up|book|arrange|hop\s+on|do)\b",
        r"\bI('d|\s+would)\s+(love|like)\s+to\s+(see|hear|learn|know|get|review)\b",
        r"\b(yeah|yes|sure|okay|ok)\s*(,|\.|\s)+(send|let's|that\s+works|that\s+sounds|I'll)\b",
        r"\bsounds\s+(good|great|interesting|like\s+a\s+plan)\b",
        r"\bwhen\s+can\s+(we|you|I)\b",
        r"\b(go\s+ahead|do\s+it|let's\s+do\s+it|I'm\s+in|count\s+me\s+in)\b",
    ],
    "information_sharing": [
        # Prospect voluntarily shares contact info — "my email is...", "you can reach me at"
        r"\b(my|our)\s+(email|number|phone|address|contact)\s+(is|address)\b",
        r"\b(here's|here\s+is)\s+(my|our)\b",
        r"\byou\s+can\s+(reach|contact|email|call)\s+(me|us)\b",
        r"\bI('ll|\s+will)\s+(send|email|forward|share)\s+(you|it|that)\b",
        r"\blet\s+me\s+give\s+you\s+(my|our)\b",
        r"\b(it's|its)\s+\S+@\S+\b",
    ],
    "risk_evaluation": [
        # Prospect evaluating risks — serious consideration of the offering
        r"\bwhat\s+if\s+(it|this|things|we)\b",
        r"\bworst\s+case\b",
        r"\bwhat\s+(are\s+the\s+)?risks?\b",
        r"\bwhat\s+happens\s+if\b",
        r"\bany\s+(downsides?|risks?|concerns?|limitations?)\b",
        r"\bcan\s+we\s+(cancel|opt\s+out|back\s+out|stop)\b",
        r"\bwhat('s|\s+is)\s+(the\s+)?guarantee\b",
    ],
    "conditional_interest": [
        # "If you could X, then..." — engaged enough to negotiate terms
        r"\bif\s+you\s+(could|can|were\s+able\s+to)\b",
        r"\bas\s+long\s+as\b",
        r"\bprovided\s+(that|you)\b",
        r"\bassuming\s+(you|it|this|that)\b",
        r"\bwould\s+you\s+be\s+(able|willing|open)\s+to\b",
        r"\bI('d|\s+would)\s+be\s+(interested|open|willing)\s+if\b",
    ],
}

# Pre-compile all patterns
BUYING_PATTERNS_COMPILED = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in BUYING_SIGNAL_PATTERNS.items()
}

# ═══════════════════════════════════════════════════════════════
# Objection / Resistance Patterns — Rackham 1988
# ═══════════════════════════════════════════════════════════════

OBJECTION_PATTERNS = {
    "direct_objection": [
        r"\bthat('s|\s+is)\s+(too\s+)?(expensive|costly|pricey)\b",
        r"\bwe\s+(can't|cannot|don't)\s+(afford|justify|approve)\b",
        r"\bthat('s|\s+is)\s+not\s+(going\s+to|gonna)\s+work\b",
        r"\bwe\s+(already|currently)\s+(have|use)\b",
        r"\bwe('re|\s+are)\s+(not\s+)?(interested|looking|ready)\b",
        r"\bI\s+don't\s+(think|see|believe)\s+(this|it|that)\b",
    ],
    "timing_objection": [
        r"\bnot\s+(right\s+)?now\b",
        r"\bmaybe\s+(later|next\s+(quarter|year|month))\b",
        r"\bthe\s+timing\s+(isn't|is\s+not)\s+(right|good|ideal)\b",
        r"\bwe('re|\s+are)\s+(not\s+)?ready\b",
        r"\bneed\s+more\s+time\b",
    ],
    "authority_objection": [
        r"\bI('d|\s+would)\s+(need|have)\s+to\s+(check|ask|run\s+it\s+by|get\s+approval)\b",
        r"\bthat('s|\s+is)\s+not\s+(my|our)\s+decision\b",
        r"\bI('m|\s+am)\s+not\s+the\s+(right|decision)\b",
        r"\bneed\s+to\s+(involve|consult|talk\s+to)\b",
    ],
    "competitor_comparison": [
        r"\bwe('re|\s+are)\s+(also\s+)?(looking|evaluating|considering)\s+(at\s+)?(other|alternative)\b",
        r"\bhow\s+(do|does)\s+(this|it|you)\s+compare\s+(to|with|against)\b",
        r"\bwhat\s+makes\s+(this|you)\s+(different|better|unique)\b",
    ],
}

OBJECTION_PATTERNS_COMPILED = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in OBJECTION_PATTERNS.items()
}

# ═══════════════════════════════════════════════════════════════
# Power Language — Lakoff 1975, O'Barr & Atkins 1982
# ═══════════════════════════════════════════════════════════════

POWERLESS_HEDGES = [
    "kind of", "sort of", "maybe", "perhaps", "i think", "i guess",
    "i suppose", "it seems", "it appears", "more or less", "possibly",
    "probably", "somewhat", "a little", "a bit", "might be",
    "could be", "i believe", "in my opinion", "if you ask me",
]

POWERLESS_TAG_QUESTIONS = [
    r",?\s*(right|isn't it|don't you think|wouldn't you say|you know)\s*\??\s*$",
    r",?\s*(doesn't it|won't it|can't it|isn't that right)\s*\??\s*$",
]
POWERLESS_TAG_COMPILED = [re.compile(p, re.IGNORECASE) for p in POWERLESS_TAG_QUESTIONS]

POWERLESS_INTENSIFIERS = [
    "so", "very", "really", "extremely", "absolutely", "totally",
    "completely", "definitely", "certainly", "honestly", "literally",
]

POWERLESS_HESITATIONS = [
    "well", "you know", "i mean", "like", "basically", "actually",
    "you see", "the thing is",
]

POWERLESS_POLITE_FORMS = [
    "if you don't mind", "i was wondering if", "would it be possible",
    "i'm sorry but", "excuse me but", "if i may", "with all due respect",
    "i hate to ask but", "if it's not too much trouble",
]


class LanguageFeatureExtractor:
    """Extract linguistic features from transcript segments."""

    def __init__(self):
        self._sentiment_ready = False

    def warm_up(self):
        """Pre-load the sentiment model so first analysis isn't slow."""
        try:
            _get_sentiment_pipeline()
        except Exception:
            pass  # VADER fallback is automatic
        # Also ensure VADER is ready as fallback
        _get_vader_analyzer()
        self._sentiment_ready = True

    def extract_all(self, segments: list[dict]) -> list[dict]:
        """
        Extract linguistic features for every segment.

        Args:
            segments: list of transcript segments from Voice Agent
                      [{speaker, start_ms, end_ms, text, words}, ...]

        Returns:
            List of feature dicts, one per segment:
            [{
                speaker_id, start_ms, end_ms, text,
                sentiment_label, sentiment_score,
                buying_signals, objection_signals,
                power_score, powerless_features,
                is_question, word_count, ...
            }, ...]
        """
        if not segments:
            return []

        # ── Batch sentiment for all segments at once (much faster) ──
        texts = [seg.get("text", "") for seg in segments]
        sentiments = self._batch_sentiment(texts)

        features_list = []
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            if not text:
                continue

            features = {
                "speaker_id": seg.get("speaker", "unknown"),
                "start_ms": seg.get("start_ms", 0),
                "end_ms": seg.get("end_ms", 0),
                "text": text,
            }

            # Sentiment
            sent = sentiments[i] if i < len(sentiments) else {"label": "NEUTRAL", "score": 0.5}
            features["sentiment_label"] = sent["label"]
            features["sentiment_score"] = sent["score"]
            # Convert to -1..+1 scale: POSITIVE → +score, NEGATIVE → -score
            if sent["label"] == "NEGATIVE":
                features["sentiment_value"] = -sent["score"]
            else:
                features["sentiment_value"] = sent["score"]

            # Buying signals
            buying = self._detect_buying_signals(text)
            features["buying_signals"] = buying["matches"]
            features["buying_signal_count"] = buying["count"]
            features["buying_categories"] = buying["categories"]

            # Objection signals
            objections = self._detect_objection_signals(text)
            features["objection_signals"] = objections["matches"]
            features["objection_signal_count"] = objections["count"]
            features["objection_categories"] = objections["categories"]

            # Power language
            power = self._score_power_language(text)
            features["power_score"] = power["score"]
            features["powerless_feature_count"] = power["powerless_count"]
            features["powerless_features_found"] = power["features_found"]
            features["power_word_count"] = power["word_count"]

            # Lexical basics
            features["word_count"] = len(text.split())
            features["is_question"] = text.rstrip().endswith("?")
            features["sentence_count"] = max(1, len(re.split(r'[.!?]+', text.strip())))

            features_list.append(features)

        return features_list

    # ── Sentiment (DistilBERT with VADER fallback) ──

    def _batch_sentiment(self, texts: list[str]) -> list[dict]:
        """
        Run sentiment analysis on a batch of texts.
        Uses DistilBERT if available, falls back to VADER (rule-based).
        Returns list of {label: POSITIVE|NEGATIVE, score: 0.0-1.0}.
        """
        if not texts:
            return []

        # Filter out empty strings
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            return [{"label": "NEUTRAL", "score": 0.5}] * len(texts)

        # Try DistilBERT first
        pipe = _get_sentiment_pipeline()
        if pipe is not None and _sentiment_backend == "distilbert":
            try:
                results = pipe(valid_texts, batch_size=32)
                output = [{"label": "NEUTRAL", "score": 0.5}] * len(texts)
                for idx, result in zip(valid_indices, results):
                    output[idx] = {
                        "label": result["label"],
                        "score": round(result["score"], 4),
                    }
                return output
            except Exception as e:
                logger.warning(f"DistilBERT batch failed: {e}, falling back to VADER")

        # VADER fallback (rule-based, no GPU needed)
        vader = _get_vader_analyzer()
        if vader is not None:
            output = [{"label": "NEUTRAL", "score": 0.5}] * len(texts)
            for idx in valid_indices:
                text = texts[idx]
                scores = vader.polarity_scores(text)
                compound = scores["compound"]
                if compound > 0.05:
                    output[idx] = {"label": "POSITIVE", "score": round(min(1.0, 0.5 + compound / 2), 4)}
                elif compound < -0.05:
                    output[idx] = {"label": "NEGATIVE", "score": round(min(1.0, 0.5 + abs(compound) / 2), 4)}
                else:
                    output[idx] = {"label": "NEUTRAL", "score": 0.5}
            return output

        # Ultimate fallback — everything neutral
        logger.warning("No sentiment backend available — returning neutral")
        return [{"label": "NEUTRAL", "score": 0.5}] * len(texts)

    # ── Buying Signals (Rackham 1988 SPIN patterns) ──

    def _detect_buying_signals(self, text: str) -> dict:
        """
        Detect buying signal keywords/phrases from SPIN Selling research.
        Returns matched categories and specific patterns.
        """
        matches = []
        categories_found = set()

        for category, patterns in BUYING_PATTERNS_COMPILED.items():
            for pattern in patterns:
                m = pattern.search(text)
                if m:
                    matches.append({
                        "category": category,
                        "matched_text": m.group(0),
                        "position": m.start(),
                    })
                    categories_found.add(category)
                    break  # One match per category per utterance is enough

        return {
            "matches": matches,
            "count": len(matches),
            "categories": sorted(categories_found),
        }

    # ── Objection Signals (Rackham 1988) ──

    def _detect_objection_signals(self, text: str) -> dict:
        """
        Detect objection/resistance patterns.
        """
        matches = []
        categories_found = set()

        for category, patterns in OBJECTION_PATTERNS_COMPILED.items():
            for pattern in patterns:
                m = pattern.search(text)
                if m:
                    matches.append({
                        "category": category,
                        "matched_text": m.group(0),
                        "position": m.start(),
                    })
                    categories_found.add(category)
                    break

        # Also count hedges as weak objection indicators
        text_lower = text.lower()
        hedge_count = sum(1 for h in POWERLESS_HEDGES if h in text_lower)
        if hedge_count >= 2 and not matches:
            matches.append({
                "category": "hedge_cluster",
                "matched_text": f"{hedge_count} hedges detected",
                "position": 0,
            })
            categories_found.add("hedge_cluster")

        return {
            "matches": matches,
            "count": len(matches),
            "categories": sorted(categories_found),
        }

    # ── Power Language (Lakoff 1975, O'Barr 1982) ──

    def _score_power_language(self, text: str) -> dict:
        """
        Score power/powerless language using Lakoff/O'Barr feature counting.

        Power score = 1.0 - (powerless_features / total_words × normalisation_factor)
        Higher = more powerful speech. Lower = more tentative/hedging.
        """
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        if word_count == 0:
            return {"score": 0.5, "powerless_count": 0, "features_found": [], "word_count": 0}

        features_found = []

        # Count hedges
        for hedge in POWERLESS_HEDGES:
            # Use word boundary matching for multi-word hedges
            count = text_lower.count(hedge)
            if count > 0:
                features_found.extend([f"hedge:{hedge}"] * count)

        # Count tag questions
        for pattern in POWERLESS_TAG_COMPILED:
            if pattern.search(text):
                features_found.append("tag_question")

        # Count intensifiers (only flagged when overused — 2+ in short utterance)
        intensifier_count = 0
        for intensifier in POWERLESS_INTENSIFIERS:
            for word in words:
                if word.strip(".,!?;:'\"") == intensifier:
                    intensifier_count += 1
        if intensifier_count >= 2 or (intensifier_count >= 1 and word_count < 8):
            features_found.extend([f"intensifier"] * intensifier_count)

        # Count hesitation forms
        for hesitation in POWERLESS_HESITATIONS:
            if hesitation in text_lower:
                features_found.append(f"hesitation:{hesitation}")

        # Count polite forms
        for polite in POWERLESS_POLITE_FORMS:
            if polite in text_lower:
                features_found.append(f"polite:{polite}")

        powerless_count = len(features_found)

        # Normalisation: ~20 words per feature is the expected "normal" rate.
        # More features per word = more powerless.
        normalisation_factor = 20.0
        raw_ratio = powerless_count / word_count * normalisation_factor
        score = max(0.0, min(1.0, 1.0 - raw_ratio))

        return {
            "score": round(score, 4),
            "powerless_count": powerless_count,
            "features_found": features_found[:10],  # Cap to prevent huge metadata
            "word_count": word_count,
        }
