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
import time
import logging
from typing import Optional

logger = logging.getLogger("nexus.language.features")

# ── Lazy-loaded sentiment model (loaded once on first call) ──
_sentiment_model = None
_sentiment_tokenizer = None
_sentiment_backend = "none"  # "onnx_int8" | "onnx_fp32" | "pytorch" | "vader" | "none"

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


def _load_sentiment_model():
    """
    Lazy-load sentiment model with cascading fallback:
    1. DistilBERT ONNX INT8 (fastest, ~1-3ms/sample)
    2. DistilBERT ONNX FP32 (fast, ~3-5ms/sample)
    3. DistilBERT PyTorch (slow, ~15ms/sample)
    4. VADER (rule-based, no model needed)
    Set SENTIMENT_BACKEND=vader to skip all models.
    """
    import os
    global _sentiment_model, _sentiment_tokenizer, _sentiment_backend

    if _sentiment_backend != "none":
        return  # Already loaded or explicitly set

    force_vader = os.environ.get("SENTIMENT_BACKEND", "auto").lower() == "vader"
    if force_vader:
        logger.info("SENTIMENT_BACKEND=vader — skipping DistilBERT.")
        _sentiment_backend = "vader"
        return

    # Try ONNX models first (INT8 then FP32)
    model_paths = [
        ("models/distilbert-onnx-int8", "onnx_int8"),
        ("models/distilbert-onnx", "onnx_fp32"),
    ]
    # Also check under /app/models inside Docker
    for prefix in ["", "/app/"]:
        for model_dir, backend_name in model_paths:
            full_path = prefix + model_dir
            onnx_file = os.path.join(full_path, "model.onnx")
            if os.path.exists(onnx_file):
                try:
                    from optimum.onnxruntime import ORTModelForSequenceClassification
                    from transformers import AutoTokenizer

                    _sentiment_tokenizer = AutoTokenizer.from_pretrained(full_path)
                    _sentiment_model = ORTModelForSequenceClassification.from_pretrained(
                        full_path, provider="CPUExecutionProvider",
                    )
                    _sentiment_backend = backend_name
                    logger.info(f"Loaded sentiment model: {backend_name} from {full_path}")
                    # Warmup
                    inputs = _sentiment_tokenizer(
                        "test", return_tensors="np", padding=True, truncation=True, max_length=128
                    )
                    _sentiment_model(**inputs)
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {backend_name} from {full_path}: {e}")

    # Fallback to PyTorch DistilBERT
    try:
        from transformers import pipeline as tf_pipeline
        logger.info("Loading DistilBERT sentiment model (PyTorch fallback)...")
        _sentiment_model = tf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            truncation=True,
            max_length=512,
        )
        _sentiment_backend = "pytorch"
        logger.warning(
            "Using PyTorch DistilBERT (slow). "
            "Run scripts/convert_model_to_onnx.py for 4-6x speedup."
        )
        return
    except Exception as e:
        logger.warning(f"PyTorch DistilBERT failed ({e}), falling back to VADER...")

    _sentiment_backend = "vader"


def _predict_sentiment_local(texts: list[str]) -> list[dict | None]:
    """
    Run local sentiment model on a batch of texts.
    Returns list of {"label": "POSITIVE"/"NEGATIVE", "score": float} or None.
    """
    _load_sentiment_model()

    if _sentiment_backend == "vader" or _sentiment_backend == "none":
        return [None] * len(texts)

    if _sentiment_backend.startswith("onnx"):
        import numpy as np
        results = []
        BATCH_SIZE = 32
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = _sentiment_tokenizer(
                batch, return_tensors="np", padding=True, truncation=True, max_length=128,
            )
            outputs = _sentiment_model(**inputs)
            logits = outputs.logits
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            for j in range(len(batch)):
                pred_idx = int(np.argmax(probs[j]))
                pred_score = float(probs[j][pred_idx])
                label = "POSITIVE" if pred_idx == 1 else "NEGATIVE"
                results.append({"label": label, "score": pred_score})
        return results

    if _sentiment_backend == "pytorch":
        try:
            pipe_results = _sentiment_model(texts, batch_size=16, truncation=True, max_length=128)
            return [{"label": r["label"], "score": r["score"]} for r in pipe_results]
        except Exception as e:
            logger.error(f"PyTorch sentiment failed: {e}")
            return [None] * len(texts)

    return [None] * len(texts)


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

# Allow optional adverb modifiers (actually, really, currently, etc.) between keywords
_MOD = r"(\s+\w+)?"  # optional single-word modifier

OBJECTION_PATTERNS = {
    "direct_objection": [
        r"\bthat('s|\s+is)\s+(too\s+)?(expensive|costly|pricey)\b",
        r"\bwe\s+(can't|cannot|don't)\s+(afford|justify|approve)\b",
        r"\bthat('s|\s+is)\s+not" + _MOD + r"\s+(going\s+to|gonna)\s+work\b",
        r"\bwe\s+(already|currently)\s+(have|use)\b",
        # "we are not (actually/really/currently) looking/interested/ready"
        r"\bwe('re|\s+are)\s+not" + _MOD + r"\s+(interested|looking|ready)\b",
        # "we are not looking" without negation word (catch "we aren't looking")
        r"\bwe\s+(aren't|ain't)" + _MOD + r"\s+(interested|looking|ready)\b",
        r"\bI\s+don't" + _MOD + r"\s+(think|see|believe)\s+(this|it|that)\b",
        # "not looking for (outsourced/external/any) X"
        r"\bnot" + _MOD + r"\s+looking\s+for\b",
        # "don't need / don't want"
        r"\b(we|I)\s+(don't|do\s+not)" + _MOD + r"\s+(need|want|require)\b",
    ],
    "timing_objection": [
        # "not (sure/available/free) right now" — allow modifier before "now"
        r"\bnot" + _MOD + r"\s+(right\s+)?now\b",
        r"\bmaybe\s+(later|next\s+(quarter|year|month))\b",
        r"\bthe\s+timing\s+(isn't|is\s+not)\s+(right|good|ideal)\b",
        r"\bwe('re|\s+are)\s+(not\s+)?ready\b",
        r"\bneed\s+more\s+time\b",
        # "I'm a little busy" / "I'm busy right now"
        r"\bI('m|\s+am)\s+(\w+\s+)?busy\b",
        # "can you call (me) (back) (some) other time"
        r"\bcall\s+(me\s+)?(back\s+)?(some\s+)?other\s+time\b",
        # "maybe in (the) future" / "in future maybe"
        r"\b(maybe\s+)?in\s+(the\s+)?future\b",
        r"\bI'm\s+not\s+sure\s+(right\s+)?now\b",
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
            _load_sentiment_model()
        except Exception:
            pass  # VADER fallback is automatic
        # Also ensure VADER is ready as fallback
        _get_vader_analyzer()
        self._sentiment_ready = True

    def extract_all(self, segments: list[dict]) -> list[dict]:
        """
        Extract linguistic features for every segment (includes sync LLM sentiment).

        Args:
            segments: list of transcript segments from Voice Agent
                      [{speaker, start_ms, end_ms, text, words}, ...]

        Returns:
            List of feature dicts, one per segment.
        """
        if not segments:
            return []

        # ── Batch sentiment for all segments at once (much faster) ──
        texts = [seg.get("text", "") for seg in segments]
        sentiments = self._batch_sentiment(texts)

        features_list = self._extract_non_llm_features(segments)

        # Merge sentiment into features
        for i, features in enumerate(features_list):
            sent = sentiments[i] if i < len(sentiments) else {"label": "NEUTRAL", "score": 0.0}
            features["sentiment_label"] = sent["label"]
            features["sentiment_score"] = sent["score"]
            features["sentiment_value"] = sent["score"]

        return features_list

    def extract_all_no_llm(self, segments: list[dict]) -> list[dict]:
        """
        Extract linguistic features WITHOUT LLM sentiment (fast path).
        Sentiment fields are set to neutral defaults; caller should merge
        real sentiment later via batch_sentiment_async().
        """
        features_list = self._extract_non_llm_features(segments)

        # Set neutral sentiment defaults
        for features in features_list:
            features["sentiment_label"] = "NEUTRAL"
            features["sentiment_score"] = 0.0
            features["sentiment_value"] = 0.0

        return features_list

    async def batch_sentiment_async(self, texts: list[str]) -> list[dict]:
        """
        Run sentiment analysis asynchronously via LLM (acomplete).
        Falls back to sync VADER/DistilBERT if LLM unavailable.
        Returns list of {label, score} dicts.
        """
        if not texts:
            return []

        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        neutral = {"label": "NEUTRAL", "score": 0.0}

        if not valid_texts:
            return [neutral] * len(texts)

        # ── 1. Async LLM sentiment (primary) ──
        llm_result = await self._llm_batch_sentiment_async(valid_texts, valid_indices, len(texts))
        if llm_result is not None:
            return llm_result

        # ── 2. Sync fallback (VADER → DistilBERT) — fast, no async needed ──
        return self._batch_sentiment_sync_fallback(texts, valid_indices, neutral)

    async def _llm_batch_sentiment_async(
        self,
        valid_texts: list[str],
        valid_indices: list[int],
        total_count: int,
    ) -> Optional[list[dict]]:
        """Async version of _llm_batch_sentiment using acomplete()."""
        import sys, json
        from pathlib import Path
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            from shared.utils.llm_client import acomplete, is_configured
            if not is_configured():
                return None
        except ImportError:
            return None

        BATCH_SIZE = 12
        neutral = {"label": "NEUTRAL", "score": 0.0}
        output = [neutral.copy() for _ in range(total_count)]

        system_prompt = (
            "You are a sentiment scoring engine for conversational speech. "
            "Rate the EMOTIONAL sentiment of each sentence from the SPEAKER'S perspective.\n"
            "Scale: -1.0 (speaker expressing strong displeasure/frustration) to "
            "+1.0 (speaker expressing strong approval/enthusiasm). 0.0 = neutral.\n\n"
            "CRITICAL RULES:\n"
            "- Factual descriptions of problems or situations are NEUTRAL (0.0), not negative. "
            "'We don't have time for social media' is factual (0.0), not a complaint.\n"
            "- Greetings, introductions, and self-identification are NEUTRAL (0.0). "
            "'This is Sadam calling from HOS' = 0.0.\n"
            "- Explaining a situation without emotion is NEUTRAL: "
            "'executives are so busy right now' = 0.0.\n"
            "- Only score negative when the speaker expresses personal displeasure: "
            "'This is terrible and I hate it' = -0.8.\n"
            "- Only score positive when the speaker expresses genuine enthusiasm: "
            "'sounds great awesome' = +0.7.\n\n"
            "Return ONLY a JSON array of numbers, one per sentence. No explanations."
        )

        try:
            for batch_start in range(0, len(valid_texts), BATCH_SIZE):
                batch = valid_texts[batch_start:batch_start + BATCH_SIZE]
                batch_indices = valid_indices[batch_start:batch_start + BATCH_SIZE]

                numbered = "\n".join(f"{i+1}. \"{t}\"" for i, t in enumerate(batch))
                user_prompt = f"Score these {len(batch)} sentences:\n{numbered}"

                raw = await acomplete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=200,
                    temperature=0.0,
                )

                text = raw.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

                # Tolerant JSON extraction: direct parse → bracketed substring
                # → numbered-line fallback ("1. 0.5\n2. -0.3"). Ollama llama3.1
                # occasionally returns the latter instead of a clean array.
                scores = None
                try:
                    scores = json.loads(text)
                except json.JSONDecodeError:
                    lb, rb = text.find("["), text.rfind("]")
                    if lb >= 0 and rb > lb:
                        try:
                            scores = json.loads(text[lb:rb + 1])
                        except json.JSONDecodeError:
                            pass
                if scores is None:
                    import re
                    floats = re.findall(r"-?\d*\.?\d+", text)
                    if floats:
                        try:
                            scores = [float(f) for f in floats[:len(batch)]]
                        except ValueError:
                            scores = None

                if not isinstance(scores, list):
                    logger.warning(f"LLM returned non-array sentiment: {text[:100]}")
                    continue

                for j, idx in enumerate(batch_indices):
                    if j < len(scores):
                        val = float(scores[j])
                        val = max(-1.0, min(1.0, val))
                        if val > 0.35:
                            output[idx] = {"label": "POSITIVE", "score": round(val, 4)}
                        elif val < -0.35:
                            output[idx] = {"label": "NEGATIVE", "score": round(val, 4)}
                        else:
                            output[idx] = {"label": "NEUTRAL", "score": round(val, 4)}

            logger.info(f"LLM async sentiment scored {len(valid_texts)} segments")
            return output

        except Exception as e:
            logger.warning(f"Async LLM sentiment failed: {e}, falling back")
            return None

    def _batch_sentiment_sync_fallback(
        self,
        texts: list[str],
        valid_indices: list[int],
        neutral: dict,
    ) -> list[dict]:
        """Sync VADER → DistilBERT fallback for sentiment."""
        vader = _get_vader_analyzer()
        if vader is not None:
            output = [neutral.copy() for _ in texts]
            for idx in valid_indices:
                compound = vader.polarity_scores(texts[idx])["compound"]
                if compound > 0.2:
                    output[idx] = {"label": "POSITIVE", "score": round(compound, 4)}
                elif compound < -0.2:
                    output[idx] = {"label": "NEGATIVE", "score": round(compound, 4)}
                else:
                    output[idx] = {"label": "NEUTRAL", "score": 0.0}
            return output

        # DistilBERT / ONNX fallback
        valid_texts = [texts[i] for i in valid_indices]
        start_t = time.time()
        model_results = _predict_sentiment_local(valid_texts)
        elapsed = time.time() - start_t
        if model_results and model_results[0] is not None:
            logger.info(
                f"Local sentiment: {len(valid_texts)} texts in {elapsed:.2f}s "
                f"({elapsed / max(len(valid_texts), 1) * 1000:.1f} ms/text, backend={_sentiment_backend})"
            )
            output = [neutral.copy() for _ in texts]
            for idx, result in zip(valid_indices, model_results):
                if result is None:
                    continue
                label = result["label"]
                raw = result["score"]
                signed = raw if label == "POSITIVE" else -raw
                if abs(signed) < 0.4:
                    output[idx] = {"label": "NEUTRAL", "score": 0.0}
                else:
                    output[idx] = {
                        "label": "POSITIVE" if signed > 0 else "NEGATIVE",
                        "score": round(signed, 4),
                    }
            return output

        return [neutral.copy() for _ in texts]

    def _extract_non_llm_features(self, segments: list[dict]) -> list[dict]:
        """
        Extract all non-LLM features (buying, objection, power, lexical).
        Shared by extract_all() and extract_all_no_llm().
        """
        if not segments:
            return []

        features_list = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            features = {
                "speaker_id": seg.get("speaker", "unknown"),
                "start_ms": seg.get("start_ms", 0),
                "end_ms": seg.get("end_ms", 0),
                "text": text,
            }

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

    # ── Sentiment (LLM primary → VADER fallback → DistilBERT fallback) ──

    def _batch_sentiment(self, texts: list[str]) -> list[dict]:
        """
        Run sentiment analysis on a batch of texts.

        Priority:
          1. LLM (gpt-4o-mini / Claude) — best for conversational speech
          2. VADER (rule-based) — decent fallback, with ±0.2 dead zone
          3. DistilBERT — last resort, wide neutral zone

        Returns list of {label: POSITIVE|NEGATIVE|NEUTRAL, score: -1.0 to +1.0}.
        """
        if not texts:
            return []

        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        neutral = {"label": "NEUTRAL", "score": 0.0}

        if not valid_texts:
            return [neutral] * len(texts)

        # ── 1. LLM-based sentiment (primary) ──
        llm_result = self._llm_batch_sentiment(valid_texts, valid_indices, len(texts))
        if llm_result is not None:
            return llm_result

        # ── 2. VADER fallback (rule-based, dead zone ±0.2) ──
        vader = _get_vader_analyzer()
        if vader is not None:
            output = [neutral.copy() for _ in texts]
            for idx in valid_indices:
                compound = vader.polarity_scores(texts[idx])["compound"]
                if compound > 0.2:
                    output[idx] = {"label": "POSITIVE", "score": round(compound, 4)}
                elif compound < -0.2:
                    output[idx] = {"label": "NEGATIVE", "score": round(compound, 4)}
                else:
                    output[idx] = {"label": "NEUTRAL", "score": 0.0}
            return output

        # ── 3. DistilBERT / ONNX fallback (wide neutral zone) ──
        start_t = time.time()
        model_results = _predict_sentiment_local(valid_texts)
        elapsed = time.time() - start_t
        if model_results and model_results[0] is not None:
            logger.info(
                f"Local sentiment (LLM path fallback): {len(valid_texts)} texts in {elapsed:.2f}s "
                f"({elapsed / max(len(valid_texts), 1) * 1000:.1f} ms/text, backend={_sentiment_backend})"
            )
            output = [neutral.copy() for _ in texts]
            for idx, result in zip(valid_indices, model_results):
                if result is None:
                    continue
                label = result["label"]
                raw = result["score"]
                signed = raw if label == "POSITIVE" else -raw
                if abs(signed) < 0.4:
                    output[idx] = {"label": "NEUTRAL", "score": 0.0}
                else:
                    output[idx] = {
                        "label": "POSITIVE" if signed > 0 else "NEGATIVE",
                        "score": round(signed, 4),
                    }
            return output

        logger.warning("No sentiment backend available — returning neutral")
        return [neutral.copy() for _ in texts]

    def _llm_batch_sentiment(
        self,
        valid_texts: list[str],
        valid_indices: list[int],
        total_count: int,
    ) -> Optional[list[dict]]:
        """
        Score sentiment via LLM in batches of 12.
        Returns None if LLM is not available or fails entirely.
        """
        import sys, json
        from pathlib import Path
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            from shared.utils.llm_client import complete as llm_complete, is_configured
            if not is_configured():
                return None
        except ImportError:
            return None

        BATCH_SIZE = 12
        neutral = {"label": "NEUTRAL", "score": 0.0}
        output = [neutral.copy() for _ in range(total_count)]

        system_prompt = (
            "You are a sentiment scoring engine for conversational speech. "
            "Rate the EMOTIONAL sentiment of each sentence from the SPEAKER'S perspective.\n"
            "Scale: -1.0 (speaker expressing strong displeasure/frustration) to "
            "+1.0 (speaker expressing strong approval/enthusiasm). 0.0 = neutral.\n\n"
            "CRITICAL RULES:\n"
            "- Factual descriptions of problems or situations are NEUTRAL (0.0), not negative. "
            "'We don't have time for social media' is factual (0.0), not a complaint.\n"
            "- Greetings, introductions, and self-identification are NEUTRAL (0.0). "
            "'This is Sadam calling from HOS' = 0.0.\n"
            "- Explaining a situation without emotion is NEUTRAL: "
            "'executives are so busy right now' = 0.0.\n"
            "- Only score negative when the speaker expresses personal displeasure: "
            "'This is terrible and I hate it' = -0.8.\n"
            "- Only score positive when the speaker expresses genuine enthusiasm: "
            "'sounds great awesome' = +0.7.\n\n"
            "Return ONLY a JSON array of numbers, one per sentence. No explanations."
        )

        try:
            for batch_start in range(0, len(valid_texts), BATCH_SIZE):
                batch = valid_texts[batch_start:batch_start + BATCH_SIZE]
                batch_indices = valid_indices[batch_start:batch_start + BATCH_SIZE]

                numbered = "\n".join(f"{i+1}. \"{t}\"" for i, t in enumerate(batch))
                user_prompt = f"Score these {len(batch)} sentences:\n{numbered}"

                raw = llm_complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=200,
                    temperature=0.0,
                )

                # Parse JSON array from response
                text = raw.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

                # Tolerant JSON extraction: direct parse → bracketed substring
                # → numbered-line fallback ("1. 0.5\n2. -0.3"). Ollama llama3.1
                # occasionally returns the latter instead of a clean array.
                scores = None
                try:
                    scores = json.loads(text)
                except json.JSONDecodeError:
                    lb, rb = text.find("["), text.rfind("]")
                    if lb >= 0 and rb > lb:
                        try:
                            scores = json.loads(text[lb:rb + 1])
                        except json.JSONDecodeError:
                            pass
                if scores is None:
                    import re
                    floats = re.findall(r"-?\d*\.?\d+", text)
                    if floats:
                        try:
                            scores = [float(f) for f in floats[:len(batch)]]
                        except ValueError:
                            scores = None

                if not isinstance(scores, list):
                    logger.warning(f"LLM returned non-array sentiment: {text[:100]}")
                    continue

                for j, idx in enumerate(batch_indices):
                    if j < len(scores):
                        val = float(scores[j])
                        val = max(-1.0, min(1.0, val))
                        if val > 0.35:
                            output[idx] = {"label": "POSITIVE", "score": round(val, 4)}
                        elif val < -0.35:
                            output[idx] = {"label": "NEGATIVE", "score": round(val, 4)}
                        else:
                            output[idx] = {"label": "NEUTRAL", "score": round(val, 4)}

            logger.info(f"LLM sentiment scored {len(valid_texts)} segments")
            return output

        except Exception as e:
            logger.warning(f"LLM sentiment failed: {e}, falling back to VADER/DistilBERT")
            return None

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

        # Count hedges (word-boundary matching to avoid "mankind of" → "kind of")
        import re
        for hedge in POWERLESS_HEDGES:
            pattern = r'\b' + re.escape(hedge) + r'\b'
            count = len(re.findall(pattern, text_lower))
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

        # Count hesitation forms (word-boundary matching)
        for hesitation in POWERLESS_HESITATIONS:
            pattern = r'\b' + re.escape(hesitation) + r'\b'
            if re.search(pattern, text_lower):
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
