"""
NEXUS Language Agent — Entity & Topic Extraction

Sends the full transcript to an LLM and extracts structured entities:
people, companies, products, topics (conversation phases), objections,
commitments, and key terms.

The **topics** array is the most important output — it segments the
conversation into named phases so signals can be mapped to conversation
context ("stress peaked during objection handling").
"""
import json
import logging
from typing import Optional

logger = logging.getLogger("nexus.language.entities")


def _match_timestamp(text: str, seg_texts: list[tuple[str, int]]) -> int:
    """Find the best matching segment timestamp for a text snippet."""
    query = text.lower().strip()
    # Exact substring match
    for seg_text, ts in seg_texts:
        if query in seg_text or seg_text in query:
            return ts
    # Word overlap match
    query_words = set(query.split())
    best_ts, best_overlap = 0, 0
    for seg_text, ts in seg_texts:
        seg_words = set(seg_text.split())
        overlap = len(query_words & seg_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_ts = ts
    return best_ts


class EntityExtractor:
    """Extract entities, topics, and commitments from a transcript via LLM."""

    def __init__(self):
        self._llm_ok = False
        try:
            from shared.utils.llm_client import is_configured
            self._llm_ok = is_configured()
        except ImportError:
            pass

    async def extract(
        self,
        transcript_segments: list[dict],
        content_type: str = "sales_call",
    ) -> dict:
        """
        Send full transcript to LLM and return structured entities.

        Falls back to a lightweight keyword extraction if LLM is unavailable.
        """
        if not transcript_segments:
            return self._empty()

        if not self._llm_ok:
            logger.info("LLM not configured — using lightweight entity extraction")
            return self._extract_lightweight(transcript_segments, content_type)

        try:
            return await self._extract_via_llm(transcript_segments, content_type)
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}. Falling back.")
            return self._extract_lightweight(transcript_segments, content_type)

    # ──────────────────────────────────────────
    # LLM-based extraction
    # ──────────────────────────────────────────

    async def _extract_via_llm(
        self, segments: list[dict], content_type: str
    ) -> dict:
        from shared.utils.llm_client import acomplete

        transcript_text = self._format_transcript(segments)
        duration_ms = segments[-1].get("end_ms", 0) if segments else 0
        duration_sec = duration_ms // 1000

        system_prompt = (
            "You are a conversation analyst. Extract structured information "
            "from the following transcript. Return ONLY valid JSON, no commentary."
        )

        user_prompt = f"""Analyse this {content_type} transcript ({duration_sec} seconds long) and extract entities.

TRANSCRIPT (timestamps in [M:SS] format, in milliseconds):
{transcript_text}

Return a JSON object with these keys:
{{
  "people": [
    {{"name": "string", "role": "seller|prospect|interviewer|candidate|participant", "speaker_label": "Speaker_0", "first_mention_ms": number}}
  ],
  "companies": [
    {{"name": "string", "context": "short description", "first_mention_ms": number}}
  ],
  "products_services": [
    {{"name": "string", "context": "what it is"}}
  ],
  "topics": [
    {{"name": "short phase name", "start_ms": number, "end_ms": number}}
  ],
  "objections": [
    {{"text": "exact quote from transcript", "timestamp_ms": number, "resolved": true|false, "resolved_at_ms": number}}
  ],
  "commitments": [
    {{"text": "exact quote from transcript", "speaker": "Speaker_X", "timestamp_ms": number}}
  ],
  "key_terms": ["term1", "term2"]
}}

CRITICAL RULES:
1. TOPICS: You MUST return 3-7 distinct conversation phases. Each phase covers a DIFFERENT time range. They must cover the full duration (0 to {duration_ms}ms) with no gaps.
   Example for a {duration_sec}-second sales call:
   [
     {{"name": "Introduction", "start_ms": 0, "end_ms": 5000}},
     {{"name": "Initial objection", "start_ms": 5000, "end_ms": 13000}},
     {{"name": "Pitch & discussion", "start_ms": 13000, "end_ms": 35000}},
     {{"name": "Value proposition", "start_ms": 35000, "end_ms": 44000}},
     {{"name": "Closing & next steps", "start_ms": 44000, "end_ms": {duration_ms}}}
   ]
2. TIMESTAMPS: All timestamp_ms values MUST be actual millisecond values from the transcript (NOT zero). Use the [M:SS] timestamps shown in the transcript and convert to milliseconds.
3. EXACT QUOTES: For objections, commitments, and buying signals, use the EXACT words from the transcript — do NOT paraphrase or summarize.
4. PEOPLE: Extract ALL person names mentioned anywhere in the conversation — including names in greetings, sign-offs, or mid-conversation references like "thank you Holly" or "hi John". For EACH person found, identify which Speaker_X label they correspond to. In a sales call, the person who introduces themselves and their company is the Seller; the person who answers is the Prospect.
5. OBJECTION RESOLUTION: For each objection, check whether the SAME SPEAKER who raised it later did ANY of these:
   - Agreed to next steps (shared email, scheduled a call, said "sounds good", "yes", "sure")
   - Asked specification questions showing interest (e.g., "have you worked in education?")
   - Made commitments (agreed to receive a proposal, look at materials, etc.)
   If ANY of these happened AFTER the objection, mark "resolved": true AND set "resolved_at_ms" to the timestamp (in ms) of the segment where that resolving behaviour occurred. Only mark "resolved": false if the conversation ended with the objection still standing and no positive engagement afterward; in that case set "resolved_at_ms": 0.

Return ONLY the JSON object."""

        raw = await acomplete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2000,
            temperature=0.1,
        )

        # Parse JSON from LLM response
        result = self._parse_json(raw)
        if result is None:
            logger.warning("LLM returned unparseable JSON, falling back")
            return self._extract_lightweight(segments, content_type)

        # Validate and clean
        return self._validate(result, segments)

    # ──────────────────────────────────────────
    # Lightweight fallback (no LLM)
    # ──────────────────────────────────────────

    def _extract_lightweight(
        self, segments: list[dict], content_type: str
    ) -> dict:
        """Rule-based entity extraction when LLM is unavailable."""
        if not segments:
            return self._empty()

        duration_ms = max(s.get("end_ms", 0) for s in segments)
        all_text = " ".join(s.get("text", "") for s in segments).lower()

        # Simple topic segmentation: split into equal phases
        n_phases = max(2, min(5, len(segments) // 6))
        phase_dur = duration_ms // n_phases if n_phases > 0 else duration_ms
        phase_names = self._guess_phase_names(segments, content_type)

        topics = []
        for i in range(n_phases):
            start = i * phase_dur
            end = (i + 1) * phase_dur if i < n_phases - 1 else duration_ms
            name = phase_names[i] if i < len(phase_names) else f"Phase {i+1}"
            topics.append({"name": name, "start_ms": start, "end_ms": end})

        # Extract key terms (frequent non-stopword terms)
        key_terms = self._extract_key_terms(all_text)

        # Detect names from introduction patterns
        people = []
        companies = []
        import re
        for seg in segments[:8]:
            text = seg.get("text", "")
            speaker = seg.get("speaker", "unknown")
            # "This is X calling from Y" / "My name is X from Y"
            m = re.search(
                r"(?:this is|my name is|i[''']m)\s+(\w+).*?(?:from|at|with)\s+(\w[\w\s]*)",
                text, re.IGNORECASE,
            )
            if m:
                people.append({
                    "name": m.group(1).strip(),
                    "role": "seller" if content_type == "sales_call" else "participant",
                    "speaker_label": speaker,
                    "first_mention_ms": seg.get("start_ms", 0),
                })
                companies.append({
                    "name": m.group(2).strip().rstrip(".,:;"),
                    "context": "mentioned in introduction",
                    "first_mention_ms": seg.get("start_ms", 0),
                })

        return {
            "people": people,
            "companies": companies,
            "products_services": [],
            "topics": topics,
            "objections": [],
            "commitments": [],
            "key_terms": key_terms,
        }

    def _guess_phase_names(self, segments: list[dict], content_type: str) -> list[str]:
        """Guess conversation phase names from content."""
        if content_type == "sales_call":
            return ["Introduction", "Discovery", "Pitch", "Objection Handling", "Closing"]
        if content_type == "interview":
            return ["Opening", "Background", "Technical", "Questions", "Wrap-up"]
        return ["Opening", "Discussion", "Main Topic", "Follow-up", "Closing"]

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _format_transcript(segments: list[dict]) -> str:
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            start = seg.get("start_ms", 0)
            end = seg.get("end_ms", 0)
            text = seg.get("text", "")
            m, s = divmod(start // 1000, 60)
            ts = f"[{m}:{s:02d}]"
            lines.append(f"{ts} ({start}ms-{end}ms) {speaker}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _validate(result: dict, segments: list[dict]) -> dict:
        """Validate and clean LLM output."""
        duration_ms = max((s.get("end_ms", 0) for s in segments), default=0)

        # Ensure all expected keys exist
        for key in ("people", "companies", "products_services", "topics",
                     "objections", "commitments", "key_terms"):
            if key not in result:
                result[key] = []

        # ── Fix topics ──
        topics = result.get("topics", [])
        # If LLM returned 0-1 topics or all start at 0 → auto-segment
        all_at_zero = all(t.get("start_ms", 0) == 0 for t in topics)
        if len(topics) <= 1 or all_at_zero:
            n = max(3, min(5, len(segments) // 6))
            chunk = len(segments) // n if n > 0 else len(segments)
            topics = []
            names = ["Introduction", "Discussion", "Main Topic", "Development", "Closing"]
            for i in range(n):
                s_idx = i * chunk
                e_idx = min((i + 1) * chunk, len(segments)) - 1 if i < n - 1 else len(segments) - 1
                topics.append({
                    "name": names[i] if i < len(names) else f"Phase {i+1}",
                    "start_ms": segments[s_idx].get("start_ms", 0),
                    "end_ms": segments[e_idx].get("end_ms", duration_ms),
                })
        else:
            topics.sort(key=lambda t: t.get("start_ms", 0))
            if topics[0].get("start_ms", 0) > 0:
                topics[0]["start_ms"] = 0
            if topics[-1].get("end_ms", 0) < duration_ms:
                topics[-1]["end_ms"] = duration_ms
        result["topics"] = topics

        # ── Fix zero timestamps on objections/commitments ──
        seg_texts = [(s.get("text", "").lower(), s.get("start_ms", 0)) for s in segments]

        for obj in result.get("objections", []):
            if obj.get("timestamp_ms", 0) == 0 and obj.get("text"):
                obj["timestamp_ms"] = _match_timestamp(obj["text"], seg_texts)

        for com in result.get("commitments", []):
            if com.get("timestamp_ms", 0) == 0 and com.get("text"):
                com["timestamp_ms"] = _match_timestamp(com["text"], seg_texts)

        for person in result.get("people", []):
            if person.get("first_mention_ms", 0) == 0 and person.get("name"):
                person["first_mention_ms"] = _match_timestamp(
                    person["name"], seg_texts
                )

        return result

    @staticmethod
    def _extract_key_terms(text: str) -> list[str]:
        """Extract frequent meaningful terms."""
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "i", "you", "we",
            "they", "it", "to", "of", "in", "for", "on", "with", "that",
            "this", "and", "or", "but", "not", "so", "just", "like", "um",
            "uh", "know", "think", "going", "want", "have", "has", "do",
            "does", "did", "can", "could", "would", "will", "be", "been",
            "if", "at", "from", "by", "about", "up", "out", "as", "my",
            "your", "our", "me", "us", "him", "her", "them", "what", "how",
            "when", "where", "who", "which", "there", "here", "very", "really",
            "also", "yes", "no", "yeah", "okay", "right", "well", "got",
            "get", "let", "see", "say", "said", "one", "don't", "didn't",
        }
        import re
        words = re.findall(r"\b[a-z]{3,}\b", text)
        freq: dict[str, int] = {}
        for w in words:
            if w not in stop:
                freq[w] = freq.get(w, 0) + 1
        sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_terms[:15]]

    @staticmethod
    def _empty() -> dict:
        return {
            "people": [], "companies": [], "products_services": [],
            "topics": [], "objections": [], "commitments": [], "key_terms": [],
        }
