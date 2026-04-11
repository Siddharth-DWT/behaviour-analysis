# Upload Configuration UI — Model-Aware Settings Panel

Read these files first:
- services/api_gateway/main.py (upload endpoint POST /sessions, _run_pipeline, _call_voice_agent)
- services/voiceAgent/main.py (Voice Agent /analyse endpoint)
- services/voiceAgent/transcriber.py (transcription backends + diarization cascade)
- shared/models/requests.py (VoiceAnalysisRequest model)
- shared/utils/external_apis.py (Whisper external client, AssemblyAI client, Deepgram client)
- shared/utils/parakeet_client.py (Parakeet client)
- shared/utils/assemblyai_client.py (standalone AssemblyAI client)
- dashboard/src/ (React dashboard — find the upload component)

## What We're Building

An upload settings panel that shows/hides configuration options based on the selected transcription model. Each model has different capabilities — the UI should only show options that the selected model actually supports.

## Model Priority Order (by functionality)

```
1. AssemblyAI Universal 3 Pro  — most features, cloud API, best accuracy
2. Deepgram Nova 3             — almost everything, cloud API, fast
3. Whisper Large V3 + NeMo     — language + prompt, local/GPU, good multilingual
4. Parakeet TDT 0.6B + NeMo   — fastest, fewest features, local/GPU, English-focused
5. Auto (Best Available)       — uses cascade: Parakeet → AssemblyAI → Deepgram → Whisper → Local
```

## Feature × Model Matrix

This is the source of truth for the UI. Show a feature toggle ONLY when the selected model supports it.

```python
MODEL_FEATURES = {
    "assemblyai": {
        "label": "AssemblyAI Universal 3 Pro",
        "description": "Most accurate, full features, cloud API",
        "priority": 1,
        "features": {
            "language_selection":  True,   # language_code param
            "custom_prompt":      True,   # prompt param in API (needs wiring)
            "key_terms":          True,   # word_boost param (needs wiring)
            "auto_punctuation":   True,   # punctuate param (configurable)
            "keep_filler_words":  True,   # disfluencies param (needs wiring)
            "text_formatting":    True,   # format_text param (needs wiring)
            "multichannel":       True,   # multichannel param (needs wiring)
            "word_timestamps":    True,   # always on
            "speaker_diarization": True,  # built-in speaker_labels
            "num_speakers_hint":  True,   # speakers_expected param
            "temperature":        True,   # via API (needs wiring)
        },
        "wiring_status": {
            # What's already wired vs needs code changes
            "language_selection":  "PARTIAL — language_code exists in assemblyai_client.py:54 but not passed from transcriber",
            "custom_prompt":      "NOT WIRED — add to _submit() payload in external_apis.py:763",
            "key_terms":          "NOT WIRED — add word_boost to _submit() payload",
            "auto_punctuation":   "NOT WIRED — add punctuate to _submit() payload",
            "keep_filler_words":  "NOT WIRED — add disfluencies to _submit() payload",
            "text_formatting":    "NOT WIRED — add format_text to _submit() payload",
            "multichannel":       "NOT WIRED — add multichannel to _submit() payload",
            "word_timestamps":    "WIRED — always on via words in utterances",
            "speaker_diarization":"WIRED — speaker_labels: True in _submit()",
            "num_speakers_hint":  "WIRED — speakers_expected in assemblyai_client.py:53",
            "temperature":        "NOT WIRED — add to _submit() payload",
        }
    },
    "deepgram": {
        "label": "Deepgram Nova 3",
        "description": "Fast cloud API, good diarization, most features",
        "priority": 2,
        "features": {
            "language_selection":  True,   # language param (hardcoded "en", needs parameterizing)
            "custom_prompt":      False,  # NOT supported by Deepgram API
            "key_terms":          True,   # keywords param (needs wiring)
            "auto_punctuation":   True,   # punctuate param (hardcoded "true", needs parameterizing)
            "keep_filler_words":  True,   # filler_words param (needs wiring)
            "text_formatting":    True,   # smart_format param (needs wiring)
            "multichannel":       True,   # multichannel param (needs wiring)
            "word_timestamps":    True,   # always on
            "speaker_diarization": True,  # built-in diarize=true
            "num_speakers_hint":  True,   # min_speakers/max_speakers params exist
            "temperature":        False,  # NOT supported
        },
        "wiring_status": {
            "language_selection":  "PARTIAL — param exists at external_apis.py:1001 but hardcoded 'en'",
            "key_terms":          "NOT WIRED — add keywords to params dict at line ~996",
            "auto_punctuation":   "PARTIAL — hardcoded 'true' at line 998, needs to be configurable",
            "keep_filler_words":  "NOT WIRED — add filler_words param",
            "text_formatting":    "NOT WIRED — add smart_format param",
            "multichannel":       "NOT WIRED — add multichannel param",
            "word_timestamps":    "WIRED — via utterances + words",
            "speaker_diarization":"WIRED — diarize=true",
            "num_speakers_hint":  "WIRED — min_speakers/max_speakers params exist",
        }
    },
    "whisper": {
        "label": "Whisper Large V3 + NeMo",
        "description": "Best multilingual, local GPU, custom prompts",
        "priority": 3,
        "features": {
            "language_selection":  True,   # language param exists (hardcoded "en", needs parameterizing)
            "custom_prompt":      True,   # Whisper supports initial_prompt (needs wiring)
            "key_terms":          True,   # Can append to initial_prompt as workaround
            "auto_punctuation":   False,  # Always on, cannot disable
            "keep_filler_words":  False,  # Cannot control — model decides
            "text_formatting":    False,  # Not supported
            "multichannel":       False,  # Not supported
            "word_timestamps":    True,   # Always on (word_timestamps=True)
            "speaker_diarization": True,  # Via separate pyannote/NeMo diarization
            "num_speakers_hint":  True,   # Passed to diarizer
            "temperature":        True,   # Whisper supports temperature param (needs wiring)
        },
        "wiring_status": {
            "language_selection":  "PARTIAL — param exists at external_apis.py:137 but hardcoded 'en' in transcriber calls",
            "custom_prompt":      "NOT WIRED — Whisper API supports initial_prompt, not passed",
            "key_terms":          "NOT WIRED — append to initial_prompt string",
            "word_timestamps":    "WIRED — word_timestamps=True at line 755",
            "speaker_diarization":"WIRED — via _diarize_simple cascade",
            "num_speakers_hint":  "WIRED — passed through transcribe()",
            "temperature":        "NOT WIRED — Whisper API supports temperature, not passed",
        }
    },
    "parakeet": {
        "label": "Parakeet TDT 0.6B + NeMo",
        "description": "Fastest (174× realtime), English-focused, local GPU",
        "priority": 4,
        "features": {
            "language_selection":  False,  # English only
            "custom_prompt":      False,  # Not supported by Parakeet
            "key_terms":          False,  # Not supported
            "auto_punctuation":   False,  # Always on, cannot disable
            "keep_filler_words":  False,  # Cannot control
            "text_formatting":    False,  # Not supported
            "multichannel":       False,  # Not supported
            "word_timestamps":    True,   # Always on (hardcoded word_timestamps: "true")
            "speaker_diarization": True,  # Via separate NeMo/pyannote diarization
            "num_speakers_hint":  True,   # Passed to diarizer
            "temperature":        False,  # Not supported
        },
        "wiring_status": {
            "word_timestamps":    "WIRED — hardcoded in parakeet_client.py:91",
            "speaker_diarization":"WIRED — via _diarize_simple cascade",
            "num_speakers_hint":  "WIRED — passed through transcribe()",
        }
    },
    "auto": {
        "label": "Auto (Best Available)",
        "description": "Cascade: Parakeet → AssemblyAI → Deepgram → Whisper → Local",
        "priority": 0,
        "features": {
            # Show minimal options — the cascade picks the backend
            "language_selection":  False,  # Can't control which backend is used
            "custom_prompt":      False,
            "key_terms":          False,
            "auto_punctuation":   False,  # Always on in all backends
            "keep_filler_words":  False,
            "text_formatting":    False,
            "multichannel":       False,
            "word_timestamps":    True,   # Always on in all backends
            "speaker_diarization": True,
            "num_speakers_hint":  True,
            "temperature":        False,
        }
    }
}
```

## UI Behavior — Dynamic Settings Based on Model Selection

When user selects a model, the settings sections show/hide accordingly:

```
User selects "AssemblyAI Universal 3 Pro":
  ▾ Transcription Settings
    Language:        [Auto-detect ▼]         ← shown
    Custom Prompt:   [________________]      ← shown
    Key Terms:       [________________]      ← shown
    Temperature:     [████░░░░░░ 0.0]        ← shown
  ▾ Speaker Diarization
    Num Speakers:    [Auto ▼]                ← shown
    ☐ Multichannel                           ← shown
  ▾ Transcript Formatting
    ☑ Auto Punctuation                       ← shown (toggleable)
    ☐ Keep Filler Words                      ← shown
    ☐ Text Formatting                        ← shown
    ☑ Word Timestamps                        ← shown (always on, disabled)

User selects "Parakeet TDT 0.6B + NeMo":
  ▾ Transcription Settings
    (No configurable options for this model)  ← collapsed/hidden
  ▾ Speaker Diarization
    Num Speakers:    [Auto ▼]                ← shown (only option)
  ▾ Transcript Formatting
    ☑ Auto Punctuation                       ← shown but disabled (always on)
    ☑ Word Timestamps                        ← shown but disabled (always on)
    (Filler words, text formatting not available with this model)

User selects "Whisper Large V3 + NeMo":
  ▾ Transcription Settings
    Language:        [Auto-detect ▼]         ← shown
    Custom Prompt:   [________________]      ← shown
    Key Terms:       [________________]      ← shown (appended to prompt)
    Temperature:     [████░░░░░░ 0.0]        ← shown
  ▾ Speaker Diarization
    Num Speakers:    [Auto ▼]                ← shown
  ▾ Transcript Formatting
    ☑ Auto Punctuation                       ← shown but disabled (always on)
    ☑ Word Timestamps                        ← shown but disabled (always on)
    (Filler words, multichannel, text formatting not available with this model)
```

### React Implementation Pattern

```tsx
const MODEL_FEATURES = { /* from above */ };

function UploadSettings({ config, setConfig }) {
    const selectedModel = config.model_preference || "auto";
    const features = MODEL_FEATURES[selectedModel]?.features || {};
    
    return (
        <div>
            {/* Model Selection — always shown */}
            <ModelSelector value={selectedModel} onChange={...} />
            
            {/* Transcription Settings — show if ANY text feature is available */}
            {(features.language_selection || features.custom_prompt || 
              features.key_terms || features.temperature) && (
                <CollapsibleSection title="Transcription Settings">
                    {features.language_selection && (
                        <LanguageDropdown value={config.language} onChange={...} />
                    )}
                    {features.custom_prompt && (
                        <TextArea label="Custom Prompt" value={config.custom_prompt}
                                  placeholder="ah, hmm, like, you know..." onChange={...} />
                    )}
                    {features.key_terms && (
                        <TextInput label="Key Terms" value={config.key_terms}
                                   placeholder="Metoprolol, Toyota, Acme Corp" onChange={...}
                                   helpText="Comma-separated domain terms to boost recognition" />
                    )}
                    {features.temperature && (
                        <Slider label="Temperature" value={config.temperature}
                                min={0} max={1} step={0.1} onChange={...} />
                    )}
                </CollapsibleSection>
            )}
            
            {/* Speaker Diarization — always shown (all models support it) */}
            <CollapsibleSection title="Speaker Diarization">
                <SpeakerCountDropdown value={config.num_speakers} onChange={...} />
                {features.multichannel && (
                    <Toggle label="Multichannel" checked={config.multichannel}
                            helpText="Each audio channel = one speaker" onChange={...} />
                )}
            </CollapsibleSection>
            
            {/* Transcript Formatting — show if any formatting option available */}
            {(features.auto_punctuation || features.keep_filler_words || 
              features.text_formatting) && (
                <CollapsibleSection title="Transcript Formatting">
                    <Toggle label="Auto Punctuation" 
                            checked={config.auto_punctuation}
                            disabled={!features.auto_punctuation}
                            helpText={!features.auto_punctuation ? "Always on for this model" : ""}
                            onChange={...} />
                    {features.keep_filler_words && (
                        <Toggle label="Keep Filler Words" checked={config.keep_filler_words}
                                helpText='Keep "um", "uh", "like" in transcript' onChange={...} />
                    )}
                    {features.text_formatting && (
                        <Toggle label="Text Formatting" checked={config.text_formatting}
                                helpText="Smart formatting for dates, numbers, URLs" onChange={...} />
                    )}
                    <Toggle label="Word Timestamps" checked={true} disabled={true}
                            helpText="Always enabled — per-word timing for all models" />
                </CollapsibleSection>
            )}
            
            {/* Analysis Options — always shown, independent of model */}
            <CollapsibleSection title="Analysis Options">
                <Toggle label="Behavioural Analysis" checked={config.run_behavioural}
                        helpText="Voice + Language + Conversation + Fusion agents" onChange={...} />
                <Toggle label="Entity Extraction" checked={config.run_entity_extraction}
                        helpText="People, companies, topics, objections, commitments" onChange={...} />
                <Toggle label="Knowledge Graph" checked={config.run_knowledge_graph}
                        helpText="Neo4j sync for causal chain queries" onChange={...} />
                <Slider label="Sensitivity" value={config.sensitivity}
                        min={0} max={1} step={0.1} onChange={...}
                        helpText="Higher = more signals detected, lower = fewer false positives" />
            </CollapsibleSection>
        </div>
    );
}
```

### Model Selection Card Design

Each model shown as a card (not just a dropdown) so user can see what they're choosing:

```
┌─────────────────────────────────────────────────┐
│  ○ Auto (Best Available)                         │
│    Cascade: fastest available with fallback       │
│    Features: basic (no configurable options)      │
├─────────────────────────────────────────────────┤
│  ○ AssemblyAI Universal 3 Pro           ★★★★★   │
│    Most accurate, full features, cloud API        │
│    ✓ Language ✓ Prompt ✓ Key Terms ✓ Fillers      │
│    ✓ Formatting ✓ Multichannel ✓ Temperature      │
├─────────────────────────────────────────────────┤
│  ○ Deepgram Nova 3                      ★★★★    │
│    Fast cloud API, good diarization               │
│    ✓ Language ✓ Key Terms ✓ Fillers               │
│    ✓ Formatting ✓ Multichannel                    │
├─────────────────────────────────────────────────┤
│  ○ Whisper Large V3 + NeMo              ★★★     │
│    Best multilingual, custom prompts              │
│    ✓ Language ✓ Prompt ✓ Key Terms                │
│    ✓ Temperature                                  │
├─────────────────────────────────────────────────┤
│  ● Parakeet TDT 0.6B + NeMo            ★★       │
│    Fastest (174× realtime), English only          │
│    Basic: diarization + word timestamps only      │
└─────────────────────────────────────────────────┘
```

## Backend Wiring — What Needs To Change Per Backend

### AssemblyAI — external_apis.py `_submit()` method (~line 763)

Currently sends:
```python
payload = {
    "audio_url": audio_url,
    "speaker_labels": True,
    "speech_models": ["universal-3-pro"],
}
```

Change to:
```python
def _submit(self, audio_url: str, config: dict = None) -> str:
    config = config or {}
    payload = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "speech_models": ["universal-3-pro"],
    }
    # Wire new features from config
    if config.get("language"):
        payload["language_code"] = config["language"]
    if config.get("custom_prompt"):
        payload["prompt"] = config["custom_prompt"]
    if config.get("key_terms"):
        payload["word_boost"] = config["key_terms"]
        payload["boost_param"] = "high"
    if config.get("keep_filler_words") is True:
        payload["disfluencies"] = True
    if config.get("text_formatting") is True:
        payload["format_text"] = True
    if config.get("auto_punctuation") is False:
        payload["punctuate"] = False
    if config.get("multichannel") is True:
        payload["multichannel"] = True
        payload["speaker_labels"] = False  # AssemblyAI: can't use both
    if config.get("temperature") is not None:
        payload["temperature"] = config["temperature"]
    if config.get("num_speakers"):
        payload["speakers_expected"] = config["num_speakers"]
    
    # ... rest of submit logic
```

### Deepgram — external_apis.py `DeepgramDiarizeClient` params (~line 996)

Currently sends:
```python
params = {
    "diarize": "true",
    "punctuate": "true",
    "utterances": "true",
    "model": "nova-3",
    "language": "en",
}
```

Change to:
```python
def diarize(self, audio_path: str, ..., config: dict = None) -> dict:
    config = config or {}
    params = {
        "diarize": "true",
        "punctuate": "true" if config.get("auto_punctuation", True) else "false",
        "utterances": "true",
        "model": "nova-3",
        "language": config.get("language", "en"),
    }
    # Wire new features
    if config.get("key_terms"):
        # Deepgram keywords format: "term:boost" pairs
        params["keywords"] = ",".join(f"{t}:2" for t in config["key_terms"])
    if config.get("keep_filler_words") is True:
        params["filler_words"] = "true"
    if config.get("text_formatting") is True:
        params["smart_format"] = "true"
    if config.get("multichannel") is True:
        params["multichannel"] = "true"
        params["diarize"] = "false"  # Deepgram: multichannel replaces diarize
    
    # ... rest of diarize logic
```

### Whisper — external_apis.py transcribe method (~line 133)

Currently sends:
```python
data = {
    "model": use_model,
    "language": language if language != "auto" else "",
    "word_timestamps": str(word_timestamps).lower(),
    "vad_filter": str(vad_filter).lower(),
}
```

Change to:
```python
def transcribe(self, audio_path: str, ..., config: dict = None) -> dict:
    config = config or {}
    lang = config.get("language") or language
    
    data = {
        "model": use_model,
        "language": lang if lang != "auto" else "",
        "word_timestamps": str(word_timestamps).lower(),
        "vad_filter": str(vad_filter).lower(),
    }
    # Wire new features
    if config.get("custom_prompt"):
        prompt = config["custom_prompt"]
        # Append key terms to prompt for Whisper (no native keyword boost)
        if config.get("key_terms"):
            prompt += "\n\nKey terms: " + ", ".join(config["key_terms"])
        data["initial_prompt"] = prompt
    elif config.get("key_terms"):
        data["initial_prompt"] = "Key terms: " + ", ".join(config["key_terms"])
    if config.get("temperature") is not None:
        data["temperature"] = str(config["temperature"])
    
    # ... rest of transcribe logic
```

### Parakeet — parakeet_client.py

No changes needed. Parakeet has no configurable options beyond what's already wired. The UI hides all options when Parakeet is selected.

### Transcriber — transcriber.py `transcribe()` method (~line 337)

Currently:
```python
def transcribe(self, audio_path, num_speakers=None, audio_data=None, meeting_type="sales_call"):
```

Change to:
```python
def transcribe(self, audio_path, num_speakers=None, audio_data=None, 
               meeting_type="sales_call", transcription_config=None):
    config = transcription_config or {}
    
    # If user selected a specific model, use it directly
    model_pref = config.get("model_preference")
    
    if model_pref == "assemblyai" and self._use_assemblyai:
        return self._transcribe_assemblyai(audio_path, config=config)
    elif model_pref == "deepgram" and self._use_deepgram:
        return self._transcribe_deepgram(audio_path, config=config)
    elif model_pref == "whisper" and self._use_external:
        return self._transcribe_whisper_pyannote(audio_path, config=config)
    elif model_pref == "parakeet" and self._use_parakeet:
        return self._transcribe_parakeet(audio_path)
    else:
        # Auto cascade (existing logic) — pass config to each backend
        # Each backend's method now accepts optional config param
        ...
```

Then update each `_transcribe_*` method to accept and forward `config`:

```python
def _transcribe_assemblyai(self, audio_path, config=None):
    config = config or {}
    result = self._assemblyai_client.transcribe(
        audio_path,
        speakers_expected=self._num_speakers or config.get("num_speakers"),
        config=config,  # Forward all settings
    )
    ...

def _transcribe_deepgram(self, audio_path, config=None):
    config = config or {}
    # Pass config to diarize method
    ...
```

## API Gateway Wiring

### Upload Endpoint — main.py POST /sessions (~line 605)

```python
@app.post("/sessions")
async def create_session_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(default=""),
    meeting_type: str = Form(default="sales_call"),
    num_speakers: Optional[int] = Form(default=None),
    config: str = Form(default="{}"),    # NEW — JSON string with all settings
    current_user: dict = Depends(require_role("member")),
):
    # Parse config
    try:
        config_dict = json.loads(config) if config else {}
    except json.JSONDecodeError:
        config_dict = {}
    
    transcription_config = config_dict.get("transcription", {})
    analysis_config = config_dict.get("analysis", {})
    
    # Merge num_speakers into transcription_config if provided separately
    if num_speakers and "num_speakers" not in transcription_config:
        transcription_config["num_speakers"] = num_speakers
    
    # ... existing file save + session creation ...
    
    # Persist config for re-processing and display
    # Add upload_config column to sessions table if not exists
    try:
        await _pool.execute(
            "UPDATE sessions SET upload_config = $1 WHERE id = $2",
            json.dumps(config_dict), session_id
        )
    except Exception:
        pass  # Column might not exist yet
    
    background_tasks.add_task(
        _run_pipeline,
        session_id=session_id,
        file_path=file_path,
        title=title,
        meeting_type=meeting_type,
        num_speakers=num_speakers,
        transcription_config=transcription_config,
        analysis_config=analysis_config,
    )
```

### Pipeline — forward config to voice agent

```python
async def _run_pipeline(session_id, file_path, title, meeting_type,
                        num_speakers=None, transcription_config=None, analysis_config=None):
    transcription_config = transcription_config or {}
    analysis_config = analysis_config or {}
    
    voice_result = await _call_with_retry(
        lambda: _call_voice_agent(
            session_id, str(file_path.resolve()),
            num_speakers=num_speakers,
            meeting_type=meeting_type,
            transcription_config=transcription_config,
        )
    )
    
    # Skip analysis if user chose transcription-only
    if not analysis_config.get("run_behavioural", True):
        await _try_update_status(session_id, "completed")
        return
    
    # ... rest of pipeline ...
    
    # Neo4j sync if enabled
    if analysis_config.get("run_knowledge_graph", True):
        try:
            from neo4j_sync import sync_session
            await sync_session(_pool, session_id)
        except Exception as e:
            logger.warning(f"[{session_id}] Neo4j sync skipped: {e}")
```

### Voice Agent Call — pass config through

```python
async def _call_voice_agent(session_id, file_path, num_speakers=None,
                            meeting_type="sales_call", transcription_config=None):
    payload = {
        "file_path": file_path,
        "session_id": session_id,
        "meeting_type": meeting_type,
    }
    if num_speakers is not None:
        payload["num_speakers"] = num_speakers
    if transcription_config:
        payload["transcription_config"] = transcription_config
    
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(f"{VOICE_AGENT_URL}/analyse", json=payload)
        resp.raise_for_status()
        return resp.json()
```

## Dashboard Upload Flow

### Full Upload Component Layout

```
┌──────────────────────────────────────────────────────┐
│  Upload Recording                                     │
│                                                       │
│  ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│  │  📁 Drag & drop audio/video file here           │  │
│  │     or click to browse                          │  │
│  │     Supported: WAV, MP3, M4A, FLAC, MP4, WebM  │  │
│  │     Max size: 300 MB                            │  │
│  └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │
│                                                       │
│  ── After file selected: ──                           │
│                                                       │
│  📁 quarterly_review.mp3 (14.2 MB)    [✕ Remove]     │
│                                                       │
│  Title: [quarterly_review___________________]         │
│                                                       │
│  Content Type: [Sales Call ▼]                         │
│                                                       │
│  Transcription Model:                                 │
│  ┌─────────────────────────────────────────────────┐  │
│  │ ○ Auto (Best Available)                          │  │
│  │   Fastest available with fallback cascade        │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ ○ AssemblyAI Universal 3 Pro        ★★★★★      │  │
│  │   Most accurate, full features                   │  │
│  │   Language · Prompt · Key Terms · Fillers ·       │  │
│  │   Formatting · Multichannel                      │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ ○ Deepgram Nova 3                   ★★★★       │  │
│  │   Fast, good diarization                         │  │
│  │   Language · Key Terms · Fillers ·                │  │
│  │   Formatting · Multichannel                      │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ ● Whisper Large V3                  ★★★        │  │
│  │   Best multilingual, custom prompts              │  │
│  │   Language · Prompt · Key Terms                  │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ ○ Parakeet TDT 0.6B                ★★          │  │
│  │   Fastest, English only, basic                   │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  ▸ Transcription Settings   (shows if model supports) │
│  ▸ Speaker Diarization                                │
│  ▸ Transcript Formatting    (shows if model supports) │
│  ▸ Analysis Options                                   │
│                                                       │
│  [Cancel]                         [🚀 Analyse]       │
└──────────────────────────────────────────────────────┘
```

### FormData Submission

```tsx
async function handleUpload(file: File, config: UploadConfig) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("title", config.title || file.name.replace(/\.[^.]+$/, ""));
    formData.append("meeting_type", config.meeting_type);
    
    if (config.num_speakers) {
        formData.append("num_speakers", String(config.num_speakers));
    }
    
    const configJson = {
        transcription: {
            model_preference: config.model_preference,
            language: config.language,
            custom_prompt: config.custom_prompt || null,
            key_terms: config.key_terms
                ? config.key_terms.split(",").map(t => t.trim()).filter(Boolean)
                : null,
            temperature: config.temperature,
            multichannel: config.multichannel,
            keep_filler_words: config.keep_filler_words,
            auto_punctuation: config.auto_punctuation,
            text_formatting: config.text_formatting,
            num_speakers: config.num_speakers,
        },
        analysis: {
            run_behavioural: config.run_behavioural,
            run_entity_extraction: config.run_entity_extraction,
            run_knowledge_graph: config.run_knowledge_graph,
            sensitivity: config.sensitivity,
        },
    };
    formData.append("config", JSON.stringify(configJson));
    
    const response = await fetch("/api/sessions", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
    });
    
    const data = await response.json();
    navigate(`/sessions/${data.session_id}`);
}
```

## DB Migration

```sql
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS upload_config JSONB DEFAULT '{}';
```

## Files to Create/Modify

1. **CREATE** `dashboard/src/components/UploadSettings.tsx` — settings panel with model-aware feature toggles
2. **MODIFY** `dashboard/src/components/SessionList.tsx` (or wherever upload lives) — integrate settings panel
3. **MODIFY** `shared/models/requests.py` — add TranscriptionConfig, AnalysisConfig, extend VoiceAnalysisRequest
4. **MODIFY** `services/api_gateway/main.py` — accept config JSON, pass through pipeline
5. **MODIFY** `services/voiceAgent/main.py` — read config, forward to transcriber
6. **MODIFY** `services/voiceAgent/transcriber.py` — accept config, route to selected model, pass per-backend params
7. **MODIFY** `shared/utils/external_apis.py` — wire AssemblyAI _submit() + Deepgram params + Whisper initial_prompt
8. **RUN** DB migration — add upload_config column

## Files NOT to Modify
- parakeet_client.py — no configurable options to add
- Any agent rules.py — analysis logic doesn't change
- neo4j_sync.py — graph sync doesn't change
- knowledge_store.py — RAG doesn't change
- entity_extractor.py — entity extraction doesn't change