"""
NEXUS Shared Configuration
All agents import from here for consistent DB/Redis connections.
"""
import os
from dataclasses import dataclass


@dataclass
class NexusConfig:
    """Central configuration loaded from environment variables."""

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://nexus:nexus_dev_2026@localhost:5432/nexus"
    )

    # Redis / Valkey
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # LLM Provider ("openai" or "anthropic")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "")  # empty = use provider default

    # API Keys
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # External GPU APIs (optional — accelerates transcription & TTS)
    external_whisper_url: str = os.getenv("EXTERNAL_WHISPER_URL", "")
    external_tts_url: str = os.getenv("EXTERNAL_TTS_URL", "")
    external_api_key: str = os.getenv("EXTERNAL_API_KEY", "")
    external_whisper_model: str = os.getenv("EXTERNAL_WHISPER_MODEL", "base")

    # Agent settings
    fusion_cycle_seconds: float = 10.0      # How often FUSION runs per speaker
    calibration_min_seconds: int = 60       # Minimum speech for initial baseline
    calibration_reliable_seconds: int = 180 # Speech duration for reliable baseline

    # Diarization
    diarization_mode: str = os.getenv("DIARIZATION_MODE", "auto")  # auto|pyannote|embedding|kmeans
    diarization_embedding_model: str = os.getenv(
        "DIARIZATION_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
    )

    # Redis Streams
    stream_prefix: str = "nexus:stream"
    max_stream_length: int = 10000          # Max entries per stream before trimming

    def stream_name(self, agent: str, session_id: str) -> str:
        """Generate stream name: nexus:stream:voice:{session_id}"""
        return f"{self.stream_prefix}:{agent}:{session_id}"

    @property
    def has_external_whisper(self) -> bool:
        """Check if external Whisper API is configured."""
        return bool(self.external_whisper_url and self.external_api_key)

    @property
    def has_external_tts(self) -> bool:
        """Check if external TTS API is configured."""
        return bool(self.external_tts_url and self.external_api_key)

    @property
    def has_llm_configured(self) -> bool:
        """Check if the active LLM provider has a valid API key."""
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key and not self.anthropic_api_key.startswith("sk-ant-your"))
        else:  # openai
            return bool(self.openai_api_key and not self.openai_api_key.startswith("sk-your"))

    def validate(self) -> list[str]:
        """Check for missing required configuration. Returns list of issues."""
        issues = []
        if self.llm_provider == "anthropic":
            if not self.anthropic_api_key or self.anthropic_api_key.startswith("sk-ant-your"):
                issues.append("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY not set")
        elif self.llm_provider == "openai":
            if not self.openai_api_key or self.openai_api_key.startswith("sk-your"):
                issues.append("LLM_PROVIDER=openai but OPENAI_API_KEY not set")
        else:
            issues.append(f"Unknown LLM_PROVIDER: '{self.llm_provider}' (use 'openai' or 'anthropic')")
        return issues

    def validate_external_apis(self) -> list[str]:
        """Check external API configuration. Returns list of issues."""
        issues = []
        if self.external_whisper_url and not self.external_api_key:
            issues.append("EXTERNAL_WHISPER_URL set but EXTERNAL_API_KEY missing")
        if self.external_tts_url and not self.external_api_key:
            issues.append("EXTERNAL_TTS_URL set but EXTERNAL_API_KEY missing")
        return issues


# Singleton config instance
config = NexusConfig()
