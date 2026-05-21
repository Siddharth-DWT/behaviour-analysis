# backend/agents/base.py
"""
BaseAgentService ABC and TranscriptionBackend Protocol.

All 5 in-process agent services implement BaseAgentService so the monolith
lifespan can start/stop them with a uniform loop:
    for svc in services: await svc.startup()

TranscriptionBackend is a Strategy Protocol — the Transcriber class in
services/voiceAgent/ already implements it via TRANSCRIPTION_BACKEND env var
switching (Whisper / AssemblyAI / Deepgram), so no refactoring is needed there.

Design:
  - Composition over inheritance: rule engines are constructor parameters, not
    superclasses. Services import the engines they need.
  - ABC enforces the contract (name, startup, shutdown) at class-definition time
    rather than at runtime, so missing overrides surface immediately on import.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("nexus.backend.agents")


@runtime_checkable
class TranscriptionBackend(Protocol):
    """
    Strategy interface for swappable transcription providers.

    The existing Transcriber class already implements this contract — it selects
    the provider at construction time based on TRANSCRIPTION_BACKEND env var
    (auto / whisper / assemblyai / deepgram / external_whisper).
    No changes to Transcriber are required.
    """

    def transcribe(self, file_path: str, **kwargs: Any) -> dict: ...


class BaseAgentService(ABC):
    """
    Contract for all 5 in-process analysis agent services.

    Lifecycle (called once by backend/main.py lifespan):
      startup()  — load models, allocate thread pools, warm up caches
      shutdown() — release GPU/CPU resources, drain thread pools

    The name property is the canonical identifier used in:
      - Redis key namespacing  (nexus:session:{id}:result:{name})
      - Log prefixes
      - Health-check response dict keys

    Composition pattern:
      Rule engines, feature extractors, and calibrators are passed as
      constructor arguments or built inside startup() — never via class
      inheritance. This keeps each service independently testable.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable lower-case identifier, e.g. 'voice', 'language', 'fusion'."""
        ...

    @abstractmethod
    async def startup(self) -> None:
        """
        Load models and allocate resources.
        Called sequentially (not gathered) to avoid PyTorch GIL conflicts
        when multiple services initialise CUDA context simultaneously.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources. Called on app shutdown — must not raise."""
        ...

    # ── Convenience helpers shared by all services ──────────────────────────

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        logger.log(level, "[%s] %s", self.name, msg)

    def _warn(self, msg: str) -> None:
        self._log(msg, logging.WARNING)
