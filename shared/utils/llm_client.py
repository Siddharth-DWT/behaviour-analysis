"""
NEXUS Shared — Unified LLM Client
Supports both Anthropic Claude API and OpenAI API behind a single interface.

Configure via environment variables:
  LLM_PROVIDER      = "anthropic" | "openai"  (default: "openai")
  ANTHROPIC_API_KEY  = sk-ant-...
  OPENAI_API_KEY     = sk-...

Models used:
  Anthropic: claude-sonnet-4-20250514 (configurable via LLM_MODEL)
  OpenAI:    gpt-4o-mini       (configurable via LLM_MODEL)

Usage:
    from shared.utils.llm_client import complete

    # Synchronous (blocking)
    text = complete(
        system_prompt="You are a helpful analyst.",
        user_prompt="Classify this text: ...",
        max_tokens=1000,
    )

    # Async
    text = await acomplete(
        system_prompt="You are a helpful analyst.",
        user_prompt="Classify this text: ...",
        max_tokens=1000,
    )
"""
import os
import logging
from typing import Optional

logger = logging.getLogger("nexus.llm_client")

# ── Configuration ──────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower().strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "").rstrip("/")

# Default models per provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o-mini",
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
}

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

LLM_MODEL = os.getenv("LLM_MODEL", "")  # Override; empty = use provider default

# ── Lazy-loaded clients ──
_anthropic_client = None
_openai_client = None


def _get_model() -> str:
    """Get the model name for the current provider."""
    if LLM_MODEL:
        return LLM_MODEL
    return DEFAULT_MODELS.get(LLM_PROVIDER, DEFAULT_MODELS["openai"])


def _ensure_anthropic():
    """Lazy-initialise the Anthropic client."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set"
        )

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "LLM_PROVIDER=anthropic but the 'anthropic' package is not installed. "
            "Run: pip install anthropic"
        )

    _anthropic_client = anthropic.Anthropic(api_key=api_key)
    logger.info(f"Anthropic client initialised (model: {_get_model()})")
    return _anthropic_client


def _ensure_openai():
    """Lazy-initialise the OpenAI client."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set"
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "LLM_PROVIDER=openai but the 'openai' package is not installed. "
            "Run: pip install openai"
        )

    _openai_client = OpenAI(api_key=api_key)
    logger.info(f"OpenAI client initialised (model: {_get_model()})")
    return _openai_client


# ═══════════════════════════════════════════════════════════════
# SYNCHRONOUS API
# ═══════════════════════════════════════════════════════════════

def complete(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> str:
    """
    Send a prompt to the configured LLM provider and return the text response.

    Args:
        system_prompt: System/role instructions
        user_prompt: The user message
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature (0.0 = deterministic)

    Returns:
        The model's text response as a string

    Raises:
        RuntimeError: If API key is missing
        ImportError: If provider SDK is not installed
        Exception: On API call failure
    """
    if LLM_PROVIDER == "anthropic":
        return _complete_anthropic(system_prompt, user_prompt, max_tokens, temperature)
    elif LLM_PROVIDER == "openai":
        return _complete_openai(system_prompt, user_prompt, max_tokens, temperature)
    elif LLM_PROVIDER == "ollama":
        return _complete_ollama(system_prompt, user_prompt, max_tokens, temperature)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
            f"Set LLM_PROVIDER to 'anthropic', 'openai', or 'ollama'."
        )


def _complete_anthropic(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Anthropic Claude Messages API."""
    client = _ensure_anthropic()
    model = _get_model()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )

    return response.content[0].text.strip()


def _complete_openai(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call OpenAI Chat Completions API."""
    client = _ensure_openai()
    model = _get_model()

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()


def _complete_ollama(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call Ollama /api/chat endpoint (OpenAI-compatible format)."""
    if not OLLAMA_URL:
        raise RuntimeError("LLM_PROVIDER=ollama but OLLAMA_URL is not set")

    import httpx
    model = _get_model()

    resp = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ═══════════════════════════════════════════════════════════════
# ASYNC API
# ═══════════════════════════════════════════════════════════════

# Async versions for FastAPI endpoints that want non-blocking calls.
# Falls back to running the sync version in a thread pool if the
# async SDK is not available.

async def acomplete(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> str:
    """
    Async version of complete(). Uses provider's async client if available,
    otherwise runs the sync version in an executor.
    """
    if LLM_PROVIDER == "anthropic":
        return await _acomplete_anthropic(system_prompt, user_prompt, max_tokens, temperature)
    elif LLM_PROVIDER == "openai":
        return await _acomplete_openai(system_prompt, user_prompt, max_tokens, temperature)
    elif LLM_PROVIDER == "ollama":
        return await _acomplete_ollama(system_prompt, user_prompt, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'")


async def _acomplete_anthropic(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Async call to Anthropic Claude API."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        model = _get_model()

        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )

        return response.content[0].text.strip()
    except ImportError:
        # Fallback to sync in executor
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, complete, system_prompt, user_prompt, max_tokens, temperature
        )


async def _acomplete_openai(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Async call to OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        model = _get_model()

        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content.strip()
    except ImportError:
        # Fallback to sync in executor
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, complete, system_prompt, user_prompt, max_tokens, temperature
        )


async def _acomplete_ollama(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Async call to Ollama /api/chat endpoint."""
    if not OLLAMA_URL:
        raise RuntimeError("LLM_PROVIDER=ollama but OLLAMA_URL is not set")

    import httpx
    model = _get_model()

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
        )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ═══════════════════════════════════════════════════════════════
# INTROSPECTION
# ═══════════════════════════════════════════════════════════════

def get_provider_info() -> dict:
    """Return current LLM provider configuration for health checks."""
    provider = LLM_PROVIDER
    model = _get_model()

    if provider == "anthropic":
        key_set = bool(os.getenv("ANTHROPIC_API_KEY", ""))
    elif provider == "openai":
        key_set = bool(os.getenv("OPENAI_API_KEY", ""))
    elif provider == "ollama":
        key_set = bool(OLLAMA_URL)
    else:
        key_set = False

    return {
        "provider": provider,
        "model": model,
        "api_key_configured": key_set,
        "ollama_url": OLLAMA_URL if provider == "ollama" else None,
    }


def is_configured() -> bool:
    """Check if the LLM client is properly configured (provider + key)."""
    info = get_provider_info()
    return info["api_key_configured"]


# ═══════════════════════════════════════════════════════════════
# EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

_local_embed_model = None


async def get_embedding(text: str, model: str = "") -> Optional[list[float]]:
    """
    Generate a text embedding vector for RAG search.

    Ollama provider: uses nomic-embed-text (768 dims) via Ollama API.
    OpenAI provider: uses text-embedding-3-small (1536 dims) via API.
    Anthropic provider: uses local sentence-transformers all-MiniLM-L6-v2
    (384 dims, CPU) since Anthropic has no embeddings API.
    """
    global _local_embed_model

    # Ollama embeddings (self-hosted, free)
    if LLM_PROVIDER == "ollama" and OLLAMA_URL:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={
                        "model": model or OLLAMA_EMBED_MODEL,
                        "input": text[:2000],
                    },
                )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings:
                return embeddings[0]
            logger.warning(f"Ollama embed returned empty: {list(data.keys())}")
            return None
        except Exception as e:
            logger.warning(f"Ollama embedding failed: {e}")
            return None

    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — cannot generate embeddings")
            return None
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)
            response = await client.embeddings.create(
                input=text[:8000],
                model=model or "text-embedding-3-small",
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}")
            return None

    # Anthropic / fallback: local sentence-transformers
    try:
        if _local_embed_model is None:
            from sentence_transformers import SentenceTransformer
            _local_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded local embedding model: all-MiniLM-L6-v2 (384 dims)")

        import asyncio
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: _local_embed_model.encode(text[:512]).tolist()
        )
        return embedding
    except ImportError:
        logger.warning(
            "sentence-transformers not installed for local embeddings. "
            "Install: pip install sentence-transformers"
        )
        return None
    except Exception as e:
        logger.warning(f"Local embedding failed: {e}")
        return None
