"""
Shared audio loading utility for NEXUS agents.

Handles librosa (WAV/FLAC/OGG) with ffmpeg fallback for container formats
(MP3, M4A, WebM, MP4).  ffmpeg is more reliable than pyav for mixed-codec
containers and avoids dependency on the av wheel.
"""
import logging
import os
import subprocess
import tempfile

import numpy as np

logger = logging.getLogger("nexus.audio_loader")


def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load an audio file as mono float32 at the target sample rate.

    Tries librosa/soundfile first (fast, native for WAV/FLAC/OGG).
    Falls back to ffmpeg for container formats (MP3, M4A, WebM, MP4, etc.)
    by extracting a temporary WAV and reloading it with librosa.

    Returns:
        (samples, sample_rate) — samples normalised to [-1.0, 1.0]

    Raises:
        ValueError: if no audio could be decoded.
    """
    import librosa
    import warnings

    # Fast path: librosa handles WAV, FLAC, OGG natively via soundfile
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PySoundFile failed")
            warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
            y, out_sr = librosa.load(path, sr=sr, mono=True)
            return y, out_sr
    except Exception:
        pass

    # Slow path: ffmpeg extracts audio to a temp WAV, then librosa loads it
    logger.info(f"librosa failed, falling back to ffmpeg extraction for {path}")

    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-ac", "1",                     # mono
                "-ar", str(sr),                 # target sample rate
                "-vn",                          # no video
                "-f", "wav",
                tmp_wav,
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise ValueError(
                f"ffmpeg audio extraction failed (rc={result.returncode}): "
                f"{result.stderr[-300:]}"
            )

        y, out_sr = librosa.load(tmp_wav, sr=sr, mono=True)
        return y, out_sr
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
