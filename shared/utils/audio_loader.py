"""
Shared audio loading utility for NEXUS agents.

Handles librosa (WAV/FLAC/OGG) with pyav fallback (MP3/M4A/WebM/MP4).
Correct int16/int32 normalisation to [-1, 1] float32.
"""
import numpy as np
import logging

logger = logging.getLogger("nexus.audio_loader")


def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load an audio file as mono float32 at the target sample rate.

    Tries librosa/soundfile first (fast, native).
    Falls back to pyav for container formats (MP3, M4A, WebM, MP4).

    Returns:
        (samples, sample_rate) — samples normalised to [-1.0, 1.0]

    Raises:
        ValueError: if no audio frames could be decoded.
    """
    import librosa

    # Fast path: librosa handles WAV, FLAC, OGG natively via soundfile
    try:
        y, out_sr = librosa.load(path, sr=sr, mono=True)
        return y, out_sr
    except Exception:
        pass

    # Slow path: pyav decodes any ffmpeg-supported container
    import av

    logger.info(f"librosa failed, falling back to pyav for {path}")
    container = av.open(path)
    stream = container.streams.audio[0]
    native_sr = stream.sample_rate

    samples = []
    for packet in container.demux(stream):
        if packet.dts is None:
            continue
        try:
            for frame in packet.decode():
                arr = frame.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)       # mix to mono
                arr = arr.astype(np.float32)
                # Correct normalisation per bit depth
                if frame.format.name in ("s16", "s16p"):
                    arr = arr / 32768.0          # 2^15
                elif frame.format.name in ("s32", "s32p"):
                    arr = arr / 2147483648.0     # 2^31
                samples.append(arr)
        except Exception:
            continue

    container.close()

    if not samples:
        raise ValueError(f"pyav decoded no audio frames from {path}")

    y = np.concatenate(samples).astype(np.float32)

    if native_sr != sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)

    return y, sr
