"""
NEXUS Shared — Universal Media Ingestion
Accepts any audio/video format or URL and produces a clean 16kHz mono WAV
for the Voice Agent pipeline.

Supported inputs:
  - Audio files: .wav, .mp3, .flac, .ogg, .m4a, .aac, .wma, .opus
  - Video files: .mp4, .mkv, .webm, .mov, .avi, .flv
  - URLs: YouTube, podcast RSS, direct audio/video URLs

Dependencies:
  - ffmpeg (preferred) or afconvert (macOS fallback) for conversion
  - yt-dlp (optional, for YouTube/URL downloads)

Usage:
    from shared.utils.media_ingest import prepare_audio

    # From a local file
    wav_path = await prepare_audio("/path/to/recording.mp4")

    # From a URL
    wav_path = await prepare_audio("https://youtube.com/watch?v=XXX")

    # Synchronous version
    wav_path = prepare_audio_sync("/path/to/file.mp3")
"""
import os
import sys
import logging
import subprocess
import tempfile
import asyncio
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("nexus.media_ingest")

# ── Supported formats ──
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".wmv", ".ts"}
ALL_MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Target format for Voice Agent
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = "wav"

# Output directory for downloaded/converted files
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "recordings"


def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_afconvert() -> bool:
    """Check if macOS afconvert is available."""
    try:
        result = subprocess.run(
            ["afconvert", "--help"],
            capture_output=True, timeout=5,
        )
        # afconvert returns non-zero for --help but still means it's installed
        return True
    except FileNotFoundError:
        return False


def _has_ytdlp() -> bool:
    """Check if yt-dlp is available."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except FileNotFoundError:
        # Try with full path (pip user install on macOS)
        for path in [
            os.path.expanduser("~/Library/Python/3.9/bin/yt-dlp"),
            os.path.expanduser("~/.local/bin/yt-dlp"),
        ]:
            if os.path.exists(path):
                return True
        return False


def _get_ytdlp_cmd() -> str:
    """Get the yt-dlp command (may be a full path)."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=5)
        return "yt-dlp"
    except FileNotFoundError:
        for path in [
            os.path.expanduser("~/Library/Python/3.9/bin/yt-dlp"),
            os.path.expanduser("~/.local/bin/yt-dlp"),
        ]:
            if os.path.exists(path):
                return path
    return "yt-dlp"


def _is_url(input_path: str) -> bool:
    """Check if the input looks like a URL."""
    parsed = urlparse(input_path)
    return parsed.scheme in ("http", "https", "ftp")


def _is_youtube_url(url: str) -> bool:
    """Check if the URL is a YouTube URL."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return any(h in host for h in ["youtube.com", "youtu.be", "youtube-nocookie.com"])


def _get_file_extension(filepath: str) -> str:
    """Get normalized file extension."""
    return Path(filepath).suffix.lower()


def _is_already_target_format(filepath: str) -> bool:
    """Check if file is already in our target format (16kHz mono WAV)."""
    if _get_file_extension(filepath) != ".wav":
        return False

    # Check actual WAV properties
    try:
        import wave
        with wave.open(filepath, "rb") as wf:
            return (
                wf.getframerate() == TARGET_SAMPLE_RATE
                and wf.getnchannels() == TARGET_CHANNELS
                and wf.getsampwidth() == 2  # 16-bit
            )
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# CONVERSION
# ═══════════════════════════════════════════════════════════════

def _convert_with_ffmpeg(input_path: str, output_path: str) -> str:
    """Convert any audio/video to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-sample_fmt", "s16",
        "-f", "wav",
        output_path,
    ]

    logger.info(f"Converting with ffmpeg: {input_path} → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:500]}")

    return output_path


def _convert_with_afconvert(input_path: str, output_path: str) -> str:
    """Convert audio to 16kHz mono WAV using macOS afconvert."""
    cmd = [
        "afconvert",
        "-f", "WAVE",
        "-d", f"LEI16@{TARGET_SAMPLE_RATE}",
        "-c", str(TARGET_CHANNELS),
        input_path,
        output_path,
    ]

    logger.info(f"Converting with afconvert: {input_path} → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"afconvert conversion failed: {result.stderr[:500]}")

    return output_path


def _extract_audio_with_ffmpeg(video_path: str, output_path: str) -> str:
    """Extract audio from video file and convert to 16kHz mono WAV."""
    return _convert_with_ffmpeg(video_path, output_path)


def convert_to_target_format(
    input_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Convert any audio/video file to 16kHz mono WAV.

    Args:
        input_path: Path to the input media file
        output_path: Optional output path. If None, generates one.

    Returns:
        Path to the converted WAV file
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = _get_file_extension(input_path)
    if ext not in ALL_MEDIA_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Supported: {', '.join(sorted(ALL_MEDIA_EXTENSIONS))}"
        )

    # Check if already in target format
    if _is_already_target_format(input_path):
        logger.info(f"File already in target format: {input_path}")
        return input_path

    # Generate output path if not provided
    if output_path is None:
        stem = Path(input_path).stem
        output_path = str(DEFAULT_OUTPUT_DIR / f"{stem}-16k.wav")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert using best available tool
    if _has_ffmpeg():
        return _convert_with_ffmpeg(input_path, output_path)
    elif _has_afconvert() and ext in AUDIO_EXTENSIONS:
        return _convert_with_afconvert(input_path, output_path)
    elif _has_afconvert() and ext in VIDEO_EXTENSIONS:
        raise RuntimeError(
            f"Cannot extract audio from video ({ext}) without ffmpeg. "
            f"Install ffmpeg: brew install ffmpeg"
        )
    else:
        raise RuntimeError(
            "No audio conversion tool found. "
            "Install ffmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )


# ═══════════════════════════════════════════════════════════════
# URL DOWNLOAD
# ═══════════════════════════════════════════════════════════════

def download_url(
    url: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Download audio from a URL using yt-dlp or direct HTTP download.

    Args:
        url: The URL to download from
        output_dir: Directory to save the downloaded file
        filename: Optional filename (without extension)

    Returns:
        Path to the downloaded file
    """
    if output_dir is None:
        output_dir = str(DEFAULT_OUTPUT_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if _is_youtube_url(url) or _needs_ytdlp(url):
        return _download_with_ytdlp(url, output_dir, filename)
    else:
        return _download_direct(url, output_dir, filename)


def _needs_ytdlp(url: str) -> bool:
    """Check if the URL likely needs yt-dlp (streaming platforms)."""
    hosts_needing_ytdlp = [
        "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
        "soundcloud.com", "twitch.tv", "tiktok.com", "twitter.com",
        "x.com", "facebook.com", "instagram.com",
    ]
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return any(h in host for h in hosts_needing_ytdlp)


def _download_with_ytdlp(
    url: str,
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    """Download audio from a URL using yt-dlp."""
    if not _has_ytdlp():
        raise RuntimeError(
            "yt-dlp is required for downloading from this URL. "
            "Install: pip install yt-dlp"
        )

    ytdlp_cmd = _get_ytdlp_cmd()
    fname = filename or "downloaded_audio"
    output_template = os.path.join(output_dir, f"{fname}.%(ext)s")

    # Try to download best audio format
    # Use --extract-audio if ffmpeg is available, otherwise download best format
    if _has_ffmpeg():
        cmd = [
            ytdlp_cmd,
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", output_template,
            "--no-playlist",
            url,
        ]
    else:
        # Without ffmpeg, download the best audio-only format or lowest quality video
        cmd = [
            ytdlp_cmd,
            "-f", "bestaudio/worst",
            "-o", output_template,
            "--no-playlist",
            url,
        ]

    logger.info(f"Downloading with yt-dlp: {url}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        # Try alternative: download any available format
        logger.warning(f"yt-dlp failed with preferred format, trying fallback...")
        cmd_fallback = [
            ytdlp_cmd,
            "-f", "18/worst",  # format 18 = 360p mp4 (widely available)
            "-o", output_template,
            "--no-playlist",
            url,
        ]
        result = subprocess.run(
            cmd_fallback, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlp download failed: {result.stderr[:500]}"
            )

    # Find the downloaded file (extension may vary)
    downloaded_files = list(Path(output_dir).glob(f"{fname}.*"))
    if not downloaded_files:
        raise RuntimeError("yt-dlp completed but no output file found")

    # Return the most recent file
    downloaded = str(max(downloaded_files, key=lambda f: f.stat().st_mtime))
    logger.info(f"Downloaded: {downloaded}")
    return downloaded


def _download_direct(
    url: str,
    output_dir: str,
    filename: Optional[str] = None,
) -> str:
    """Download a file directly via HTTP(S)."""
    try:
        import httpx
    except ImportError:
        import urllib.request
        # Fallback to urllib
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix or ".mp3"
        fname = filename or "downloaded_audio"
        output_path = os.path.join(output_dir, f"{fname}{ext}")

        logger.info(f"Downloading (urllib): {url}")
        urllib.request.urlretrieve(url, output_path)
        return output_path

    # Use httpx for better error handling
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or ".mp3"
    fname = filename or "downloaded_audio"
    output_path = os.path.join(output_dir, f"{fname}{ext}")

    logger.info(f"Downloading (httpx): {url}")
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        with open(output_path, "wb") as f:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

    logger.info(f"Downloaded: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════
# MAIN API
# ═══════════════════════════════════════════════════════════════

def prepare_audio_sync(
    input_path_or_url: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Synchronous version: Accept any media file or URL, return a clean 16kHz
    mono WAV path ready for the Voice Agent.

    Args:
        input_path_or_url: Local file path or URL
        output_dir: Optional output directory
        filename: Optional output filename (without extension)

    Returns:
        Absolute path to the 16kHz mono WAV file
    """
    if output_dir is None:
        output_dir = str(DEFAULT_OUTPUT_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fname = filename or "prepared_audio"

    # Step 1: If URL, download first
    if _is_url(input_path_or_url):
        logger.info(f"Input is a URL, downloading: {input_path_or_url}")
        raw_path = download_url(input_path_or_url, output_dir, fname)
    else:
        raw_path = input_path_or_url

    # Step 2: Verify file exists
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Media file not found: {raw_path}")

    # Step 3: Check if already in target format
    if _is_already_target_format(raw_path):
        logger.info(f"Audio already in target format: {raw_path}")
        return os.path.abspath(raw_path)

    # Step 4: Convert to target format
    output_path = os.path.join(output_dir, f"{fname}-16k.wav")
    converted = convert_to_target_format(raw_path, output_path)
    logger.info(f"Audio prepared: {converted}")

    return os.path.abspath(converted)


async def prepare_audio(
    input_path_or_url: str,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Async version: Accept any media file or URL, return a clean 16kHz
    mono WAV path ready for the Voice Agent.

    Args:
        input_path_or_url: Local file path or URL
        output_dir: Optional output directory
        filename: Optional output filename (without extension)

    Returns:
        Absolute path to the 16kHz mono WAV file
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        prepare_audio_sync,
        input_path_or_url,
        output_dir,
        filename,
    )


# ═══════════════════════════════════════════════════════════════
# INTROSPECTION
# ═══════════════════════════════════════════════════════════════

def get_capabilities() -> dict:
    """Return a dict of available media ingestion capabilities."""
    return {
        "ffmpeg_available": _has_ffmpeg(),
        "afconvert_available": _has_afconvert(),
        "ytdlp_available": _has_ytdlp(),
        "supported_audio_formats": sorted(AUDIO_EXTENSIONS),
        "supported_video_formats": sorted(VIDEO_EXTENSIONS),
        "url_download": _has_ytdlp() or True,  # direct HTTP always works
        "youtube_download": _has_ytdlp(),
    }
