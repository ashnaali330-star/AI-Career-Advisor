"""Local-only neural speech helpers for the career chat.

Microphone bytes are written to a temporary file only while Whisper decodes
them and are removed immediately afterward. Model files live in the user's
local application cache, outside the project checkout.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import io
import os
from pathlib import Path
import re
import tempfile
from urllib.request import urlretrieve


_MAX_AUDIO_BYTES = 20 * 1024 * 1024
# Kokoro v1.0 voice IDs supported by the project's pinned voice pack. Keeping
# this list static avoids loading/downloading the ONNX engine during Streamlit
# page rendering; the engine is initialized lazily in the background on Listen.
_KOKORO_VOICES = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella",
    "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)
_MODEL_URLS = {
    "kokoro-v1.0.int8.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin",
}


def split_speech_chunks(text: str, max_chars: int = 220) -> list[str]:
    """Split already-clean speech into short, natural chunks for progressive TTS."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            clauses = re.split(r"(?<=[,;:])\s+", sentence)
        else:
            clauses = [sentence]
        for clause in clauses:
            if len(clause) > max_chars:
                words = clause.split()
                pieces, piece = [], ""
                for word in words:
                    candidate = f"{piece} {word}".strip()
                    if len(candidate) > max_chars and piece:
                        pieces.append(piece)
                        piece = word
                    else:
                        piece = candidate
                if piece:
                    pieces.append(piece)
            else:
                pieces = [clause]
            for piece in pieces:
                candidate = f"{current} {piece}".strip()
                if current and len(candidate) > max_chars:
                    chunks.append(current)
                    current = piece
                else:
                    current = candidate
        if current and current[-1:] in ".!?;:":
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _voice_cache_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    return base / "AI-Career-Advisor" / "voice-models"


def _download_if_missing(name: str, url: str) -> Path:
    folder = _voice_cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if target.exists() and target.stat().st_size > 1024 * 1024:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    try:
        urlretrieve(url, partial)
        if partial.stat().st_size < 1024 * 1024:
            raise RuntimeError("The voice model download was incomplete.")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


@lru_cache(maxsize=1)
def _kokoro_engine():
    from kokoro_onnx import Kokoro

    model = _download_if_missing(*("kokoro-v1.0.int8.onnx", _MODEL_URLS["kokoro-v1.0.int8.onnx"]))
    voices = _download_if_missing(*("voices-v1.0.bin", _MODEL_URLS["voices-v1.0.bin"]))
    return Kokoro(str(model), str(voices))


def warm_speech_engine():
    """Load the cached Kokoro runtime; callers should run this in a worker."""
    return _kokoro_engine()


def available_neural_voices() -> list[str]:
    """Return supported voice IDs without loading the slow model at page render."""
    return list(_KOKORO_VOICES)


def synthesize_speech(text: str, voice: str = "af_sarah", style: str = "Natural", speed: float = 1.0) -> bytes:
    """Generate a WAV response with Kokoro ONNX and return it in memory."""
    import soundfile as sf

    engine = _kokoro_engine()
    voices = set(engine.get_voices())
    if voice not in voices:
        voice = "af_sarah" if "af_sarah" in voices else sorted(voices)[0]
    style_speed = {"Natural": 1.0, "Friendly": 1.02, "Professional": 0.96, "Calm": 0.90, "Warm": 0.94}
    final_speed = min(1.25, max(0.75, float(speed) * style_speed.get(style, 1.0)))
    samples, sample_rate = engine.create(text, voice=voice, speed=final_speed, lang="en-us")
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="WAV")
    return output.getvalue()


@lru_cache(maxsize=1)
def _whisper_model():
    from faster_whisper import WhisperModel

    # The tiny English model is intentionally CPU/int8 to fit ordinary laptops.
    return WhisperModel("tiny.en", device="cpu", compute_type="int8")


def transcribe_audio(audio: bytes, mime_type: str = "audio/webm") -> str:
    """Transcribe a browser recording locally, then delete its temporary file."""
    if not audio:
        raise ValueError("The microphone recording was empty. Try recording again.")
    if len(audio) > _MAX_AUDIO_BYTES:
        raise ValueError("That recording is too long to process. Try a shorter question.")
    suffix = ".wav" if "wav" in (mime_type or "").lower() else ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="career-voice-", suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio)
            temp_path = Path(temp_file.name)
        segments, _info = _whisper_model().transcribe(str(temp_path), beam_size=1, vad_filter=True)
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        if not transcript:
            raise ValueError("No speech was recognized. Try speaking clearly and a little closer to the microphone.")
        return transcript
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _transcription_executor() -> ThreadPoolExecutor:
    """Use one background worker so cold Whisper loading never blocks chat sends."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="career-voice")


def transcribe_audio_async(audio: bytes, mime_type: str = "audio/webm"):
    """Schedule local transcription without blocking the Streamlit script thread."""
    return _transcription_executor().submit(transcribe_audio, audio, mime_type)
