#!/usr/bin/env python3
"""Prepare reference recordings and synthesize new text with XTTS-v2."""
import argparse
from contextlib import ExitStack
import json
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def split_text(text, max_chars):
    """Split paragraphs into bounded chunks, preferring sentence/word ends."""
    if max_chars < 1:
        raise ValueError("--chunk-chars must be positive.")
    chunks = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        remaining = " ".join(paragraph.split())
        while len(remaining) > max_chars:
            window = remaining[:max_chars + 1]
            # Include closing quotes in a sentence boundary when possible.
            ends = [match.end() for match in re.finditer(
                r'''[.!?]["'”’\)\]]*(?=\s)|[。！？]["'”’\)\]]*''', window
            ) if match.end() <= max_chars]
            if ends:
                end = ends[-1]
            else:
                spaces = [match.start() for match in re.finditer(r"\s", window)]
                end = spaces[-1] if spaces else max_chars
            chunks.append(remaining[:end].strip())
            remaining = remaining[end:].lstrip()
        if remaining:
            chunks.append(remaining)
    return chunks


def merge_wavs(paths, output, pause_ms):
    """Stream matching PCM WAV chunks into one WAV with interchunk silence."""
    if not paths:
        raise ValueError("No audio chunks to merge.")
    if not isinstance(pause_ms, int) or pause_ms < 0:
        raise ValueError("--pause-ms must be a nonnegative integer.")
    audio_format = None
    with ExitStack() as stack:
        for index, path in enumerate(paths):
            with wave.open(str(path), "rb") as chunk:
                current = (chunk.getnchannels(), chunk.getsampwidth(), chunk.getframerate())
                if chunk.getcomptype() != "NONE" or chunk.getsampwidth() != 2:
                    raise ValueError("Generated chunks must be 16-bit PCM WAV audio.")
                if chunk.getnframes() == 0:
                    raise ValueError("Model produced empty audio.")
                if audio_format is None:
                    audio_format = current
                    merged = stack.enter_context(wave.open(str(output), "wb"))
                    merged.setnchannels(current[0])
                    merged.setsampwidth(current[1])
                    merged.setframerate(current[2])
                elif current != audio_format:
                    raise ValueError("Generated chunks have incompatible WAV formats.")
                if index:
                    frames_left = round(current[2] * pause_ms / 1000)
                    while frames_left:
                        frames = min(frames_left, 65536)
                        merged.writeframesraw(b"\0" * frames * current[0] * current[1])
                        frames_left -= frames
                expected_bytes = chunk.getnframes() * current[0] * current[1]
                copied_bytes = 0
                while data := chunk.readframes(65536):
                    merged.writeframesraw(data)
                    copied_bytes += len(data)
                if copied_bytes != expected_bytes:
                    raise ValueError("Model produced truncated audio.")


def prepare(args):
    if not shutil.which("ffmpeg"):
        raise ValueError("Install FFmpeg and put it on PATH first.")
    if not 0 <= args.start or not 3 <= args.seconds <= 30:
        raise ValueError("Use --start >= 0 and --seconds between 3 and 30.")
    inputs = [Path(p).resolve() for p in args.input]
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / f"ref_{i:02d}.wav" for i in range(1, len(inputs) + 1)]
    if any(p.exists() for p in destinations):
        raise ValueError("Reference outputs already exist; choose a new --out-dir.")
    records = []
    # Convert all inputs successfully before copying any into the destination.
    with tempfile.TemporaryDirectory() as tmp:
        for i, (source, destination) in enumerate(zip(inputs, destinations)):
            converted = Path(tmp) / destination.name
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", str(args.start), "-i", str(source),
                "-t", str(args.seconds), "-vn", "-ac", "1", "-ar", "24000",
                "-c:a", "pcm_s16le", str(converted),
            ], check=True)
            with wave.open(str(converted), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            if duration < 3:
                raise ValueError(f"{source.name}: selected segment is shorter than 3 seconds.")
            records.append({"source": str(source), "reference": str(destination),
                            "start_seconds": args.start, "duration_seconds": duration})
        for destination in destinations:
            shutil.copyfile(Path(tmp) / destination.name, destination)
    print(json.dumps(records, indent=2))
    print("Listen to every reference before synthesis; retain clean speech from one speaker.")


def synthesize(args, progress=None):
    text = (Path(args.text_file).read_text(encoding="utf-8")
            if args.text_file else args.text).strip()
    if not text:
        raise ValueError("Text cannot be empty.")
    chunk_chars = getattr(args, "chunk_chars", None)
    pause_ms = getattr(args, "pause_ms", None)
    if chunk_chars is None and pause_ms is not None:
        raise ValueError("--pause-ms requires --chunk-chars.")
    if pause_ms is not None and (not isinstance(pause_ms, int) or pause_ms < 0):
        raise ValueError("--pause-ms must be a nonnegative integer.")
    chunks = split_text(text, chunk_chars) if chunk_chars is not None else [text]
    pause_ms = 200 if pause_ms is None else pause_ms
    references = [Path(p).resolve() for p in args.refs]
    for path in references:
        if not path.is_file():
            raise ValueError(f"Missing reference: {path}")
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError(f"Run prepare first: {path} must be mono 16-bit PCM WAV.")
            if wav.getnframes() / wav.getframerate() < 3:
                raise ValueError(f"Reference too short: {path}")
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".wav":
        raise ValueError("Output must have a .wav extension.")
    if output.exists():
        raise ValueError("Output exists; choose a different filename.")
    import random
    try:
        import numpy as np
        import torch
        from TTS.api import TTS
    except ImportError as exc:
        raise ValueError("Speech dependencies are missing. Follow the README installation steps "
                         "for PyTorch and requirements.txt.") from exc
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install matching GPU PyTorch or use --device cpu.")
    print(f"Loading {MODEL} on {device}; first run downloads model weights.")
    tts = TTS(MODEL).to(device)
    if tts.languages and args.language not in tts.languages:
        raise ValueError(f"Supported languages: {tts.languages}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A temporary output prevents an interrupted synthesis leaving a final WAV.
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        generated = Path(tmp) / "generated.wav"
        if progress:
            progress(0, len(chunks))
        with torch.inference_mode():
            if chunk_chars is None:
                tts.tts_to_file(text=text, speaker_wav=[str(p) for p in references],
                                language=args.language, file_path=str(generated),
                                split_sentences=True)
                if progress:
                    progress(1, 1)
            else:
                audio_chunks = []
                for index, chunk in enumerate(chunks, start=1):
                    print(f"Generating chunk {index}/{len(chunks)} ({len(chunk)} characters)",
                          flush=True)
                    chunk_path = Path(tmp) / f"chunk_{index:05d}.wav"
                    tts.tts_to_file(text=chunk, speaker_wav=[str(p) for p in references],
                                   language=args.language, file_path=str(chunk_path),
                                   split_sentences=True)
                    audio_chunks.append(chunk_path)
                    if progress:
                        progress(index, len(chunks))
                merge_wavs(audio_chunks, generated, pause_ms)
        with wave.open(str(generated), "rb") as wav:
            if wav.getnframes() == 0:
                raise ValueError("Model produced empty audio.")
            print(f"Generated {wav.getnframes() / wav.getframerate():.1f}s at {wav.getframerate()} Hz")
        generated.replace(output)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Extract mono 24 kHz reference WAVs using FFmpeg")
    prep.add_argument("--input", nargs="+", required=True)
    prep.add_argument("--out-dir", default="references")
    prep.add_argument("--start", type=float, default=0)
    prep.add_argument("--seconds", type=float, default=12)
    prep.set_defaults(func=prepare)
    synth = commands.add_parser("synthesize", help="Generate new speech using reference WAVs")
    synth.add_argument("--refs", nargs="+", required=True)
    source = synth.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    synth.add_argument("--output", default="generated.wav")
    synth.add_argument("--language", default="en")
    synth.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    synth.add_argument("--seed", type=int, default=42)
    synth.add_argument("--chunk-chars", type=int,
                       help="Enable long narration with chunks of at most this many characters")
    synth.add_argument("--pause-ms", type=int,
                       help="Silence between narration chunks in ms (default: 200; requires --chunk-chars)")
    synth.set_defaults(func=synthesize)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, EOFError, subprocess.CalledProcessError, wave.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
