#!/usr/bin/env python3
"""Prepare reference recordings and synthesize new text with XTTS-v2."""
import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


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


def synthesize(args):
    text = (Path(args.text_file).read_text(encoding="utf-8")
            if args.text_file else args.text).strip()
    if not text:
        raise ValueError("Text cannot be empty.")
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
    import numpy as np
    import torch
    from TTS.api import TTS
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
        with torch.inference_mode():
            tts.tts_to_file(text=text, speaker_wav=[str(p) for p in references],
                            language=args.language, file_path=str(generated),
                            split_sentences=True)
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
    synth.set_defaults(func=synthesize)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, subprocess.CalledProcessError, wave.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
