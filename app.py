#!/usr/bin/env python3
"""A local browser interface for reference preparation and voice generation."""

import argparse
import math
import subprocess
import tempfile
import wave
from pathlib import Path

import voice


LANGUAGES = [
    ("English", "en"), ("Spanish", "es"), ("French", "fr"),
    ("German", "de"), ("Italian", "it"), ("Portuguese", "pt"),
    ("Polish", "pl"), ("Turkish", "tr"), ("Russian", "ru"),
    ("Dutch", "nl"), ("Czech", "cs"), ("Arabic", "ar"),
    ("Chinese", "zh-cn"), ("Hungarian", "hu"), ("Korean", "ko"),
    ("Japanese", "ja"), ("Hindi", "hi"),
]


def _integer(value, label, minimum=0):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a whole number.")
    if int(value) != value or value < minimum:
        raise ValueError(f"{label} must be a whole number of at least {minimum}.")
    return int(value)


def prepare_recordings(inputs, start, seconds, work_dir):
    """Prepare one isolated set of uploaded recordings and return its WAV paths."""
    if not inputs:
        raise ValueError("Upload at least one recording first.")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (start, seconds)):
        raise ValueError("Start time and clip length must be finite numbers.")
    directory = Path(tempfile.mkdtemp(prefix="references-", dir=work_dir))
    try:
        voice.prepare(argparse.Namespace(
            input=inputs, out_dir=directory, start=start, seconds=seconds,
        ))
        return [str(path) for path in sorted(directory.glob("ref_*.wav"))]
    except Exception:
        # This directory belongs only to this operation.
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        raise


def generate_speech(refs, text, language, device, seed, chunking,
                    chunk_chars, pause_ms, work_dir, progress=None):
    """Generate a unique output file using the same backend as the CLI."""
    if not refs:
        raise ValueError("Prepare recordings above or upload reference WAVs first.")
    if not text or not text.strip():
        raise ValueError("Enter the text you want to hear.")
    seed = _integer(seed, "Seed")
    if seed > 2**32 - 1:
        raise ValueError("Seed must be between 0 and 4294967295.")
    if chunking:
        chunk_chars = _integer(chunk_chars, "Chunk length", minimum=1)
        pause_ms = _integer(pause_ms, "Pause length")
    directory = Path(tempfile.mkdtemp(prefix="speech-", dir=work_dir))
    output = directory / "generated.wav"
    try:
        voice.synthesize(argparse.Namespace(
            refs=refs, text=text, text_file=None, language=language,
            device=device, seed=seed, output=output,
            chunk_chars=chunk_chars if chunking else None,
            pause_ms=pause_ms if chunking else None,
        ), progress=progress)
        return str(output)
    except Exception:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
        raise


def create_app(work_dir):
    """Build the app; keep work_dir alive until the server has stopped."""
    import gradio as gr

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    def show_error(exc):
        if isinstance(exc, ImportError):
            message = (
                "A speech dependency is missing. Install requirements-ui.txt and "
                "the hardware-appropriate PyTorch packages described in README.md. "
                f"Details: {exc}"
            )
        elif isinstance(exc, subprocess.CalledProcessError):
            message = "FFmpeg could not read this recording. Try another audio file or segment."
        else:
            message = str(exc)
        raise gr.Error(message) from exc

    def prepare(inputs, start, seconds):
        try:
            paths = prepare_recordings(inputs, start, seconds, work_dir)
            choices = [(Path(path).name, path) for path in paths]
            return (
                paths, paths, gr.update(choices=choices, value=paths[0]), paths[0],
                f"Prepared {len(paths)} reference(s). Listen below, then enter your text in step 2.",
            )
        except (ValueError, OSError, EOFError, subprocess.CalledProcessError, wave.Error) as exc:
            show_error(exc)

    def generate(refs, text, language, device, seed, chunking, chunk_chars, pause_ms,
                 progress=gr.Progress()):
        try:
            progress(0, desc="Loading the voice model and generating speech…")
            path = generate_speech(
                refs, text, language, device, seed, chunking, chunk_chars, pause_ms,
                work_dir, progress=lambda completed, total: progress(
                    completed / total, desc=f"Generated chunk {completed} of {total}…",
                ),
            )
            with wave.open(path, "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            progress(1, desc="Ready to listen")
            return path, path, f"Ready · {duration:.1f} seconds. Download the WAV to keep it."
        except (ValueError, OSError, EOFError, ImportError, RuntimeError,
                subprocess.CalledProcessError, wave.Error) as exc:
            show_error(exc)

    with gr.Blocks(title="Voice studio") as app:
        gr.Markdown(
            "# Voice studio\n"
            "Turn your recordings into a reference voice, then bring new text to life. "
            "Audio is processed on the machine running this app."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 1. Prepare your voice\n"
                            "Use clean recordings of one speaker. Start with one to three "
                            "clips, without background music or overlapping voices.")
                recordings = gr.File(
                    label="Source recordings", file_count="multiple", type="filepath",
                    file_types=["audio"],
                )
                with gr.Row():
                    start = gr.Number(label="Start at (seconds)", value=0, minimum=0)
                    seconds = gr.Slider(
                        label="Clip length (seconds)", minimum=3, maximum=30,
                        value=12, step=1,
                    )
                prepare_button = gr.Button("Prepare references", variant="secondary")
                prepare_status = gr.Markdown("Prepared clips will appear here.")
                prepared_files = gr.File(label="Prepared WAVs", file_count="multiple",
                                         interactive=False)
                preview_choice = gr.Dropdown(label="Choose a reference to listen to", choices=[])
                reference_audio = gr.Audio(label="Reference preview", interactive=False)

            with gr.Column():
                gr.Markdown("## 2. Create speech\n"
                            "Prepared references are added automatically. You can also "
                            "upload existing mono, 16-bit PCM WAVs of at least 3 seconds.")
                refs = gr.File(
                    label="Reference WAVs", file_count="multiple", type="filepath",
                    file_types=[".wav"],
                )
                text = gr.Textbox(
                    label="What should the voice say?", lines=7,
                    placeholder="Type or paste your text here…",
                )
                language = gr.Dropdown(label="Language of the text", choices=LANGUAGES, value="en")
                with gr.Accordion("Generation settings", open=False):
                    device = gr.Dropdown(
                        label="Processing device", choices=["auto", "cpu", "cuda"], value="auto",
                        info="Auto uses an NVIDIA GPU when available. CPU generation can be slow.",
                    )
                    seed = gr.Number(label="Seed", value=42, minimum=0, precision=0,
                                     info="Use the same seed to compare recordings and settings.")
                    chunking = gr.Checkbox(label="Split long text into manageable chunks", value=True)
                    chunk_chars = gr.Number(label="Maximum characters per chunk", value=200,
                                            minimum=1, precision=0)
                    pause_ms = gr.Number(label="Pause between chunks (milliseconds)", value=200,
                                         minimum=0, precision=0)
                    gr.Markdown("Use a smaller chunk size if the model reports a text limit. "
                                "Set the pause to 0 for no added silence.")
                generate_button = gr.Button("Generate speech", variant="primary")
                generation_status = gr.Markdown("Your generated speech will appear here.")
                generated_audio = gr.Audio(label="Listen to your result", interactive=False)
                download = gr.File(label="Download WAV", interactive=False)

        gr.Markdown(
            "**First run:** model weights are downloaded; keep the terminal open for any "
            "model setup or license prompt. Review the "
            "[XTTS-v2 model terms](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt). "
            "Use recordings you are entitled to use and label shared outputs as synthetic. "
            "Download files you want to keep before closing the app."
        )
        prepare_button.click(
            prepare, inputs=[recordings, start, seconds],
            outputs=[prepared_files, refs, preview_choice, reference_audio, prepare_status],
        )
        preview_choice.change(lambda path: path, inputs=preview_choice, outputs=reference_audio)
        generate_button.click(
            generate, inputs=[refs, text, language, device, seed, chunking, chunk_chars, pause_ms],
            outputs=[generated_audio, download, generation_status], concurrency_limit=1,
        )
        chunking.change(
            lambda enabled: (gr.update(interactive=enabled), gr.update(interactive=enabled)),
            inputs=chunking, outputs=[chunk_chars, pause_ms],
        )
    return app.queue(default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address (default: local only)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="voice-studio-") as work_dir:
            app = create_app(work_dir)
            app.launch(server_name=args.host, server_port=args.port, share=False)
    except ImportError as exc:
        parser.exit(1, f"Missing interface dependency: {exc}. Install requirements-ui.txt first.\n")


if __name__ == "__main__":
    main()
