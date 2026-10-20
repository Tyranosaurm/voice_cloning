# Python voice simulation workflow

Generate new speech from text, conditioned on recordings of a target voice.
This uses pretrained XTTS-v2 through the maintained `coqui-tts` package. It
does not train a new model or convert the voice of an existing spoken performance.
No transcripts are required for the reference recordings.

## 1. Install

Use Python 3.11 in a fresh virtual environment. Install FFmpeg separately and
verify `ffmpeg -version`. On Ubuntu/Debian: `sudo apt install ffmpeg`.
On macOS with Homebrew: `brew install ffmpeg`. On Windows, install FFmpeg and
add its bin directory to PATH.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
uv pip install torch torchaudio torchcodec --torch-backend=auto
python -m pip install -r requirements.txt
python -m pip check
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.
The automatic backend command follows the current Coqui installation guide.
If GPU detection fails, choose matching Torch/torchaudio packages using the
official PyTorch installer: https://pytorch.org/get-started/locally/
CPU inference is supported but can be slow. CUDA requires an NVIDIA GPU and a
compatible driver. This script selects CUDA when available, otherwise CPU;
Apple Silicon uses CPU here. FFmpeg is also used by the audio decoding stack.

## 2. Select recordings

Start with one to three clean clips of roughly 6–15 seconds each. This is a
practical starting point, not a guaranteed quality threshold. Use the same
speaker throughout, without music, other speakers, heavy echo, clipping, or
long silence. Match the desired accent, pace, and delivery where possible.
Use recordings you are entitled to use. Label shared outputs as synthetic.

## 3. Prepare samples

```bash
python voice.py prepare --input sample1.wav sample2.m4a --out-dir references --seconds 12
```

This extracts the first 12 seconds from each recording and converts it to
24 kHz mono 16-bit PCM WAV, named `ref_01.wav`, `ref_02.wav`, etc. Inputs may be
any format your FFmpeg supports. Add `--start 5` to start five seconds into
each recording. For different offsets, prepare each input into a separate
directory. Outputs are not overwritten. Conversion does not remove noise,
detect speech, identify speakers, or repair clipping: listen to each result
and select a better segment if needed.

## 4. Generate speech

```bash
python voice.py synthesize --refs references/ref_01.wav references/ref_02.wav --text-file example.txt --language en --output output/demo.wav
```

Or generate a short phrase directly:

```bash
python voice.py synthesize --refs references/ref_01.wav --text "Hello, this is my generated voice." --output output/hello.wav
```

The first synthesis downloads model weights and may prompt for acceptance of
the model terms. Review those terms yourself; the script does not accept them
automatically. Subsequent runs use the cached model. Reference audio is used
by local inference; no hosted speech API is called by this script.

Use `--device cpu` or `--device cuda` to override automatic selection.
Sentence splitting is enabled. For long narration, generate a paragraph at a
time and listen for missing words, repetitions, and changes in delivery.
Very long unpunctuated sentences may exceed model limits; break them up.
An unchanged seed helps comparisons but does not guarantee identical output
across hardware or software versions.

## 5. Evaluate and improve

Listen to the reference and generated clips side by side. Check word accuracy,
voice resemblance, accent, pauses, and audible artifacts. Compare the same
short text with one reference versus several. Better reference selection is
the first adjustment; similarity and emotional delivery are not guaranteed.
After a successful local run, record the resolved environment:

```bash
python -m pip freeze > requirements-lock.txt
```

These requirements specify a package version range, not a fully locked and
end-to-end certified environment. No user audio was supplied when this package
was created. CLI parsing, conversion, and invalid-input handling were checked;
model download, inference, GPU compatibility, and voice quality were not tested.

## Optional fine-tuning

If conditioning alone is insufficient, the official XTTS documentation provides
a GPT encoder fine-tuning demo. That is a separate workflow requiring dataset
preparation, corrected transcripts, training, and evaluation. This package
does not implement fine-tuning.

## Model terms and sources

XTTS-v2 weights use the Coqui Public Model License; the Python library's license
does not replace the model terms. Review the license for your intended use,
particularly before commercial deployment.

- Model card: https://huggingface.co/coqui/XTTS-v2
- Model license: https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt
- Installation: https://coqui-tts.readthedocs.io/en/latest/installation.html
- XTTS inference and fine-tuning: https://coqui-tts.readthedocs.io/en/latest/models/xtts.html

Documentation consulted September 7, 2026.
