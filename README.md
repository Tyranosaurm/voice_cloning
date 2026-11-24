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

### Optional local web interface

After installing the hardware-appropriate Torch packages above, install the
optional interface dependencies and launch the app:

```bash
python -m pip install -r requirements-ui.txt
python app.py
```

Open `http://127.0.0.1:7860` in your browser. The app listens locally and does
not create a public sharing link by default. Use `--host` and `--port` to
choose a different listening address or port, for example
`python app.py --port 7861`.

Upload your recordings, choose the start offset and clip duration, and prepare
the reference WAVs. Preview the prepared clips before reusing them in the
speech-generation controls. Enter text in the multiline editor, choose a
language, device, and seed, and enable the long-text controls when needed.
The generated WAV can be played in the app and downloaded.
Download files you want to keep before closing the app; its work directory is
temporary. Long-text chunking is enabled by default in the interface and can
be switched off in Generation settings.

Launching the interface does not load or download XTTS. The first generation
loads the model and may download its weights. Review the model terms below;
they are not accepted automatically. The CLI workflow in the following
sections remains available.

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
Sentence splitting is enabled. The default command generates speech in one
model call, as before.
An unchanged seed helps comparisons but does not guarantee identical output
across hardware or software versions.

### Long narration

Add `--chunk-chars` to generate a longer script as smaller chunks and combine
them into one WAV:

```bash
python voice.py synthesize --refs references/ref_01.wav --text-file example.txt --language en --chunk-chars 200 --pause-ms 200 --output output/narration.wav
```

`--chunk-chars` must be a positive integer. Blank lines start new chunks;
within each paragraph, whitespace is normalized and splitting prefers sentence
boundaries, then word boundaries. An unbroken string longer than the limit is
split at the character limit. The model loads once for the run, and progress is
reported for each chunk. Each generated chunk still uses the model's sentence
splitting. A character limit is only a heuristic: model token limits vary by
language, so reduce `--chunk-chars` if a passage still exceeds them.

`--pause-ms` adds silence between every chunk, with no added silence at the
beginning or end. It defaults to 200 milliseconds when chunking is enabled;
use `--pause-ms 0` to disable added pauses. It must be a nonnegative integer
and requires `--chunk-chars`. The final WAV is saved only after all chunks
have been generated and validated successfully. Existing output files are
not overwritten.

Listen through the assembled narration for missing words, repetitions, abrupt
transitions, and changes in delivery. Chunking does not guarantee consistent
voice quality across a long passage.

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

## Tests

Run the standard-library test suite from the project directory:

```bash
python -m unittest discover -s tests -v
```

The tests use generated WAV fixtures and a fake speech backend; they do not
require model downloads or the inference dependencies. They check text
chunking, WAV assembly, and synthesis workflow behavior. They do not evaluate
actual XTTS inference or voice quality.

The interface was also checked with Gradio 6.32.0 in a desktop and mobile
browser layout: uploads, FFmpeg preparation, reference reuse, WAV downloads,
settings, and validation messages. Those checks used a generated test tone,
not speaker recordings or real model inference.

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
