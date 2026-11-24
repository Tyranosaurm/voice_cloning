"""Narration workflow checks using real PCM files and a fake model backend."""
import argparse
import contextlib
import io
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import voice


def write_wav(path, samples, *, rate=24000, channels=1, width=2):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        if width == 2:
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        else:
            wav.writeframes(bytes(samples))


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        return (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                wav.getnframes(), wav.readframes(wav.getnframes()))


class SplitTextTests(unittest.TestCase):
    def test_normalizes_whitespace_and_keeps_paragraphs_separate(self):
        self.assertEqual(
            voice.split_text("  First\t paragraph.\nMore text.\r\n\r\n Second one!  ", 100),
            ["First paragraph. More text.", "Second one!"],
        )

    def test_chunk_bound_preserves_unbroken_and_multilingual_text(self):
        for text in (
            "A sentence. Another sentence! Then a much longer ending without punctuation",
            "A " + "x" * 73 + " last word",
            "这是没有空格的长文本。它也应该完整保留！最后一段内容。",
        ):
            with self.subTest(text=text):
                chunks = voice.split_text(text, 17)
                self.assertTrue(chunks)
                self.assertTrue(all(0 < len(chunk) <= 17 for chunk in chunks))
                self.assertEqual(
                    "".join("".join(chunks).split()), "".join(text.split())
                )

    def test_empty_text_and_invalid_chunk_limit(self):
        self.assertEqual(voice.split_text(" \n\t ", 10), [])
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                voice.split_text("hello", limit)


class MergeWavsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_concatenates_frames_and_inserts_only_interchunk_silence(self):
        first, second = self.root / "one.wav", self.root / "two.wav"
        output = self.root / "combined.wav"
        write_wav(first, [100, 200, 300])
        write_wav(second, [-100, -200])

        voice.merge_wavs([first, second], output, 200)

        channels, width, rate, frames, data = read_wav(output)
        self.assertEqual((channels, width, rate, frames), (1, 2, 24000, 4805))
        self.assertEqual(
            data, struct.pack("<3h", 100, 200, 300)
            + b"\0\0" * 4800 + struct.pack("<2h", -100, -200)
        )

    def test_zero_pause_and_single_chunk_add_no_silence(self):
        first, second = self.root / "one.wav", self.root / "two.wav"
        write_wav(first, [1, 2])
        write_wav(second, [3, 4])
        output = self.root / "combined.wav"
        voice.merge_wavs([first, second], output, 0)
        self.assertEqual(read_wav(output)[4], struct.pack("<4h", 1, 2, 3, 4))
        single = self.root / "single.wav"
        voice.merge_wavs([first], single, 200)
        self.assertEqual(read_wav(single)[4], struct.pack("<2h", 1, 2))

    def test_rejects_incompatible_or_empty_chunks(self):
        first, second = self.root / "one.wav", self.root / "two.wav"
        write_wav(first, [1, 2])
        for kwargs, samples in (
            ({"rate": 22050}, [3, 4]),
            ({"channels": 2}, [3, 4]),
            ({"width": 1}, [128, 128]),
            ({}, []),
        ):
            with self.subTest(kwargs=kwargs, samples=samples):
                write_wav(second, samples, **kwargs)
                with self.assertRaises(ValueError):
                    voice.merge_wavs([first, second], self.root / "out.wav", 200)
        with self.assertRaises(ValueError):
            voice.merge_wavs([], self.root / "empty.wav", 200)

    def test_rejects_invalid_first_chunk_without_masking_the_error(self):
        first = self.root / "first.wav"
        for samples, width in (([], 2), ([128], 1)):
            with self.subTest(samples=samples, width=width):
                write_wav(first, samples, width=width)
                with self.assertRaises(ValueError):
                    voice.merge_wavs([first], self.root / "out.wav", 200)

    def test_rejects_truncated_audio_frames(self):
        first = self.root / "first.wav"
        write_wav(first, [1, 2, 3, 4])
        first.write_bytes(first.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "truncated"):
            voice.merge_wavs([first], self.root / "out.wav", 200)


class SynthesizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference.wav"
        write_wav(self.reference, [0] * (24000 * 3))
        self.output = self.root / "generated.wav"

        self.tts = mock.Mock()
        self.tts.to.return_value = self.tts
        self.tts.languages = ["en"]
        self.constructor = mock.Mock(return_value=self.tts)
        numpy = types.ModuleType("numpy")
        numpy.random = types.SimpleNamespace(seed=mock.Mock())
        torch = types.ModuleType("torch")
        torch.manual_seed = mock.Mock()
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.inference_mode = contextlib.nullcontext
        tts_package = types.ModuleType("TTS")
        tts_api = types.ModuleType("TTS.api")
        tts_api.TTS = self.constructor
        tts_package.api = tts_api
        modules = {"numpy": numpy, "torch": torch, "TTS": tts_package,
                   "TTS.api": tts_api}
        self.dependencies = mock.patch.dict(sys.modules, modules)
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)

        def generate(**kwargs):
            marker = self.tts.tts_to_file.call_count * 100
            write_wav(kwargs["file_path"], [marker] * 4)

        self.tts.tts_to_file.side_effect = generate

    def args(self, **changes):
        values = dict(text="Hello there.", text_file=None, refs=[str(self.reference)],
                      output=str(self.output), language="en", device="cpu", seed=42,
                      chunk_chars=None, pause_ms=None)
        values.update(changes)
        return argparse.Namespace(**values)

    def synthesize(self, **changes):
        with contextlib.redirect_stdout(io.StringIO()):
            voice.synthesize(self.args(**changes))

    def test_existing_short_text_path_remains_one_model_call(self):
        self.synthesize(text="  Hello,  there!  ")

        self.constructor.assert_called_once_with(voice.MODEL)
        self.tts.to.assert_called_once_with("cpu")
        self.tts.tts_to_file.assert_called_once()
        call = self.tts.tts_to_file.call_args.kwargs
        self.assertEqual(call["text"], "Hello,  there!")
        self.assertEqual(call["speaker_wav"], [str(self.reference)])
        self.assertEqual(call["language"], "en")
        self.assertIs(call["split_sentences"], True)
        self.assertEqual(read_wav(self.output)[4], struct.pack("<4h", *([100] * 4)))

    def test_chunked_text_loads_once_and_joins_with_default_pause(self):
        self.synthesize(text="First paragraph.\n\nSecond paragraph.", chunk_chars=32)

        self.constructor.assert_called_once()
        calls = self.tts.tts_to_file.call_args_list
        self.assertEqual([call.kwargs["text"] for call in calls],
                         ["First paragraph.", "Second paragraph."])
        self.assertTrue(all(call.kwargs["split_sentences"] for call in calls))
        self.assertEqual(read_wav(self.output)[4],
                         struct.pack("<4h", *([100] * 4)) + b"\0\0" * 4800
                         + struct.pack("<4h", *([200] * 4)))

    def test_utf8_file_and_explicit_zero_pause(self):
        source = self.root / "script.txt"
        source.write_text("第一段。\n\n第二段。", encoding="utf-8")
        self.synthesize(text=None, text_file=str(source), chunk_chars=10, pause_ms=0)

        self.assertEqual([call.kwargs["text"] for call in self.tts.tts_to_file.call_args_list],
                         ["第一段。", "第二段。"])
        self.assertEqual(read_wav(self.output)[3], 8)

    def test_invalid_chunk_options_fail_before_model_loading(self):
        for options in (
            {"chunk_chars": 0}, {"chunk_chars": -10}, {"pause_ms": 100},
            {"chunk_chars": 32, "pause_ms": -1},
            {"chunk_chars": 32, "pause_ms": float("nan")},
            {"chunk_chars": 32, "pause_ms": float("inf")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.synthesize(**options)
        self.constructor.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved_without_loading_model(self):
        self.output.write_bytes(b"keep this output")
        with self.assertRaises(ValueError):
            self.synthesize(chunk_chars=32)
        self.assertEqual(self.output.read_bytes(), b"keep this output")
        self.constructor.assert_not_called()

    def test_failed_later_chunk_does_not_publish_or_leave_temporary_files(self):
        def generate(**kwargs):
            if self.tts.tts_to_file.call_count == 2:
                raise RuntimeError("simulated inference failure")
            write_wav(kwargs["file_path"], [100] * 4)

        self.tts.tts_to_file.side_effect = generate
        with self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
            self.synthesize(text="First.\n\nSecond.", chunk_chars=20)
        self.assertEqual(set(self.root.iterdir()), {self.reference})

    def test_invalid_later_audio_does_not_publish(self):
        for bad_kind in ("empty", "sample_rate", "truncated"):
            with self.subTest(bad_kind=bad_kind):
                self.tts.tts_to_file.reset_mock()

                def generate(**kwargs):
                    second = self.tts.tts_to_file.call_count == 2
                    samples = [] if second and bad_kind == "empty" else [100] * 4
                    rate = 22050 if second and bad_kind == "sample_rate" else 24000
                    write_wav(kwargs["file_path"], samples, rate=rate)
                    if second and bad_kind == "truncated":
                        path = Path(kwargs["file_path"])
                        path.write_bytes(path.read_bytes()[:-2])

                self.tts.tts_to_file.side_effect = generate
                with self.assertRaises(ValueError):
                    self.synthesize(text="First.\n\nSecond.", chunk_chars=20)
                self.assertEqual(set(self.root.iterdir()), {self.reference})


if __name__ == "__main__":
    unittest.main()
