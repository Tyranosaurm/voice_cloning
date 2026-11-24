"""UI handlers use the existing backend without requiring a browser or model."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class AppHandlerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work_dir = Path(self.temp.name)

    def test_prepare_reuses_backend_and_separates_operations(self):
        def prepare(args):
            for index in range(len(args.input)):
                (args.out_dir / f"ref_{index + 1:02d}.wav").touch()

        with patch.object(app.voice, "prepare", side_effect=prepare) as backend:
            first = app.prepare_recordings(["one.wav", "two.m4a"], 2, 12, self.work_dir)
            second = app.prepare_recordings(["one.wav"], 0, 6, self.work_dir)
        self.assertEqual([Path(path).name for path in first], ["ref_01.wav", "ref_02.wav"])
        self.assertNotEqual(Path(first[0]).parent, Path(second[0]).parent)
        self.assertEqual(backend.call_args_list[0].args[0].start, 2)
        self.assertEqual(backend.call_args_list[0].args[0].seconds, 12)

    def test_generate_forwards_settings_and_progress(self):
        progress = lambda completed, total: None

        def synthesize(args, progress=None):
            args.output.write_bytes(b"wav")

        with patch.object(app.voice, "synthesize", side_effect=synthesize) as backend:
            path = app.generate_speech(
                ["ref.wav"], "Hello world.", "fr", "cpu", 42, True, 180, 300,
                self.work_dir, progress=progress,
            )
        args = backend.call_args.args[0]
        self.assertEqual(args.refs, ["ref.wav"])
        self.assertEqual((args.language, args.device, args.seed), ("fr", "cpu", 42))
        self.assertEqual((args.chunk_chars, args.pause_ms), (180, 300))
        self.assertIs(backend.call_args.kwargs["progress"], progress)
        self.assertTrue(Path(path).is_file())

    def test_short_mode_ignores_unused_chunk_controls(self):
        with patch.object(app.voice, "synthesize") as backend:
            app.generate_speech(
                ["ref.wav"], "Hello.", "en", "auto", 42, False, None, None,
                self.work_dir,
            )
        args = backend.call_args.args[0]
        self.assertIsNone(args.chunk_chars)
        self.assertIsNone(args.pause_ms)

    def test_validation_fails_before_backend(self):
        invalid = [
            ([], "Hello", 42, 200, 200),
            (["ref.wav"], "  ", 42, 200, 200),
            (["ref.wav"], "Hello", -1, 200, 200),
            (["ref.wav"], "Hello", 2**32, 200, 200),
            (["ref.wav"], "Hello", float("nan"), 200, 200),
            (["ref.wav"], "Hello", 42, 0, 200),
            (["ref.wav"], "Hello", 42, 200, -1),
        ]
        with patch.object(app.voice, "synthesize") as backend:
            for refs, text, seed, chunk_chars, pause_ms in invalid:
                with self.subTest(refs=refs, text=text, seed=seed,
                                  chunk_chars=chunk_chars, pause_ms=pause_ms):
                    with self.assertRaises(ValueError):
                        app.generate_speech(
                            refs, text, "en", "auto", seed, True, chunk_chars, pause_ms,
                            self.work_dir,
                        )
            backend.assert_not_called()
        self.assertEqual(list(self.work_dir.iterdir()), [])

    def test_failed_operations_remove_their_work_files(self):
        with patch.object(app.voice, "prepare", side_effect=ValueError("bad recording")):
            with self.assertRaisesRegex(ValueError, "bad recording"):
                app.prepare_recordings(["bad.wav"], 0, 12, self.work_dir)
        with patch.object(app.voice, "synthesize", side_effect=ValueError("bad reference")):
            with self.assertRaisesRegex(ValueError, "bad reference"):
                app.generate_speech(
                    ["bad.wav"], "Hello", "en", "cpu", 42, True, 200, 200,
                    self.work_dir,
                )
        self.assertEqual(list(self.work_dir.iterdir()), [])


    def test_cleared_or_nonfinite_timing_fields_show_validation_error(self):
        with patch.object(app.voice, "prepare") as backend:
            for start, seconds in [(None, 12), (0, None), (float("nan"), 12),
                                   (0, float("inf"))]:
                with self.subTest(start=start, seconds=seconds):
                    with self.assertRaisesRegex(ValueError, "finite numbers"):
                        app.prepare_recordings(["sample.wav"], start, seconds, self.work_dir)
            backend.assert_not_called()
        self.assertEqual(list(self.work_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
