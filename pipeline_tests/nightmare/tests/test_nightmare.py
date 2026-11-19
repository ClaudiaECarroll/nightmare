import contextlib
import importlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "code/evaluation"))

core = importlib.import_module("core")
booknlp_loader = importlib.import_module("loaders.booknlp")


class EvaluationTests(unittest.TestCase):
    def test_saved_predictions_reproduce_all_saved_results(self):
        gold = core.load_gold_strings(
            REPO / "benchmark/final_benchmark.tsv", REPO / "benchmark/manual_samples"
        )
        predictions = booknlp_loader.load_predictions(PROJECT / "data/intermediate/BookNLP_output")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            core.run_evaluation(predictions, gold, tmp)
            expected = PROJECT / "data/results/booknlp"
            actual = Path(tmp)
            self.assertEqual(
                (actual / "evaluation_results.txt").read_bytes(),
                (expected / "evaluation_results.txt").read_bytes(),
            )
            for path in sorted((expected / "per_book").glob("*.json")):
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8")),
                    json.loads((actual / "per_book" / path.name).read_text(encoding="utf-8")),
                )

    def test_header_only_predictions_and_missing_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            book = root / "example"
            book.mkdir()
            (book / "example.quotes").write_text("quote\n", encoding="utf-8")
            predictions = booknlp_loader.load_predictions(root)
            self.assertEqual(predictions, {"example": []})
            with self.assertRaisesRegex(ValueError, "missing predictions"):
                core.run_evaluation(predictions, {"example": [], "missing": []}, root / "results")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                core.run_evaluation(predictions, {"other": []}, root / "results")

    def test_ambiguous_prediction_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp) / "example"
            book.mkdir()
            for name in ("example.quotes", "example.txt.quotes"):
                (book / name).write_text("quote\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                booknlp_loader.load_predictions(tmp)


class RunnerTests(unittest.TestCase):
    def test_runner_uses_custom_models_and_fresh_per_book_output(self):
        spec = importlib.util.spec_from_file_location(
            "nightmare_runner", PROJECT / "code/classification/BookNLP/run_nightmare.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        calls = []
        torch = types.ModuleType("torch")
        original_load = lambda *args, **kwargs: {
            "bert.embeddings.position_ids": 1, "other": 2
        }
        torch.load = original_load
        booknlp = types.ModuleType("booknlp")
        booknlp_module = types.ModuleType("booknlp.booknlp")

        class FakeBookNLP:
            def __init__(self, language, params):
                calls.append((language, params))
                self.assert_patched = torch.load("weights")

            def process(self, input_file, output_dir, book_id):
                calls.append((input_file, output_dir, book_id, self.assert_patched))
                (Path(output_dir) / (book_id + ".quotes")).write_text(
                    "quote\n", encoding="utf-8"
                )

        booknlp_module.BookNLP = FakeBookNLP
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples, models, bert, output = (root / name for name in
                                              ("samples", "models", "bert", "output"))
            for directory in (samples, models, bert):
                directory.mkdir()
            (samples / "123_manual_sample.txt").write_text("text", encoding="utf-8")
            for name in (
                "entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model",
                "coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model",
                "speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model",
            ):
                (models / name).touch()
            args = ["run_nightmare", "--input-dir", str(samples), "--model-dir",
                    str(models), "--bert-model-path", str(bert), "--output-dir", str(output)]
            modules = {"torch": torch, "booknlp": booknlp, "booknlp.booknlp": booknlp_module}
            with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "argv", args):
                runner.main()
                self.assertIs(torch.load, original_load)
                self.assertEqual(calls[0][1]["pipeline"], "entity,quote")
                self.assertEqual(calls[0][1]["model"], "custom")
                self.assertEqual(calls[1][2:], ("123_manual_sample", {"other": 2}))
                self.assertEqual(booknlp_loader.load_predictions(output), {"123_manual_sample": []})
                with self.assertRaisesRegex(FileExistsError, "fresh --output-dir"):
                    runner.main()


if __name__ == "__main__":
    unittest.main()
