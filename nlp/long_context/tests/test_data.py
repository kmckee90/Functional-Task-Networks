from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from long_context_bench.config import artifact_path
from long_context_bench.data import build_locomo_records, build_niah_records, prepare_datasets


class DataTests(unittest.TestCase):
    def test_niah_generation_is_deterministic(self):
        spec = {
            "type": "niah",
            "name": "niah_test",
            "train_size": 3,
            "eval_size": 2,
            "context_lengths": [64],
            "needle_depths": [0.25, 0.75],
        }
        a = list(build_niah_records(spec, "train", 123))
        b = list(build_niah_records(spec, "train", 123))
        self.assertEqual([x.to_json() for x in a], [x.to_json() for x in b])
        self.assertEqual(len(a), 3)
        self.assertIn(a[0].answers[0], a[0].prompt)
        self.assertIn(a[0].answers[0], a[0].target)

    def test_prepare_writes_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = {
                "run": {"seed": 1},
                "data": {
                    "processed_dir": str(Path(td) / "processed"),
                    "datasets": [
                        {
                            "type": "niah",
                            "name": "tiny",
                            "train_size": 2,
                            "eval_size": 1,
                            "context_lengths": [32],
                        }
                    ],
                },
            }
            counts = prepare_datasets(cfg)
            self.assertEqual(counts, {"train": 2, "eval": 1})
            self.assertTrue((Path(td) / "processed" / "train.jsonl").exists())

    def test_artifact_root_resolves_only_relative_paths(self):
        cfg = {"run": {"artifact_root": "/tmp/lc_artifacts"}}
        self.assertEqual(
            artifact_path(cfg, "runs/example"),
            Path("/tmp/lc_artifacts/runs/example"),
        )
        self.assertEqual(
            artifact_path(cfg, "/home/dev/datasets/locomo/data/locomo10.json"),
            Path("/home/dev/datasets/locomo/data/locomo10.json"),
        )

    def test_locomo_local_jsonl_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "locomo.jsonl"
            row = {
                "messages": [
                    {"speaker": "A", "text": "I moved to Seattle in May."},
                    {"speaker": "B", "text": "Noted."},
                ],
                "qa": [{"question": "Where did A move?", "answer": "Seattle"}],
            }
            path.write_text(json.dumps(row) + "\n")
            spec = {
                "type": "locomo",
                "name": "locomo_test",
                "path": str(path),
                "train_size": 10,
            }
            records = list(build_locomo_records(spec, "train", 5))
            self.assertEqual(len(records), 1)
            self.assertIn("Seattle", records[0].prompt)
            self.assertIn("Seattle", records[0].target)


if __name__ == "__main__":
    unittest.main()
