import ast
from pathlib import Path
import tempfile
import unittest

import pandas as pd

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ExampleMergeTests(unittest.TestCase):
    def merge_function(self, directory):
        # Exercise the file merger without importing the optional training libraries.
        source = ast.parse((EXAMPLES / "train_pairrank_with_diagnostics.py").read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                        and n.name == "merge_csv_files")
        namespace = {"pd": pd, "FEATURE_DIR": directory, "SHARD_IDS": [1, 2]}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<example merger>", "exec"), namespace)
        return namespace["merge_csv_files"]

    def test_bundles_from_different_shards_remain_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            shard = pd.DataFrame({"bundle_id": [1, 1, 2, 2], "label": [1, 0, 1, 0],
                                  "pair_role": ["positive_paired", "negative_paired"] * 2})
            for task in (1, 2):
                shard.to_csv(directory / f"genotype_training_dataoceanian{task}.csv", index=False)
            merge = self.merge_function(directory)
            merge("oceanian")
            path = directory / "genotype_training_dataoceanian.csv"
            first = pd.read_csv(path)
            self.assertEqual(first.bundle_id.nunique(), 4)
            self.assertTrue((first.groupby("bundle_id").size() == 2).all())
            self.assertTrue((first.groupby("bundle_id").label.sum() == 1).all())
            merge("oceanian")
            pd.testing.assert_frame_equal(first, pd.read_csv(path))

    def test_missing_shards_raise(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                self.merge_function(Path(temp))("oceanian")
