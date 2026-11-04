import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from deep_introgression import Classifier, FEATURES
from deep_introgression.cli import main
from deep_introgression.vcf import window_features

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = pd.read_csv(EXAMPLES / "toy_features.csv")
        cls.reference = json.loads((ROOT / "tests/reference_scores.json").read_text())
        cls.oceanian = Classifier("oceanian")

    def test_prepared_vcf_matches_original_statistic_functions(self):
        actual = pd.DataFrame(window_features(EXAMPLES / "toy.vcf"))
        self.assertEqual(actual.window.tolist(), ["1_50000", "50001_100000"])
        pd.testing.assert_frame_equal(actual, self.features, check_dtype=False,
                                      atol=1e-8, rtol=1e-10)

    def test_checkpoints_against_original_keras_and_scalers(self):
        for name, expected in self.reference.items():
            population, checkpoint = name.split("/")
            with self.subTest(name=name):
                model = Classifier(population, checkpoint)
                actual = model.predict(self.features)
                np.testing.assert_allclose(actual.predicted_probs, expected, atol=2e-6, rtol=0)
                self.assertEqual("pred" in actual, checkpoint == "empirical")
                if checkpoint == "empirical":
                    threshold = 0.976012647 if population == "oceanian" else 0.973326921
                    self.assertEqual(model.threshold, threshold)
                    np.testing.assert_array_equal(actual.pred, np.asarray(expected) > threshold)

    def test_feature_order_and_missing_empirical_columns(self):
        altered = self.features.drop(columns=["human_singletons", "human_high_freq_sites"])
        altered = altered[altered.columns[::-1]]
        pd.testing.assert_frame_equal(self.oceanian.predict(altered), self.oceanian.predict(self.features))
        with self.assertRaisesRegex(ValueError, "human_den_shared_sites"):
            self.oceanian.predict(altered.drop(columns=["human_den_shared_sites"]))
        altered.loc[0, "human_den_shared_sites"] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.oceanian.predict(altered)

    def test_strict_cutoff(self):
        value = float(self.oceanian.predict(self.features).predicted_probs.iloc[0])
        at_boundary = Classifier("oceanian", threshold=value)
        self.assertEqual(at_boundary.predict(self.features).pred.iloc[0], 0)
        with self.assertRaisesRegex(ValueError, "threshold"):
            Classifier("oceanian", threshold=float("nan"))

    def test_cli_from_vcf_and_empty_input(self):
        with tempfile.TemporaryDirectory() as temp:
            features = Path(temp) / "features.csv"
            scores = Path(temp) / "scores.csv"
            main(["features", str(EXAMPLES / "toy.vcf"), str(features)])
            main(["predict", "--population", "east-asian", str(features), str(scores)])
            output = pd.read_csv(scores)
            self.assertEqual(output.window.tolist(), self.features.window.tolist())
            np.testing.assert_allclose(output.predicted_probs, self.reference["east-asian/empirical"],
                                       atol=2e-6, rtol=0)
            # Re-score a table already containing calls with a score-only checkpoint.
            main(["predict", "--population", "oceanian", "--checkpoint", "fine-tuned",
                  str(scores), str(features)])
            self.assertNotIn("pred", pd.read_csv(features))
            self.features.iloc[:0].to_csv(features, index=False)
            main(["predict", "--population", "oceanian", str(features), str(scores)])
            self.assertEqual(len(pd.read_csv(scores)), 0)

    def test_invalid_input_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "bad.csv"
            destination = Path(temp) / "previous.csv"
            destination.write_text("old output")
            with source.open("w", newline="") as f:
                csv.writer(f).writerow([*FEATURES, FEATURES[0]])
            with self.assertRaises(SystemExit):
                main(["predict", "--population", "east-asian", str(source), str(destination)])
            self.assertEqual(destination.read_text(), "old output")

    def test_unphased_and_missing_genotypes(self):
        original = (EXAMPLES/"toy.vcf").read_text().splitlines()
        header = [line for line in original if line.startswith("#")]
        for invalid in ["0/1", ".|1"]:
            with self.subTest(gt=invalid), tempfile.TemporaryDirectory() as temp:
                record = next(line.split("\t") for line in original if not line.startswith("#"))
                record[9] = invalid
                path = Path(temp) / "bad.vcf"
                path.write_text("\n".join([*header, "\t".join(record)]) + "\n")
                with self.assertRaises(ValueError):
                    list(window_features(path))
