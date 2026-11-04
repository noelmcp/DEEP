"""Score window-level features with the trained DEEP models."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).with_name("models")
FEATURES = tuple(json.loads((ASSETS / "features.json").read_text(encoding="utf8")))
ZERO_IF_ABSENT = {"human_singletons", "human_high_freq_sites"}
POPULATIONS = {"oceanian": ("test", 0.976012647), "east-asian": ("tibetan_test", 0.973326921)}
CHECKPOINTS = ("empirical", "fine-tuned")


class Classifier:
    """Load one model and its training scaler for repeated predictions."""

    def __init__(self, population, checkpoint="empirical", threshold=None):
        if population not in POPULATIONS:
            raise ValueError(f"Unknown population {population!r}. Choose from {tuple(POPULATIONS)}.")
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint {checkpoint!r}. Choose from {CHECKPOINTS}.")
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("The threshold must be a finite number from 0 to 1.")
        run, cutoff = POPULATIONS[population]
        prefix = "pretrain_" if checkpoint == "empirical" else ""
        self.filename = f"{prefix}{run}.keras"
        self.threshold = cutoff if threshold is None and checkpoint == "empirical" else threshold
        scaler = json.loads((ASSETS / "scalers.json").read_text(encoding="utf8"))[population]
        self._mean = np.asarray(scaler["mean"], dtype=np.float64)
        self._scale = np.asarray(scaler["scale"], dtype=np.float64)
        from tensorflow.keras.models import load_model

        self._model = load_model(ASSETS / self.filename, compile=False)

    def predict(self, frame):
        """Return aligned score and optional call columns for a pandas DataFrame."""
        if not frame.columns.is_unique:
            raise ValueError("Feature table contains duplicate column names.")
        missing = [f for f in FEATURES if f not in frame and f not in ZERO_IF_ABSENT]
        if missing:
            raise ValueError("Missing required DEEP features: " + ", ".join(missing))
        features = frame.reindex(columns=FEATURES, fill_value=0)
        try:
            values = features.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("DEEP features must contain numeric values.") from exc
        invalid = ~np.isfinite(values)
        if invalid.any():
            row, column = np.argwhere(invalid)[0]
            raise ValueError(f"Non-finite feature {FEATURES[column]!r} at row {frame.index[row]!r}.")
        scaled = ((values - self._mean) / self._scale).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled feature values exceed the model's numeric range.")
        scores = np.empty(len(frame), dtype=np.float32)
        for start in range(0, len(frame), 1024):
            batch = slice(start, start + 1024)
            scores[batch] = self._model(scaled[batch], training=False).numpy().ravel()
        if not np.isfinite(scores).all():
            raise ValueError("Model produced a non-finite score. Check the feature ranges.")
        result = pd.DataFrame({"predicted_probs": scores}, index=frame.index)
        if self.threshold is not None:
            result["pred"] = (scores > self.threshold).astype(int)
        return result
