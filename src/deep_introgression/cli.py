"""Command-line entry points for feature extraction and scoring."""

import argparse
import csv
import os
from pathlib import Path
import tempfile

import pandas as pd

from .predict import CHECKPOINTS, FEATURES, POPULATIONS, Classifier


def _predict(args, dest):
    with args.input.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
        if len(header) != len(set(header)):
            raise ValueError("Feature table contains duplicate column names.")
        if not header:
            raise ValueError("Feature table needs a CSV header.")
        source.seek(0)
        classifier = Classifier(args.population, args.checkpoint, args.threshold)
        print_header = True
        for chunk in pd.read_csv(source, chunksize=8192):
            predictions = classifier.predict(chunk)
            keep = chunk.drop(columns=["predicted_probs", "pred"], errors="ignore")
            pd.concat([keep, predictions], axis=1).to_csv(dest, index=False, header=print_header)
            print_header = False


def _features(args, dest):
    from .vcf import window_features

    writer = csv.DictWriter(dest, fieldnames=["chromosome", "window", *FEATURES])
    writer.writeheader()
    writer.writerows(window_features(args.input, args.denisovan))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="deep", description="Score 50-kb windows with DEEP.")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Score a DEEP feature CSV")
    predict.add_argument("--population", choices=POPULATIONS, required=True)
    predict.add_argument("--checkpoint", choices=CHECKPOINTS, default="empirical")
    predict.add_argument("--threshold", type=float, help="Override the decision threshold")
    features = commands.add_parser("features", help="Extract model features from a prepared VCF")
    features.add_argument("--denisovan", default="Denisova", help="Denisovan sample ID")
    for sub in (predict, features):
        sub.add_argument("input", type=Path)
        sub.add_argument("output", type=Path, help="Destination CSV")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose separate input and output files.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", newline="", delete=False,
                                         dir=args.output.parent) as dest:
            temporary = Path(dest.name)
            if args.command == "features":
                _features(args, dest)
            else:
                _predict(args, dest)
        os.replace(temporary, args.output)
    except (OSError, ImportError, ValueError) as exc:
        parser.exit(2, f"deep: {exc}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
