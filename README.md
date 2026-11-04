# DEEP

**Deep ancestry Estimation through Efficient Proxies** scores 50-kb genomic windows for evidence of Denisovan-mediated superarchaic introgression. Select the Oceanian model for the Oceanian analysis or the East Asian model for the Tibetan and Han analyses. A positive call identifies a candidate window for follow-up. [McAllister, Zöllner & Zhang (2026)](https://doi.org/10.64898/2026.06.25.734355)

## Install from GitHub and run both models

Use a Python 3.10–3.12 environment with Git installed. The package includes all four trained models, the input schema and the fitted scalers. Run the following in a working directory of your choice.

```sh
python -m pip install "git+https://github.com/noelmcp/DEEP.git"
curl -fL https://raw.githubusercontent.com/noelmcp/DEEP/main/examples/toy_features.csv -o toy_features.csv
deep predict --population oceanian toy_features.csv oceanian_scores.csv
deep predict --population east-asian toy_features.csv east_asian_scores.csv
```

Each output CSV preserves the input columns and adds the classifier's sigmoid score (`predicted_probs`) and its binary call (`pred`). A call is 1 when the score strictly exceeds the population cutoff. The synthetic example gives these scores rounded to six decimal places.

| Chromosome 1 window | Oceanian score | East Asian score |
| --- | ---: | ---: |
| `1_50000` | 0.895511 | 0.882294 |
| `50001_100000` | 0.972854 | 0.944584 |

Both models return `pred = 0` for both windows. If you already cloned the repository, you can install with `python -m pip install .` and use `examples/toy_features.csv` as the input.

## Start from a prepared VCF

On Linux, install the VCF reader and try the included synthetic panel. It contains seven artificial diploid humans and one artificial Denisovan. Two windows have segregating human sites and produce output. A third contains only Denisovan variation and is skipped.

```sh
python -m pip install "deep-introgression[vcf] @ git+https://github.com/noelmcp/DEEP.git"
curl -fL https://raw.githubusercontent.com/noelmcp/DEEP/main/examples/toy.vcf -o toy.vcf
deep features toy.vcf features.csv
deep predict --population east-asian features.csv scores.csv
```

Supply a sorted GRCh37/hg19 autosomal VCF or VCF.gz containing only the target humans and one Denisovan. For multiple autosomes, order them numerically. Use biallelic SNPs, complete diploid genotypes and phased human heterozygotes. The Denisovan can be unphased. Its sample ID defaults to `Denisova` and can be changed with `--denisovan ID`. The empirical panels used seven target diploids (14 haplotypes). Scores on other panel sizes require independent calibration.

Start after cohort selection, singleton filtering and genomic masking. To match a study panel, use its larger ascertainment panel for singleton filtering before subsetting to the final individuals, and apply the same accessibility and site masks. These upstream choices affect the scores. The extractor folds allele counts across humans and the Denisovan, retains sites polymorphic in humans, and computes the original haplotype, diploid and Denisovan-sharing statistics. The `window` field uses 1-based, inclusive coordinates, with the last window clipped to the GRCh37 chromosome length.

## Start from an existing feature CSV

Provide one row per window with a header and unscaled, numeric DEEP features. Both populations use the same 80 feature names. The package orders the columns and applies the selected population's training means and scales. Metadata such as `chromosome` and `window` passes through. Previously calculated `predicted_probs` and `pred` columns are replaced by the new result.

Two inputs, `human_singletons` and `human_high_freq_sites`, were absent in the empirical extractor output. The scoring notebook inserted zeros before scaling. The package does the same when these columns are absent and honors supplied values when present. It requires all other model inputs and reports non-finite values. The synthetic feature CSV shows the complete schema. [Ordered features](src/deep_introgression/models/features.json)

## Choose a checkpoint

| Population option | Empirical checkpoint (default) | Empirical cutoff | Fine-tuned checkpoint |
| --- | --- | ---: | --- |
| `oceanian` | `oceanian-empirical.keras` | 0.976012647 | `oceanian-fine-tuned.keras` |
| `east-asian` | `east-asian-empirical.keras` | 0.973326921 | `east-asian-fine-tuned.keras` |

The empirical defaults follow the prediction notebook and reproduce the saved cohort scores. The fine-tuned models produce a different score distribution. Select one with `--checkpoint fine-tuned` to get scores only. Supply `--threshold VALUE` to generate calls at an explicitly chosen cutoff for either checkpoint.

```sh
deep predict --population east-asian --checkpoint fine-tuned toy_features.csv fine_tuned.csv
```

## Python

```python
import pandas as pd
from deep_introgression import Classifier

windows = pd.read_csv("toy_features.csv")
model = Classifier("oceanian")
scores = model.predict(windows)
print(windows[["chromosome", "window"]].join(scores))
```

## Simulation and training examples

The examples directory includes the Oceanian and East Asian simulation loops, metadata-aware training statistics, and the two-stage model-training script. Its guide describes the required mutation-fit inputs, editable paths, array jobs, preprocessing, and generated artifacts. [Full workflow examples](examples/README.md)

## Reproducibility

The four Keras archives retain their original weights. The numeric scaler files preserve the fitted scikit-learn means and scales without requiring a pickle at runtime. The three cohort scripts shared the same statistic functions. The packaged extractor uses those functions with the original 50-kb windows, 10-kb subwindows and 5-kb occupancy bins.

Run `python -m unittest discover -s tests -v` after installing `.[vcf]`. The tests cover the synthetic VCF, independently calculated features, all four checkpoints, and malformed inputs. [Model fingerprints and validation](VALIDATION.md)

## License

The MIT License covers the DEEP code and bundled model weights, including all four Keras checkpoints. [License text](LICENSE)
