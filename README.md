# DEEP

**Deep ancestry Estimation through Efficient Proxies** scores 50-kb genomic windows for evidence of Denisovan-mediated superarchaic introgression. Select the Oceanian model for the Oceanian analysis or the East Asian model for the Tibetan and Han analyses. A positive call identifies a candidate window for follow-up. [McAllister, Zöllner & Zhang (2026)](https://doi.org/10.64898/2026.06.25.734355)

## Run the example

From the repository root, install into a Python 3.10–3.12 environment. The model files, input schema and fitted scalers are included.

```sh
python -m pip install .
deep predict --population oceanian examples/toy_features.csv scores.csv
```

Open `scores.csv` to see one row per window. The input columns are preserved. `predicted_probs` is the classifier's sigmoid score and `pred` is 1 when the score strictly exceeds the population cutoff. Each score describes the whole window. It does not estimate an individual's fraction of superarchaic ancestry.

## Start from a prepared VCF

On Linux, install the VCF reader and try the included synthetic panel. It contains seven artificial diploid humans and one artificial Denisovan. Two windows have segregating human sites and produce output. A third contains only Denisovan variation and is skipped.

```sh
python -m pip install ".[vcf]"
deep features examples/toy.vcf features.csv
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
| `oceanian` | `pretrain_test.keras` | 0.976012647 | `test.keras` |
| `east-asian` | `pretrain_tibetan_test.keras` | 0.973326921 | `tibetan_test.keras` |

The empirical defaults follow the prediction notebook and reproduce the saved cohort scores. The fine-tuned models produce a different score distribution. Select one with `--checkpoint fine-tuned` to get scores only. Supply `--threshold VALUE` to generate calls at an explicitly chosen cutoff for either checkpoint.

```sh
deep predict --population east-asian --checkpoint fine-tuned examples/toy_features.csv fine_tuned.csv
```

## Python

```python
import pandas as pd
from deep_introgression import Classifier

windows = pd.read_csv("examples/toy_features.csv")
model = Classifier("oceanian")
scores = model.predict(windows)
print(windows[["chromosome", "window"]].join(scores))
```

## Reproducibility

The four Keras archives retain their original weights. The numeric scaler files preserve the fitted scikit-learn means and scales without requiring a pickle at runtime. The three cohort scripts shared the same statistic functions. The packaged extractor uses those functions with the original 50-kb windows, 10-kb subwindows and 5-kb occupancy bins.

Run `python -m unittest discover -s tests -v` after installing `.[vcf]`. The tests cover the synthetic VCF, independently calculated features, all four checkpoints, and malformed inputs. [Model fingerprints and validation](VALIDATION.md)

## License

The MIT License covers the DEEP code and bundled model weights, including all four Keras checkpoints. [License text](LICENSE)
