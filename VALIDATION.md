# Validation against the archived DEEP analysis

The installed wheel was compared with the saved empirical tables on November 4, 2026. Each table supplied the raw feature values, archived sigmoid scores and binary calls. The package loaded its own model files and numeric scalers. Every call agreed at the population-specific cutoff.

The archived-table comparison used TensorFlow 2.19 on Linux. A fresh Python 3.10 environment independently installed TensorFlow 2.20, Keras 3.12, NumPy 2.1, pandas 2.3, SciPy 1.15 and cyvcf2 0.32. All seven regression tests and the command-line VCF-to-score example passed in that environment.

| Cohort | Windows scored | Candidate windows | Different calls | Maximum absolute score difference |
| --- | ---: | ---: | ---: | ---: |
| Oceanian | 51,726 | 130 | 0 | 2.02 × 10⁻⁷ |
| Tibetan | 52,771 | 323 | 0 | 2.07 × 10⁻⁷ |
| Han | 52,704 | 231 | 0 | 2.08 × 10⁻⁷ |

For each cohort, the first 20 scorable windows in its filtered chromosome 22 VCF were processed independently. All 80 input values per window agreed with the corresponding saved feature row within absolute tolerance 10⁻⁷ and relative tolerance 10⁻⁸. This checks extraction from the prepared panels. Cohort selection, VCF merging, masks and larger-panel filtering precede this comparison.

Both the training notebook and prediction notebook save or load more than one checkpoint. The empirical prediction cell selected the `pretrain_` files. The population feature lists were byte-identical. Each population has its own fitted `StandardScaler`. The alternative pair was saved after pairwise fine-tuning. The empirical cutoffs in the README apply to the empirical checkpoints.

## Recheck a saved table

```python
import numpy as np
import pandas as pd
from deep_introgression import Classifier

reference = pd.read_csv("df_empirical.csv")
result = Classifier("oceanian").predict(reference)
np.testing.assert_allclose(result.predicted_probs, reference.predicted_probs, atol=2e-6, rtol=0)
np.testing.assert_array_equal(result.pred, reference.pred)
```

For `tibetan_df_empirical.csv` and `han_df_empirical.csv`, select `east-asian`. The empirical tables stay with the original analysis. Repository tests instead use an artificial panel whose expected features were calculated by the original Oceanian extraction functions. The four reference score vectors were calculated using the original Keras files and scikit-learn scaler objects independently of the packaged classifier.

## Packaged models and original artifacts

The packaged model names identify the population and checkpoint. Each SHA-256 below applies to both names in its row.

| Packaged file | Original file | SHA-256 |
| --- | --- | --- |
| `oceanian-empirical.keras` | `pretrain_test.keras` | `2facd7dd91f031745940ce2efa22327a647f3e0c02e882d756ec4026ab95b2fb` |
| `east-asian-empirical.keras` | `pretrain_tibetan_test.keras` | `b6a8edab970d321391d2c39436c1c9f3f387f177c14827a9c75d4d1e9c553f06` |
| `oceanian-fine-tuned.keras` | `test.keras` | `32a463b564b7c51e432741b3faa81d4202f1246bb93f735e6fe47849f8e3fdef` |
| `east-asian-fine-tuned.keras` | `tibetan_test.keras` | `d0f5675890b8c04977610561e0fff647e304a96007ff3edfc14bca3dcccb97d5` |

| Original preprocessing file | SHA-256 |
| --- | --- |
| `feature_columnstest.json`, `feature_columnstibetan_test.json` | `bcf11fa881be6a2f209b72e6408991341365d408e06f2fef5d1f39b503462200` |
| `scalertest.pkl` | `1207bfdfe9281f2b24f50eb2e1dcee62c99ae925efa13316444c3f8a888551db` |
| `scalertibetan_test.pkl` | `530a3505df4279ab70d64f710451d671890c2b58111d33ed36501746b843806a` |

The Keras archives and feature list are included directly. The scaler means and scales are serialized as double-precision numbers in `scalers.json`.
