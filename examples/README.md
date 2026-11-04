# DEEP workflow examples

These scripts show the simulation, feature-extraction, and training workflow. They use editable settings at the top of each file.

| Stage | Script | Output |
| --- | --- | --- |
| Oceanian simulations | `robustSim.py` | Genotype and metadata NPZ archives |
| East Asian simulations | `TIB_Sim.py` | Genotype and metadata NPZ archives |
| Training statistics | `generate_training_data_with_pair_metadata.py` | Feature CSV for each simulation task |
| Training and evaluation | `train_pairrank_with_diagnostics.py` | Two checkpoints, scaler, feature order, and diagnostics |

## Prepare the environment and paths

Use Python 3.10–3.12. From a clone of DEEP, install the package and the extra simulation and training dependencies.

```sh
python -m pip install .
python -m pip install -r examples/requirements.txt
```

The commands below use a Linux shell. Replace each `/path/to/...` value with your own location. You can also edit the corresponding placeholder at the top of the scripts.

```sh
export DEEP_WORK_DIR=/path/to/your/deep_work
export DEEP_DATA_SUFFIX=oceanian
export DEEP_MUTATION_FIT=/path/to/your/oceanian_mutation_fit.json
export SLURM_ARRAY_TASK_ID=1
```

| Setting | Meaning |
| --- | --- |
| `DEEP_WORK_DIR` | Writable parent directory for `genotypes/`, `features/`, and `trained_models/` |
| `DEEP_DATA_SUFFIX` | Run label shared by all three stages, such as `oceanian` or `east-asian` |
| `DEEP_MUTATION_FIT` | Population-specific fitted mutation-model JSON used by the simulator |
| `SLURM_ARRAY_TASK_ID` | Task number appended to the genotype and feature filenames. The local default is `1` |

Supply your population-specific mutation-fit JSON. Its top-level `mutation_model` object must contain `mu0`, `base_gamma_shape`, `base_gamma_scale`, `p_hot`, `hot_gamma_shape`, and `hot_gamma_scale`. The gamma draws supply multipliers of `mu0`, a per-base, per-generation rate. The two shape and scale pairs must be positive, and `p_hot` gives the probability of using the hotspot component. Both component parameter pairs are read even when `p_hot` is zero.

## 1. Generate simulated windows

Review `TARGET_WITH`, `TARGET_WITHOUT`, and `TARGET_ROBUST` before running. The supplied research settings request 50,000 accepted windows of each label. For a small plumbing check, edit the first two values to `10` and retain `TARGET_ROBUST = 0`. Acceptance depends on realized ancestry, so the number of attempted simulations can be much larger.

```sh
python examples/robustSim.py
```

For the East Asian run, select its suffix and mutation fit and use the other simulator.

```sh
export DEEP_DATA_SUFFIX=east-asian
export DEEP_MUTATION_FIT=/path/to/your/east-asian_mutation_fit.json
python examples/TIB_Sim.py
```

Each simulator draws 50-kb windows, a mutation rate, and demographic parameters. Within a paired attempt, the two simulations share a nuisance-parameter bundle. The positive branch enables ghost-to-Denisovan admixture and requires a carrier haplotype among the retained humans. Acceptance occurs independently for the two labels. Complete retained pairs share a `bundle_id` and the roles `positive_paired` and `negative_paired`.

The entry points simulate 20 human diploids, remove minor-allele singletons across those human haplotypes, and select seven intact diploids plus one Denisovan. Both supplied base profiles select the `severe` bottleneck setting. The explicit demographic events in each `simulate_one_instance` function specify its timing and population sizes. Setting `TARGET_ROBUST` above zero adds examples from the other background profiles.

For suffix `oceanian` and task `1`, the simulator writes the following files under `DEEP_WORK_DIR/genotypes/`.

```text
with_ghost_genotypesoceanian1.npz
without_ghost_genotypesoceanian1.npz
with_ghost_metadataoceanian1.npz
without_ghost_metadataoceanian1.npz
```

Genotype archives contain arrays named `arr_0`, `arr_1`, and so on. Each array has one row per variant and 17 columns. Column 0 holds the position within the simulated window, columns 1–14 hold the human haplotypes in adjacent diploid pairs, and columns 15–16 hold the Denisovan alleles. Each metadata archive contains a same-length array of dictionaries under the key `metadata`. Its order matches the corresponding genotype archive.

## 2. Extract statistics and preserve pairing metadata

Use the same work directory, suffix, and task ID as the simulation.

```sh
python examples/generate_training_data_with_pair_metadata.py
```

The script folds alleles jointly across the humans and Denisovan, keeps human-polymorphic sites, and computes diversity, pairwise differences, within-individual heterozygosity differences, and Denisovan-sharing statistics. It appends labels and simulation metadata. Windows with no retained human polymorphism are skipped.

The output is `DEEP_WORK_DIR/features/genotype_training_data<SUFFIX><TASK_ID>.csv`. The metadata fields `human_singletons` and `human_high_freq_sites` come from the simulator's retained-panel counts. The supplied training recipe numbers the largest pairwise and within-individual differences in ascending order within that group of largest values. These conventions are recorded in the source comments.

For array jobs, run simulation and extraction once per task ID. Keep each run's suffix consistent across both stages.

## 3. Train a model

Select the suffix to train and review `SHARD_IDS` in the training script. It defaults to tasks 1–25 and merges the available CSVs. For a single local task, use `SHARD_IDS = [1]`. For a small execution check, lower `PRETRAIN_EPOCHS` and `PAIR_FINETUNE_EPOCHS` to `1`. The supplied caps are 1,000 and 150 epochs, with early stopping.

```sh
python examples/train_pairrank_with_diagnostics.py
```

The training stage performs the following operations.

1. Merge feature shards and prefix each bundle ID with its task ID. Simulators restart bundle numbering for every task, so this prefix keeps independent draws distinct.
2. Split bundles approximately 80/10/10 into training, validation, and test sets.
3. Select numeric `human_` and `combined_` inputs, apply the explicit feature exclusions, and fit `StandardScaler` on the training split.
4. Train a 128/64/32-unit network with ReLU, dropout, L2 regularization, and sigmoid output using binary cross-entropy.
5. Select the pretraining checkpoint by validation AUC, then fine-tune using a weighted pair-ranking loss, binary cross-entropy, and regularization. Only retained pairs with both roles enter pair-ranking.
6. Select fine-tuned weights by validation pair accuracy, with mean score difference as the tie-breaker. Save scores, threshold evaluations, and pair diagnostics.

Use enough independent bundles and retained pairs to populate all three splits. Very small simulation trials are useful for checking files and statistics. Model fitting and low-FPR evaluation require substantially larger datasets. The stochastic simulation and training produce fresh weights on each run.

### Training outputs

Files are written under `DEEP_WORK_DIR/trained_models/`.

| File pattern | Contents |
| --- | --- |
| `<SUFFIX>-empirical.keras` | Selected binary-cross-entropy pretraining checkpoint |
| `<SUFFIX>-fine-tuned.keras` | Selected pair-ranking checkpoint |
| `scaler<SUFFIX>.pkl` | Fitted training scaler |
| `feature_columns<SUFFIX>.json` | Exact ordered model inputs |
| `X_*` and `y_*` NumPy arrays | Scaled train/validation/test inputs and labels |
| `row_predictions<SUFFIX>.csv` | Final-checkpoint scores, split membership, statistics, and metadata |
| `val_pair_diagnostics<SUFFIX>.csv`, `test_pair_diagnostics<SUFFIX>.csv` | Retained-pair score and feature differences |
| `history<SUFFIX>.npy` | Pretraining and fine-tuning histories |
| `fixed_fpr_summary<SUFFIX>.npy` | Thresholds, confusion matrices, rates, and correlations |

Pair-diagnostic CSVs are written for splits containing complete retained pairs. The history and summary files hold dictionaries and can be read with `np.load(path, allow_pickle=True).item()`.

The original training threshold routine uses `score >= threshold`. Its historically named `allow_one_fp` mode chooses the second-largest negative score, which can include two or more negatives at that boundary. Inspect the recorded confusion-matrix counts when interpreting each achieved false-positive rate. Thresholds in these training diagnostics apply to the final checkpoint used to produce the scores.

## 4. Process empirical data and run the packaged classifiers

The prepared-VCF workflow uses the installed `deep features` command. The input panel requires cohort selection, the appropriate larger-panel singleton ascertainment, genomic masking, and complete phased human genotypes before feature extraction. See the repository guide for the input specification. [Prepared VCF workflow](../README.md#start-from-a-prepared-vcf)

From the repository root, the bundled synthetic example can be run as follows.

```sh
python -m pip install ".[vcf]"
deep features examples/toy.vcf features.csv
deep predict --population oceanian features.csv oceanian_scores.csv
deep predict --population east-asian features.csv east_asian_scores.csv
```

The `deep predict` population options load the released checkpoints and matching preprocessing assets. Newly trained checkpoints have their own scaler, feature order, and validation thresholds. Preserve those artifacts together when evaluating a new training run.
