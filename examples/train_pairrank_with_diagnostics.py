"""Train a DEEP classifier and export paired diagnostics.

Merge feature shards, split by nuisance bundle, fit the scaler on training rows,
pretrain with binary cross-entropy, and fine-tune on retained matched pairs.
Validation-negative scores define the recorded operational thresholds.

Set the work directory, dataset suffix, and shard IDs below. Training writes
fresh checkpoints, preprocessing artifacts, split arrays, and diagnostics.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC


# Set the work directory and dataset suffix used by the feature extractor.
WORK_DIR = Path(os.environ.get("DEEP_WORK_DIR", "/path/to/your/deep_work"))
DATA_SUFF = os.environ.get("DEEP_DATA_SUFFIX", "oceanian")
FEATURE_DIR = WORK_DIR / "features"
# These IDs correspond to SLURM_ARRAY_TASK_ID. A single local run writes shard 1.
SHARD_IDS = range(1, 26)


def merge_csv_files(data_suff):
    """Merge available task CSVs while keeping independent bundle IDs distinct."""
    base_name = f"genotype_training_data{data_suff}"
    data_frames = []
    for file_id in SHARD_IDS:
        file_name = FEATURE_DIR / f"{base_name}{file_id}.csv"
        if file_name.exists():
            print(f"Reading file: {file_name}")
            frame = pd.read_csv(file_name)
            # Simulators restart bundle IDs in each task. Qualify IDs by shard so
            # pairing and grouped splits refer to the original nuisance draw.
            frame["bundle_id"] = f"{file_id}_" + frame["bundle_id"].astype(str)
            data_frames.append(frame)
        else:
            print(f"File not found: {file_name}")
    if not data_frames:
        raise FileNotFoundError(f"No task CSVs for {data_suff!r} in {FEATURE_DIR}.")
    merged_df = pd.concat(data_frames, ignore_index=True)
    output_file = FEATURE_DIR / f"{base_name}.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as: {output_file}")


# ============================================================
# User-set parameters
# ============================================================
FPR_TARGET = 1 / 55000
THRESHOLD_MODE = "allow_one_fp"
MODEL_SELECT_MONITOR = "val_auc"
MODEL_SELECT_MODE = "max" if MODEL_SELECT_MONITOR == "val_auc" else "min"

# Pair-ranking fine-tuning
PAIR_FINETUNE_EPOCHS = 150
PAIR_BATCH_SIZE = 128
PAIR_FINETUNE_LR = 1e-4
PAIR_PATIENCE = 15
PAIR_MARGIN = 0.02
PAIR_LOSS_WEIGHT = 0.5
PAIR_BCE_WEIGHT = 1.0

# Base BCE pretraining
PRETRAIN_EPOCHS = 1000
PRETRAIN_BATCH_SIZE = 32


# ============================================================
# Paths
# ============================================================
SIM_CSV = FEATURE_DIR / f"genotype_training_data{DATA_SUFF}.csv"
MODEL_DIR = WORK_DIR / "trained_models"

BASE_MODEL_PATH = MODEL_DIR / f"{DATA_SUFF}-empirical.keras"
FINAL_MODEL_PATH = MODEL_DIR / f"{DATA_SUFF}-fine-tuned.keras"
SCALER_PATH = MODEL_DIR / f"scaler{DATA_SUFF}.pkl"
FEATURES_PATH = MODEL_DIR / f"feature_columns{DATA_SUFF}.json"
SUMMARY_PATH = MODEL_DIR / f"fixed_fpr_summary{DATA_SUFF}.npy"
VAL_PAIR_DIAG_PATH = MODEL_DIR / f"val_pair_diagnostics{DATA_SUFF}.csv"
TEST_PAIR_DIAG_PATH = MODEL_DIR / f"test_pair_diagnostics{DATA_SUFF}.csv"
ROW_PRED_PATH = MODEL_DIR / f"row_predictions{DATA_SUFF}.csv"
HISTORY_PATH = MODEL_DIR / f"history{DATA_SUFF}.npy"


# ============================================================
# Helper functions
# ============================================================
def metrics_at_threshold(y_true, y_score, threshold):
    """Compute counts and rates using the training script's score >= threshold convention."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    tpr = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "accuracy": float(accuracy),
    }



def threshold_from_negative_tail(y_true, y_score, mode="allow_one_fp"):
    """Choose an order-statistic cutoff from validation-negative scores.

    zero_fp uses the next representable value above the largest score.
    The historical allow_one_fp mode uses the second-largest score. With the >=
    comparison it can count two or more negatives at the cutoff, including ties.
    Use the returned confusion-matrix counts to interpret the achieved FPR."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    neg_scores = np.sort(y_score[y_true == 0])
    if neg_scores.size == 0:
        return np.inf

    if mode == "zero_fp":
        return float(np.nextafter(neg_scores[-1], np.inf))

    if mode == "allow_one_fp":
        if neg_scores.size == 1:
            return float(neg_scores[-1])
        return float(neg_scores[-2])

    raise ValueError(f"Unknown mode: {mode}")



def threshold_for_target_fpr(y_true, y_score, target_fpr):
    """Select the historical order-statistic cutoff for a target validation FPR.

    The >= comparison and tied scores can produce more false positives than the
    integer target. The reported validation and test metrics give the observed rates."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    neg_scores = np.sort(y_score[y_true == 0])
    n_neg = neg_scores.size
    if n_neg == 0:
        return np.inf

    max_fp = int(np.floor(target_fpr * n_neg))
    if max_fp <= 0:
        return float(np.nextafter(neg_scores[-1], np.inf))

    idx = max(0, n_neg - max_fp - 1)
    return float(neg_scores[idx])



def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }



def create_binary_classification_model(input_dim):
    """Build the 128/64/32-unit network with ReLU, dropout, L2 penalties, and sigmoid output."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, kernel_regularizer=l2(1e-4)),
        ReLU(),
        Dropout(0.15),
        Dense(64, kernel_regularizer=l2(1e-4)),
        ReLU(),
        Dropout(0.10),
        Dense(32, kernel_regularizer=l2(1e-4)),
        ReLU(),
        Dropout(0.05),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=[
            "accuracy",
            AUC(name="auc"),
            AUC(curve="PR", name="pr_auc")
        ]
    )
    return model



def split_bundle_ids(bundle_labels, random_state=42):
    """Split nuisance bundles into approximately 80/10/10 percent train/validation/test.

    Stratify by whether a bundle contains a positive when group counts permit."""
    bundle_ids = bundle_labels.index.to_numpy()
    labels = bundle_labels.to_numpy()

    try:
        train_ids, temp_ids, train_y, temp_y = train_test_split(
            bundle_ids, labels,
            test_size=0.2,
            stratify=labels,
            random_state=random_state,
        )
    except ValueError:
        train_ids, temp_ids, train_y, temp_y = train_test_split(
            bundle_ids, labels,
            test_size=0.2,
            random_state=random_state,
        )

    try:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=0.5,
            stratify=temp_y,
            random_state=random_state,
        )
    except ValueError:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=0.5,
            random_state=random_state,
        )

    return set(train_ids), set(val_ids), set(test_ids)



def build_pair_frame(df_split, score_col="score"):
    needed = ["bundle_id", "pair_role", score_col, "human_mean_allele_frequency", "human_num_sites"]
    available = [c for c in needed if c in df_split.columns]
    tmp = df_split[available].copy()

    pos = tmp[tmp["pair_role"] == "positive_paired"].copy()
    neg = tmp[tmp["pair_role"] == "negative_paired"].copy()

    if pos.empty or neg.empty:
        return pd.DataFrame()

    pos = pos.rename(columns={
        score_col: "score_pos",
        "human_mean_allele_frequency": "human_mean_allele_frequency_pos",
        "human_num_sites": "human_num_sites_pos",
    })
    neg = neg.rename(columns={
        score_col: "score_neg",
        "human_mean_allele_frequency": "human_mean_allele_frequency_neg",
        "human_num_sites": "human_num_sites_neg",
    })

    merged = pos.merge(neg, on="bundle_id", how="inner")
    if merged.empty:
        return merged

    merged["delta_score"] = merged["score_pos"] - merged["score_neg"]
    if "human_mean_allele_frequency_pos" in merged.columns and "human_mean_allele_frequency_neg" in merged.columns:
        merged["delta_human_mean_allele_frequency"] = (
            merged["human_mean_allele_frequency_pos"] - merged["human_mean_allele_frequency_neg"]
        )
    if "human_num_sites_pos" in merged.columns and "human_num_sites_neg" in merged.columns:
        merged["delta_human_num_sites"] = merged["human_num_sites_pos"] - merged["human_num_sites_neg"]
    return merged



def extract_pair_arrays(df_split, X_scaled):
    """Select retained positive_paired and negative_paired rows sharing a bundle ID."""
    if "pair_role" not in df_split.columns:
        return None, None, pd.DataFrame()

    work = df_split[["bundle_id", "pair_role"]].copy()
    work["row_pos"] = np.arange(len(work))

    pos = work[work["pair_role"] == "positive_paired"][["bundle_id", "row_pos"]].rename(columns={"row_pos": "pos_row"})
    neg = work[work["pair_role"] == "negative_paired"][["bundle_id", "row_pos"]].rename(columns={"row_pos": "neg_row"})
    pairs = pos.merge(neg, on="bundle_id", how="inner")

    if pairs.empty:
        return None, None, pairs

    X_pos = X_scaled[pairs["pos_row"].to_numpy()]
    X_neg = X_scaled[pairs["neg_row"].to_numpy()]
    return X_pos, X_neg, pairs



def paired_eval(model, X_pos, X_neg):
    if X_pos is None or X_neg is None or len(X_pos) == 0:
        return {"pair_count": 0, "pair_accuracy": np.nan, "mean_delta": np.nan, "median_delta": np.nan}

    pos_score = model.predict(X_pos, verbose=0).ravel()
    neg_score = model.predict(X_neg, verbose=0).ravel()
    delta = pos_score - neg_score

    return {
        "pair_count": int(len(delta)),
        "pair_accuracy": float(np.mean(delta > 0)),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
    }


if __name__ == "__main__":
    merge_csv_files(DATA_SUFF)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # ============================================================
    # Load the dataset
    # ============================================================
    df = pd.read_csv(SIM_CSV)

    required_meta = {"bundle_id", "pair_role"}
    missing_meta = required_meta - set(df.columns)
    if missing_meta:
        raise ValueError(
            "Training CSV is missing pairing metadata columns: "
            f"{sorted(missing_meta)}. Rebuild the CSV with the metadata-aware feature script first."
        )

    # Select numeric human_ and combined_ statistics, then apply the exclusions.
    # Simulation labels, demographic parameters, and bundle IDs remain metadata.
    EXCLUDE_FEATURE_COLS = {
        "human_num_sites",
        "combined_num_sites",
        "human_num_singletons",
        "combined_num_singletons",
        "combined_total_sites",
        "combined_num_individuals",
        "combined_prop_segregating_sites",
        "human_site_frequency_spectrum",
        "combined_site_frequency_spectrum",

    }

    all_candidate_feature_cols = [
        c for c in df.columns
        if (c.startswith("human_") or c.startswith("combined_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    missing_excluded_cols = sorted(EXCLUDE_FEATURE_COLS - set(df.columns))
    excluded_feature_cols = sorted(set(all_candidate_feature_cols) & EXCLUDE_FEATURE_COLS)

    feature_cols = [
        c for c in all_candidate_feature_cols
        if c not in EXCLUDE_FEATURE_COLS
    ]

    if not feature_cols:
        raise ValueError("No numeric human_/combined_ feature columns found after explicit exclusions.")

    print("\nExplicitly excluded feature columns:")
    if excluded_feature_cols:
        for c in excluded_feature_cols:
            print(f"  - {c}")
    else:
        print("  None of the requested exclusions were present among candidate features.")

    if missing_excluded_cols:
        print("\nRequested excluded columns not found in dataframe:")
        for c in missing_excluded_cols:
            print(f"  - {c}")

    print(f"\nCandidate feature count before exclusion: {len(all_candidate_feature_cols)}")
    print(f"Excluded feature count: {len(excluded_feature_cols)}")
    print(f"Retained feature count after exclusion: {len(feature_cols)}")
    # Keep raw index for later joins / diagnostics.
    df = df.reset_index(drop=True).copy()
    df["_row_id"] = np.arange(len(df))

    # Group split by bundle_id to avoid pair leakage across train/val/test.
    bundle_labels = df.groupby("bundle_id")["label"].max().astype(int)
    train_ids, val_ids, test_ids = split_bundle_ids(bundle_labels)

    train_df = df[df["bundle_id"].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df["bundle_id"].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df[df["bundle_id"].isin(test_ids)].copy().reset_index(drop=True)

    X_train_raw = train_df[feature_cols].to_numpy(dtype=float)
    X_val_raw = val_df[feature_cols].to_numpy(dtype=float)
    X_test_raw = test_df[feature_cols].to_numpy(dtype=float)

    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    # Fit scaler on training split only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save the scaled split arrays and labels for reproducible downstream diagnostics.
    np.save(MODEL_DIR / f"X_train{DATA_SUFF}.npy", X_train)
    np.save(MODEL_DIR / f"y_train{DATA_SUFF}.npy", y_train)
    np.save(MODEL_DIR / f"X_val{DATA_SUFF}.npy", X_val)
    np.save(MODEL_DIR / f"y_val{DATA_SUFF}.npy", y_val)
    np.save(MODEL_DIR / f"X_test{DATA_SUFF}.npy", X_test)
    np.save(MODEL_DIR / f"y_test{DATA_SUFF}.npy", y_test)

    print(f"Loaded {len(df)} rows from {SIM_CSV}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Train / Val / Test rows: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Train / Val / Test bundles: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
    if "pair_role" in df.columns:
        print("Pair-role counts:")
        print(df["pair_role"].value_counts(dropna=False).to_string())


    # ============================================================
    # Pretrain with binary cross-entropy on the training split
    # ============================================================
    input_dim = X_train.shape[1]
    model = create_binary_classification_model(input_dim)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=12,
            restore_best_weights=False,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        ),
        ModelCheckpoint(
            filepath=str(BASE_MODEL_PATH),
            monitor=MODEL_SELECT_MONITOR,
            mode=MODEL_SELECT_MODE,
            save_best_only=True,
            verbose=0,
        ),
    ]

    history_pre = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=PRETRAIN_EPOCHS,
        batch_size=PRETRAIN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    best_model = tf.keras.models.load_model(BASE_MODEL_PATH)


    # ============================================================
    # Pairwise fine-tuning on matched training bundles
    # ============================================================
    X_pos_train, X_neg_train, train_pairs = extract_pair_arrays(train_df, X_train)
    X_pos_val, X_neg_val, val_pairs = extract_pair_arrays(val_df, X_val)
    X_pos_test, X_neg_test, test_pairs = extract_pair_arrays(test_df, X_test)

    pair_history = {
        "train_pair_loss": [],
        "train_pair_acc": [],
        "val_pair_acc": [],
        "val_mean_delta": [],
    }

    if X_pos_train is not None and len(X_pos_train) > 0:
        optimizer = tf.keras.optimizers.Adam(learning_rate=PAIR_FINETUNE_LR)
        bce_loss = tf.keras.losses.BinaryCrossentropy()

        best_weights = best_model.get_weights()
        best_val_pair_acc = -np.inf
        best_val_mean_delta = -np.inf
        patience_counter = 0

        n_pairs = len(X_pos_train)
        indices = np.arange(n_pairs)

        for epoch in range(PAIR_FINETUNE_EPOCHS):
            np.random.shuffle(indices)
            epoch_losses = []
            epoch_accs = []

            for start in range(0, n_pairs, PAIR_BATCH_SIZE):
                batch_idx = indices[start:start + PAIR_BATCH_SIZE]
                xb_pos = tf.convert_to_tensor(X_pos_train[batch_idx], dtype=tf.float32)
                xb_neg = tf.convert_to_tensor(X_neg_train[batch_idx], dtype=tf.float32)

                with tf.GradientTape() as tape:
                    s_pos = best_model(xb_pos, training=True)
                    s_neg = best_model(xb_neg, training=True)
                    delta = s_pos - s_neg

                    pair_loss = tf.reduce_mean(tf.nn.softplus(-(delta - PAIR_MARGIN)))
                    class_loss = 0.5 * (
                        bce_loss(tf.ones_like(s_pos), s_pos) + bce_loss(tf.zeros_like(s_neg), s_neg)
                    )
                    reg_loss = tf.add_n(best_model.losses) if best_model.losses else 0.0
                    total_loss = PAIR_LOSS_WEIGHT * pair_loss + PAIR_BCE_WEIGHT * class_loss + reg_loss

                grads = tape.gradient(total_loss, best_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, best_model.trainable_variables))

                epoch_losses.append(float(total_loss.numpy()))
                epoch_accs.append(float(tf.reduce_mean(tf.cast(delta > 0, tf.float32)).numpy()))

            val_pair_metrics = paired_eval(best_model, X_pos_val, X_neg_val)

            pair_history["train_pair_loss"].append(float(np.mean(epoch_losses)))
            pair_history["train_pair_acc"].append(float(np.mean(epoch_accs)))
            pair_history["val_pair_acc"].append(val_pair_metrics["pair_accuracy"])
            pair_history["val_mean_delta"].append(val_pair_metrics["mean_delta"])

            improved = False
            if np.isfinite(val_pair_metrics["pair_accuracy"]):
                if val_pair_metrics["pair_accuracy"] > best_val_pair_acc:
                    improved = True
                elif val_pair_metrics["pair_accuracy"] == best_val_pair_acc and np.isfinite(val_pair_metrics["mean_delta"]):
                    improved = val_pair_metrics["mean_delta"] > best_val_mean_delta

            if improved:
                best_val_pair_acc = val_pair_metrics["pair_accuracy"]
                best_val_mean_delta = val_pair_metrics["mean_delta"]
                best_weights = best_model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"Pair fine-tune epoch {epoch + 1}: "
                f"train_pair_loss={pair_history['train_pair_loss'][-1]:.6f}, "
                f"train_pair_acc={pair_history['train_pair_acc'][-1]:.6f}, "
                f"val_pair_acc={val_pair_metrics['pair_accuracy']}, "
                f"val_mean_delta={val_pair_metrics['mean_delta']}"
            )

            if patience_counter >= PAIR_PATIENCE:
                break

        # Restore the checkpoint maximizing validation pair accuracy, then mean score gap.
        best_model.set_weights(best_weights)
    else:
        print("No matched training pairs found; skipping pairwise fine-tuning.")

    best_model.save(FINAL_MODEL_PATH)


    # ============================================================
    # Score all splits with the final checkpoint
    # ============================================================
    train_pred = best_model.predict(X_train, verbose=0).ravel()
    val_pred = best_model.predict(X_val, verbose=0).ravel()
    test_pred = best_model.predict(X_test, verbose=0).ravel()

    train_df["score"] = train_pred
    val_df["score"] = val_pred
    test_df["score"] = test_pred

    np.save(MODEL_DIR / f"val_pred{DATA_SUFF}.npy", val_pred)
    np.save(MODEL_DIR / f"test_pred{DATA_SUFF}.npy", test_pred)

    row_pred_df = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ], ignore_index=True)
    row_pred_df.to_csv(ROW_PRED_PATH, index=False)


    # ============================================================
    # Calibrate the final checkpoint from validation-negative scores
    # ============================================================
    thr_target_empirical = threshold_for_target_fpr(y_val, val_pred, FPR_TARGET)
    thr_zero_fp = threshold_from_negative_tail(y_val, val_pred, mode="zero_fp")
    thr_allow_one = threshold_from_negative_tail(y_val, val_pred, mode="allow_one_fp")

    if THRESHOLD_MODE == "zero_fp":
        thr_main = thr_zero_fp
    elif THRESHOLD_MODE == "allow_one_fp":
        thr_main = thr_allow_one
    else:
        raise ValueError("THRESHOLD_MODE must be 'zero_fp' or 'allow_one_fp'")

    val_metrics_main = metrics_at_threshold(y_val, val_pred, thr_main)
    test_metrics_main = metrics_at_threshold(y_test, test_pred, thr_main)
    val_metrics_target = metrics_at_threshold(y_val, val_pred, thr_target_empirical)
    test_metrics_target = metrics_at_threshold(y_test, test_pred, thr_target_empirical)
    val_metrics_zero = metrics_at_threshold(y_val, val_pred, thr_zero_fp)
    test_metrics_zero = metrics_at_threshold(y_test, test_pred, thr_zero_fp)
    val_metrics_one = metrics_at_threshold(y_val, val_pred, thr_allow_one)
    test_metrics_one = metrics_at_threshold(y_test, test_pred, thr_allow_one)


    # ============================================================
    # Compare paired score differences with allele-frequency and site-count differences
    # ============================================================
    val_pair_diag = build_pair_frame(val_df, score_col="score")
    test_pair_diag = build_pair_frame(test_df, score_col="score")

    if not val_pair_diag.empty:
        val_pair_diag.to_csv(VAL_PAIR_DIAG_PATH, index=False)
    if not test_pair_diag.empty:
        test_pair_diag.to_csv(TEST_PAIR_DIAG_PATH, index=False)

    val_pair_metrics = paired_eval(best_model, X_pos_val, X_neg_val)
    test_pair_metrics = paired_eval(best_model, X_pos_test, X_neg_test)

    val_corr_maf = safe_corr(
        val_pair_diag.get("delta_human_mean_allele_frequency", []),
        val_pair_diag.get("delta_score", []),
    )
    test_corr_maf = safe_corr(
        test_pair_diag.get("delta_human_mean_allele_frequency", []),
        test_pair_diag.get("delta_score", []),
    )

    val_corr_sites = safe_corr(
        val_pair_diag.get("delta_human_num_sites", []),
        val_pair_diag.get("delta_score", []),
    )
    test_corr_sites = safe_corr(
        test_pair_diag.get("delta_human_num_sites", []),
        test_pair_diag.get("delta_score", []),
    )


    # ============================================================
    # Save histories and summary
    # ============================================================
    history_dict = {
        "pretrain_history": history_pre.history,
        "pair_history": pair_history,
    }
    np.save(HISTORY_PATH, history_dict, allow_pickle=True)

    summary_dict = {
        "data_suff": DATA_SUFF,
        "fpr_target": float(FPR_TARGET),
        "threshold_mode": THRESHOLD_MODE,
        "model_select_monitor": MODEL_SELECT_MONITOR,
        "n_features": int(len(feature_cols)),
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_train_bundles": int(len(train_ids)),
        "n_val_bundles": int(len(val_ids)),
        "n_test_bundles": int(len(test_ids)),
        "feature_columns": feature_cols,
        "threshold_target_empirical": float(thr_target_empirical),
        "threshold_zero_fp": float(thr_zero_fp),
        "threshold_allow_one_fp": float(thr_allow_one),
        "val_metrics_main": val_metrics_main,
        "test_metrics_main": test_metrics_main,
        "val_metrics_target_empirical": val_metrics_target,
        "test_metrics_target_empirical": test_metrics_target,
        "val_metrics_zero_fp": val_metrics_zero,
        "test_metrics_zero_fp": test_metrics_zero,
        "val_metrics_allow_one_fp": val_metrics_one,
        "test_metrics_allow_one_fp": test_metrics_one,
        "val_pair_metrics": val_pair_metrics,
        "test_pair_metrics": test_pair_metrics,
        "val_delta_score_vs_delta_maf": val_corr_maf,
        "test_delta_score_vs_delta_maf": test_corr_maf,
        "val_delta_score_vs_delta_num_sites": val_corr_sites,
        "test_delta_score_vs_delta_num_sites": test_corr_sites,
    }
    np.save(SUMMARY_PATH, summary_dict, allow_pickle=True)


    # ============================================================
    # Final evaluation printout
    # ============================================================
    print("\nMain operational threshold:")
    print(thr_main)

    print("\nValidation metrics at main threshold:")
    for k, v in val_metrics_main.items():
        print(f"{k}: {v}")

    print("\nTest metrics at main threshold:")
    for k, v in test_metrics_main.items():
        print(f"{k}: {v}")

    print("\nValidation paired metrics:")
    for k, v in val_pair_metrics.items():
        print(f"{k}: {v}")

    print("\nTest paired metrics:")
    for k, v in test_pair_metrics.items():
        print(f"{k}: {v}")

    print("\nValidation delta_score ~ delta_mean_AF:")
    for k, v in val_corr_maf.items():
        print(f"{k}: {v}")

    print("\nTest delta_score ~ delta_mean_AF:")
    for k, v in test_corr_maf.items():
        print(f"{k}: {v}")

    print("\nValidation delta_score ~ delta_num_sites:")
    for k, v in val_corr_sites.items():
        print(f"{k}: {v}")

    print("\nTest delta_score ~ delta_num_sites:")
    for k, v in test_corr_sites.items():
        print(f"{k}: {v}")

    print("\nSaved final model to:", FINAL_MODEL_PATH)
    print("Saved scaler to:", SCALER_PATH)
    print("Saved feature columns to:", FEATURES_PATH)
    print("Saved summary to:", SUMMARY_PATH)
    print("Saved row predictions to:", ROW_PRED_PATH)
    if not val_pair_diag.empty:
        print("Saved validation pair diagnostics to:", VAL_PAIR_DIAG_PATH)
    if not test_pair_diag.empty:
        print("Saved test pair diagnostics to:", TEST_PAIR_DIAG_PATH)
