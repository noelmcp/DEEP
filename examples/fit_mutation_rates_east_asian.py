#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit East Asian (Tibetan) mutation-rate heterogeneity from empirical site counts.

For each candidate, draw a mutation rate for each window, simulate ancestry and
mutations with msprime, apply optional site retention, remove singletons in the
oversampled human panel, and retain whole diploid pairs for the final panel.
Count sites polymorphic in the retained humans and compare the simulated count
distribution with the empirical distribution using quantiles and summary losses.
Ghost introgression is absent when --low_ghost and --high_ghost are both zero.

Inputs and settings
-------------------
Install the dependencies from examples/requirements.txt after installing DEEP.
Supply an empirical feature CSV with at least 25 usable windows and an integer
count column, such as human_num_sites or combined_num_sites from the empirical
preparation notebook. Counts must be unscaled. Non-finite values and counts below
--min_sites are excluded from the empirical target. A window is 50 kb by default.

Match the empirical cohort's ascertainment panel, final sample size, filtering,
and demographic assumptions when choosing fitting settings. Both fitters default
to 50 oversampled human diploids and seven retained diploids, plus one Denisovan;
the packaged training-simulation examples use 20 when run as scripts.
The East Asian command uses minor-allele singleton removal. Its
--base_background_bottleneck_mode option controls the base_background
profile and defaults to severe.

The five grids specify mu0 (per base per generation), base gamma shape, hotspot
probability, hotspot mean multiplier, and hotspot gamma shape. The base multiplier
has mean 1. The hotspot multiplier has mean hot_mean. Use positive rates, shapes,
and means, with hotspot probabilities in [0, 1]. A grid may be one value or a
comma-separated list. Exactly three tokens with an integer third token mean
start, stop, number of evenly spaced values; for example 1e-8,2e-8,3.
The Cartesian product determines the number of candidates, and --n_reps is the
number of successful simulated windows per candidate (default 500).

Small execution check from the repository root
----------------------------------------------
Replace the input/output placeholders. This two-candidate, five-replicate command
checks execution only; choose a suitable grid and replication count for fitting.

    python examples/fit_mutation_rates_east_asian.py \
        --empirical_csv /path/to/your/empirical_features.csv \
        --counts_col human_num_sites \
        --out_json /path/to/your/east-asian_mutation_fit.json \
        --mu_grid 1e-8,2e-8 --kbase_grid 2 --phot_grid 0 \
        --hotmean_grid 5 --khot_grid 2 --n_reps 5 --seed 42 \
        --base_background_bottleneck_mode severe --fail_fast

Output
------
The JSON contains mutation_model, the empirical summaries, the best candidate,
top-ranked candidates, simulation diagnostics, and the supplied arguments.
Its six mutation_model fields are consumed by TIB_Sim.py: mu0,
base_gamma_shape, base_gamma_scale, p_hot, hot_gamma_shape, and hot_gamma_scale.
Set DEEP_MUTATION_FIT to your output JSON when running that simulator. Output files
are overwritten if --out_json already exists. No empirical observations or fitted
parameters are embedded in this script."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import msprime
except Exception:  # pragma: no cover
    msprime = None


# -----------------------------
# General utilities
# -----------------------------

def _safe_seed(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    try:
        return abs(int(seed))
    except Exception:
        return None


def _msprime_seed(rng: np.random.Generator) -> int:
    """msprime requires a positive 32-bit-ish integer seed."""
    return int(rng.integers(1, 2**31 - 1))


def parse_grid(s: str) -> np.ndarray:
    """
    Parse a grid specification.

    Supported:
      "a,b,n"  -> linspace(a, b, n) if the 3rd token is an int
      otherwise explicit comma-separated list
    """
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    if len(parts) == 3:
        a = float(parts[0])
        b = float(parts[1])
        try:
            n = int(parts[2])
            if n <= 1:
                return np.array([a], dtype=float)
            return np.linspace(a, b, n, dtype=float)
        except ValueError:
            return np.array([float(x) for x in parts], dtype=float)
    return np.array([float(x) for x in parts], dtype=float)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def q(x: np.ndarray, p: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, p))


def rel_sq(a: float, b: float, eps: float = 1e-12) -> float:
    return float(((float(a) - float(b)) / (float(b) + eps)) ** 2)


def _draw_uniform_or_scalar(x, rng: np.random.Generator) -> float:
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return float(rng.uniform(float(x[0]), float(x[1])))
    return float(x)


# -----------------------------
# Mutation model
# -----------------------------

@dataclass(frozen=True)
class ModelParams:
    """One candidate's rate scale and gamma-mixture parameters.

    mu0 is a per-base, per-generation rate. The base multiplier is Gamma(k_base,
    1/k_base); the hotspot multiplier is Gamma(k_hot, hot_mean/k_hot). p_hot is
    the probability of selecting the hotspot component for a window."""
    mu0: float
    k_base: float
    p_hot: float
    hot_mean: float
    k_hot: float


def draw_window_mu(rng: np.random.Generator, params: ModelParams) -> float:
    """Draw one window's rate from the base/hotspot gamma mixture.

    Before optional background multipliers, the expected rate is
    mu0 * ((1 - p_hot) + p_hot * hot_mean)."""
    k_base = max(float(params.k_base), 1e-12)
    k_hot = max(float(params.k_hot), 1e-12)
    p_hot = min(max(float(params.p_hot), 0.0), 1.0)
    hot_mean = max(float(params.hot_mean), 1e-12)

    if rng.random() < p_hot:
        M = rng.gamma(shape=k_hot, scale=hot_mean / k_hot)
    else:
        M = rng.gamma(shape=k_base, scale=1.0 / k_base)
    return float(params.mu0) * float(M)


# -----------------------------
# Demographic and site-retention profiles
# -----------------------------

def get_background_profiles(base_background_bottleneck_mode: str = "severe") -> Dict[str, Dict]:
    """Return the East Asian demographic and site-retention profiles.

    base_background_bottleneck_mode controls the base_background profile and
    defaults to severe. Other profiles set their own modes. Ghost introgression
    is configured separately by --low_ghost and --high_ghost."""
    if base_background_bottleneck_mode not in {"none", "severe"}:
        raise ValueError("base_background_bottleneck_mode must be 'none' or 'severe'.")

    return {
        "base_background": {
            "den_eas_range": (0.02, 0.10),
            "ned_eas_range": (0.02, 0.05),
            "force_hot_mutation": False,
            "eas_bottleneck_mode": base_background_bottleneck_mode,
        },
        "den_rich_background": {
            "den_eas_range": (0.08, 0.20),
            "ned_eas_range": (0.01, 0.05),
            "force_hot_mutation": False,
            "eas_bottleneck_mode": "none",
        },
        "ned_rich_background": {
            "den_eas_range": (0.02, 0.08),
            "ned_eas_range": (0.05, 0.12),
            "force_hot_mutation": False,
            "eas_bottleneck_mode": "none",
        },
        "dual_archaic_background": {
            "den_eas_range": (0.08, 0.20),
            "ned_eas_range": (0.05, 0.12),
            "force_hot_mutation": False,
            "eas_bottleneck_mode": "none",
        },
        "bottlenecked_background": {
            "den_eas_range": (0.02, 0.10),
            "ned_eas_range": (0.02, 0.05),
            "force_hot_mutation": False,
            "eas_bottleneck_mode": "severe",
        },
        "mut_hot_background": {
            "den_eas_range": (0.02, 0.10),
            "ned_eas_range": (0.02, 0.05),
            "force_hot_mutation": True,
            "eas_bottleneck_mode": "none",
        },
        "compound_hard_background": {
            "den_eas_range": (0.08, 0.20),
            "ned_eas_range": (0.05, 0.12),
            "force_hot_mutation": True,
            "eas_bottleneck_mode": "severe",
        },
        "humanpoly_muthot_background": {
            "den_eas_range": (0.02, 0.10),
            "ned_eas_range": (0.02, 0.05),
            "force_hot_mutation": False,
            "mutation_multiplier_range": (1.5, 2.5),
            "eas_bottleneck_mode": "none",
            "callability_mode": "human_poly_enriched",
            "keep_human_poly": (0.90, 1.00),
            "keep_shared": (0.80, 0.95),
            "keep_den_only": (0.10, 0.30),
            "keep_human_fixed": (0.40, 0.70),
            "keep_other": (0.35, 0.60),
        },
        "humanpoly_dual_archaic_background": {
            "den_eas_range": (0.08, 0.20),
            "ned_eas_range": (0.05, 0.12),
            "force_hot_mutation": False,
            "mutation_multiplier_range": (1.5, 2.5),
            "eas_bottleneck_mode": "none",
            "callability_mode": "human_poly_enriched",
            "keep_human_poly": (0.90, 1.00),
            "keep_shared": (0.80, 0.95),
            "keep_den_only": (0.10, 0.30),
            "keep_human_fixed": (0.40, 0.70),
            "keep_other": (0.35, 0.60),
        },
    }


def draw_nuisance_bundle(
    profile: Dict,
    rng: np.random.Generator,
    mutation_rate: float,
    generation_time: float,
    sequence_length: int,
    recombination_rate: float,
) -> Dict:
    """Draw one nuisance/background bundle, using a supplied mutation rate."""
    generation_time = float(generation_time)
    profile = dict(profile)

    # Apply any additional mutation-rate multiplier specified by this background.
    # base_background applies no extra multiplier.
    mutation_rate = float(mutation_rate)
    if profile.get("force_hot_mutation", False):
        mutation_rate *= float(rng.uniform(2.0, 10.0))
    if "mutation_multiplier_range" in profile:
        lo, hi = profile["mutation_multiplier_range"]
        mutation_rate *= float(rng.uniform(float(lo), float(hi)))

    bundle = {
        "mutation_rate": float(mutation_rate),
        "denisova_3_age": float((((76.2 + 51.6) / 2.0) * 1000.0) / generation_time),
        "sequence_length": int(sequence_length),
        "recombination_rate": float(recombination_rate),
        "generation_time": float(generation_time),
        "T_OOA": float(rng.uniform(50_000, 60_000) / generation_time),
        "T_AMH": float(140_000 / generation_time),
        "T_ANC": float(220_000 / generation_time),
        "T_ARC": float(600_000 / generation_time),
        "T_ND": float(rng.uniform(350_000, 600_000) / generation_time),
        "T_EAS": float(rng.uniform(35_000, 49_500) / generation_time),
        "N_ARC": float(rng.uniform(1000, 6000)),
        "N_AFR": 12300.0,
        "N_OOA": 2100.0,
        "N_EAS": float(rng.uniform(6000, 10000)),
        "N_GHO": float(rng.uniform(1000, 10000)),
        "N_DEN": float(rng.uniform(1000, 4500)),
        "N_NED": float(rng.uniform(1000, 4500)),
        "profile_name": profile.get("_profile_name", "unknown"),
        "callability_mode": profile.get("callability_mode", "none"),
        "mutation_multiplier_range": profile.get("mutation_multiplier_range", None),
        "force_hot_mutation": bool(profile.get("force_hot_mutation", False)),
    }

    bundle["T_GHO"] = float(rng.uniform(bundle["T_ARC"] + 5000.0, bundle["T_ARC"] + 50_000.0))
    bundle["IT_DEN_EAS"] = float(rng.uniform(bundle["T_EAS"] + 1.0, bundle["T_OOA"] - 1.0))
    bundle["IT_NED_EAS"] = float(rng.uniform(bundle["T_EAS"] + 1.0, bundle["T_OOA"] - 1.0))

    lo = bundle["denisova_3_age"] + 1.0
    hi = bundle["T_ND"] - 1.0
    if hi <= lo:
        hi = lo + 1.0
    bundle["IT_GHO_DEN"] = float(rng.uniform(lo, hi))

    den_rng = profile.get("den_eas_range", (0.02, 0.10))
    ned_rng = profile.get("ned_eas_range", (0.02, 0.05))
    bundle["M_DEN_EAS"] = float(rng.uniform(float(den_rng[0]), float(den_rng[1])))
    bundle["M_NED_EAS"] = float(rng.uniform(float(ned_rng[0]), float(ned_rng[1])))

    bottleneck_mode = profile.get("eas_bottleneck_mode", "none")
    if bottleneck_mode == "default":
        bottleneck_mode = "none"
    if bottleneck_mode not in {"none", "severe"}:
        raise ValueError(f"Unknown eas_bottleneck_mode: {bottleneck_mode}")
    bundle["eas_bottleneck_mode"] = bottleneck_mode

    if bottleneck_mode == "none":
        bundle["bottleneck_size"] = None
        bundle["intermediate_size"] = None
    else:
        bundle["bottleneck_size"] = 100.0
        bundle["intermediate_size"] = float(bundle["N_EAS"] / 2.0)

    if profile.get("callability_mode", None) == "human_poly_enriched":
        bundle["keep_human_poly"] = _draw_uniform_or_scalar(profile.get("keep_human_poly", 1.0), rng)
        bundle["keep_shared"] = _draw_uniform_or_scalar(profile.get("keep_shared", bundle["keep_human_poly"]), rng)
        bundle["keep_den_only"] = _draw_uniform_or_scalar(profile.get("keep_den_only", 1.0), rng)
        bundle["keep_human_fixed"] = _draw_uniform_or_scalar(profile.get("keep_human_fixed", 1.0), rng)
        bundle["keep_other"] = _draw_uniform_or_scalar(profile.get("keep_other", 1.0), rng)
    else:
        bundle["keep_human_poly"] = None
        bundle["keep_shared"] = None
        bundle["keep_den_only"] = None
        bundle["keep_human_fixed"] = None
        bundle["keep_other"] = None

    return bundle


def build_demography_and_samples(
    bundle: Dict,
    n_eas_oversample_diploids: int,
    low_ghost: float,
    high_ghost: float,
    rng: np.random.Generator,
) -> Tuple["msprime.Demography", List["msprime.SampleSet"], float]:
    """Build the demographic history and sample the oversampled EAS panel.

    Event times are generations before the present. SampleSet sizes are diploid
    individuals because simulate_one_processed_count uses ploidy=2. One ancient
    Denisovan diploid is sampled after the human panel in the sample list."""
    if msprime is None:
        raise RuntimeError("msprime is not available.")

    if high_ghost < low_ghost:
        raise ValueError("high_ghost must be >= low_ghost")
    if high_ghost == 0.0 and low_ghost == 0.0:
        M_GHO_DEN = 0.0
    elif high_ghost == low_ghost:
        M_GHO_DEN = float(low_ghost)
    else:
        M_GHO_DEN = float(rng.uniform(float(low_ghost), float(high_ghost)))

    demography = msprime.Demography()

    demography.add_population(name="YRI", description="Yoruba", initial_size=bundle["N_AFR"])
    demography.add_population(name="GHO", description="Ghost", initial_size=bundle["N_GHO"])
    demography.add_population(name="OOA", description="OOA", initial_size=bundle["N_OOA"])
    demography.add_population(name="AMH", description="AMH", initial_size=bundle["N_AFR"])
    demography.add_population(name="EAS", description="East Asian sampled population", initial_size=bundle["N_EAS"])
    demography.add_population(name="NED", description="Neanderthal", initial_size=bundle["N_NED"])
    demography.add_population(name="DEN", description="Denisovan", initial_size=bundle["N_DEN"])
    demography.add_population(name="ANC", description="ANC", initial_size=7300.0)
    demography.add_population(name="ANC_OLD", description="ANC_OLD", initial_size=7300.0)
    demography.add_population(name="COMMON", description="COMMON", initial_size=7300.0)
    demography.add_population(name="ARC", description="ARC", initial_size=bundle["N_ARC"])
    demography.add_population_split(time=bundle["T_OOA"], derived=["EAS"], ancestral="OOA")
    demography.add_symmetric_migration_rate_change(time=bundle["T_OOA"], populations=["YRI", "OOA"], rate=25e-5)
    demography.add_population_split(time=bundle["T_AMH"], derived=["YRI", "OOA"], ancestral="AMH")
    demography.add_population_split(time=bundle["T_ND"], derived=["DEN", "NED"], ancestral="ARC")
    demography.add_population_split(time=bundle["T_ANC"], derived=["AMH"], ancestral="ANC")
    demography.add_population_split(time=bundle["T_ARC"], derived=["ANC", "ARC"], ancestral="ANC_OLD")
    demography.add_population_split(time=bundle["T_GHO"], derived=["GHO", "ANC_OLD"], ancestral="COMMON")

    demography.add_mass_migration(time=bundle["IT_DEN_EAS"], source="EAS", dest="DEN", proportion=bundle["M_DEN_EAS"])
    demography.add_mass_migration(time=bundle["IT_NED_EAS"], source="EAS", dest="NED", proportion=bundle["M_NED_EAS"])
    if M_GHO_DEN > 0.0:
        demography.add_mass_migration(time=bundle["IT_GHO_DEN"], source="DEN", dest="GHO", proportion=M_GHO_DEN)

    # Severe EAS size changes, measured backward from the present:
    # present-day EAS starts at N_EAS, changes to N_EAS/2 at 50 generations,
    # and changes to bottleneck_size=100 at T_OOA - 50 generations.
    if bundle["eas_bottleneck_mode"] == "severe":
        demography.add_population_parameters_change(
            time=bundle["T_OOA"] - 50.0,
            initial_size=bundle["bottleneck_size"],
            population="EAS",
        )
        demography.add_population_parameters_change(
            time=50.0,
            initial_size=bundle["intermediate_size"],
            population="EAS",
        )

    demography.sort_events()

    samples = [
        msprime.SampleSet(num_samples=int(n_eas_oversample_diploids), population="EAS", time=0.0),
        msprime.SampleSet(num_samples=1, population="DEN", time=float(bundle["denisova_3_age"])),
    ]
    return demography, samples, float(M_GHO_DEN)


# -----------------------------
# Filtering and count extraction
# -----------------------------

def apply_human_poly_callable_profile(
    G_full: np.ndarray,
    positions: np.ndarray,
    rng: np.random.Generator,
    bundle: Dict,
    n_den_haps: int = 2,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Retain sites probabilistically under an optional callability profile.

    Classify each row using the oversampled humans and the final Denisovan columns.
    Human-polymorphic, shared, Denisovan-only, and human-fixed-alternate sites can
    have different retention probabilities. Without this profile, keep all rows.
    Return the retained genotypes, aligned positions, and number of removed sites."""
    if bundle.get("callability_mode", None) != "human_poly_enriched":
        return G_full, positions, 0

    if G_full.shape[0] == 0:
        return G_full, positions, 0

    n_total_haps = G_full.shape[1]
    n_human_haps = n_total_haps - int(n_den_haps)
    if n_human_haps <= 0:
        return G_full, positions, 0

    H = G_full[:, :n_human_haps]
    D = G_full[:, n_human_haps:]

    h_ct = H.sum(axis=1)
    d_ct = D.sum(axis=1)

    human_poly = (h_ct > 0) & (h_ct < n_human_haps)
    den_present = d_ct > 0

    shared = human_poly & den_present
    human_poly_only = human_poly & (~den_present)
    den_only = (h_ct == 0) & den_present
    human_fixed_alt = (h_ct == n_human_haps)
    other = ~(shared | human_poly_only | den_only | human_fixed_alt)

    p_keep = np.full(G_full.shape[0], float(bundle.get("keep_other", 1.0)), dtype=float)
    p_keep[human_poly_only] = float(bundle.get("keep_human_poly", 1.0))
    p_keep[shared] = float(bundle.get("keep_shared", bundle.get("keep_human_poly", 1.0)))
    p_keep[den_only] = float(bundle.get("keep_den_only", 1.0))
    p_keep[human_fixed_alt] = float(bundle.get("keep_human_fixed", 1.0))
    p_keep[other] = float(bundle.get("keep_other", 1.0))
    p_keep = np.clip(p_keep, 0.0, 1.0)

    keep_mask = rng.random(G_full.shape[0]) < p_keep
    return G_full[keep_mask, :], positions[keep_mask], int(np.sum(~keep_mask))


def remove_oversample_eas_singletons(
    G_full: np.ndarray,
    positions: np.ndarray,
    n_eas_oversample_diploids: int,
    singleton_mode: str = "derived",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Remove sites with human minor-allele count 1 in the oversampled panel.

    Pass singleton_mode='minor', as main() does explicitly. The function's
    standalone default 'derived' is unsupported by its validation. The Denisovan
    alleles do not contribute to the human singleton count."""
    if singleton_mode != "minor":
        raise ValueError("singleton_mode must be 'minor' for this workflow.")
    if G_full.shape[0] == 0:
        return G_full, positions, 0

    n_human_haps = 2 * int(n_eas_oversample_diploids)
    H = G_full[:, :n_human_haps]
    human_alt_ct = H.sum(axis=1)

    singleton_mask = (human_alt_ct == 1) | (human_alt_ct == n_human_haps - 1)

    keep_mask = ~singleton_mask
    return G_full[keep_mask, :], positions[keep_mask], int(np.sum(singleton_mask))


def downsample_eas_diploid_pairs(
    G_full: np.ndarray,
    positions: np.ndarray,
    rng: np.random.Generator,
    n_eas_oversample_diploids: int,
    n_eas_final_diploids: int,
    n_den_haps: int = 2,
) -> np.ndarray:
    """Choose whole EAS diploids without replacement and retain the Denisovan.

    Adjacent human haplotypes belong to the same individual. Keep each selected
    pair together, append the two Denisovan alleles, and prepend site positions.
    Seven humans and one Denisovan therefore produce a 17-column matrix."""
    n_over = int(n_eas_oversample_diploids)
    n_final = int(n_eas_final_diploids)
    if n_final > n_over:
        raise ValueError("n_eas_final_diploids cannot exceed n_eas_oversample_diploids.")

    n_human_haps_over = 2 * n_over
    expected_cols = n_human_haps_over + int(n_den_haps)
    if G_full.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} genotype columns, got {G_full.shape[1]}.")

    chosen_diploids = rng.choice(n_over, size=n_final, replace=False)
    rng.shuffle(chosen_diploids)
    human_cols = np.array([c for d in chosen_diploids for c in (2 * int(d), 2 * int(d) + 1)], dtype=int)
    den_cols = np.arange(n_human_haps_over, n_human_haps_over + int(n_den_haps), dtype=int)
    keep_cols = np.concatenate([human_cols, den_cols])

    if G_full.shape[0] == 0:
        return np.empty((0, 1 + keep_cols.size), dtype=float)

    mat = np.column_stack([positions.astype(float), G_full[:, keep_cols].astype(np.int16)])
    return mat


def count_processed_sites_like_training_stats(genotype_matrix_with_positions: np.ndarray, n_den_haps: int = 2) -> int:
    """Count sites polymorphic in the final human panel.

    Column 0 contains positions, followed by paired human haplotypes and the last
    n_den_haps Denisovan columns. Fold alleles across all retained samples, then
    count rows containing both alleles among the humans. This corresponds to
    human_num_sites and combined_num_sites in the empirical statistics. Singleton
    removal has already occurred in the oversampled panel; a site may become a
    singleton after downsampling and still contribute to this count."""
    if genotype_matrix_with_positions.shape[0] == 0:
        return 0

    G = genotype_matrix_with_positions[:, 1:].astype(np.int16, copy=True)
    n_total_haps = G.shape[1]
    if n_total_haps <= int(n_den_haps):
        return 0

    ac = G.sum(axis=1)
    flip = ac > (n_total_haps / 2.0)
    if np.any(flip):
        G[flip, :] = 1 - G[flip, :]

    human = G[:, : n_total_haps - int(n_den_haps)]
    human_poly = np.any(human == 1, axis=1) & np.any(human == 0, axis=1)
    return int(np.sum(human_poly))


def simulate_one_processed_count(
    params: ModelParams,
    rng: np.random.Generator,
    profile: Dict,
    args: argparse.Namespace,
) -> Tuple[int, Dict]:
    """Simulate and filter one window, returning its human site count and diagnostics.

    The order is ancestry, mutations, optional callability retention, oversampled
    human singleton removal, whole-diploid downsampling, and human-polymorphic site
    counting. Ghost introgression follows args.low_ghost and args.high_ghost."""
    mu = draw_window_mu(rng, params)
    bundle = draw_nuisance_bundle(
        profile=profile,
        rng=rng,
        mutation_rate=mu,
        generation_time=float(args.generation_time),
        sequence_length=int(args.sequence_length),
        recombination_rate=float(args.recombination_rate),
    )

    demography, samples, M_GHO_DEN = build_demography_and_samples(
        bundle=bundle,
        n_eas_oversample_diploids=int(args.n_eas_oversample_diploids),
        low_ghost=float(args.low_ghost),
        high_ghost=float(args.high_ghost),
        rng=rng,
    )

    ts_anc = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=int(args.sequence_length),
        recombination_rate=float(args.recombination_rate),
        ploidy=2,
        record_migrations=False,
        random_seed=_msprime_seed(rng),
    )

    ts = msprime.sim_mutations(
        ts_anc,
        rate=float(bundle["mutation_rate"]),
        model=msprime.BinaryMutationModel(),
        random_seed=_msprime_seed(rng),
    )

    G_full = ts.genotype_matrix().astype(np.int16, copy=False)
    positions = np.asarray(ts.tables.sites.position, dtype=float)

    G_full, positions, callable_removed = apply_human_poly_callable_profile(
        G_full=G_full,
        positions=positions,
        rng=rng,
        bundle=bundle,
        n_den_haps=int(args.n_haps_den),
    )

    G_full, positions, singletons_removed = remove_oversample_eas_singletons(
        G_full=G_full,
        positions=positions,
        n_eas_oversample_diploids=int(args.n_eas_oversample_diploids),
        singleton_mode=str(args.singleton_mode),
    )

    mat = downsample_eas_diploid_pairs(
        G_full=G_full,
        positions=positions,
        rng=rng,
        n_eas_oversample_diploids=int(args.n_eas_oversample_diploids),
        n_eas_final_diploids=int(args.n_eas_diploids),
        n_den_haps=int(args.n_haps_den),
    )
    count = count_processed_sites_like_training_stats(mat, n_den_haps=int(args.n_haps_den))

    info = {
        "mu": float(mu),
        "actual_mutation_rate": float(bundle["mutation_rate"]),
        "M_GHO_DEN": float(M_GHO_DEN),
        "singletons_removed": int(singletons_removed),
        "callable_removed": int(callable_removed),
        "N_EAS": float(bundle["N_EAS"]),
        "M_DEN_EAS": float(bundle["M_DEN_EAS"]),
        "M_NED_EAS": float(bundle["M_NED_EAS"]),
        "eas_bottleneck_mode": bundle["eas_bottleneck_mode"],
    }
    return int(count), info


def simulate_counts_for_params(
    params: ModelParams,
    profile: Dict,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, Dict]:
    """Collect n_reps successful window simulations for one candidate.

    Zero-site windows are included. Retry failed simulations up to the attempt
    limit, or stop at the first error with --fail_fast. Report failure counts,
    removed sites, and mean mutation rate alongside the simulated counts."""
    rng = np.random.default_rng(_safe_seed(seed))
    counts = np.empty(int(args.n_reps), dtype=np.int64)
    n_fail = 0
    singleton_removed_values: List[int] = []
    callable_removed_values: List[int] = []
    mu_values: List[float] = []

    i = 0
    attempts = 0
    max_attempts = max(int(args.n_reps) * 5, int(args.n_reps) + 100)
    while i < int(args.n_reps):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Too many failed attempts for params={params}. "
                f"Accepted {i}/{args.n_reps}, failed {n_fail}."
            )
        try:
            c, info = simulate_one_processed_count(params=params, rng=rng, profile=profile, args=args)
        except Exception as e:
            n_fail += 1
            if bool(args.fail_fast):
                raise
            continue
        counts[i] = int(c)
        singleton_removed_values.append(int(info.get("singletons_removed", 0)))
        callable_removed_values.append(int(info.get("callable_removed", 0)))
        mu_values.append(float(info.get("actual_mutation_rate", np.nan)))
        i += 1

    diag = {
        "n_fail": int(n_fail),
        "attempts": int(attempts),
        "mean_oversample_singletons_removed": float(np.mean(singleton_removed_values)) if singleton_removed_values else 0.0,
        "mean_callable_removed": float(np.mean(callable_removed_values)) if callable_removed_values else 0.0,
        "mean_actual_mutation_rate": float(np.nanmean(mu_values)) if mu_values else float("nan"),
    }
    return counts, diag


# -----------------------------
# Losses and summaries
# -----------------------------

def compute_stats(counts: np.ndarray, upper_thresh: Sequence[float]) -> Dict[str, float]:
    """Summarize count moments, quantiles, maximum, and strict exceedance rates."""
    counts = np.asarray(counts, dtype=float)
    out: Dict[str, float] = {}
    out["mean"] = float(counts.mean()) if counts.size else 0.0
    out["var"] = float(counts.var(ddof=0)) if counts.size else 0.0
    out["q50"] = q(counts, 0.50)
    out["q75"] = q(counts, 0.75)
    out["q90"] = q(counts, 0.90)
    out["q95"] = q(counts, 0.95)
    out["q99"] = q(counts, 0.99)
    out["q999"] = q(counts, 0.999)
    out["max"] = float(counts.max()) if counts.size else 0.0
    for t in upper_thresh:
        out[f"p_gt_{int(t)}"] = float(np.mean(counts > float(t))) if counts.size else 0.0
    return out


def quantile_grid_loss(
    sim_counts: np.ndarray,
    emp_counts: np.ndarray,
    probs: Sequence[float],
    tail_power: float,
    w_quantiles: float,
) -> Tuple[float, Dict]:
    """Compare simulated and empirical quantiles with relative squared error.

    Keep unique probabilities strictly between 0 and 1. Normalize the quantile
    weights to mean 1; tail_power controls their distribution across the chosen
    probabilities. A nonpositive w_quantiles disables this component."""
    if float(w_quantiles) <= 0:
        return 0.0, {"probs": list(probs), "sim": [], "emp": [], "weights": []}

    probs_arr = np.array(sorted(set(float(p) for p in probs)), dtype=float)
    probs_arr = probs_arr[(probs_arr > 0.0) & (probs_arr < 1.0)]
    if probs_arr.size == 0:
        raise ValueError("quantile_probs must contain values between 0 and 1.")

    sim_qs = np.quantile(sim_counts, probs_arr)
    emp_qs = np.quantile(emp_counts, probs_arr)

    lo = float(np.min(probs_arr))
    hi = float(np.max(probs_arr))
    x = (probs_arr - lo) / max(hi - lo, 1e-12)

    # Mild tail emphasis without letting the last few quantiles dominate.
    # tail_power=0 gives equal extra weight; larger values tilt upward.
    w = 0.5 + (x ** float(tail_power))
    w = w / (w.mean() + 1e-12)

    err = ((sim_qs - emp_qs) / (emp_qs + 1e-12)) ** 2
    loss = float(w_quantiles) * float(np.mean(w * err))
    details = {
        "probs": [float(p) for p in probs_arr],
        "sim": [float(v) for v in sim_qs],
        "emp": [float(v) for v in emp_qs],
        "weights": [float(v) for v in w],
    }
    return loss, details


def compute_loss(
    sim_counts: np.ndarray,
    emp_counts: np.ndarray,
    sim_stats: Dict[str, float],
    emp_stats: Dict[str, float],
    upper_thresh: Sequence[float],
    args: argparse.Namespace,
) -> Tuple[float, Dict]:
    """Sum weighted moment, named-quantile, exceedance, and quantile-grid losses.

    When an empirical exceedance probability is below min_exceed_prob, use the
    squared simulated probability to avoid dividing by a near-zero target."""
    loss = 0.0

    # Optional broad moments / named quantiles. Defaults can be set low so the
    # quantile grid remains the primary criterion.
    loss += float(args.w_mean) * rel_sq(sim_stats["mean"], emp_stats["mean"])
    loss += float(args.w_var) * rel_sq(sim_stats["var"], emp_stats["var"])
    loss += float(args.w_q50) * rel_sq(sim_stats["q50"], emp_stats["q50"])
    loss += float(args.w_q75) * rel_sq(sim_stats["q75"], emp_stats["q75"])
    loss += float(args.w_q90) * rel_sq(sim_stats["q90"], emp_stats["q90"])
    loss += float(args.w_q99) * rel_sq(sim_stats["q99"], emp_stats["q99"])
    loss += float(args.w_q999) * rel_sq(sim_stats["q999"], emp_stats["q999"])

    exceed_terms = {}
    if float(args.w_exceed) > 0 and len(upper_thresh) > 0:
        for t in upper_thresh:
            key = f"p_gt_{int(t)}"
            ps = sim_stats[key]
            pe = emp_stats[key]
            if pe < float(args.min_exceed_prob):
                term = float(args.w_exceed) * (ps ** 2)
            else:
                term = float(args.w_exceed) * rel_sq(ps, pe)
            loss += term
            exceed_terms[key] = float(term)

    qloss, qdetails = quantile_grid_loss(
        sim_counts=sim_counts,
        emp_counts=emp_counts,
        probs=parse_float_list(args.quantile_probs),
        tail_power=float(args.tail_power),
        w_quantiles=float(args.w_quantiles),
    )
    loss += qloss

    details = {
        "quantile_grid_loss": float(qloss),
        "quantile_grid": qdetails,
        "exceed_terms": exceed_terms,
    }
    return float(loss), details


def candidate_iter(
    mu_grid: np.ndarray,
    kbase_grid: np.ndarray,
    phot_grid: np.ndarray,
    hotmean_grid: np.ndarray,
    khot_grid: np.ndarray,
) -> Iterable[ModelParams]:
    """Yield the Cartesian product of the five supplied mutation-parameter grids."""
    for mu0, k_base, p_hot, hot_mean, k_hot in itertools.product(
        mu_grid,
        kbase_grid,
        phot_grid,
        hotmean_grid,
        khot_grid,
    ):
        yield ModelParams(
            mu0=float(mu0),
            k_base=float(k_base),
            p_hot=float(p_hot),
            hot_mean=float(hot_mean),
            k_hot=float(k_hot),
        )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    """Read the empirical counts, evaluate candidates, and write the best fit as JSON."""
    ap = argparse.ArgumentParser()

    # Supply your empirical count CSV and choose an output JSON path.
    # --min_sites filters the empirical target; it does not filter simulated counts.
    ap.add_argument("--empirical_csv", required=True)
    ap.add_argument("--counts_col", required=True)
    ap.add_argument("--min_sites", type=int, default=0)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--n_reps", type=int, default=500)

    # Grids use parse_grid(): one value, a list, or start,stop,number_of_values.
    ap.add_argument("--mu_grid", required=True)
    ap.add_argument("--kbase_grid", required=True)
    ap.add_argument("--phot_grid", required=True)
    ap.add_argument("--hotmean_grid", required=True)
    ap.add_argument("--khot_grid", required=True)

    ap.add_argument("--upper_thresh", default="80,100,120,150,180,200,250")
    ap.add_argument(
        "--quantile_probs",
        default="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.975,0.99",
        help="Comma-separated quantile grid used for the primary fit.",
    )

    # Window length is in bases, recombination is per base per generation,
    # and generation_time converts demographic ages in years to generations.
    ap.add_argument("--sequence_length", type=int, default=50_000)
    ap.add_argument("--recombination_rate", type=float, default=1e-8)
    ap.add_argument("--generation_time", type=float, default=25.0)

    # Match the empirical ascertainment panel and retained human panel.
    # n_haps_human must be twice the retained diploid count; one Denisovan has two haplotypes.
    ap.add_argument("--n_eas_diploids", type=int, default=7)
    ap.add_argument("--n_eas_oversample_diploids", type=int, default=50)
    ap.add_argument("--n_haps_human", type=int, default=14)
    ap.add_argument("--n_haps_den", type=int, default=2)
    ap.add_argument("--singleton_mode", choices=["minor"], default="minor")

    ap.add_argument("--calibration_profile", type=str, default="base_background")
    # This option selects the base profile's demographic bottleneck mode.
    ap.add_argument("--base_background_bottleneck_mode", choices=["none", "severe"], default="severe")
    ap.add_argument("--low_ghost", type=float, default=0.0)
    ap.add_argument("--high_ghost", type=float, default=0.0)

    # The master seed controls candidate subsampling and per-candidate simulation seeds.
    # --fail_fast is useful for diagnosing input or demographic errors in a small trial.
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fail_fast", action="store_true")
    ap.add_argument("--max_grid_evals", type=int, default=None, help="Optional random subset of full grid for testing.")
    ap.add_argument("--keep_top", type=int, default=25, help="Number of top candidates to keep in JSON.")
    ap.add_argument("--progress_every", type=int, default=25, help="Print progress every N evaluated candidates.")

    # Loss weights. Defaults emphasize the quantile grid and overall shape, not
    # the extreme upper tail.
    ap.add_argument("--w_mean", type=float, default=1.0)
    ap.add_argument("--w_var", type=float, default=0.25)
    ap.add_argument("--w_q50", type=float, default=0.5)
    ap.add_argument("--w_q75", type=float, default=0.5)
    ap.add_argument("--w_q90", type=float, default=0.25)
    ap.add_argument("--w_q99", type=float, default=0.10)
    ap.add_argument("--w_q999", type=float, default=0.0)
    ap.add_argument("--w_exceed", type=float, default=0.0)
    ap.add_argument("--min_exceed_prob", type=float, default=5e-4)
    ap.add_argument("--w_quantiles", type=float, default=3.0)
    ap.add_argument("--tail_power", type=float, default=1.0)

    # Compatibility options: accepted and recorded in args, but unused by this fitter.
    ap.add_argument("--C_override", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n_calib", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--mu_calib", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--callable_keep_prob", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--p2_keep_sim", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--disable_doubleton_approx", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--enable_doubleton_approx", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--n_sep1_diploids", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n_sep2_diploids", type=int, default=None, help=argparse.SUPPRESS)

    args = ap.parse_args()

    if msprime is None:
        raise RuntimeError("msprime is not available. Run this on the environment where msprime is installed.")

    if 2 * int(args.n_eas_diploids) != int(args.n_haps_human):
        raise ValueError(
            f"Expected n_haps_human == 2*n_eas_diploids; got "
            f"{args.n_haps_human} vs {2 * int(args.n_eas_diploids)}."
        )
    if int(args.n_eas_oversample_diploids) < int(args.n_eas_diploids):
        raise ValueError("n_eas_oversample_diploids must be >= n_eas_diploids.")
    if float(args.low_ghost) != 0.0 or float(args.high_ghost) != 0.0:
        print("[warn] You are fitting with nonzero ghost introgression; for base negative fitting use 0/0.")

    seed = _safe_seed(args.seed)
    master_rng = np.random.default_rng(seed)

    df = pd.read_csv(args.empirical_csv)
    if args.counts_col not in df.columns:
        raise ValueError(f"counts_col '{args.counts_col}' not found in CSV.")

    emp_counts = df[args.counts_col].to_numpy(dtype=float)
    emp_counts = emp_counts[np.isfinite(emp_counts)]
    emp_counts = emp_counts[emp_counts >= float(args.min_sites)].astype(np.int64)
    if emp_counts.size < 25:
        raise ValueError(f"Too few empirical windows after filtering: n={emp_counts.size}")

    upper_thresh = sorted(set(parse_float_list(args.upper_thresh)))
    emp_stats = compute_stats(emp_counts, upper_thresh)
    emp_stats["n"] = int(emp_counts.size)

    profiles = get_background_profiles(base_background_bottleneck_mode=str(args.base_background_bottleneck_mode))
    if args.calibration_profile not in profiles:
        raise ValueError(f"Unknown calibration_profile: {args.calibration_profile}")
    profile = dict(profiles[str(args.calibration_profile)])
    profile["_profile_name"] = str(args.calibration_profile)

    mu_grid = parse_grid(args.mu_grid)
    kbase_grid = parse_grid(args.kbase_grid)
    phot_grid = parse_grid(args.phot_grid)
    hotmean_grid = parse_grid(args.hotmean_grid)
    khot_grid = parse_grid(args.khot_grid)

    # Materialize the complete grid before optionally selecting a random subset.
    all_candidates = list(candidate_iter(mu_grid, kbase_grid, phot_grid, hotmean_grid, khot_grid))
    full_grid_size = len(all_candidates)
    if args.max_grid_evals is not None and int(args.max_grid_evals) < full_grid_size:
        idx = master_rng.choice(full_grid_size, size=int(args.max_grid_evals), replace=False)
        all_candidates = [all_candidates[int(i)] for i in idx]
        print(f"[info] using random subset of grid: {len(all_candidates)} / {full_grid_size}")

    print(
        f"[info] empirical n={emp_counts.size} mean={emp_stats['mean']:.3f} "
        f"q50={emp_stats['q50']:.2f} q75={emp_stats['q75']:.2f} "
        f"q90={emp_stats['q90']:.2f} q99={emp_stats['q99']:.2f} "
        f"q999={emp_stats['q999']:.2f} max={emp_stats['max']:.2f}"
    )
    print(
        f"[info] direct simulation fit; no C_eff calibration. profile={args.calibration_profile} "
        f"base_background_bottleneck_mode={args.base_background_bottleneck_mode} "
        f"low_ghost={args.low_ghost} high_ghost={args.high_ghost} "
        f"singleton_mode={args.singleton_mode}"
    )
    print(f"[info] grid size = {len(all_candidates)} candidate(s); n_reps per candidate = {args.n_reps}")

    best = None
    top: List[Dict] = []

    for cand_i, params in enumerate(all_candidates, start=1):
        # Each candidate gets its own reproducible simulation seed from the master RNG.
        cand_seed = _msprime_seed(master_rng)
        sim_counts, diag = simulate_counts_for_params(params=params, profile=profile, args=args, seed=cand_seed)
        sim_stats = compute_stats(sim_counts, upper_thresh)
        loss, loss_details = compute_loss(
            sim_counts=sim_counts,
            emp_counts=emp_counts,
            sim_stats=sim_stats,
            emp_stats=emp_stats,
            upper_thresh=upper_thresh,
            args=args,
        )

        record = {
            "mu0": float(params.mu0),
            "k_base": float(params.k_base),
            "p_hot": float(params.p_hot),
            "hot_mean": float(params.hot_mean),
            "k_hot": float(params.k_hot),
            "loss": float(loss),
            "sim_stats": sim_stats,
            "diagnostics": diag,
            "loss_details": loss_details,
        }

        top.append(record)
        top = sorted(top, key=lambda d: d["loss"])[: int(args.keep_top)]

        if best is None or loss < best["loss"]:
            best = record
            extras = " ".join(
                [
                    f"p_gt_{int(t)}={sim_stats[f'p_gt_{int(t)}']:.4g}/{emp_stats[f'p_gt_{int(t)}']:.4g}"
                    for t in upper_thresh[:4]
                ]
            )
            print(
                f"[best {cand_i}/{len(all_candidates)}] "
                f"mu0={best['mu0']:.3e} k_base={best['k_base']:.4g} "
                f"p_hot={best['p_hot']:.4g} hot_mean={best['hot_mean']:.4g} "
                f"k_hot={best['k_hot']:.4g} loss={best['loss']:.5g} "
                f"sim_mean={sim_stats['mean']:.2f} sim_q50={sim_stats['q50']:.2f} "
                f"sim_q90={sim_stats['q90']:.2f} sim_q99={sim_stats['q99']:.2f} "
                f"singletons_removed_mean={diag['mean_oversample_singletons_removed']:.2f} "
                + extras
            )
        elif int(args.progress_every) > 0 and cand_i % max(1, int(args.progress_every)) == 0:
            print(f"[info] evaluated {cand_i}/{len(all_candidates)} candidates; current best loss={best['loss']:.5g}")

    if best is None:
        raise RuntimeError("No candidate was evaluated successfully.")

    # Export gamma scales as well as shapes for the training simulators.
    # The base component has mean 1, and the hotspot component has mean hot_mean.
    mutation_model = {
        "mu0": float(best["mu0"]),
        "base_gamma_shape": float(best["k_base"]),
        "base_gamma_scale": float(1.0 / max(float(best["k_base"]), 1e-12)),
        "p_hot": float(best["p_hot"]),
        "hot_gamma_shape": float(best["k_hot"]),
        "hot_gamma_scale": float(float(best["hot_mean"]) / max(float(best["k_hot"]), 1e-12)),
    }

    out = {
        "mutation_model": mutation_model,
        "empirical_stats": emp_stats,
        "best": best,
        "top_candidates": top,
        "calibration": {
            "method": "none_direct_msprime_quantile_grid",
            "note": "C_eff calibration is intentionally not used. Counts are produced by direct simulation under each candidate mutation model, then processed with oversample-singleton filtering and final-sample site-count extraction.",
        },
        "simulation_fit": {
            "method": "direct_msprime_quantile_grid_east_asian",
            "counts_col": str(args.counts_col),
            "profile_name": str(args.calibration_profile),
            "base_background_bottleneck_mode": str(args.base_background_bottleneck_mode),
            "low_ghost": float(args.low_ghost),
            "high_ghost": float(args.high_ghost),
            "singleton_mode": str(args.singleton_mode),
            "n_eas_diploids": int(args.n_eas_diploids),
            "n_eas_oversample_diploids": int(args.n_eas_oversample_diploids),
            "n_haps_human": int(args.n_haps_human),
            "n_haps_den": int(args.n_haps_den),
            "sequence_length": int(args.sequence_length),
            "recombination_rate": float(args.recombination_rate),
            "generation_time": float(args.generation_time),
            "full_grid_size": int(full_grid_size),
            "evaluated_grid_size": int(len(all_candidates)),
        },
        "args": vars(args),
    }

    # Create the requested output directory and save the fit plus its diagnostics.
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"\nWrote: {args.out_json}")
    print(
        f"Best mutation_model: mu0={mutation_model['mu0']:.6e}, "
        f"base_gamma_shape={mutation_model['base_gamma_shape']:.6g}, "
        f"base_gamma_scale={mutation_model['base_gamma_scale']:.6g}, "
        f"p_hot={mutation_model['p_hot']:.6g}, "
        f"hot_gamma_shape={mutation_model['hot_gamma_shape']:.6g}, "
        f"hot_gamma_scale={mutation_model['hot_gamma_scale']:.6g}"
    )


if __name__ == "__main__":
    main()
