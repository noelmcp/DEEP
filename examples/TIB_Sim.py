"""Generate East Asian training windows and matched-attempt metadata for DEEP.

Draw 50-kb windows under shared demographic and mutation-rate settings, simulate
ghost-positive and ghost-negative histories, filter singletons in the larger
human panel, and retain seven diploid humans plus one Denisovan.

Set the paths and run settings below before execution. The example guide explains
the mutation-fit JSON, NPZ layout, and following feature-extraction stage.
"""

import os
import json
from pathlib import Path
import numpy as np
import msprime

# Run configuration. Replace the /path/to placeholders or set the environment variables.
WORK_DIR = Path(os.environ.get("DEEP_WORK_DIR", "/path/to/your/deep_work"))
MUTATION_FIT_PATH = Path(os.environ.get(
    "DEEP_MUTATION_FIT", "/path/to/your/east-asian_mutation_fit.json"
))
GENOTYPE_DIR = WORK_DIR / "genotypes"
data_suff = os.environ.get("DEEP_DATA_SUFFIX", "east-asian")
# Use the same task ID during simulation and feature extraction. Local runs use 1.
array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "1")

# Counts are accepted windows per class. Decrease them for a small trial.
TARGET_WITH = 50000
TARGET_WITHOUT = 50000
# Each additional background receives this many negatives, with proportional positives.
TARGET_ROBUST = 0


# ----------------------------
# Utilities for pop metadata
# ----------------------------
def _pop_name(pop):
    md = getattr(pop, "metadata", None)
    if isinstance(md, dict) and "name" in md:
        return md["name"]
    return None


def _is_human_pop_name(name: str) -> bool:
    if name is None:
        return False
    return name.startswith("EAS")


# ----------------------------
# Mutation model
# ----------------------------
def _load_mutation_model_fit(
    fit_path=MUTATION_FIT_PATH
):
    """Read the fitted gamma-mixture mutation-rate parameters for this population."""
    with open(fit_path, encoding="utf8") as handle:
        fit = json.load(handle)
    mm = fit["mutation_model"]
    MU0 = mm["mu0"]

    K_BASE = mm["base_gamma_shape"]
    S_BASE = mm["base_gamma_scale"]

    P_HOT = mm["p_hot"]
    K_HOT = mm["hot_gamma_shape"]
    S_HOT = mm["hot_gamma_scale"]

    return MU0, K_BASE, S_BASE, P_HOT, K_HOT, S_HOT


MU0, K_BASE, S_BASE, P_HOT, K_HOT, S_HOT = _load_mutation_model_fit()


def draw_window_mu(rng: np.random.Generator) -> float:
    """Draw a mutation rate per base per generation from the fitted mixture."""
    if rng.random() < P_HOT:
        M = rng.gamma(shape=K_HOT, scale=S_HOT)
    else:
        M = rng.gamma(shape=K_BASE, scale=S_BASE)
    return MU0 * M


def draw_window_mu_forced_hot(rng: np.random.Generator, hot_multiplier_range=(2.0, 10.0)) -> float:
    base_mu = draw_window_mu(rng)
    mult = rng.uniform(hot_multiplier_range[0], hot_multiplier_range[1])
    return base_mu * mult


def _draw_uniform_or_scalar(x, rng):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return float(rng.uniform(float(x[0]), float(x[1])))
    return float(x)


# ----------------------------
# Background nuisance profile library
# ----------------------------
# Both labels draw from this demographic and mutation-rate profile library.
# The paired-attempt generator reuses one nuisance draw for its two simulations.
#
# eas_bottleneck_mode:
#   "none"   -> EAS remains constant-size at N_EAS.
#   "severe" -> EAS has a recent explicit bottleneck.
#
# The supplied base_background selects "severe". Extra backgrounds are used
# when TARGET_ROBUST is positive or explicit profile weights request them.
# ----------------------------

def get_background_profiles():
    return {
        "base_background": {
            "den_eas_range": (0.005, 0.02),
            "ned_eas_range": (0.02, 0.05),
            "force_hot_mutation": False,
            "eas_bottleneck_mode": "severe",
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


def draw_nuisance_bundle(profile, rng):
    """Draw shared demographic parameters, times in generations, and a window mutation rate."""
    generation_time = 25

    if profile.get("force_hot_mutation", False):
        mutation_rate = draw_window_mu_forced_hot(rng)
    else:
        mutation_rate = draw_window_mu(rng)

    if "mutation_multiplier_range" in profile:
        mutation_rate *= rng.uniform(
            float(profile["mutation_multiplier_range"][0]),
            float(profile["mutation_multiplier_range"][1])
        )

    eas_bottleneck_mode = profile.get("eas_bottleneck_mode", "none")
    if eas_bottleneck_mode == "default":
        # Backwards-compatible alias for scripts that still pass the old name.
        eas_bottleneck_mode = "none"

    bundle = {
        "mutation_rate": float(mutation_rate),
        "denisova_3_age": float((((76.2 + 51.6) / 2) * 1000) / generation_time),
        "sequence_length": int(5e4),
        "recombination_rate": 1e-8,
        "T_OOA": float(rng.uniform(50000, 60000) / generation_time),
        "T_AMH": float(140e3 / generation_time),
        "T_ANC": float(220e3 / generation_time),
        "T_ARC": float(600e3 / generation_time),
        "T_ND": float(rng.uniform(350e3, 600e3) / generation_time),
        "T_EAS": float(rng.uniform(35000, 49500) / generation_time),
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
        "eas_bottleneck_mode": eas_bottleneck_mode,
    }

    bundle["T_GHO"] = float(rng.uniform(bundle["T_ARC"] + 5000, bundle["T_ARC"] + 50000))

    bundle["IT_DEN_EAS"] = float(rng.uniform(bundle["T_EAS"] + 1, bundle["T_OOA"] - 1))
    bundle["IT_NED_EAS"] = float(rng.uniform(bundle["T_EAS"] + 1, bundle["T_OOA"] - 1))
    bundle["IT_GHO_DEN"] = float(rng.uniform(bundle["denisova_3_age"] + 1, bundle["T_ND"] - 1))

    den_rng = profile.get("den_eas_range", (0.005, 0.02))
    ned_rng = profile.get("ned_eas_range", (0.02, 0.05))
    bundle["M_DEN_EAS"] = float(rng.uniform(den_rng[0], den_rng[1]))
    bundle["M_NED_EAS"] = float(rng.uniform(ned_rng[0], ned_rng[1]))

    if eas_bottleneck_mode == "none":
        bundle["bottleneck_size"] = None
        bundle["intermediate_size"] = None
    elif eas_bottleneck_mode == "severe":
        bundle["bottleneck_size"] = 100
        bundle["intermediate_size"] = bundle["N_EAS"]/2
    else:
        raise ValueError(f"Unknown eas_bottleneck_mode: {eas_bottleneck_mode}")

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


def simulate_one_instance(
    low_ghost=0.005,
    high_ghost=0.02,
    eas_diploids=7,
    rng=None,
    profile=None,
    nuisance_bundle=None,
):
    """Simulate ancestry with migration records, add binary mutations, and return metadata.

    The matrix contains position followed by human haplotypes and two Denisovan
    haplotypes. msprime supplies random seeds internally for ancestry and mutations."""
    if rng is None:
        rng = np.random.default_rng()
    if profile is None:
        profile = {}

    if nuisance_bundle is None:
        nuisance_bundle = draw_nuisance_bundle(profile, rng)

    if high_ghost < low_ghost:
        raise ValueError("high_ghost must be >= low_ghost")

    if high_ghost == 0.0 and low_ghost == 0.0:
        M_GHO_DEN = 0.0
    elif high_ghost == low_ghost:
        M_GHO_DEN = float(low_ghost)
    else:
        M_GHO_DEN = float(rng.uniform(low_ghost, high_ghost))

    current_eas_size = nuisance_bundle["N_EAS"]


    demography = msprime.Demography()

    demography.add_population(name="YRI", description="Yoruba", initial_size=nuisance_bundle["N_AFR"])
    demography.add_population(name="GHO", description="Ghost", initial_size=nuisance_bundle["N_GHO"])
    demography.add_population(name="OOA", description="OOA", initial_size=nuisance_bundle["N_OOA"])
    demography.add_population(name="AMH", description="AMH", initial_size=nuisance_bundle["N_AFR"])
    demography.add_population(name="NED", description="Neanderthal", initial_size=nuisance_bundle["N_NED"])
    demography.add_population(name="DEN", description="Denisovan", initial_size=nuisance_bundle["N_DEN"])
    demography.add_population(name="ANC", description="ANC", initial_size=7300)
    demography.add_population(name="ANC_OLD", description="ANC_OLD", initial_size=7300)
    demography.add_population(name="COMMON", description="COMMON", initial_size=7300)
    demography.add_population(name="ARC", description="ARC", initial_size=nuisance_bundle["N_ARC"])
    demography.add_population(name="EAS", description="East Asian", initial_size=current_eas_size)

    demography.add_population_split(time=nuisance_bundle["T_OOA"], derived=["EAS"], ancestral="OOA")
    demography.add_symmetric_migration_rate_change(time=nuisance_bundle["T_OOA"], populations=["YRI", "OOA"], rate=25e-5)
    demography.add_population_split(time=nuisance_bundle["T_AMH"], derived=["YRI", "OOA"], ancestral="AMH")
    demography.add_population_split(time=nuisance_bundle["T_ND"], derived=["DEN", "NED"], ancestral="ARC")
    demography.add_population_split(time=nuisance_bundle["T_ANC"], derived=["AMH"], ancestral="ANC")
    demography.add_population_split(time=nuisance_bundle["T_ARC"], derived=["ANC", "ARC"], ancestral="ANC_OLD")
    demography.add_population_split(time=nuisance_bundle["T_GHO"], derived=["GHO", "ANC_OLD"], ancestral="COMMON")

    demography.add_mass_migration(
        time=nuisance_bundle["IT_DEN_EAS"],
        source="EAS",
        dest="DEN",
        proportion=nuisance_bundle["M_DEN_EAS"]
    )
    demography.add_mass_migration(
        time=nuisance_bundle["IT_NED_EAS"],
        source="EAS",
        dest="NED",
        proportion=nuisance_bundle["M_NED_EAS"]
    )
    if M_GHO_DEN > 0:
        demography.add_mass_migration(
            time=nuisance_bundle["IT_GHO_DEN"],
            source="DEN",
            dest="GHO",
            proportion=M_GHO_DEN
        )

    if nuisance_bundle["eas_bottleneck_mode"] == "severe":
        demography.add_population_parameters_change(
            time=nuisance_bundle["T_OOA"]-50,
            initial_size=nuisance_bundle["bottleneck_size"],
            population="EAS",
        )
        demography.add_population_parameters_change(
            time=50,
            initial_size=nuisance_bundle["intermediate_size"],
            population="EAS",
        )

    demography.sort_events()

    samples = [
        msprime.SampleSet(num_samples=eas_diploids, population="EAS", time=0),
        msprime.SampleSet(num_samples=1, population="DEN", time=nuisance_bundle["denisova_3_age"]),
    ]

    ts_anc = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=nuisance_bundle["sequence_length"],
        recombination_rate=nuisance_bundle["recombination_rate"],
        ploidy=2,
        record_migrations=True,
    )

    ts = msprime.sim_mutations(ts_anc, rate=nuisance_bundle["mutation_rate"], model="binary")

    G_full = ts.genotype_matrix()
    pos_full = ts.tables.sites.position

    if nuisance_bundle.get("callability_mode", None) == "human_poly_enriched":
        n_total_haps = G_full.shape[1]
        n_human_haps = n_total_haps - 2

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

        p_keep = np.full(G_full.shape[0], nuisance_bundle["keep_other"], dtype=float)
        p_keep[human_poly_only] = nuisance_bundle["keep_human_poly"]
        p_keep[shared] = nuisance_bundle["keep_shared"]
        p_keep[den_only] = nuisance_bundle["keep_den_only"]
        p_keep[human_fixed_alt] = nuisance_bundle["keep_human_fixed"]
        p_keep = np.clip(p_keep, 0.0, 1.0)

        keep_mask = rng.random(G_full.shape[0]) < p_keep
        G = G_full[keep_mask, :]
        pos = pos_full[keep_mask].reshape(-1, 1)
    else:
        G = G_full
        pos = pos_full.reshape(-1, 1)

    mat = np.hstack([pos, G])

    sim_info = {
        "mutation_rate": float(nuisance_bundle["mutation_rate"]),
        "M_DEN_EAS": float(nuisance_bundle["M_DEN_EAS"]),
        "M_NED_EAS": float(nuisance_bundle["M_NED_EAS"]),
        "M_GHO_DEN": float(M_GHO_DEN),
        "IT_DEN_EAS": float(nuisance_bundle["IT_DEN_EAS"]),
        "IT_NED_EAS": float(nuisance_bundle["IT_NED_EAS"]),
        "IT_GHO_DEN": float(nuisance_bundle["IT_GHO_DEN"]),
        "N_EAS": float(nuisance_bundle["N_EAS"]),
        "N_DEN": float(nuisance_bundle["N_DEN"]),
        "N_NED": float(nuisance_bundle["N_NED"]),
        "T_EAS": float(nuisance_bundle["T_EAS"]),
        "T_ND": float(nuisance_bundle["T_ND"]),
        "background_profile_name": nuisance_bundle["profile_name"],
        "callability_mode": nuisance_bundle["callability_mode"],
        "mutation_multiplier_range": nuisance_bundle["mutation_multiplier_range"],
        "keep_human_poly": nuisance_bundle["keep_human_poly"],
        "keep_shared": nuisance_bundle["keep_shared"],
        "keep_den_only": nuisance_bundle["keep_den_only"],
        "keep_human_fixed": nuisance_bundle["keep_human_fixed"],
        "keep_other": nuisance_bundle["keep_other"],
        "eas_bottleneck_mode": nuisance_bundle["eas_bottleneck_mode"],
        "bottleneck_size": nuisance_bundle["bottleneck_size"],
        "intermediate_size": nuisance_bundle["intermediate_size"],
    }

    return ts, mat, sim_info


# ----------------------------
# Ghost ancestry detection
# ----------------------------
def ghost_human_haplotype_nodes(ts):
    """Identify sampled human nodes descending from recorded Denisovan-to-ghost migrations."""
    ghost_pop_id = None
    den_pop_id = None
    human_pop_ids = set()

    for i, pop in enumerate(ts.populations()):
        name = _pop_name(pop)
        if name == "GHO":
            ghost_pop_id = i
        elif name == "DEN":
            den_pop_id = i
        elif _is_human_pop_name(name):
            human_pop_ids.add(i)

    if ghost_pop_id is None or den_pop_id is None or not human_pop_ids:
        return set()

    human_nodes = {
        n.id for n in ts.nodes()
        if n.is_sample() and n.population in human_pop_ids
    }

    humans_with_ghost = set()
    for mig in ts.tables.migrations:
        if mig.source == den_pop_id and mig.dest == ghost_pop_id:
            midpoint = (mig.left + mig.right) / 2.0
            tree = ts.at(midpoint)
            desc = set(tree.samples(mig.node))
            humans_with_ghost |= (desc & human_nodes)

    return humans_with_ghost


def analyze_ghost_introgression(ts):
    """Summarize ghost ancestry using the midpoint of each recorded migration interval.

    Return the number of intervals reaching humans, the number of human carrier
    haplotypes, and whether the sampled Denisovan carries ghost ancestry."""
    ghost_pop_id = None
    den_pop_id = None
    human_pop_ids = set()

    for i, pop in enumerate(ts.populations()):
        name = _pop_name(pop)
        if name == "GHO":
            ghost_pop_id = i
        elif name == "DEN":
            den_pop_id = i
        elif _is_human_pop_name(name):
            human_pop_ids.add(i)

    if ghost_pop_id is None or den_pop_id is None or not human_pop_ids:
        return 0, 0, False

    human_nodes = {
        n.id for n in ts.nodes()
        if n.is_sample() and n.population in human_pop_ids
    }
    den_nodes = {
        n.id for n in ts.nodes()
        if n.is_sample() and n.population == den_pop_id
    }

    ghost_tract_count = 0
    humans_with_ghost = set()
    denisovan_has_ghost = False

    for mig in ts.tables.migrations:
        if mig.source == den_pop_id and mig.dest == ghost_pop_id:
            midpoint = (mig.left + mig.right) / 2.0
            tree = ts.at(midpoint)
            desc = set(tree.samples(mig.node))

            hits = desc & human_nodes
            if hits:
                ghost_tract_count += 1
                humans_with_ghost |= hits

            if (not denisovan_has_ghost) and (desc & den_nodes):
                denisovan_has_ghost = True

    return ghost_tract_count, len(humans_with_ghost), denisovan_has_ghost


# ----------------------------
# Proxy summaries
# ----------------------------
def compute_archaic_like_proxies(genotype_matrix_with_positions, final_human_diploids=7):
    """Calculate retained-panel counts for simulation metadata.

    human_singletons and human_high_freq_sites enter the training feature table
    through metadata. They use the binary alleles before the later feature folding."""
    G = genotype_matrix_with_positions[:, 1:]
    human = G[:, :2 * final_human_diploids]
    den = G[:, 2 * final_human_diploids:2 * final_human_diploids + 2]

    human_alt_ct = human.sum(axis=1)
    den_alt_ct = den.sum(axis=1)

    total_sites = int(G.shape[0])

    den_shared_sites = int(np.sum((den_alt_ct >= 1) & (human_alt_ct >= 1)))
    den_shared_sites_2plus = int(np.sum((den_alt_ct >= 1) & (human_alt_ct >= 2)))
    den_shared_sites_4plus = int(np.sum((den_alt_ct >= 1) & (human_alt_ct >= 4)))
    human_singletons = int(np.sum(human_alt_ct == 1))
    human_high_freq_sites = int(np.sum(human_alt_ct >= 7))

    return {
        "total_sites": total_sites,
        "den_shared_sites": den_shared_sites,
        "den_shared_sites_2plus": den_shared_sites_2plus,
        "den_shared_sites_4plus": den_shared_sites_4plus,
        "human_singletons": human_singletons,
        "human_high_freq_sites": human_high_freq_sites,
    }



# ----------------------------
# Diploid-pair and singleton-filter helpers
# ----------------------------
def get_human_diploid_pairs(ts):
    sample_nodes = list(ts.samples())
    eas_nodes = [
        n for n in sample_nodes
        if _pop_name(ts.population(ts.node(n).population)) == "EAS"
    ]

    if len(eas_nodes) % 2 != 0:
        raise ValueError("EAS sample node count is not even.")

    return [(eas_nodes[i], eas_nodes[i + 1]) for i in range(0, len(eas_nodes), 2)]


def filter_oversample_human_singletons(
    ts,
    genotype_matrix_with_positions,
    singleton_mode="derived",
):
    """Remove singleton sites in the oversampled human panel before downsampling.

    The derived mode drops alt count 1. The minor mode also drops alt count
    n_human_haplotypes - 1. Counts use the human columns of the oversampled matrix.
    Return the filtered matrix and number of removed sites."""
    if singleton_mode not in ("derived", "minor"):
        raise ValueError("singleton_mode must be 'derived' or 'minor'.")

    if genotype_matrix_with_positions.shape[0] == 0:
        return genotype_matrix_with_positions, 0

    G = genotype_matrix_with_positions[:, 1:]
    sample_nodes = list(ts.samples())

    human_sample_indices = []
    for i, n in enumerate(sample_nodes):
        pop_name = _pop_name(ts.population(ts.node(n).population))
        if _is_human_pop_name(pop_name):
            human_sample_indices.append(i)

    if not human_sample_indices:
        raise ValueError("No EAS sample columns found for singleton filtering.")

    H = G[:, human_sample_indices]
    human_alt_ct = H.sum(axis=1)
    n_human_haps = H.shape[1]

    if singleton_mode == "derived":
        singleton_mask = human_alt_ct == 1
    else:
        singleton_mask = (human_alt_ct == 1) | (human_alt_ct == n_human_haps - 1)

    keep_mask = ~singleton_mask
    n_removed = int(np.sum(singleton_mask))
    return genotype_matrix_with_positions[keep_mask, :], n_removed


# ----------------------------
# Downsampling after oversample-defined singleton filtering
# ----------------------------
def downsample_eas_pairs_atleast(
    ts,
    genotype_matrix_with_positions,
    n_eas_final=7,
    min_carrier_haplotypes=4,
    rng=None,
    singleton_mode="derived",
):
    """Sample intact diploids after singleton filtering and require carrier haplotypes.

    The generator rejects a draw that retains fewer than min_carrier_haplotypes.
    The returned k_kept counts carriers in the retained human sample."""
    if rng is None:
        rng = np.random.default_rng()

    filtered_mat, n_singletons_removed = filter_oversample_human_singletons(
        ts,
        genotype_matrix_with_positions,
        singleton_mode=singleton_mode,
    )

    eas_pairs = get_human_diploid_pairs(ts)

    if len(eas_pairs) < n_eas_final:
        raise ValueError("Not enough EAS diploids.")

    carriers = ghost_human_haplotype_nodes(ts)

    chosen_idx = rng.choice(len(eas_pairs), size=n_eas_final, replace=False)
    chosen_pairs = [eas_pairs[i] for i in chosen_idx]
    rng.shuffle(chosen_pairs)

    chosen_human_nodes = [n for pair in chosen_pairs for n in pair]
    k_kept = sum(n in carriers for n in chosen_human_nodes)

    if k_kept < min_carrier_haplotypes:
        raise ValueError("Chosen diploids do not contain enough ghost-carrying haplotypes.")

    sample_nodes = list(ts.samples())
    den_nodes = [n for n in sample_nodes if _pop_name(ts.population(ts.node(n).population)) == "DEN"]
    if len(den_nodes) != 2:
        raise ValueError(f"Expected 2 Denisovan haplotypes, found {len(den_nodes)}.")

    kept_nodes = chosen_human_nodes + sorted(den_nodes)

    node_to_sample_index = {node_id: idx for idx, node_id in enumerate(sample_nodes)}
    kept_sample_indices = [node_to_sample_index[n] for n in kept_nodes]

    keep_cols = [0] + [1 + idx for idx in kept_sample_indices]
    new_mat = filtered_mat[:, keep_cols]

    return new_mat, k_kept, n_singletons_removed


def downsample_no_ghost_eas_fixed(
    ts,
    genotype_matrix_with_positions,
    n_eas_final=7,
    rng=None,
    singleton_mode="derived",
):
    """Sample intact human diploids after singleton filtering and append the Denisovan."""
    if rng is None:
        rng = np.random.default_rng()

    filtered_mat, n_singletons_removed = filter_oversample_human_singletons(
        ts,
        genotype_matrix_with_positions,
        singleton_mode=singleton_mode,
    )

    eas_pairs = get_human_diploid_pairs(ts)

    if len(eas_pairs) < n_eas_final:
        raise ValueError("Not enough EAS diploids.")

    chosen_idx = rng.choice(len(eas_pairs), size=n_eas_final, replace=False)
    chosen_pairs = [eas_pairs[i] for i in chosen_idx]
    rng.shuffle(chosen_pairs)

    chosen_human_nodes = [n for pair in chosen_pairs for n in pair]

    sample_nodes = list(ts.samples())
    den_nodes = [n for n in sample_nodes if _pop_name(ts.population(ts.node(n).population)) == "DEN"]
    if len(den_nodes) != 2:
        raise ValueError(f"Expected 2 Denisovan haplotypes, found {len(den_nodes)}.")

    kept_nodes = chosen_human_nodes + sorted(den_nodes)

    node_to_sample_index = {node_id: idx for idx, node_id in enumerate(sample_nodes)}
    kept_sample_indices = [node_to_sample_index[n] for n in kept_nodes]

    keep_cols = [0] + [1 + idx for idx in kept_sample_indices]
    new_mat = filtered_mat[:, keep_cols]

    return new_mat, n_singletons_removed


# ----------------------------
# Count allocation helpers
# ----------------------------
def _should_log_accept(accepted_count, log_every=100, initial_n=10):
    return accepted_count <= initial_n or (log_every > 0 and accepted_count % log_every == 0)



def _format_accept_line(prefix, accepted_count, target_n, background_name, bundle_id, sim_info,
                        ghost_tract_count, num_haps_with_ghost, denisovan_has_ghost,
                        proxy, k_kept=None, pair_role=None, singletons_removed=None):
    parts = [
        prefix,
        f"accepted={accepted_count}",
        f"target={target_n}",
        f"background={background_name}",
        f"bundle_id={bundle_id}",
    ]
    if pair_role is not None:
        parts.append(f"pair_role={pair_role}")
    parts.extend([
        f"ghost_tracts={ghost_tract_count}",
        f"humans_with_ghost={num_haps_with_ghost}",
        f"denisovan_has_ghost={denisovan_has_ghost}",
        f"den_shared_sites={proxy['den_shared_sites']}",
        f"den_shared_sites_2plus={proxy['den_shared_sites_2plus']}",
        f"den_shared_sites_4plus={proxy['den_shared_sites_4plus']}",
        f"human_high_freq_sites={proxy['human_high_freq_sites']}",
        f"human_singletons_after_downsample={proxy['human_singletons']}",
        f"total_sites={proxy['total_sites']}",
        f"mutation_rate={sim_info['mutation_rate']:.3e}",
        f"M_DEN_EAS={sim_info['M_DEN_EAS']:.4f}",
        f"M_NED_EAS={sim_info['M_NED_EAS']:.4f}",
        f"M_GHO_DEN={sim_info['M_GHO_DEN']:.4f}",
        f"eas_bottleneck_mode={sim_info['eas_bottleneck_mode']}",
        f"bottleneck_size={sim_info['bottleneck_size']}",
        f"intermediate_size={sim_info['intermediate_size']}",
        f"callability_mode={sim_info['callability_mode']}",
    ])
    if k_kept is not None:
        parts.append(f"k_kept={k_kept}")
    if singletons_removed is not None:
        parts.append(f"oversample_singletons_removed={singletons_removed}")
    return ", ".join(parts)


def _allocate_counts_from_weights(total, weights_dict):
    if total < 0:
        raise ValueError("total must be >= 0")

    positive_items = [(k, float(v)) for k, v in weights_dict.items() if float(v) > 0]
    out = {k: 0 for k in weights_dict}

    if total == 0 or not positive_items:
        return out

    keys = [k for k, _ in positive_items]
    weights = np.array([v for _, v in positive_items], dtype=float)
    weights = weights / weights.sum()

    raw = total * weights
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())

    frac = raw - base
    order = np.argsort(-frac)
    for i in range(remainder):
        base[order[i]] += 1

    for k, c in zip(keys, base):
        out[k] = int(c)
    return out


# ----------------------------
# Example generation helpers
# ----------------------------
def _build_metadata(sim_info, proxy, label, ghost_tract_count, num_haps_with_ghost, denisovan_has_ghost,
                    bundle_id=None, pair_role=None, k_kept=None, singletons_removed=None,
                    singleton_mode="derived", n_eas_final=None):
    md = {
        **sim_info,
        **proxy,
        "label": int(label),
        "ghost_tract_count": int(ghost_tract_count),
        "num_haps_with_ghost": int(num_haps_with_ghost),
        "denisovan_has_ghost": bool(denisovan_has_ghost),
        "bundle_id": bundle_id,
        "pair_role": pair_role,
        "singleton_mode": singleton_mode,
    }
    if k_kept is not None:
        md["k_kept"] = int(k_kept)
    if singletons_removed is not None:
        md["oversample_singletons_removed"] = int(singletons_removed)
    if n_eas_final is not None:
        md["n_eas_final"] = int(n_eas_final)
    return md



def generate_negative_class(
    background_name,
    target_n,
    profiles,
    rng,
    n_eas_final,
    eas_oversample_diploids_without,
    bundle_id_start=0,
    singleton_mode="derived",
):
    genotype_list = []
    metadata_list = []
    attempts = 0
    next_bundle_id = bundle_id_start

    prof = dict(profiles[background_name])
    prof["_profile_name"] = background_name

    while len(genotype_list) < target_n:
        attempts += 1
        bundle_id = next_bundle_id
        next_bundle_id += 1

        nuisance_bundle = draw_nuisance_bundle(prof, rng)

        ts, genotype_matrix, sim_info = simulate_one_instance(
            low_ghost=0.0,
            high_ghost=0.0,
            eas_diploids=eas_oversample_diploids_without,
            rng=rng,
            profile=prof,
            nuisance_bundle=nuisance_bundle,
        )

        try:
            new_mat, singletons_removed = downsample_no_ghost_eas_fixed(
                ts,
                genotype_matrix,
                n_eas_final=n_eas_final,
                rng=rng,
                singleton_mode=singleton_mode,
            )
        except ValueError:
            continue

        ghost_tract_count, num_haps_with_ghost, denisovan_has_ghost = analyze_ghost_introgression(ts)
        if ghost_tract_count != 0 or num_haps_with_ghost != 0:
            continue

        proxy = compute_archaic_like_proxies(
            new_mat,
            final_human_diploids=n_eas_final,
        )

        genotype_list.append(new_mat)
        metadata_list.append(
            _build_metadata(
                sim_info, proxy, 0, ghost_tract_count, num_haps_with_ghost, denisovan_has_ghost,
                bundle_id=bundle_id,
                pair_role="negative_unpaired",
                singletons_removed=singletons_removed,
                singleton_mode=singleton_mode,
                n_eas_final=n_eas_final,
            )
        )

    return genotype_list, metadata_list, attempts, next_bundle_id



def generate_positive_class(
    background_name,
    target_n,
    profiles,
    rng,
    n_eas_final,
    eas_oversample_diploids_with,
    num_with_tract,
    low_ghost_with,
    high_ghost_with,
    bundle_id_start=0,
    singleton_mode="derived",
):
    genotype_list = []
    metadata_list = []
    attempts = 0
    next_bundle_id = bundle_id_start

    prof = dict(profiles[background_name])
    prof["_profile_name"] = background_name

    while len(genotype_list) < target_n:
        attempts += 1
        bundle_id = next_bundle_id
        next_bundle_id += 1

        nuisance_bundle = draw_nuisance_bundle(prof, rng)

        ts, genotype_matrix, sim_info = simulate_one_instance(
            low_ghost=low_ghost_with,
            high_ghost=high_ghost_with,
            eas_diploids=eas_oversample_diploids_with,
            rng=rng,
            profile=prof,
            nuisance_bundle=nuisance_bundle,
        )

        try:
            new_mat, k_kept, singletons_removed = downsample_eas_pairs_atleast(
                ts,
                genotype_matrix,
                n_eas_final=n_eas_final,
                min_carrier_haplotypes=num_with_tract,
                rng=rng,
                singleton_mode=singleton_mode,
            )
        except ValueError:
            continue

        ghost_tract_count, num_haps_with_ghost, denisovan_has_ghost = analyze_ghost_introgression(ts)
        proxy = compute_archaic_like_proxies(
            new_mat,
            final_human_diploids=n_eas_final,
        )

        genotype_list.append(new_mat)
        metadata_list.append(
            _build_metadata(
                sim_info, proxy, 1, ghost_tract_count, num_haps_with_ghost, denisovan_has_ghost,
                bundle_id=bundle_id,
                pair_role="positive_unpaired",
                k_kept=k_kept,
                singletons_removed=singletons_removed,
                singleton_mode=singleton_mode,
                n_eas_final=n_eas_final,
            )
        )

    return genotype_list, metadata_list, attempts, next_bundle_id



def generate_paired_class_examples(
    background_name,
    target_pairs,
    profiles,
    rng,
    n_eas_final,
    eas_oversample_diploids_with,
    eas_oversample_diploids_without,
    num_with_tract,
    low_ghost_with,
    high_ghost_with,
    bundle_id_start=0,
    singleton_mode="derived",
):
    """Attempt both labels using a shared nuisance draw and attempt-level bundle ID.

    Each label is accepted independently until its quota is filled. Some bundle IDs
    therefore have one retained row. Pair-ranking later uses IDs with both roles."""
    pos_genotypes = []
    pos_meta = []
    neg_genotypes = []
    neg_meta = []
    attempts = 0
    next_bundle_id = bundle_id_start

    prof = dict(profiles[background_name])
    prof["_profile_name"] = background_name

    while len(pos_genotypes) < target_pairs or len(neg_genotypes) < target_pairs:
        attempts += 1
        bundle_id = next_bundle_id
        next_bundle_id += 1
        nuisance_bundle = draw_nuisance_bundle(prof, rng)

        if len(neg_genotypes) < target_pairs:
            ts_neg, genotype_matrix_neg, sim_info_neg = simulate_one_instance(
                low_ghost=0.0,
                high_ghost=0.0,
                eas_diploids=eas_oversample_diploids_without,
                rng=rng,
                profile=prof,
                nuisance_bundle=nuisance_bundle,
            )
            try:
                new_mat_neg, neg_singletons_removed = downsample_no_ghost_eas_fixed(
                    ts_neg,
                    genotype_matrix_neg,
                    n_eas_final=n_eas_final,
                    rng=rng,
                    singleton_mode=singleton_mode,
                )
                neg_ghost_tract_count, neg_num_haps_with_ghost, neg_denisovan_has_ghost = analyze_ghost_introgression(ts_neg)
                if neg_ghost_tract_count == 0 and neg_num_haps_with_ghost == 0:
                    proxy_neg = compute_archaic_like_proxies(
                        new_mat_neg,
                        final_human_diploids=n_eas_final,
                    )
                    neg_genotypes.append(new_mat_neg)
                    neg_meta.append(
                        _build_metadata(
                            sim_info_neg, proxy_neg, 0,
                            neg_ghost_tract_count, neg_num_haps_with_ghost, neg_denisovan_has_ghost,
                            bundle_id=bundle_id,
                            pair_role="negative_paired",
                            singletons_removed=neg_singletons_removed,
                            singleton_mode=singleton_mode,
                            n_eas_final=n_eas_final,
                        )
                    )
            except ValueError:
                pass

        if len(pos_genotypes) < target_pairs:
            ts_pos, genotype_matrix_pos, sim_info_pos = simulate_one_instance(
                low_ghost=low_ghost_with,
                high_ghost=high_ghost_with,
                eas_diploids=eas_oversample_diploids_with,
                rng=rng,
                profile=prof,
                nuisance_bundle=nuisance_bundle,
            )
            try:
                new_mat_pos, k_kept, pos_singletons_removed = downsample_eas_pairs_atleast(
                    ts_pos,
                    genotype_matrix_pos,
                    n_eas_final=n_eas_final,
                    min_carrier_haplotypes=num_with_tract,
                    rng=rng,
                    singleton_mode=singleton_mode,
                )
                pos_ghost_tract_count, pos_num_haps_with_ghost, pos_denisovan_has_ghost = analyze_ghost_introgression(ts_pos)
                proxy_pos = compute_archaic_like_proxies(
                    new_mat_pos,
                    final_human_diploids=n_eas_final,
                )
                pos_genotypes.append(new_mat_pos)
                pos_meta.append(
                    _build_metadata(
                        sim_info_pos, proxy_pos, 1,
                        pos_ghost_tract_count, pos_num_haps_with_ghost, pos_denisovan_has_ghost,
                        bundle_id=bundle_id,
                        pair_role="positive_paired",
                        k_kept=k_kept,
                        singletons_removed=pos_singletons_removed,
                        singleton_mode=singleton_mode,
                        n_eas_final=n_eas_final,
                    )
                )
                accepted_count = len(pos_genotypes)
                if _should_log_accept(accepted_count):
                    print(_format_accept_line(
                        prefix="Accepted paired positive",
                        accepted_count=accepted_count,
                        target_n=target_pairs,
                        background_name=background_name,
                        bundle_id=bundle_id,
                        sim_info=sim_info_pos,
                        ghost_tract_count=pos_ghost_tract_count,
                        num_haps_with_ghost=pos_num_haps_with_ghost,
                        denisovan_has_ghost=pos_denisovan_has_ghost,
                        proxy=proxy_pos,
                        k_kept=k_kept,
                        pair_role="positive_paired",
                        singletons_removed=pos_singletons_removed,
                    ))
            except ValueError:
                pass

    return pos_genotypes, pos_meta, neg_genotypes, neg_meta, attempts, next_bundle_id

# ----------------------------
# Main driver
# ----------------------------

def simulate_and_save(
    target_with,
    target_without,
    target_robust,
    num_with_tract,
    n_eas_final=7,
    eas_oversample_diploids_with=50,
    eas_oversample_diploids_without=50,
    low_ghost_with=0.02,
    high_ghost_with=0.10,
    with_profile_weights=None,
    singleton_mode="derived",
):
    """Generate accepted genotype matrices and write four NPZ archives for one task.

    Genotype archives store arr_0, arr_1, and subsequent arrays. Each corresponding
    metadata archive stores a same-length metadata array in that order."""
    if n_eas_final != 7:
        raise ValueError("For current downstream layout, n_eas_final must equal 7.")
    if eas_oversample_diploids_with < n_eas_final:
        raise ValueError("eas_oversample_diploids_with must be >= n_eas_final.")
    if eas_oversample_diploids_without < n_eas_final:
        raise ValueError("eas_oversample_diploids_without must be >= n_eas_final.")

    rng = np.random.default_rng()
    profiles = get_background_profiles()

    negative_targets = {
        "base_background": target_without,
        "den_rich_background": target_robust,
        "ned_rich_background": target_robust,
        "dual_archaic_background": target_robust,
        "bottlenecked_background": target_robust,
        "mut_hot_background": target_robust,
        "compound_hard_background": target_robust,
        "humanpoly_muthot_background": target_robust,
        "humanpoly_dual_archaic_background": target_robust,
    }

    if with_profile_weights is None:
        positive_targets = _allocate_counts_from_weights(target_with, negative_targets)
    else:
        bad = [k for k in with_profile_weights if k not in profiles]
        if bad:
            raise ValueError(
                "with_profile_weights must use background profile names from get_background_profiles(). "
                f"Invalid keys: {bad}"
            )
        positive_targets = _allocate_counts_from_weights(target_with, with_profile_weights)

    simulation_counter = 0
    next_bundle_id = 1

    with_genotype_list = []
    without_genotype_list = []
    with_meta = []
    without_meta = []

    paired_targets = {
        k: min(positive_targets.get(k, 0), negative_targets.get(k, 0))
        for k in profiles
    }

    remaining_positive_targets = dict(positive_targets)
    remaining_negative_targets = dict(negative_targets)

    for background_name in profiles:
        n_pairs = paired_targets.get(background_name, 0)
        if n_pairs <= 0:
            continue

        print(f"Generating paired examples for {background_name}: target_pairs={n_pairs}")
        pos_g, pos_m, neg_g, neg_m, attempts, next_bundle_id = generate_paired_class_examples(
            background_name=background_name,
            target_pairs=n_pairs,
            profiles=profiles,
            rng=rng,
            n_eas_final=n_eas_final,
            eas_oversample_diploids_with=eas_oversample_diploids_with,
            eas_oversample_diploids_without=eas_oversample_diploids_without,
            num_with_tract=num_with_tract,
            low_ghost_with=low_ghost_with,
            high_ghost_with=high_ghost_with,
            bundle_id_start=next_bundle_id,
            singleton_mode=singleton_mode,
        )
        with_genotype_list.extend(pos_g)
        with_meta.extend(pos_m)
        without_genotype_list.extend(neg_g)
        without_meta.extend(neg_m)

        remaining_positive_targets[background_name] -= len(pos_g)
        remaining_negative_targets[background_name] -= len(neg_g)
        simulation_counter += attempts

        print(
            f"Finished paired {background_name}: "
            f"accepted_pos={len(pos_g)}, accepted_neg={len(neg_g)}, attempts={attempts}"
        )

    for background_name in profiles:
        target_n = remaining_negative_targets.get(background_name, 0)
        if target_n <= 0:
            continue

        print(f"Generating extra negatives for {background_name}: target={target_n}")
        g_list, m_list, attempts, next_bundle_id = generate_negative_class(
            background_name=background_name,
            target_n=target_n,
            profiles=profiles,
            rng=rng,
            n_eas_final=n_eas_final,
            eas_oversample_diploids_without=eas_oversample_diploids_without,
            bundle_id_start=next_bundle_id,
            singleton_mode=singleton_mode,
        )
        without_genotype_list.extend(g_list)
        without_meta.extend(m_list)
        simulation_counter += attempts
        print(f"Finished extra negatives {background_name}: accepted={len(g_list)}, attempts={attempts}")

    for background_name in profiles:
        target_n = remaining_positive_targets.get(background_name, 0)
        if target_n <= 0:
            continue

        print(f"Generating extra positives for {background_name}: target={target_n}")
        g_list, m_list, attempts, next_bundle_id = generate_positive_class(
            background_name=background_name,
            target_n=target_n,
            profiles=profiles,
            rng=rng,
            n_eas_final=n_eas_final,
            eas_oversample_diploids_with=eas_oversample_diploids_with,
            num_with_tract=num_with_tract,
            low_ghost_with=low_ghost_with,
            high_ghost_with=high_ghost_with,
            bundle_id_start=next_bundle_id,
            singleton_mode=singleton_mode,
        )
        with_genotype_list.extend(g_list)
        with_meta.extend(m_list)
        simulation_counter += attempts
        print(f"Finished extra positives {background_name}: accepted={len(g_list)}, attempts={attempts}")

    GENOTYPE_DIR.mkdir(parents=True, exist_ok=True)
    out_with = GENOTYPE_DIR / f"with_ghost_genotypes{data_suff}{array_task_id}.npz"
    out_without = GENOTYPE_DIR / f"without_ghost_genotypes{data_suff}{array_task_id}.npz"
    out_with_meta = GENOTYPE_DIR / f"with_ghost_metadata{data_suff}{array_task_id}.npz"
    out_without_meta = GENOTYPE_DIR / f"without_ghost_metadata{data_suff}{array_task_id}.npz"

    np.savez(out_with, *with_genotype_list)
    np.savez(out_without, *without_genotype_list)
    np.savez(out_with_meta, metadata=np.array(with_meta, dtype=object))
    np.savez(out_without_meta, metadata=np.array(without_meta, dtype=object))

    print(
        f"Finished: total sims run={simulation_counter}, "
        f"with ghost={len(with_genotype_list)}, without ghost={len(without_genotype_list)}"
    )
    print(f"Saved: {out_with}")
    print(f"Saved: {out_without}")
    print(f"Saved: {out_with_meta}")
    print(f"Saved: {out_without_meta}")


if __name__ == "__main__":
    simulate_and_save(
        target_with=TARGET_WITH,
        target_without=TARGET_WITHOUT,
        target_robust=TARGET_ROBUST,
        num_with_tract=1,
        n_eas_final=7,
        eas_oversample_diploids_with=20,
        eas_oversample_diploids_without=20,
        low_ghost_with=0.02,
        high_ghost_with=0.10,
        singleton_mode="minor",
    )
