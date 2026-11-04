"""Window statistics shared by the Oceanian, Han and Tibetan empirical pipelines."""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew


def tajimas_d_from_counts(k_counts, n_chrom):
    """
    Compute Tajima's D from allele counts.
    """
    seg = (k_counts > 0) & (k_counts < n_chrom)
    S = np.sum(seg)

    if n_chrom < 2 or S == 0:
        return 0.0

    a1 = np.sum(1.0 / np.arange(1, n_chrom))
    a2 = np.sum(1.0 / (np.arange(1, n_chrom) ** 2))
    b1 = (n_chrom + 1) / (3 * (n_chrom - 1))
    b2 = 2 * (n_chrom**2 + n_chrom + 3) / (9 * n_chrom * (n_chrom - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n_chrom + 2) / (a1 * n_chrom) + a2 / (a1**2)
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)

    k = k_counts.astype(float)
    pi_site = 2.0 * k * (n_chrom - k) / (n_chrom * (n_chrom - 1))
    pi_total = np.sum(pi_site)
    theta_w = S / a1
    var_d = e1 * S + e2 * S * (S - 1)

    return float((pi_total - theta_w) / np.sqrt(var_d)) if var_d > 0 else 0.0


def _gini(x):
    """
    Compute Gini coefficient of a 1D array.
    """
    x = np.asarray(x, float)

    if x.size == 0:
        return 0.0

    mean_x = x.mean()
    if mean_x == 0:
        return 0.0

    x_sorted = np.sort(x)
    n = x_sorted.size

    g = (
        2.0
        * np.sum((np.arange(1, n + 1) * x_sorted))
        / (n * np.sum(x_sorted))
    ) - (n + 1) / n

    return float(g)


def _entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()

    if total <= 0:
        return 0.0

    p = counts / total
    p = p[p > 0]

    return float(-np.sum(p * np.log(p)))


def _safe_moments(x):
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(np.mean(x)),
        float(np.var(x)),
        float(np.nan_to_num(skew(x), nan=0.0)),
        float(np.nan_to_num(kurtosis(x), nan=0.0)),
    )


def _safe_quantile(x, q):
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return 0.0

    return float(np.quantile(x, q))


def _longest_run_info(positions, binary_vec):
    """
    Compute run summaries for a binary vector in site order.
    """
    positions = np.asarray(positions)
    binary_vec = np.asarray(binary_vec).astype(int)

    if binary_vec.size == 0 or np.sum(binary_vec) == 0:
        return 0, 0.0, 0, 0.0

    one_idx = np.where(binary_vec == 1)[0]

    if one_idx.size == 0:
        return 0, 0.0, 0, 0.0

    runs = []
    start = one_idx[0]
    prev = one_idx[0]

    for idx in one_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = idx
            prev = idx

    runs.append((start, prev))

    run_sites = [b - a + 1 for a, b in runs]
    run_bp = []

    for a, b in runs:
        if a == b:
            run_bp.append(1.0)
        else:
            run_bp.append(float(positions[b] - positions[a]))

    return (
        int(np.max(run_sites)),
        float(np.max(run_bp)),
        int(len(runs)),
        float(np.mean(run_sites)),
    )


def _compute_denisovan_conditioned_hap_features(
    human_positions,
    human_geno,
    den_haps,
    total_length_bp=50000,
):
    """
    Denisovan-conditioned features on human haplotypes.
    """
    stats = {}

    n_sites = human_geno.shape[0]
    n_haps = human_geno.shape[1]

    zero_keys = [
        "den_shared_sites",
        "den_shared_sites_h1",
        "den_shared_sites_h2",
        "den_shared_sites_h3_4",
        "den_shared_sites_h5_7",
        "den_shared_sites_h8plus",
        "mean_human_count_given_den_shared",
        "var_human_count_given_den_shared",
        "min_hap_den_distance",
        "mean_hap_den_distance",
        "var_hap_den_distance",
        "q10_hap_den_distance",
        "top1_closest_hap_den",
        "top2_closest_hap_den",
        "top3_closest_hap_den",
        "mean_den_shared_per_hap",
        "var_den_shared_per_hap",
        "skew_den_shared_per_hap",
        "kurt_den_shared_per_hap",
        "max_den_shared_per_hap",
        "gini_den_shared_per_hap",
        "entropy_den_shared_per_hap",
        "den_shared_nn_mean",
        "den_shared_nn_var",
        "den_shared_nn_skew",
        "den_shared_nn_kurt",
        "den_shared_max_sites_in_5kb",
        "den_shared_num_occupied_5kb_windows",
        "den_shared_longest_run_bp_all",
        "den_shared_longest_run_sites_all",
        "mean_longest_run_bp_per_hap",
        "var_longest_run_bp_per_hap",
        "mean_longest_run_sites_per_hap",
        "var_longest_run_sites_per_hap",
        "max_longest_run_bp_per_hap",
        "max_longest_run_sites_per_hap",
        "mean_num_runs_per_hap",
        "var_num_runs_per_hap",
        "mean_run_size_per_hap",
        "var_run_size_per_hap",
    ]

    if n_sites == 0 or den_haps is None or den_haps.shape[0] != n_sites:
        for key in zero_keys:
            stats[key] = 0.0
        return stats

    den_present = np.any(den_haps == 1, axis=1)
    human_counts = np.sum(human_geno, axis=1)

    den_shared_mask = den_present & (human_counts > 0)
    den_shared_counts = human_counts[den_shared_mask]

    stats["den_shared_sites"] = int(np.sum(den_shared_mask))
    stats["den_shared_sites_h1"] = int(np.sum(den_shared_counts == 1))
    stats["den_shared_sites_h2"] = int(np.sum(den_shared_counts == 2))
    stats["den_shared_sites_h3_4"] = int(
        np.sum((den_shared_counts >= 3) & (den_shared_counts <= 4))
    )
    stats["den_shared_sites_h5_7"] = int(
        np.sum((den_shared_counts >= 5) & (den_shared_counts <= 7))
    )
    stats["den_shared_sites_h8plus"] = int(np.sum(den_shared_counts >= 8))

    stats["mean_human_count_given_den_shared"] = (
        float(np.mean(den_shared_counts)) if den_shared_counts.size > 0 else 0.0
    )
    stats["var_human_count_given_den_shared"] = (
        float(np.var(den_shared_counts)) if den_shared_counts.size > 0 else 0.0
    )

    # Human haplotype to Denisovan affinity.
    dists = []
    for h in range(n_haps):
        hvec = human_geno[:, h]
        d1 = np.mean(hvec != den_haps[:, 0]) * n_sites
        d2 = np.mean(hvec != den_haps[:, 1]) * n_sites
        dists.append(min(d1, d2))

    dists = np.asarray(dists, dtype=float)

    stats["min_hap_den_distance"] = float(np.min(dists)) if dists.size > 0 else 0.0
    stats["mean_hap_den_distance"] = float(np.mean(dists)) if dists.size > 0 else 0.0
    stats["var_hap_den_distance"] = float(np.var(dists)) if dists.size > 0 else 0.0
    stats["q10_hap_den_distance"] = _safe_quantile(dists, 0.10)

    sorted_d = np.sort(dists)
    stats["top1_closest_hap_den"] = float(sorted_d[0]) if sorted_d.size >= 1 else 0.0
    stats["top2_closest_hap_den"] = float(sorted_d[1]) if sorted_d.size >= 2 else 0.0
    stats["top3_closest_hap_den"] = float(sorted_d[2]) if sorted_d.size >= 3 else 0.0

    # Concentration across human haplotypes.
    if np.sum(den_shared_mask) > 0:
        shared_per_hap = np.sum(human_geno[den_shared_mask, :], axis=0).astype(float)
    else:
        shared_per_hap = np.zeros(n_haps, dtype=float)

    m, v, s, k = _safe_moments(shared_per_hap)

    stats["mean_den_shared_per_hap"] = m
    stats["var_den_shared_per_hap"] = v
    stats["skew_den_shared_per_hap"] = s
    stats["kurt_den_shared_per_hap"] = k
    stats["max_den_shared_per_hap"] = (
        float(np.max(shared_per_hap)) if shared_per_hap.size > 0 else 0.0
    )
    stats["gini_den_shared_per_hap"] = _gini(shared_per_hap)
    stats["entropy_den_shared_per_hap"] = _entropy_from_counts(shared_per_hap)

    # Positional clustering of Denisovan-shared sites.
    den_shared_pos = np.sort(human_positions[den_shared_mask])

    if den_shared_pos.size >= 2:
        nn = np.diff(den_shared_pos)
        m, v, s, k = _safe_moments(nn)

        stats["den_shared_nn_mean"] = m
        stats["den_shared_nn_var"] = v
        stats["den_shared_nn_skew"] = s
        stats["den_shared_nn_kurt"] = k
    else:
        stats["den_shared_nn_mean"] = 0.0
        stats["den_shared_nn_var"] = 0.0
        stats["den_shared_nn_skew"] = 0.0
        stats["den_shared_nn_kurt"] = 0.0

    if den_shared_pos.size > 0:
        bins_5kb = (den_shared_pos // 5000).astype(int)
        uniq_bins, counts_bins = np.unique(bins_5kb, return_counts=True)

        stats["den_shared_max_sites_in_5kb"] = int(np.max(counts_bins))
        stats["den_shared_num_occupied_5kb_windows"] = int(uniq_bins.size)
    else:
        stats["den_shared_max_sites_in_5kb"] = 0
        stats["den_shared_num_occupied_5kb_windows"] = 0

    # Run / tractiness features.
    longest_bp = []
    longest_sites = []
    num_runs = []
    mean_run_sites = []

    if np.sum(den_shared_mask) > 0:
        den_shared_positions = human_positions[den_shared_mask]
        den_shared_human = human_geno[den_shared_mask, :]

        global_longest_bp = 0.0
        global_longest_sites = 0

        for h in range(n_haps):
            l_sites, l_bp, n_runs, mean_sites = _longest_run_info(
                den_shared_positions,
                den_shared_human[:, h],
            )

            longest_bp.append(l_bp)
            longest_sites.append(l_sites)
            num_runs.append(n_runs)
            mean_run_sites.append(mean_sites)

            if l_bp > global_longest_bp:
                global_longest_bp = l_bp
            if l_sites > global_longest_sites:
                global_longest_sites = l_sites

        stats["den_shared_longest_run_bp_all"] = float(global_longest_bp)
        stats["den_shared_longest_run_sites_all"] = int(global_longest_sites)

    else:
        longest_bp = [0.0] * n_haps
        longest_sites = [0.0] * n_haps
        num_runs = [0.0] * n_haps
        mean_run_sites = [0.0] * n_haps

        stats["den_shared_longest_run_bp_all"] = 0.0
        stats["den_shared_longest_run_sites_all"] = 0

    stats["mean_longest_run_bp_per_hap"] = float(np.mean(longest_bp))
    stats["var_longest_run_bp_per_hap"] = float(np.var(longest_bp))
    stats["mean_longest_run_sites_per_hap"] = float(np.mean(longest_sites))
    stats["var_longest_run_sites_per_hap"] = float(np.var(longest_sites))
    stats["max_longest_run_bp_per_hap"] = (
        float(np.max(longest_bp)) if len(longest_bp) > 0 else 0.0
    )
    stats["max_longest_run_sites_per_hap"] = (
        float(np.max(longest_sites)) if len(longest_sites) > 0 else 0.0
    )
    stats["mean_num_runs_per_hap"] = float(np.mean(num_runs))
    stats["var_num_runs_per_hap"] = float(np.var(num_runs))
    stats["mean_run_size_per_hap"] = float(np.mean(mean_run_sites))
    stats["var_run_size_per_hap"] = float(np.var(mean_run_sites))

    return stats


def extract_statistics_both(
    genotypes,
    total_length_bp=50000,
    window_size_bp=10000,
    top_within_sample=5,
    top_num_diff=5,
):
    """
    Run both human-only haploid and combined diploid analyses on one genotype matrix.
    """
    positions = genotypes[:, 0]
    genotype_data = genotypes[:, 1:]

    def reencode_minor_major(geno):
        geno = geno.astype(np.int16)
        reencoded = np.empty_like(geno)

        for i in range(geno.shape[0]):
            site = geno[i]
            derived_count = np.sum(site)

            if derived_count > geno.shape[1] / 2:
                reencoded[i] = np.where(site == 1, 0, 1)
            else:
                reencoded[i] = site

        return reencoded

    genotype_data = reencode_minor_major(genotype_data)

    has_one = np.any(genotype_data == 1, axis=1)
    has_zero = np.any(genotype_data == 0, axis=1)
    mask = has_one & has_zero

    genotype_data = genotype_data[mask]
    positions = positions[mask]

    raw_human = genotype_data[:, :-2]
    raw_den = genotype_data[:, -2:]

    has_one_hum = np.any(raw_human == 1, axis=1)
    has_zero_hum = np.any(raw_human == 0, axis=1)
    nonzero_mask = has_one_hum & has_zero_hum

    human_positions = positions[nonzero_mask]
    human_genotypes = raw_human[nonzero_mask, :]
    den_at_human_sites = raw_den[nonzero_mask, :]

    human_stats = _compute_haploid_stats(
        human_positions,
        human_genotypes,
        total_length_bp,
        window_size_bp,
        top_within_sample,
        top_num_diff,
        den_haps=den_at_human_sites,
    )

    genotype_data = genotype_data[nonzero_mask, :]
    positions = positions[nonzero_mask]

    if genotype_data.shape[0] == 0:
        return None

    num_human_haps = genotype_data.shape[1] - 2

    if num_human_haps % 2 != 0:
        raise ValueError(
            "The number of human haplotypes, excluding Denisovan, must be even."
        )

    human_diploids = np.sum(
        genotype_data[:, :num_human_haps].reshape(genotype_data.shape[0], -1, 2),
        axis=2,
    )

    den_dip = np.sum(genotype_data[:, -2:], axis=1, keepdims=True)
    combined_diploids = np.hstack([human_diploids, den_dip])

    combined_stats = _compute_diploid_stats(
        positions,
        combined_diploids,
        total_length_bp,
        window_size_bp,
        top_within_sample,
        top_num_diff,
    )

    return {
        "human_only_stats": human_stats,
        "combined_stats": combined_stats,
    }


def _compute_haploid_stats(
    positions,
    geno,
    total_length_bp,
    window_size_bp,
    top_within_sample,
    top_num_diff,
    den_haps=None,
):
    """
    Compute summary statistics for haploid human genotype data.
    """
    stats = {}

    total_sites, num_haplotypes = geno.shape

    stats["num_sites"] = total_sites
    stats["num_haplotypes"] = num_haplotypes

    if total_sites == 0:
        stats["mean_allele_frequency"] = 0.0
        stats["var_allele_frequency"] = 0.0
        stats["heterozygosity"] = 0.0
        stats["prop_segregating_sites"] = 0.0
        stats["site_frequency_spectrum"] = [0] * (num_haplotypes + 1)
        stats["tajimas_d"] = 0.0
        stats["num_singletons"] = 0
        stats["mean_pairwise_diff"] = 0.0
        stats["var_pairwise_diff"] = 0.0
        stats["pairwise_diff_skewness"] = 0.0
        stats["pairwise_diff_kurtosis"] = 0.0

        for i in range(1, top_num_diff + 1):
            stats[f"top_pairwise_diff_{i}"] = 0.0

        if den_haps is not None:
            den_stats = _compute_denisovan_conditioned_hap_features(
                positions,
                geno,
                den_haps,
                total_length_bp=total_length_bp,
            )
            stats.update(den_stats)

        return stats

    allele_freq = np.mean(geno, axis=1)

    stats["mean_allele_frequency"] = float(np.mean(allele_freq))
    stats["var_allele_frequency"] = float(np.var(allele_freq))

    heterozygosity = 2 * allele_freq * (1 - allele_freq)
    stats["heterozygosity"] = float(np.mean(heterozygosity))

    segregating = np.logical_and(np.any(geno > 0, axis=1), np.any(geno == 0, axis=1))
    S_total = np.sum(segregating)
    stats["prop_segregating_sites"] = (
        float(S_total) / total_sites if total_sites > 0 else 0.0
    )

    derived_counts = np.sum(geno, axis=1)
    sfs = np.bincount(derived_counts.astype(np.int64), minlength=num_haplotypes + 1)

    stats["site_frequency_spectrum"] = sfs.tolist()
    stats["tajimas_d"] = tajimas_d_from_counts(derived_counts, n_chrom=geno.shape[1])
    stats["num_singletons"] = int(sfs[1]) if len(sfs) > 1 else 0

    pdist_values = pdist(geno.T, metric="hamming") * total_sites

    stats["mean_pairwise_diff"] = (
        float(pdist_values.mean()) if pdist_values.size > 0 else 0.0
    )
    stats["var_pairwise_diff"] = (
        float(pdist_values.var()) if pdist_values.size > 0 else 0.0
    )
    stats["pairwise_diff_skewness"] = (
        float(np.nan_to_num(skew(pdist_values), nan=0.0))
        if pdist_values.size > 0
        else 0.0
    )
    stats["pairwise_diff_kurtosis"] = (
        float(np.nan_to_num(kurtosis(pdist_values), nan=0.0))
        if pdist_values.size > 0
        else 0.0
    )

    top_diffs = np.sort(pdist_values)[-top_num_diff:] if pdist_values.size > 0 else np.array([])

    for i in range(1, top_num_diff + 1):
        stats[f"top_pairwise_diff_{i}"] = (
            float(top_diffs[-i]) if top_diffs.size >= i else 0.0
        )

    if den_haps is not None:
        den_stats = _compute_denisovan_conditioned_hap_features(
            positions,
            geno,
            den_haps,
            total_length_bp=total_length_bp,
        )
        stats.update(den_stats)

    return stats


def _compute_diploid_stats(
    positions,
    geno,
    total_length_bp,
    window_size_bp,
    top_within_sample,
    top_num_diff,
):
    """
    Compute summary statistics for diploid genotype data.
    """
    stats = {}

    total_sites, num_indiv = geno.shape

    stats["total_sites"] = total_sites
    stats["num_individuals"] = num_indiv
    stats["num_sites"] = total_sites

    if total_sites == 0:
        stats["mean_allele_frequency"] = 0.0
        stats["var_allele_frequency"] = 0.0
        stats["expected_heterozygosity"] = 0.0
        stats["observed_heterozygosity"] = 0.0
        stats["prop_segregating_sites"] = 0.0
        stats["site_frequency_spectrum"] = [0] * (2 * num_indiv + 1)
        stats["num_singletons"] = 0
        stats["tajimas_d"] = 0.0
        stats["mean_pairwise_diff"] = 0.0
        stats["var_pairwise_diff"] = 0.0
        stats["pairwise_diff_skewness"] = 0.0
        stats["pairwise_diff_kurtosis"] = 0.0

        for i in range(1, top_num_diff + 1):
            stats[f"top_pairwise_diff_{i}"] = 0.0

        stats["mean_within_individual_window_diff"] = 0.0
        stats["var_within_individual_window_diff"] = 0.0
        stats["within_individual_window_diff_skewness"] = 0.0
        stats["within_individual_window_diff_kurtosis"] = 0.0

        for i in range(1, top_within_sample + 1):
            stats[f"top_within_individual_window_diff_{i}"] = 0.0

        return stats

    allele_freq = np.sum(geno, axis=1) / (2 * num_indiv)

    stats["mean_allele_frequency"] = float(np.mean(allele_freq))
    stats["var_allele_frequency"] = float(np.var(allele_freq))

    exp_het = 2 * allele_freq * (1 - allele_freq)
    obs_het = np.mean(geno == 1, axis=1)

    stats["expected_heterozygosity"] = float(np.mean(exp_het))
    stats["observed_heterozygosity"] = float(np.mean(obs_het))

    allele_counts = np.sum(geno, axis=1)

    S_total = np.sum(np.logical_and(allele_counts > 0, allele_counts < 2 * num_indiv))
    stats["prop_segregating_sites"] = (
        float(S_total) / total_sites if total_sites > 0 else 0.0
    )

    sfs = np.bincount(allele_counts.astype(np.int64), minlength=2 * num_indiv + 1)

    stats["site_frequency_spectrum"] = sfs.tolist()
    stats["num_singletons"] = int(sfs[1]) if len(sfs) > 1 else 0

    allele_counts = np.sum(geno, axis=1).astype(int)
    stats["tajimas_d"] = tajimas_d_from_counts(
        allele_counts,
        n_chrom=2 * geno.shape[1],
    )

    pdist_values = pdist(geno.T, metric="cityblock")

    stats["mean_pairwise_diff"] = (
        float(pdist_values.mean()) if pdist_values.size > 0 else 0.0
    )
    stats["var_pairwise_diff"] = (
        float(pdist_values.var()) if pdist_values.size > 0 else 0.0
    )
    stats["pairwise_diff_skewness"] = (
        float(np.nan_to_num(skew(pdist_values), nan=0.0))
        if pdist_values.size > 0
        else 0.0
    )
    stats["pairwise_diff_kurtosis"] = (
        float(np.nan_to_num(kurtosis(pdist_values), nan=0.0))
        if pdist_values.size > 0
        else 0.0
    )

    top_diffs = np.sort(pdist_values)[-top_num_diff:] if pdist_values.size > 0 else np.array([])

    for i in range(1, top_num_diff + 1):
        stats[f"top_pairwise_diff_{i}"] = (
            float(top_diffs[-i]) if top_diffs.size >= i else 0.0
        )

    # Within-individual window differences.
    window_start = np.floor(positions.min() / total_length_bp) * total_length_bp
    window_end = window_start + total_length_bp
    bin_boundaries = np.arange(window_start, window_end + window_size_bp, window_size_bp)

    within_diff = []

    for indiv in range(num_indiv):
        indiv_data = geno[:, indiv]
        window_hets = []

        for i in range(len(bin_boundaries) - 1):
            start = bin_boundaries[i]
            end = bin_boundaries[i + 1]

            idx = np.where((positions >= start) & (positions < end))[0]
            window = indiv_data[idx]

            if window.size > 0:
                window_hets.append(np.mean(window == 1))

        for i in range(len(window_hets)):
            for j in range(i + 1, len(window_hets)):
                within_diff.append(abs(window_hets[i] - window_hets[j]))

    if within_diff:
        within_diff = np.array(within_diff)

        stats["mean_within_individual_window_diff"] = float(within_diff.mean())
        stats["var_within_individual_window_diff"] = float(within_diff.var())
        stats["within_individual_window_diff_skewness"] = float(
            np.nan_to_num(skew(within_diff), nan=0.0)
        )
        stats["within_individual_window_diff_kurtosis"] = float(
            np.nan_to_num(kurtosis(within_diff), nan=0.0)
        )

        top_within = np.sort(within_diff)[-top_within_sample:]

        for i in range(1, top_within_sample + 1):
            stats[f"top_within_individual_window_diff_{i}"] = (
                float(top_within[-i]) if top_within.size >= i else 0.0
            )

    else:
        stats["mean_within_individual_window_diff"] = 0.0
        stats["var_within_individual_window_diff"] = 0.0
        stats["within_individual_window_diff_skewness"] = 0.0
        stats["within_individual_window_diff_kurtosis"] = 0.0

        for i in range(1, top_within_sample + 1):
            stats[f"top_within_individual_window_diff_{i}"] = 0.0

    return stats
