"""Extract the model inputs from a prepared, chromosome-sorted GRCh37 panel."""

from itertools import groupby

import numpy as np

from ._statistics import extract_statistics_both
from .predict import FEATURES, ZERO_IF_ABSENT

WINDOW_SIZE = 50_000
# GRCh37 primary autosomes, as in the study's extraction scripts.
LENGTHS = (249250621, 243199373, 198022430, 191154276, 180915260, 171115067,
           159138663, 146364022, 141213431, 135534747, 135006516, 133851895,
           115169878, 107349540, 102531392, 90354753, 81195210, 78077248,
           59128983, 63025520, 48129895, 51304566)
AUTOSOMES = {str(i): i for i in range(1, 23)}
BASES = {"A", "C", "G", "T"}


def _variants(reader, den_index):
    sample_order = [i for i in range(len(reader.samples)) if i != den_index] + [den_index]
    previous = (0, 0)
    for record in reader:
        chrom = AUTOSOMES.get(record.CHROM.removeprefix("chr"))
        if chrom is None:
            raise ValueError(f"Expected a GRCh37 autosome, found {record.CHROM!r}.")
        pos = record.POS
        if (chrom, pos) < previous or not 1 <= pos <= LENGTHS[chrom - 1]:
            raise ValueError(f"Unexpected position or record order at {chrom}:{pos}.")
        previous = chrom, pos
        if len(record.ALT) != 1 or record.REF not in BASES or record.ALT[0] not in BASES:
            raise ValueError(f"Expected a biallelic SNP at {chrom}:{pos}.")
        calls = record.genotypes
        alleles = []
        for i in sample_order:
            gt = calls[i]
            if len(gt) != 3 or gt[0] not in (0, 1) or gt[1] not in (0, 1):
                raise ValueError(f"Expected complete diploid genotypes at {chrom}:{pos}.")
            if i != den_index and gt[0] != gt[1] and not gt[2]:
                raise ValueError(f"Unphased human heterozygote at {chrom}:{pos}.")
            alleles.extend(gt[:2])
        start = (pos - 1) // WINDOW_SIZE * WINDOW_SIZE + 1
        yield (chrom, start), [pos, *alleles]


def window_features(path, denisovan="Denisova"):
    """Yield one feature dict per window with human polymorphism."""
    try:
        from cyvcf2 import VCF
    except ImportError as exc:
        raise ImportError('Install VCF support with: python -m pip install ".[vcf]"') from exc
    reader = VCF(str(path))
    try:
        if denisovan not in reader.samples or len(reader.samples) < 2:
            raise ValueError("The VCF must contain the named Denisovan and phased target humans.")
        records = _variants(reader, reader.samples.index(denisovan))
        for (chrom, start), group in groupby(records, key=lambda row: row[0]):
            matrix = np.asarray([row for _, row in group], dtype=int)
            stats = extract_statistics_both(matrix)
            if stats is None:
                continue
            end = min(start + WINDOW_SIZE - 1, LENGTHS[chrom - 1])
            fields = {f"human_{k}": v for k, v in stats["human_only_stats"].items()}
            fields.update({f"combined_{k}": v for k, v in stats["combined_stats"].items()})
            fields.update(dict.fromkeys(ZERO_IF_ABSENT, 0.0))
            yield {"chromosome": chrom, "window": f"{start}_{end}",
                   **{name: fields[name] for name in FEATURES}}
    finally:
        reader.close()
