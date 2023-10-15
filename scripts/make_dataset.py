import math
import subprocess
import tempfile
from pathlib import Path

import bioframe as bf
import pandas as pd
import numpy as np
from tqdm import tqdm

from gpn.data import filter_length, filter_defined, get_balanced_intervals, Genome, load_table, make_windows, get_seq


# assemblies_path = "./analysis/arabidopsis/input/assembly_list/brassicales_annotated_filt.tsv"
assemblies_path = "./assemblies/mus_assemblies.tsv"
assemblies = pd.read_csv(assemblies_path, sep="\t", index_col=0)
assemblies["Assembly Name"] = assemblies["Assembly Name"].str.replace(" ", "_")



# train, val, test
splits = ["train", "validation", "test"]
split_proportions = [1, 0, 0]
# whitelist_validation_chroms= ["NC_003075.7"]    # Arabidopsis thaliana chr4
# whitelist_test_chroms = ["NC_003076.8"]         # Arabidopsis thaliana chr5
whitelist_validation_chroms= ["NC_000083.7"]    # mus 17
whitelist_test_chroms = ["NC_000084.7"]         # mus 18

chroms_subset = None
# chroms_subset = [
#     "NC_000067.7",  # train
#     "NC_000083.7",  # val
#     "NC_000084.7"   # test
# ]
samples_per_file = 1_000_000
window_size = 512
# stride = 256
stride = 64
step_size = window_size - stride
add_rc = False
balanced = False

subsample_to_target = False
target_assembly = "GCF_000001735.4"


# download data from ncbi, extract data from archive, rename and compress data agai

data_dir = Path("./data/mus")
data_dir.mkdir(parents=True, exist_ok=True)

genome_dir = data_dir / "genome"
genome_dir.mkdir(parents=True, exist_ok=True)
annotation_dir = data_dir / "annotation"
annotation_dir.mkdir(parents=True, exist_ok=True)
#
intervals_dir = data_dir / "intervals"
#
dataset_dir = data_dir / "dataset" / f"w_{window_size}_s_{step_size}_rc_{add_rc}_balanced_{balanced}"
merged_dataset_dir = data_dir / "merged_dataset" / f"w_{window_size}_s_{step_size}_rc_{add_rc}_balanced_{balanced}"


def get_assembly_genome_annotation_and_order(assembly: str, overwrite: bool = False):
    """ """
    with tempfile.TemporaryDirectory() as tmp_dir:
        #
        tmp_dir = Path(tmp_dir)
        tmp_genome_path = tmp_dir / "ncbi_dataset/data" / assembly / f"{assembly}_{row['Assembly Name']}_genomic.fna"
        tmp_annotation_path = tmp_dir / "ncbi_dataset/data" / assembly / "genomic.gff"
        #
        output_genome_path = genome_dir / f"{assembly}.fa.gz"
        output_annotation_path = annotation_dir / f"{assembly}.gff.gz"
        #
        if output_genome_path.exists() and output_annotation_path.exists() and not overwrite:
            print(f"assembly: {assembly} already present, skipping")
            return

        dl_cmd = f"datasets download genome accession {assembly} --include genome,gff3 --no-progressbar"
        unzip_cmd = "unzip ncbi_dataset.zip"
        gzip_genome_cmd = f"gzip -c {tmp_genome_path} > {output_genome_path}"
        gzip_annotation_cmd = f"gzip -c {tmp_annotation_path} > {output_annotation_path}"
        # p = subprocess.Popen(dl_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, shell=True)
        subprocess.check_call(dl_cmd, cwd=tmp_dir, shell=True)
        subprocess.check_call(unzip_cmd, cwd=tmp_dir, shell=True)
        subprocess.check_call(gzip_genome_cmd, shell=True)
        subprocess.check_call(gzip_annotation_cmd, shell=True)


def build_and_save_assembly_intervals(assembly: str, window_size: int):
    """ """
    intervals_assembly_dir = data_dir / f"intervals/{window_size}/{assembly}"
    intervals_assembly_dir.mkdir(parents=True, exist_ok=True)
    #
    fasta_path = genome_dir / f"{assembly}.fa.gz"
    genome = Genome(path=str(fasta_path))
    #
    intervals_defined_path = intervals_assembly_dir / "defined.parquet"
    #
    intervals = genome.get_all_intervals()
    intervals = bf.merge(bf.sanitize_bedframe(intervals))
    intervals = filter_length(intervals, min_interval_len=window_size)

    # remove part with "N"
    defined_intervals = filter_defined(intervals, genome)
    defined_intervals = filter_length(defined_intervals, min_interval_len=window_size)
    defined_intervals.to_parquet(intervals_defined_path, index=False)

    # Make balanced intervals
    intervals_balanced_path = intervals_assembly_dir / "balanced.parquet"
    output_annotation_path = annotation_dir / f"{assembly}.gff.gz"
    annotation = load_table(str(output_annotation_path))
    defined_intervals = load_table(str(intervals_defined_path))

    balanced_intervals = get_balanced_intervals(
        defined_intervals, annotation, int(window_size),
        promoter_upstream=1000,
    )
    balanced_intervals.to_parquet(intervals_balanced_path, index=False)
    #
    return defined_intervals, balanced_intervals


def build_and_save_assembly_dataset(
    assembly,
    window_size: int,
    step_size: int,
    use_balanced_intervals: bool = True,
    add_rc: bool = False
):
    """ """
    intervals_assembly_dir = data_dir / f"intervals/{window_size}/{assembly}"
    if use_balanced_intervals:
        intervals_path = intervals_assembly_dir / "balanced.parquet"
    else:
        intervals_path = intervals_assembly_dir / "defined.parquet"

    #
    intervals = pd.read_parquet(intervals_path)
    
    if chroms_subset:
        intervals = intervals[intervals.chrom.isin(chroms_subset)]

    fasta_path = genome_dir / f"{assembly}.fa.gz"
    genome = Genome(path=str(fasta_path), subset_chroms=chroms_subset)

    # intervals = balanced_intervals
    intervals = make_windows(
        intervals, window_size, step_size,
        add_rc=add_rc
    )
    # shuffle
    intervals = intervals.sample(frac=1.0, random_state=42)
    intervals["assembly"] = assembly
    intervals = intervals[["assembly", "chrom", "start", "end", "strand"]]
    intervals = get_seq(intervals, genome)

    # make split according to chromozome
    chroms = intervals.chrom.unique()
    chrom_split = np.random.choice(
        splits, p=split_proportions, size=len(chroms),
    )
    chrom_split[np.isin(chroms, whitelist_validation_chroms)] = "validation"
    chrom_split[np.isin(chroms, whitelist_test_chroms)] = "test"
    chrom_split = pd.Series(chrom_split, index=chroms)

    chrom_intervals_split = chrom_split[intervals.chrom]

    for split in splits:
        assembly_dataset_dir = dataset_dir / assembly
        assembly_dataset_dir.mkdir(parents=True, exist_ok=True)
        intervals_split_path = assembly_dataset_dir / f"{split}.parquet"
        intervals_split = intervals[(chrom_intervals_split == split).values]
        # if intervals_split.size > 0:
        intervals_split.to_parquet(intervals_split_path, index=False)


def merge_datasets():
    """ """
    for split in splits:
        #
        split_parquet_paths = list(dataset_dir.rglob(f"*/{split}.parquet"))
        split_intervals = pd.concat(
            tqdm((pd.read_parquet(path) for path in split_parquet_paths)),
            ignore_index=True,
        ).sample(frac=1, random_state=42)

        #
        if subsample_to_target and split == "train":
            n_target = (split_intervals.assembly == target_assembly).sum()
            split_intervals = split_intervals.groupby("assembly").sample(
                n=n_target, random_state=42
            ).sample(frac=1, random_state=42)
            print(split_intervals["assembly"].value_counts())

        #
        total = len(split_intervals)
        print(f"Number of sample for {split} dataset: {total}")
        n_shards = math.ceil(total / samples_per_file)
        assert n_shards < 10000

        for i, start in enumerate(range(0, total, samples_per_file)):
            merged_split_dir = merged_dataset_dir / split
            merged_split_dir.mkdir(parents=True, exist_ok=True)
            merged_path = merged_split_dir / f"shard_{i:05}.jsonl.zst"
            split_intervals.iloc[start:start + samples_per_file].to_json(
                merged_path, orient="records", lines=True,
                compression={'method': 'zstd', 'threads': -1}
            )


if __name__ == "__main__":

    for assembly, row in assemblies.iterrows():
        print(f"assembly: {assembly}")
        get_assembly_genome_annotation_and_order(assembly)

    for assembly in assemblies.index:
        print(assembly)
        #
        build_and_save_assembly_intervals(assembly, window_size=window_size)
        #
        build_and_save_assembly_dataset(
            assembly,
            window_size=window_size,
            step_size=step_size,
            use_balanced_intervals=balanced,
            add_rc=add_rc
        )

    merge_datasets()