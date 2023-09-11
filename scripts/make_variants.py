from pathlib import Path
import re

import numpy as np
import pandas as pd
from cyvcf2 import VCF

from gpn.utils import Genome
import h5py

data_dir = Path("./data/")
genome_path = data_dir / "genome.fa.gz"
genome = Genome(str(genome_path))


def make_simulated_variants():
    """ """
    chrom = "5"
    start = 3500000
    end = start + 1_000_000
    nucleotides = list("ACGT")

    rows = []
    for pos in range(start, end):
        ref = genome.get_nuc(chrom, pos).upper()
        for alt in nucleotides:
            if alt == ref:
                continue
            rows.append([chrom, pos, '.', ref, alt, '.', '.', '.'])

    df = pd.DataFrame(data=rows)
    print(df)

    # paths
    simulated_variant_dir = data_dir / "simulated_variants"
    simulated_variant_dir.mkdir(parents=True, exist_ok=True)
    variant_vcf_path = simulated_variant_dir / "variants.vcf.gz"
    variant_path = simulated_variant_dir / "variants.parquet"

    # save stuff
    df.to_csv(variant_vcf_path, sep="\t", index=False, header=False)
    df[[0, 1, 3, 4]].rename(
        columns={0: "chrom", 1: "pos", 3: "ref", 4: "alt"}
    ).to_parquet(variant_path, index=False)


"""
python -m gpn.run_vep \
    /mnt/shared_thomas/gpn/data/simulated_variants/variants.parquet \
    /mnt/shared_thomas/gpn/data/genome.fa.gz \
    512 \
    /mnt/shared_thomas/gpn/models/GPN_Arabidopsis_multispecies/ConvNet_batch200_weight0.1/ \
    ./data/simulated_variants/vep/ConvNet_batch200_weight0.1.parquet \
    --per-device-batch-size 4000 --is-file \
    --dataloader-num-workers 16
"""

"""
vcftools --gzvcf data/variants/all.vcf.gz --counts --out tmp --min-alleles 2 --max-alleles 2 --remove-indels \
    && mv tmp.frq.count data/variants/filt.bed \
    && gzip data/variants/filt.bed

"""

def find_ref_alt_AC(row):
    """ """
    ref = genome.get_nuc(row.chrom, row.pos).upper()
    assert(ref == row.ref_count[0])
    alt, AC = row.alt_count.split(":")
    AC = int(AC)
    return ref, alt, AC


def process_1001_variants(variants_path):
    """ """
    variants = pd.read_csv(
        variants_path,
        sep="\t",
        header=0,
        names=["chrom", "pos", "N_ALLELES", "AN", "ref_count", "alt_count"]
    ).drop(columns="N_ALLELES")
    variants.chrom = variants.chrom.astype(str)
    #
    variants["ref"], variants["alt"], variants["AC"] = \
        zip(*variants.progress_apply(find_ref_alt_AC, axis=1))

    variants = variants[["chrom", "pos", "ref", "alt", "AC", "AN"]]
    #
    return variants



def get_ensembl_vep(ensembl_vep_path):
    """
    Download arabidopsis ensembl vep from here:
    https://ftp.ensemblgenomes.org/pub/plants/release-55/variation/vcf/arabidopsis_thaliana/arabidopsis_thaliana_incl_consequences.vcf.gz

    """
    i = 0
    rows = []
    for variant in VCF(ensembl_vep_path):
        if variant.INFO.get("TSA") != "SNV":
            continue
        if len(variant.ALT) > 1:
            continue
        if variant.FILTER is not None:
            continue  # this is supposed to mean PASS
        
        VEP = variant.INFO.get("VE").split(",")
        consequence = ",".join(np.unique([transcript_vep.split("|")[0] for transcript_vep in VEP]))
        rows.append([variant.CHROM, variant.POS, variant.REF, variant.ALT[0], consequence])
        i += 1
        if i % 100000 == 0:
            print(i)

    ensembl_vep = pd.DataFrame(data=rows, columns=["chrom", "pos", "ref", "alt", "consequence"])
    #
    return ensembl_vep


def make_1001_genome_variants():
    """
    
    1. download all 1001 genome variant vcf file from
        https://1001genomes.org/data/GMI-MPI/releases/v3.1/1001genomes_snp-short-indel_only_ACGTN.vcf.gz
    
    2. Run this command to filter:
        vcftools --gzvcf data/variants/all.vcf.gz --counts --out tmp --min-alleles 2 --max-alleles 2 --remove-indels \
            && mv tmp.frq.count data/variants/filt.bed \
            && gzip data/variants/filt.bed
    
    3. Download ensembl vep from
        https://ftp.ensemblgenomes.org/pub/plants/release-55/variation/vcf/arabidopsis_thaliana/arabidopsis_thaliana_incl_consequences.vcf.gz
    
    4. apply this fct :)

    """

    #
    variants_path = data_dir / "variants" / "filt.bed.gz"
    variants = process_1001_variants(variants_path)

    #
    ensembl_vep_path = data_dir / "arabidopsis_thaliana_incl_consequences.vcf.gz" 
    ensembl_vep = get_ensembl_vep(ensembl_vep_path)

    # merge both dataframe to have variants with information about the variant
    variants = variants.merge(ensembl_vep, how="left", on=["chrom", "pos", "ref", "alt"])
    variants_all_path = data_dir / "variants" / "all.parquet"
    variants.to_parquet(variants_all_path)



"""
wget https://aragwas.1001genomes.org/api/genotypes/download -O genotype.zip \
    && unzip genotype.zip \
    && mv GENOTYPES/4. data/variants/genotype_matrix.
"""

def filter_variants_with_genotype():
    """ """
    genotype_path = "data/variants/genotype_matrix.hdf5"
    
    f = h5py.File(genotype_path, 'r')
    n_snps, n_accessions = f["snps"].shape
    AC = f["snps"][:].sum(axis=1)
    pos = f["positions"][:]
    chrom = np.empty_like(pos, dtype=str)
    for c, (left, right) in zip(
        f["positions"].attrs["chrs"], f['positions'].attrs['chr_regions']
    ):
        chrom[left:right] = str(c)

    variants = pd.DataFrame({"chrom": chrom, "pos": pos, "AC": AC})
    variants["AF"] = variants.AC / n_accessions 

    variant_info = pd.read_parquet(input[1])
    variants = variants.merge(
        variant_info.drop(columns=["AC", "AN"]), how="inner", on=["chrom", "pos"]
    )
    variants = variants[["chrom", "pos", "ref", "alt", "AC", "AF", "consequence"]]
    print(variants)

    TOTAL_SIZE = 512
    variants["start"] = variants.pos - TOTAL_SIZE // 2
    variants["end"] = variants.start + TOTAL_SIZE
    variants["strand"] = "+"

    def check_seq(w):
        seq = genome.get_seq(w.chrom, w.start, w.end).upper()
        if len(seq) != TOTAL_SIZE: return False
        if re.search("[^ACTG]", seq) is not None: return False
        return True

    variants = variants[variants.progress_apply(check_seq, axis=1)]
    variants.drop(columns=["start", "end", "strand"], inplace=True)

    print(variants)
    variants_all_dir = data_dir / "variants" / "all"
    variants_all_dir.mkdir(parents=True, exist_ok=True)
    variants.to_parquet(variants_all_dir / "variants.parquet", index=False)
    
    # split by chrom
    gb = variants.groupby("chrom")
    for chrom, df in gb:
        chrom_dir = data_dir / "variants/chrom/"
        chrom_dir.mkdir(parents=True, exist_ok=True)
        chrom_variants_path = chrom_dir / f"chrom_{chrom}.parquet"
        df.to_parquet(chrom_variants_path)
        
        
"""
python -m gpn.run_vep \
    /mnt/shared_thomas/gpn/data/variants/chrom/chrom_5.parquet \
    /mnt/shared_thomas/gpn/data/genome.fa.gz \
    512 \
    /mnt/shared_thomas/gpn/models/GPN_Arabidopsis_multispecies/ConvNet_batch200_weight0.1/ \
    ./data/variants/chrom/vep/chrom_5_ConvNet_batch200_weight0.1.parquet \
    --per-device-batch-size 2048 --is-file \
    --dataloader-num-workers 16
"""

conservation_path = "data/conservation/phastCons.bedGraph.gz"
variants_path = "data/variants/chrom/chrom_5.parquet"

variants = pd.read_parquet(variants_path)

conservation = pd.read_csv(
    conservation_path,
    sep="\t", header=None,
    names=["chrom", "pos", "end", "score"],
).drop(columns=["end"])

conservation.pos += 1
conservation.chrom = conservation.chrom.str.replace("Chr", "")
conservation.score *= -1

variants = variants.merge(conservation, how="left", on=["chrom", "pos"])
variants = variants[["score"]]
variants.to_parquet(output[0], index=False)