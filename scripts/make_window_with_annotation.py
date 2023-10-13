from typing import List
import more_itertools
import pandas as pd
from pathlib import Path

import bioframe as bf
from gpn.data import make_windows, filter_length, load_table, Genome


WINDOW_SIZE = 512
EMBEDDING_WINDOW_SIZE = 100

    
def get_repeat_intervals_from_fasta(fasta_path: str):
    """ """
    genome = Genome(path=str(fasta_path))
    repeats_intervals = []
    for chr, seq in genome._genome.items():
        start = None
        end = None
        for i, b in enumerate(seq):
            if start is None and b.islower():
                start = i
            elif start is not None and b.isupper():
                end = i
            #
            if start is not None and end is not None:
                repeats_intervals.append((chr, start, end))
                start = None
                end = None
    #
    repeat_intervals_df = pd.DataFrame(repeats_intervals, columns=["chrom", "start", "end"])
    repeat_intervals_df["size"] = repeat_intervals_df["end"] - repeat_intervals_df["start"]
    #
    return repeat_intervals_df


def get_annotation_expanded(gtf_path: str, repeats_path: str):
    """
    exemple feature count for gtf file of Arabidopsis
        exon               313952
        CDS                286067
        five_prime_UTR      56384
        mRNA                48359
        three_prime_UTR     48308
        gene                27655
        ncRNA_gene           5178
        lnc_RNA              3879
        tRNA                  689
        ncRNA                 377
        miRNA                 325
        snoRNA                287
        snRNA                  82
        rRNA                   15
        chromosome              7
    """
    gtf = load_table(gtf_path)
    #
    repeats = pd.read_csv(repeats_path, sep="\t") \
        .rename(columns=dict(genoStart="start", genoEnd="end"))
    repeats["chrom"] = repeats.genoName.str.replace("chr", "")
    repeats = repeats[["chrom", "start", "end", "strand"]]
    #
    repeats = bf.merge(repeats).drop(columns="n_intervals")
    repeats["feature"] = "Repeat"
    gtf = pd.concat([gtf, repeats], ignore_index=True)
    gtf_intergenic = bf.subtract(gtf.query('feature=="chromosome"'),
                                 gtf[gtf.feature.isin(["gene", "ncRNA_gene", "Repeat"])])
    # gtf_intergenic = bf.subtract(gtf.query('feature=="chromosome"'),
    #                              gtf[gtf.feature.isin(["gene", "ncRNA_gene"])])
    gtf_intergenic["feature"] = "intergenic"
    gtf = pd.concat([gtf, gtf_intergenic], ignore_index=True)

    gtf_exon = gtf[gtf["feature"] == "exon"]
    gtf_exon["transcript_id"] = gtf_exon["attribute"].str.split(";").str[0].str.split(":").str[-1]

    def get_transcript_introns(df_transcript):
        df_transcript = df_transcript.sort_values("start")
        exon_pairs = more_itertools.pairwise(df_transcript.loc[:, ["start", "end"]].values)
        introns = [[e1[1], e2[0]] for e1, e2 in exon_pairs]
        introns = pd.DataFrame(introns, columns=["start", "end"])
        introns["chrom"] = df_transcript.chrom.iloc[0]
        return introns

    gtf_introns = gtf_exon.groupby("transcript_id").apply(get_transcript_introns).reset_index() \
        .drop_duplicates(subset=["chrom", "start", "end"])
    gtf_introns["feature"] = "intron"
    #
    gtf = pd.concat([gtf, gtf_introns], ignore_index=True)
    gtf["start"] = gtf["start"].astype(int)
    gtf["end"] = gtf["end"].astype(int)

    return gtf


def make_embedding_windows(gtf: pd.DataFrame, genome: Genome, features_of_interest: List[str] = None) -> pd.DataFrame:
    """ """
    if features_of_interest is None:
        features_of_interest = [
            "intergenic",
            'CDS',
            'intron',
            # "exon",
            'three_prime_UTR',
            'five_prime_UTR',
            "ncRNA_gene",
            "Repeat",
        ]

    defined_intervals = genome.get_defined_intervals()
    defined_intervals = filter_length(defined_intervals, WINDOW_SIZE)
    windows = make_windows(defined_intervals, WINDOW_SIZE, EMBEDDING_WINDOW_SIZE)
    windows.rename(columns={"start": "full_start", "end": "full_end"}, inplace=True)
    #
    windows["start"] = (windows.full_start + windows.full_end) // 2 - EMBEDDING_WINDOW_SIZE // 2
    windows["end"] = windows.start + EMBEDDING_WINDOW_SIZE

    for f in features_of_interest:
        print(f)
        windows = bf.coverage(windows, gtf[gtf.feature == f])
        windows.rename(columns=dict(coverage=f), inplace=True)

    windows = windows[(windows[features_of_interest] >= 0.66 * EMBEDDING_WINDOW_SIZE).sum(axis=1) == 1]
    # windows = windows[(windows[features_of_interest] == EMBEDDING_WINDOW_SIZE).sum(axis=1) == 1]
    windows["Region"] = windows[features_of_interest].idxmax(axis=1)
    windows.drop(columns=features_of_interest, inplace=True)

    windows.rename(columns={"start": "center_start", "end": "center_end"}, inplace=True)
    windows.rename(columns={"full_start": "start", "full_end": "end"}, inplace=True)
    print(windows)
    #
    return windows


def get_id_to_chrom(seq_report_path: str):
    """ """
    df = pd.read_csv(seq_report_path, sep="\t")
    return {
        row["RefSeq seq accession"]: row["Chromosome name"]
        for _, row in df.iterrows()
        if row["RefSeq seq accession"].startswith("NC_")
    }


def main():
    """ """
    #
    # genome_dir = Path("data/genome")
    # assembly = "GCF_000001735.4"
    # gtf_path = "data/Arabidopsis_thaliana.TAIR10.55.gff3.gz"
    # repeats_path = "./analysis/arabidopsis/input/repeats.bed.gz"
    # mus_seq_report_path = "data/mus/mus_chroms.tsv"
    # expanded_annotation_path = "data/annotation/expanded.parquet"

    genome_dir = Path("data/mus/genome")
    assembly = "GCF_000001635.27"
    gtf_path = "data/mus/Mus_musculus.GRCm39.110.chr.gff3.gz"
    # gtf_path = "data/mus/annotation/GCF_000001635.27.gff.gz"
    repeats_path = "data/mus/mus.repeats.bed.gz"
    expanded_annotation_path = "data/mus/annotation/expanded.parquet"
    mus_seq_report_path = "data/mus/mus_chroms.tsv"
    
    id_to_chrom = get_id_to_chrom(mus_seq_report_path)
    
    fasta_path = genome_dir / f"{assembly}.fa.gz"
    genome = Genome(path=str(fasta_path))
    for key in genome._genome.keys():
        value = genome._genome.pop(key)
        if key in id_to_chrom:
            genome._genome[id_to_chrom[key]] = value

    gtf = get_annotation_expanded(
        gtf_path=gtf_path, repeats_path=repeats_path
    )
    gtf.to_parquet(expanded_annotation_path, index=False)
    # gtf = pd.read_parquet(expanded_annotation_path)

    #
    features_of_interest = [
        "intergenic",
        'CDS',
        'intron',
        'three_prime_UTR',
        'five_prime_UTR',
        "ncRNA_gene",
        "Repeat",
    ]
    windows = make_embedding_windows(
        gtf, genome=genome, features_of_interest=features_of_interest
    )
    
    #
    chrom_to_id = {v: k for k, v in id_to_chrom.items()}
    windows["chrom_name"] = windows["chrom"]
    windows["chrom"] = windows["chrom"].map(lambda x: chrom_to_id[x])
    
    embedding_folder = Path("data/mus/embeddings")
    embedding_folder.mkdir(exist_ok=True)
    embedding_windows_path = embedding_folder / "windows.parquet"
    windows.to_parquet(embedding_windows_path, index=False)
    embedding_windows_sample_path = embedding_folder / "windows.sample.parquet"
    windows.sample(frac=0.1).to_parquet(embedding_windows_sample_path, index=False)
    #
    for chrom, chrom_df in windows.groupby("chrom_name"):
        embedding_windows_chr = embedding_folder / f"windows_chr_{chrom}_bis.parquet"
        chrom_df.to_parquet(embedding_windows_chr, index=False)


if __name__ == "__main__":
    main()
    
    """
    python -m gpn.get_embeddings \
        /mnt/shared_thomas/gpn/data/embeddings/windows.parquet \
        /mnt/shared_thomas/gpn/data/genome/GCF_000001735.4.fa.gz \
        512 \
        /mnt/shared_thomas/gpn/models/GPN_Arabidopsis_multispecies/ConvNet_batch200_weight0.1 \
        data/embeddings/ConvNet_batch200_weight0.1_NO_AVG_REV.embbedings.parquet \
         --per-device-batch-size 2048 --is-file \
        --dataloader-num-workers 16
    
    python -m gpn.get_embeddings \
        /mnt/shared_thomas/gpn/data/embeddings/windows.parquet \
        /mnt/shared_thomas/gpn/data/genome/GCF_000001735.4.fa.gz \
        512 \
        /mnt/shared_thomas/gpn/models/GPN_Arabidopsis_multispecies/MyConvNet_12layers_batch256_weight0.1 \
        data/embeddings/MyConvNet_12layers_batch256_weight0.1.embbedings.parquet \
        --per-device-batch-size 4000 --is-file \
        --dataloader-num-workers 16
    
    python -m gpn.get_embeddings \
        /mnt/shared_thomas/gpn/data/embeddings/windows.parquet \
        /mnt/shared_thomas/gpn/data/genome/GCF_000001735.4.fa.gz \
        100 \
        /mnt/shared_thomas/gpn/models/GPN_Arabidopsis_multispecies/MyConvNet_batch1000_no_weight \
        data/embeddings/MyConvNet_batch1000_no_weight_w100.embbedings.parquet \
        --per-device-batch-size 2048 --is-file \
        --dataloader-num-workers 16

    # mus
    nohup python -m gpn.get_embeddings \
        /mnt/shared_thomas/gpn/data/mus/embeddings/windows.sample.parquet \
        /mnt/shared_thomas/gpn/data/mus/genome/GCF_000001635.27.fa.gz \
        100 \
        /mnt/shared_thomas/gpn/models/mus/MyConvNet_batch1000_w_0.5_no_balanced_ds \
        data/mus/embeddings/MyConvNet_batch1000_w_0.5_no_balanced_ds.sample.embbedings.parquet \
        --per-device-batch-size 2048 --is-file \
        --dataloader-num-workers 16
    
    nohup python -m gpn.get_embeddings \
        data/mus/embeddings/windows.sample.parquet \
        data/mus/genome/GCF_000001635.27.fa.gz \
        100 \
        models/mus/MyConvNet_batch1000_w_0.5_no_balanced_ds \
        data/mus/embeddings/MyConvNet_batch1000_w_0.5_no_balanced_ds_no_rev.sample.embbedings.parquet \
        --per-device-batch-size 2048 --is-file \
        --dataloader-num-workers 16 \
        --no-avg-rev

        
    # embedding with no balanced dataset
    python -m gpn.get_embeddings \
        data/embeddings/windows.parquet \
        data/genome/GCF_000001735.4.fa.gz \
        100 \
        models/GPN_Arabidopsis_multispecies/MyConvNet_batch1000_no_weight_no_balanced_ds \
        data/embeddings/MyConvNet_batch1000_no_weight_no_balanced_ds.embbedings.parquet \
        --per-device-batch-size 2048 --is-file \
        --dataloader-num-workers 16
    
    # TRY WITH DNABERT2
    python -m gpn.get_embeddings \
        data/embeddings/windows.parquet \
        data/genome/GCF_000001735.4.fa.gz \
        100 \
        models/GPN_Arabidopsis_multispecies/MyConvNet_batch1000_no_weight_no_balanced_ds \
        data/embeddings/MyConvNet_batch1000_no_weight_no_balanced_ds.embbedings.parquet \
        --per-device-batch-size 2048 --is-file \
        --dataloader-num-workers 16

    """
    
