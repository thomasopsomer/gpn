import more_itertools
import pandas as pd
from pathlib import Path

import bioframe as bf
from gpn.make_dataset_mlm import make_windows

from gpn.define_intervals import filter_length
from gpn.utils import load_table, Genome


genome_dir = Path("data/genome")


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


def get_annotation_expanded(gtf_path: str, repeat_path: str):
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
    # rename chroms
    id_to_chrom_map = {
        "1": "NC_003070.9" ,
        "2": "NC_003071.7",
        "3": "NC_003074.8",
        "4": "NC_003075.7",
        "5": "NC_003076.8",
        "Mt": "NC_037304.1",
        "M": "NC_037304.1",
        "Pt": "NC_000932.1",
        "C": "NC_000932.1"
    }
    gtf["chrom"] = gtf["chrom"].map(lambda x: id_to_chrom_map[x])
    #
    # repeats_bed_path = "./analysis/arabidopsis/input/repeats.bed.gz"
    repeats = pd.read_csv(repeat_path, sep="\t") \
        .rename(columns=dict(genoStart="start", genoEnd="end"))
    repeats["chrom"] = repeats.genoName.map(lambda x: id_to_chrom_map.get(x.replace("Chr", "")))
    # repeats.chrom = repeats.chrom.map(lambda x: id_to_chrom_map.get(x))
    repeats = repeats[["chrom", "start", "end"]]
    #
    repeats = bf.merge(repeats).drop(columns="n_intervals")
    repeats["feature"] = "Repeat"
    gtf = pd.concat([gtf, repeats], ignore_index=True)

    gtf_intergenic = bf.subtract(gtf.query('feature=="chromosome"'),
                                 gtf[gtf.feature.isin(["gene", "ncRNA_gene", "Repeat"])])
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

    gtf_introns = gtf_exon.groupby("transcript_id").apply(get_transcript_introns).reset_index().drop_duplicates(
        subset=["chrom", "start", "end"])
    gtf_introns["feature"] = "intron"
    gtf = pd.concat([gtf, gtf_introns], ignore_index=True)
    gtf["start"] = gtf["start"].astype(int)
    gtf["end"] = gtf["end"].astype(int)

    return gtf


def make_embedding_windows(gtf: pd.DataFrame, genome: Genome) -> pd.DataFrame:
    """ """
    WINDOW_SIZE = 512
    EMBEDDING_WINDOW_SIZE = 100

    defined_intervals = genome.get_defined_intervals()
    defined_intervals = filter_length(defined_intervals, WINDOW_SIZE)
    windows = make_windows(defined_intervals, WINDOW_SIZE, EMBEDDING_WINDOW_SIZE)
    windows.rename(columns={"start": "full_start", "end": "full_end"}, inplace=True)

    windows["start"] = (windows.full_start + windows.full_end) // 2 - EMBEDDING_WINDOW_SIZE // 2
    windows["end"] = windows.start + EMBEDDING_WINDOW_SIZE

    features_of_interest = [
        "intergenic",
        'CDS',
        'intron',
        'three_prime_UTR',
        'five_prime_UTR',
        "ncRNA_gene",
        "Repeat",
    ]

    for f in features_of_interest:
        print(f)
        windows = bf.coverage(windows, gtf[gtf.feature == f])
        windows.rename(columns=dict(coverage=f), inplace=True)

    windows = windows[(windows[features_of_interest] == EMBEDDING_WINDOW_SIZE).sum(axis=1) == 1]
    windows["Region"] = windows[features_of_interest].idxmax(axis=1)
    windows.drop(columns=features_of_interest, inplace=True)

    windows.rename(columns={"start": "center_start", "end": "center_end"}, inplace=True)
    windows.rename(columns={"full_start": "start", "full_end": "end"}, inplace=True)
    print(windows)
    #
    return windows


def main():
    """ """

    assembly = "GCF_000001735.4"
    fasta_path = genome_dir / f"{assembly}.fa.gz"
    genome = Genome(path=str(fasta_path))
    #
    gtf_path = "data/Arabidopsis_thaliana.TAIR10.55.gff3.gz"
    repeats_path = "./analysis/arabidopsis/input/repeats.bed.gz"

    gtf = get_annotation_expanded(gtf_path=gtf_path)
    expanded_annotation_path = "data/annotation/expanded.parquet"
    gtf.to_parquet(expanded_annotation_path, index=False)
    # gtf = pd.read_parquet(expanded_annotation_path)

    #
    windows = make_embedding_windows(gtf, genome=genome)
    embedding_folder = Path("data/embeddings")
    embedding_folder.mkdir(exist_ok=True)
    embedding_windows_path = embedding_folder / "windows.parquet"
    windows.to_parquet(embedding_windows_path, index=False)


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
    """
    
