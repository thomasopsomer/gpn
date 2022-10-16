import gzip
from Bio import SeqIO
import pandas as pd


def load_fasta(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as handle:
        return SeqIO.to_dict(SeqIO.parse(handle, "fasta"))


def load_table(path):
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    if 'csv' in path:
        df = pd.read_csv(path)
    elif 'tsv' in path:
        df = pd.read_csv(path, sep='\t')
    elif 'vcf' in path:
        df = pd.read_csv(
            path, sep="\t", header=None, comment="#", usecols=[0,1,2,3,4],
        ).rename(cols={0: 'chrom', 1: 'pos', 2: 'id', 3: 'ref', 4: 'alt'})
        df.pos -= 1
    df.chrom = df.chrom.astype(str)
    return df
