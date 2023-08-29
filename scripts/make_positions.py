from pathlib import Path
from gpn.define_intervals import filter_length
from gpn.utils import Genome
import bioframe as bf
import pandas as pd

data_dir = Path("./data")


CHROMS = ["1", "2", "3", "4", "5"]
WINDOW_SIZE = 512


def find_positions(interval):
    """ """
    df = pd.DataFrame(dict(pos=range(interval.start, interval.end)))
    df["chrom"] = interval.chrom
    df.pos += 1  # we'll treat as 1-based
    return df


def main():
    
    assembly = "GCF_000001735.4"
    fasta_path = data_dir / "genome" / f"{assembly}.fa.gz"
    genome = Genome(path=str(fasta_path))

    CHROMS = [
        "NC_003070.9",  # 1
        "NC_003071.7",  # 2
        "NC_003074.8",  # 3
        "NC_003075.7",  # 4
        "NC_003076.8",  # 5
    ]
    genome.filter_chroms(CHROMS)

    intervals = bf.expand(genome.get_defined_intervals(), pad=-WINDOW_SIZE//2)
    intervals = filter_length(intervals, 1)
    print(intervals)

    positions = pd.concat(
        intervals.progress_apply(find_positions, axis=1).values, ignore_index=True
    )
    print(positions)
    
    whole_genome_dir = Path("./data/whole_genome")
    whole_genome_dir.mkdir(parents=True, exist_ok=True)
    position_path = whole_genome_dir / "positions.parquet"
    positions.to_parquet(position_path, index=False)


if __name__ == "__main__":
    main()