
import gpn.model
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd

from gpn.data import Genome, make_windows
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from functools import partial
import torch
from tqdm import tqdm
from pathlib import Path
import math
# from transformers import DataCollatorForLanguageModeling


def get_id_to_chrom(seq_report_path: str):
    """ """
    df = pd.read_csv(seq_report_path, sep="\t")
    return {
        row["RefSeq seq accession"]: row["Chromosome name"]
        for _, row in df.iterrows()
        if row["RefSeq seq accession"].startswith("NC_")
    }
    

def prepare_dataset_items(examples, tokenizer, genome: Genome):
    """ """
    chrom, start, end = examples["chrom"], examples["start"], examples["end"]
    n = len(chrom)
    #
    sequences = [
        genome.get_seq(chrom[i], start[i], end[i])
        for i in range(n)
    ]
    #
    res = tokenizer(
        sequences,
        padding=False,
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_special_tokens_mask=False,
    )
    res["loss_weight"] = np.ones_like(res["input_ids"], dtype=float)
    #
    return res


def collate_center_mask(examples, tokenizer, window_size: int, mask_width: int):
    """ """
    batch = {
        key: (
            torch.stack([torch.tensor(example[key]) for example in examples], dim=0)
            if not isinstance(value, str) else [example[key] for example in examples]
        )
        for key, value in examples[0].items()
    }
    
    inputs = batch["input_ids"]
    labels = batch["input_ids"].clone()
    labels[:] = -100
    #
    #
    mask_start = window_size // 2 - mask_width // 2
    mask_end = window_size // 2 + math.ceil(mask_width / 2)
    #
    labels[:, mask_start:mask_end] = inputs[:, mask_start:mask_end]
    inputs[:, mask_start:mask_end] = tokenizer.mask_token_id
    #
    batch["input_ids"] = inputs
    batch["labels"] = labels 
    return batch


def collate_mask_every_k_base(examples, tokenizer, k: int = 8):
    """ """
    batch = {
        key: (
            torch.stack([torch.tensor(example[key]) for example in examples], dim=0)
            if not isinstance(value, str) else [example[key] for example in examples]
        )
        for key, value in examples[0].items()
    }
    
    inputs = batch["input_ids"]
    labels = batch["input_ids"].clone()
    labels[:] = -100
    #
    b, l = inputs.shape
    masked_indices = torch.arange(k, l - k, k, dtype=int)
    #
    labels[:, masked_indices] = inputs[:, masked_indices]
    inputs[:, masked_indices] = tokenizer.mask_token_id
    #
    batch["input_ids"] = inputs
    batch["labels"] = labels 
    return batch
  

def send_dict_to_device(data: dict, device):
    """ """
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    return data


def get_loss_per_nuc(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int):
    """ """
    # no reduction to get loss for every token
    loss_fct = CrossEntropyLoss(reduction="none")
    labels = labels.view(-1)
    b, l, v = logits.shape
    loss = loss_fct(logits.view(-1, vocab_size), labels)
    loss = loss.view((b, l))
    return loss


import numpy as np

def make_interval_k_windows(interval, window_size, k: int):
    """ """
    starts = []
    for i in range(k):
        starts.append(np.arange(interval.start + i, interval.end - window_size + 1, window_size - k))
    starts = np.sort(np.hstack(starts))
    #
    windows = pd.DataFrame(dict(start=starts))
    windows["end"] = windows.start + window_size
    windows["chrom"] = interval.chrom
    windows = windows[["chrom", "start", "end"]]  # just re-ordering
    windows["strand"] = "+"
    return windows


def make_k_windows(intervals, window_size, k):
    return pd.concat(
        intervals.progress_apply(
            lambda interval: make_interval_k_windows(interval, window_size, k=k), axis=1,
        ).values,
        ignore_index=True,
    )


def main():
    """ """
    # MODEL = "models/mus/MyConvNet_batch1000_no_weight_no_balanced_ds"
    MODEL = "models/mus/MyConvNet_batch256_l6_h256_no_weight_no_balanced_ds"
    # MODEL = "models/mus/MyConvNet_w1000_b256_l6_h256_no_weight_no_balanced_ds"
    # MODEL = "models/mus/MyConvNet_batch1000_12layers_no_weight_no_balanced_ds"
    # MODEL = "models/mus/MyConvNet_w512_b200_l24_h512_no_weight_no_balanced_ds"
    # MODEL = "models/GPN_Arabidopsis_multispecies/MyConvNet_batch1000_no_weight_no_balanced_ds"
    # MODEL = "songlab/gpn-brassicales"
    # MODEL = "models/GPN_Arabidopsis_multispecies/MyConvNet_batch1000_no_weight"

    WINDOW_SIZE = 1024
    STEP_SIZE = 2
    device = "cuda"
    num_worker = 6
    batch_size = 1000
    sample = False
    mode = "every_k"
    k = 8

    #
    position_loss_dir = Path(f"data/mus/proba/{MODEL.split('/')[-1]}")
    # position_loss_dir = Path(f"data/brassicales/proba/{MODEL.split('/')[-1]}")
    if sample:
        position_loss_dir = position_loss_dir / "sample"
    position_loss_dir.mkdir(exist_ok=True, parents=True)
    #
    seq_report_path = "data/mus/mus_chroms.tsv"
    genome_path = "data/mus/genome/GCF_000001635.27.fa.gz"
    # seq_report_path = "data/brassicales/arabidopsis_seq_report.tsv"
    # genome_path = "data/brassicales/genome/GCF_000001735.4.fa.gz"
    #
    tokenizer = AutoTokenizer.from_pretrained(MODEL)    
    model = AutoModelForMaskedLM.from_pretrained(MODEL)
    model.to(device)
    #
    chroms = get_id_to_chrom(seq_report_path)
    # chroms = ["NC_000083.7", "NC_000084.7"]

    # genome = Genome(genome_path)
    # chroms = list(genome._genome.keys())
    #
    for chrom in chroms:
        
        if mode == "center":
            chrom_position_loss_path = position_loss_dir / f"proba_chrom_{chrom}_step_{STEP_SIZE}_w_{WINDOW_SIZE}.parquet"
        elif mode == "every_k":
            chrom_position_loss_path = position_loss_dir / f"proba_chrom_{chrom}_every_k_{k}_w_{WINDOW_SIZE}.parquet"
        
        if chrom_position_loss_path.exists():
            print(f"skipping chrom: {chrom} because already computed.")
            continue

        genome = Genome(genome_path, subset_chroms=[chrom])
        print(f"Processing chrom {chrom}...")
        intervals = genome.get_defined_intervals()
        if mode == "center":
            windows = make_windows(intervals, WINDOW_SIZE, STEP_SIZE, add_rc=False)
        elif mode == "every_k":
            windows = make_k_windows(intervals, window_size=WINDOW_SIZE, k=k)
        
        if sample:
            windows = windows[:1000000]
        n_windows = len(windows)
        print(f"Predicting over {n_windows} windows...")
        #
        ds = Dataset.from_pandas(windows).to_iterable_dataset(num_shards=num_worker)
        #
        unused_columns = [c for c in windows if c not in ["input_ids", "start"]]
        ds = ds.map(
            lambda batch: prepare_dataset_items(
                batch,
                genome=genome,
                tokenizer=tokenizer,
            ),
            batched=True,
            # num_proc=6,
            remove_columns=unused_columns,
        )

        collator = partial(collate_center_mask, tokenizer=tokenizer, window_size=WINDOW_SIZE, mask_width=STEP_SIZE)
        # collator = DataCollatorForLanguageModeling(
        #     tokenizer=tokenizer,
        #     mlm_probability=0.15,
        # )
        if mode == "center":
            collator = partial(collate_center_mask, tokenizer=tokenizer, window_size=WINDOW_SIZE, mask_width=STEP_SIZE)
        elif mode == "every_k":
            collator = partial(collate_mask_every_k_base, tokenizer=tokenizer, k=k)

        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collator, num_workers=num_worker, shuffle=False)


        positions, probas = [], []
        with torch.no_grad():
            for batch in tqdm(dl, total=n_windows // batch_size):
                b_start = batch.pop("start").tolist()
                batch_length = len(b_start)
                batch = send_dict_to_device(batch, device)
                out = model(**batch)
                # break
                logits = out.logits.detach().cpu()
                # loss = get_loss_per_nuc(
                #     logits=logits,
                #     labels=batch["labels"].cpu(),
                #     vocab_size=tokenizer.vocab_size
                # )
                preds = torch.softmax(logits, dim=-1).max(dim=-1)

                if mode == "center":
                    mask_start = WINDOW_SIZE // 2 - STEP_SIZE // 2
                    # keep just loss on the masked nucleotides
                    for i in range(batch_length):
                        for j in range(STEP_SIZE):
                            positions.append(b_start[i] + mask_start + j)
                            probas.append(float(preds.values[i, mask_start + j]))
                            # loss_scores.append((
                            #     # float(loss[i, mask_start + j]),
                            #     float(preds.values[i, mask_start + j]),
                            #     tokenizer.decode(preds.indices[i, mask_start + j])
                            # ))
                elif mode == "every_k":
                    masked_indices = list(range(k, WINDOW_SIZE - k, k))
                    for i in range(batch_length):
                        for j in masked_indices:
                            positions.append(b_start[i] + j)
                            probas.append(float(preds.values[i, j]))
                            # loss_scores.append((
                            #     # float(loss[i, j]),
                            #     float(preds.values[i, j]),
                            #     tokenizer.decode(preds.indices[i, j])
                            # ))
                    
        #
        position_loss = pd.DataFrame(probas, index=positions, columns=("proba",))
        position_loss.sort_index(inplace=True)
        position_loss = position_loss.reindex(
            pd.RangeIndex(
                start=position_loss.index.min(),
                stop=position_loss.index.max() + 1
            )
        )
        position_loss["proba"] = position_loss.proba.astype("float32")
        position_loss.to_parquet(chrom_position_loss_path)
        
    
if __name__ == "__main__":
    main()