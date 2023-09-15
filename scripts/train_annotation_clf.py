from collections import Counter
import logging
import os
from pprint import pprint
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (CONFIG_MAPPING, AutoConfig,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          EvalPrediction, HfArgumentParser, set_seed)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer import Trainer, TrainingArguments

from gpn.model import MyConvNetConfig, MyConvNetModel, MyConvNetPreTrainedModel
from gpn.utils import Genome
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)

MODEL_TYPES = ["MyConvNet", "ConvNet"]


# ### MODELING

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        x = features.mean(axis=1)  # mean pooling
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class ConvNetForSequenceClassification(MyConvNetPreTrainedModel):
    """ """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MyConvNetModel(config)
        self.classifier = ClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor]]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        hidden_state = self.model(input_ids=input_ids).last_hidden_state
        logits = self.classifier(hidden_state)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


AutoModelForSequenceClassification.register(MyConvNetConfig, ConvNetForSequenceClassification)


def get_ds_from_windows_df(windows_df: pd.DataFrame, split_to_chrom: dict):
    """ """
    ds_dict = {
        split: Dataset.from_pandas(
            windows_df[windows_df.chrom.isin(chroms)]
        )
        for split, chroms in split_to_chrom.items()
    }
    ds = DatasetDict(ds_dict)
    ds.set_format("torch")
    return ds


def compute_metrics(eval_preds: EvalPrediction):
    """ """
    # predictions = np.argmax(logits, axis=-1)
    # preds = eval_preds.predictions > 0.5
    preds = eval_preds.predictions.argmax(axis=-1)
    labels = eval_preds.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    #
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def get_opposite_strand(strand: str):
    """ """
    if strand == "+":
        return "-"
    else:
        return "+"


def prepare_dataset_items(
    batch,
    genome: Genome,
    tokenizer,
    label2id: dict,
    add_rev_strand: bool = True,
):
    """
    get sequences and labels from annotated windows
    """
    chrom, start, end, strand = batch["chrom"], batch["start"], batch["end"], batch["strand"]
    n = len(chrom)
    sequences = [
        genome.get_seq(chrom[i], start[i], end[i], strand[i]) for i in range(n)
    ]
    # labels = torch.zeros(n, len(label2id), dtype=int)
    # for i, x in enumerate(batch["Region"]):
    #     labels[i, label2id[x]] = 1
    labels = [label2id[x] for x in batch["Region"]]
    if add_rev_strand:
        sequences += [
            genome.get_seq(chrom[i], start[i], end[i], get_opposite_strand(strand[i]))
            for i in range(n)
        ]
        # labels = torch.cat([labels, labels])
        labels += labels
    #
    res = tokenizer(
        sequences,
        padding=False,
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_special_tokens_mask=False,
    )
    res["labels"] = labels
    return res


def get_dataset_label_distribution(ds: DatasetDict, id2labels: dict):
    """ """
    split_dist = {}
    for split in ["train", "eval", "test"]:
        counts = torch.bincount(ds[split]["labels"])
        split_dist[split] = {
            label: int(counts[i]) for i, label in id2labels.items()
        }
    return split_dist


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    ann_windows_path: str = field(
        metadata={"help": ""}
    )
    genome_path: str = field(
        default=None, metadata={"help": ""}
    )
    do_test: bool = field(default=False)
    add_rev_strand: bool = field(
        default=True, metadata={"help": "Whether to add reverse seq or not (with the same label as the original strand)"}
    )



model_args = ModelArguments(
    # model_name_or_path="models/GPN_Arabidopsis_multispecies/MyConvNet_12layers_batch256_weight0.1",
    model_name_or_path="models/annotation_clf/GPN_12layers_no_pretrained_b256_no_rev",
    tokenizer_name="gonzalobenegas/tokenizer-dna-mlm",
)

training_args = TrainingArguments(
    fp16=True,
    dataloader_num_workers=6,
    seed=42,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    run_name="test_ann_clf_pretrained",
    output_dir="models/annotation_clf/test_ann_clf_pretrained",
    per_device_train_batch_size=256,
    per_device_eval_batch_size=512,
    learning_rate=1e-3,    
)

data_args = DataTrainingArguments(
    ann_windows_path="./data/embeddings/windows.parquet",
    genome_path="./data/genome/GCF_000001735.4.fa.gz",
    add_rev_strand=False
)


def main():
    """ """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    set_seed(training_args.seed)

    # data stuff
    genome = Genome(data_args.genome_path)
    windows_df = pd.read_parquet(data_args.ann_windows_path)
    # windows_df = windows_df.sample(frac=0.1)
    
    split_to_chrom = {
        "train": [
            "NC_003070.9",
            "NC_003071.7",
            "NC_003074.8",
            "NC_037304.1",
            "NC_000932.1"
            # "1", "2", "3"
        ],
        "test": [
            # "4"
            "NC_003076.8"
        ],
        "eval": [
            "NC_003075.7"
            # "5"
        ]
    }
    
    ds = get_ds_from_windows_df(windows_df, split_to_chrom)
    labels = list(windows_df.Region.unique())
    
    #
    config_kwargs = {
        "num_labels": len(labels),
        "id2label": {i: label for i, label in enumerate(labels)},
        "label2id": {label: i for i, label in enumerate(labels)},
    }

    
    # config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type](**config_kwargs)
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    
    #
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path
    )

    #
    
    if model_args.model_name_or_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            # num_labels=len(labels),
            problem_type="single_label_classification",
            **config_kwargs
        )
    else:
        model = AutoModelForSequenceClassification.from_config(config)
    
    # prepare dataset
    unused_columns = [c for c in ds["train"].features.keys() if c not in ["input_ids", "labels"]]

    ds = ds.map(
        lambda batch: prepare_dataset_items(
            batch,
            genome=genome,
            tokenizer=tokenizer,
            add_rev_strand=data_args.add_rev_strand,
            label2id=config.label2id
        ),
        batched=True,
        remove_columns=unused_columns,
        num_proc=os.cpu_count()
    )
    # if training_args.do_train:
    #     train_ds = ds["train"].map(
    #         lambda batch: prepare_dataset_items(
    #             batch,
    #             genome=genome,
    #             tokenizer=tokenizer,
    #             add_rev_strand=data_args.add_rev_strand,
    #             label2id=config.label2id
    #         ),
    #         batched=True,
    #         remove_columns=unused_columns,
    #         num_proc=os.cpu_count()
    #     )
    # else:
    #     train_ds = None

    # if training_args.do_eval:
    #     eval_ds = ds["eval"].map(
    #         lambda batch: prepare_dataset_items(
    #             batch,
    #             genome=genome,
    #             tokenizer=tokenizer,
    #             add_rev_strand=False,
    #             label2id=config.label2id
    #         ),
    #         batched=True,
    #         remove_columns=unused_columns,
    #         num_proc=os.cpu_count()
    #     )
    # else:
    #     eval_ds = None
    
    ds = ds.shuffle()
    
    train_ds = ds["train"] if training_args.do_train else None
    eval_ds = ds["eval"] if training_args.do_eval else None

    #
    pprint(get_dataset_label_distribution(ds, model.config.id2label))

    #
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics
    )
    
    if training_args.do_train:
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result = trainer.train()
        trainer.save_model()    # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    if data_args.do_test:
        logger.info("*** Test ***")

        test_output = trainer.predict(test_dataset=ds.get("test"))
        metrics = test_output.metrics
        print(metrics)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        print(classification_report(
            test_output.label_ids,
            test_output.predictions.argmax(axis=-1),
            target_names=model.config.label2id.keys()
        ))

        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 10))
        
        ConfusionMatrixDisplay.from_predictions(
            y_true=test_output.label_ids,
            y_pred=test_output.predictions.argmax(axis=-1),
            # display_labels=labels,
            normalize="true",
            ax=ax
        )
        
        # ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.xaxis.set_ticklabels(labels, rotation=45, ha='right')
        ax.yaxis.set_ticklabels(labels)
        ax.set_title(
            f"Confusion matrix on region annotation task"
        )
        model_name = model_args.model_name_or_path.split("/")[-1]
        plt.savefig(f"confusion_matrix_{model_name}.png")

        

        # p, r, f1, s = precision_recall_fscore_support(
        #     test_output.label_ids,
        #     test_output.predictions.argmax(axis=-1),
        # )
        # per_label_metrics = {}
        # for i, label in model.config.id2label.items():
        #     per_label_metrics[label] = {
        #         "precision": p[i],
        #         "recall": r[i],
        #         "f1": f1[i],
        #         # "support": s[i]
        #     }
        

if __name__ == "__main__":
    main()    



"""
python scripts/train_annotation_clf.py --do_train --do_eval \
    --ann_windows_path ./data/embeddings/windows.parquet \
    --genome_path ./data/genome/GCF_000001735.4.fa.gz \
    --fp16 --report_to wandb \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --dataloader_num_workers 6 \
    --seed 42 \
    --num_train_epochs 10  \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --run_name GPN_small_pretrained_b256_no_rev \
    --output_dir models/annotation_clf/GPN_12layers_pretrained_b256_no_rev \
    --model_type MyConvNet \
    --log_level info \
    --learning_rate 1e-3 \
    --model_name_or_path models/GPN_Arabidopsis_multispecies/MyConvNet_12layers_batch256_weight0.1 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 1024 \
    --add_rev_strand False


python scripts/train_annotation_clf.py --do_train --do_eval \
    --ann_windows_path ./data/embeddings/windows.parquet \
    --genome_path ./data/genome/GCF_000001735.4.fa.gz \
    --fp16 --report_to wandb \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --dataloader_num_workers 6 \
    --seed 42 \
    --num_train_epochs 15  \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_strategy epoch \
    --save_total_limit 10 \
    --run_name GPN_12layers_no_pretrained_b256_no_rev \
    --output_dir models/annotation_clf/GPN_12layers_no_pretrained_b256_no_rev \
    --model_type MyConvNet \
    --log_level info \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 1024 \
    --add_rev_strand False \
    --config_overrides n_layers=13


python scripts/train_annotation_clf.py --do_train --do_eval \
    --ann_windows_path ./data/embeddings/windows.parquet \
    --genome_path ./data/genome/GCF_000001735.4.fa.gz \
    --fp16 --report_to wandb \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --dataloader_num_workers 6 \
    --seed 42 \
    --save_strategy epoch \
    --num_train_epochs 4  \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_strategy epoch \
    --save_total_limit 10 \
    --run_name GPN_12layers_pretrained_b256_with_rev \
    --output_dir models/annotation_clf/GPN_12layers_pretrained_b256_with_rev \
    --model_type MyConvNet \
    --model_name_or_path models/GPN_Arabidopsis_multispecies/MyConvNet_12layers_batch256_weight0.1 \
    --log_level info \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 1024 \
    --add_rev_strand True \
    --config_overrides n_layers=13
    
    
GPN_DIR=/mnt/shared_thomas/gpn
python scripts/train_annotation_clf.py --do_train --do_eval \
    --ann_windows_path $GPN_DIR/data/embeddings/windows.parquet \
    --genome_path $GPN_DIR/data/genome/GCF_000001735.4.fa.gz \
    --fp16 --report_to wandb \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --dataloader_num_workers 6 \
    --seed 42 \
    --save_strategy epoch \
    --num_train_epochs 4  \
    --evaluation_strategy steps --eval_steps 1000 \
    --save_strategy epoch \
    --save_total_limit 10 \
    --run_name GPN_24layers_pretrained_b200_no_rev \
    --output_dir models/annotation_clf/GPN_24layers_pretrained_b200_no_rev \
    --model_type MyConvNet \
    --model_name_or_path $GPN_DIR/models/GPN_Arabidopsis_multispecies/MyConvNet_25layers_batch200_weight0.1 \
    --log_level info \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 200 \
    --per_device_eval_batch_size 1024 \
    --add_rev_strand False


"""