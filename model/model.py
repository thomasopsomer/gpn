import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, cross_entropy
import torch.nn.functional as F
from transformers import get_scheduler, AutoModel, PretrainedConfig
from torch.optim import AdamW
import torchmetrics


def calculate_auroc(outputs, n_output, output_classes):
    preds = torch.cat([output["logits"] for output in outputs])
    targets = torch.cat([output["Y"] for output in outputs])
    n_regions = len(preds) // 2

    preds = (preds[:n_regions] + preds[n_regions:]) / 2
    if len(outputs) > 2:  # except for the sanity check
        assert(torch.allclose(targets[:n_regions], targets[n_regions:]))
    targets = targets[:n_regions]
    aurocs = torch.empty(n_output)
    for i in range(n_output):
        if targets[:, i].sum() < 30:
            aurocs[i] = float("nan")
        else:
            aurocs[i] = torchmetrics.functional.auroc(preds[:, i], targets[:, i])
    res = (
        aurocs.nanmedian(),
        aurocs[output_classes["dnase"]].nanmedian(),
        aurocs[output_classes["tf"]].nanmedian(),
        aurocs[output_classes["histone"]].nanmedian(),
        {f"feature_{i}": auroc for i, auroc in enumerate(aurocs)}
    )
    return res


class DeepSEAModel(pl.LightningModule):
    def __init__(
        self,
        n_input=None,
        n_output=None,
        learning_rate=None,
        reduce_lr_on_plateau_patience=None,
        output_classes=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_input = n_input
        self.n_output = n_output
        self.lr = lr
        self.reduce_lr_on_plateau_patience = reduce_lr_on_plateau_patience
        self.output_classes = output_classes

        self.Conv1 = nn.Conv1d(in_channels=self.n_input, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(53*960, 925)
        self.Linear2 = nn.Linear(925, self.n_output)

    def forward(self, x):
        x = one_hot(x, num_classes=self.n_input).float()
        x = torch.transpose(x, 1, 2)
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = x.view(-1, 53*960)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x

    def loss(self, logits, Y):
        res = F.binary_cross_entropy_with_logits(logits, Y.float(), pos_weight=torch.full(Y.shape[1], 8.0))
        return res

    def training_step(self, batch, batch_idx):
        X, Y = batch
        logits = self(X)
        loss = self.loss(logits, Y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        logits = self(**X)
        return {"logits": logits, "Y": Y}

    def validation_step_end(self, outputs):
        return outputs

    def validation_epoch_end(self, outputs):
        m1, m2, m3, m4, _ = calculate_auroc(outputs, self.n_output, self.output_classes)
        self.log("val_neg_median_auroc", -m1)  # negative so it's a minimization problem
        self.log("val_median_auroc_dnase", m2)
        self.log("val_median_auroc_tf", m3)
        self.log("val_median_auroc_histone", m4)

    def test_step(self, batch, batch_idx):
        X, Y = batch
        logits = self(**X)
        return {"logits": logits, "Y": Y}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):
        m1, m2, m3, m4, aurocs = calculate_auroc(outputs, self.n_output, self.output_classes)
        self.log("test_neg_median_auroc", -m1)  # negative so it's a minimization problem
        self.log("test_median_auroc_dnase", m2)
        self.log("test_median_auroc_tf", m3)
        self.log("test_median_auroc_histone", m4)
        self.log_dict(aurocs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_on_plateau_patience,
            factor=0.1,
            threshold=0.0,
            threshold_mode="abs",
            verbose=True,
        )
        monitor = "val_neg_median_auroc"
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor=monitor)




#class DeepSEAModel(pl.LightningModule):
#    def __init__(
#        self,
#        language_model_name,
#        n_output=919,
#        num_training_steps=None,
#        num_warmup_steps=None,
#        lr=None,
#        **kwargs,
#    ):
#        super().__init__()
#        self.save_hyperparameters()
#
#        self.language_model_name = language_model_name
#        self.n_output = n_output
#        self.num_training_steps = num_training_steps
#        self.num_warmup_steps = num_warmup_steps
#        self.lr = lr
#
#        self.language_model = AutoModel.from_pretrained(language_model_name)
#        #config = PretrainedConfig.get_config_dict(language_model_name)
#        #self.language_model = AutoModel.from_config(config)
#        self.hidden_size = PretrainedConfig.get_config_dict(language_model_name)[0]["hidden_size"]
#        self.dropout = nn.Dropout(0.1)
#        #self.classifier = nn.Linear(self.hidden_size, n_output)
#        self.classifier = nn.Linear(3*self.hidden_size, n_output)
#
#        #for i in range(919):
#        #    setattr(self, f"val_AUROC_{i}", AUROC())
#
#    def forward(
#        self,
#        input_ids=None,
#        attention_mask=None,
#        #token_type_ids=None,
#    ):
#        #print(input_ids.shape)
#        #batch_size = len(input_ids)
#        #print(input_ids)
#        #print(attention_mask)
#        #raise Exception('debug')
#
#        input_ids = input_ids.unfold(1, 400, 300).reshape(-1, 400)
#        attention_mask = attention_mask.unfold(1, 400, 300).reshape(-1, 400)
#
#        x = self.language_model(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"]
#        #print(x)
#        #raise Exception("debug")
#        #print(x.shape)
#        x = x.view(-1, 3, self.hidden_size).view(-1, 3 * self.hidden_size)
#        #print(x.shape)
#        #raise Exception("debug")
#
#        x = self.dropout(x)
#        x = self.classifier(x)
#        return x
#
#    def loss(self, logits, Y):
#        res = F.binary_cross_entropy_with_logits(logits, Y.float())
#        return res
#
#    def training_step(self, batch, batch_idx):
#        X, Y = batch
#        logits = self(**X)
#        loss = self.loss(logits, Y)
#        self.log("train_loss", loss)
#        return loss
#
#    def validation_step(self, batch, batch_idx):
#        X, Y = batch
#        logits = self(**X)
#        loss = self.loss(logits, Y)
#        self.log("val_loss", loss, sync_dist=True)
#        return {"logits": logits, "Y": Y}
#
#    def validation_step_end(self, outputs):
#        return outputs
#
#    def validation_epoch_end(self, outputs):
#        m1, m2, m3, _ = calculate_auroc(outputs)
#        self.log("val_median_auroc_dnase", m1)
#        self.log("val_median_auroc_tf", m2)
#        self.log("val_median_auroc_histone", m3)
#
#    def test_step(self, batch, batch_idx):
#        X, Y = batch
#        logits = self(**X)
#        loss = self.loss(logits, Y)
#        self.log("test_loss", loss, sync_dist=True)
#        return {"logits": logits, "Y": Y}
#
#    def test_step_end(self, outputs):
#        return outputs
#
#    def test_epoch_end(self, outputs):
#        m1, m2, m3, aurocs = calculate_auroc(outputs)
#        self.log("test_median_auroc_dnase", m1)
#        self.log("test_median_auroc_tf", m2)
#        self.log("test_median_auroc_histone", m3)
#        self.log_dict(aurocs)
#
#    def configure_optimizers(self):
#        optimizer = AdamW(self.parameters(), lr=self.lr)
#        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
#        scheduler = {
#            "scheduler": lr_scheduler,
#            "interval": "step",
#            "frequency": 1,
#        }
#        #print(lr_scheduler.get_last_lr())
#        #raise Exception("debug")
#
#        #monitor = "val_loss"
#        #return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor=monitor)
#        return dict(optimizer=optimizer, lr_scheduler=scheduler)