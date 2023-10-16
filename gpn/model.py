import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import (
    AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification,
    PretrainedConfig, PreTrainedModel
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput
)
from typing import Optional, Tuple, Union

from .modules import (
    TransposeLayer, ConvLayer, OneHotEmbedding, get_dilation_schedule,
    GPNEmbedding, FocalLoss
)


class MLMHead(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(self, hidden_state):
        return self.decoder(hidden_state)


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


class ConvNetConfig(PretrainedConfig):
    model_type = "ConvNet"

    def __init__(
        self,
        vocab_size=7,
        hidden_size=512,
        n_layers=25,
        kernel_size=9,
        dilation_double_every=1,
        dilation_max=32,
        dilation_cycle=6,
        initializer_range=0.02,
        # for classification head:
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        #
        one_hot: bool = True,
        pad_token_id: int = 3,
        mlm_loss: str = "default",
        mlm_loss_focal_gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_double_every = dilation_double_every
        self.dilation_max = dilation_max
        self.dilation_cycle = dilation_cycle
        self.initializer_range = initializer_range
        # for classification head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.one_hot = one_hot
        self.pad_token_id = pad_token_id
        self.mlm_loss = mlm_loss
        self.mlm_loss_focal_gamma = mlm_loss_focal_gamma


class ConvNetPreTrainedModel(PreTrainedModel):
    config_class = ConvNetConfig
    base_model_prefix = "model"
    #supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ConvNetModel(ConvNetPreTrainedModel):
    def __init__(
        self,
        config: ConvNetConfig,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config
        #
        self.embedding = OneHotEmbedding(config.hidden_size)
        self.dilation_schedule = get_dilation_schedule(config)

        conv_layers = [
            ConvLayer(
                hidden_size=config.hidden_size,
                kernel_size=config.kernel_size,
                dilation=self.dilation_schedule[i],
            )
            for i in range(config.n_layers)
        ]
        self.encoder = nn.Sequential(*conv_layers)
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return BaseModelOutput(last_hidden_state=x)


class ConvNetForMaskedLM(ConvNetPreTrainedModel):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.config = config
        self.model = ConvNetModel(config)
        self.cls = MLMHead(config)
        self.post_init()

    def forward(self, input_ids=None, labels=None, loss_weight=None, **kwargs):
        hidden_state = self.model(input_ids=input_ids, **kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            labels = labels.view(-1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels)
            loss_weight = loss_weight.view(-1)
            loss_weight[labels==-100] = 0.0
            loss = (loss * loss_weight / loss_weight.sum()).sum()
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )


class MyConvNetConfig(ConvNetConfig):
    model_type = "MyConvNet"


class MyConvNetPreTrainedModel(PreTrainedModel):
    config_class = MyConvNetConfig
    base_model_prefix = "model"
    #supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MyConvNetModel(MyConvNetPreTrainedModel):
    """ """

    def __init__(
        self,
        config: MyConvNetConfig,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config

        if config.one_hot:
            self.embedding = OneHotEmbedding(config.vocab_size)
            first_conv = nn.Sequential(
                TransposeLayer(),
                nn.Conv1d(
                    in_channels=config.vocab_size,
                    out_channels=config.hidden_size,
                    padding="same",
                    kernel_size=config.kernel_size,
                ),
                TransposeLayer(),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size),
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                padding_idx=config.pad_token_id
            )
            first_conv = ConvLayer(
                hidden_size=config.hidden_size,
                kernel_size=config.kernel_size,
                # dilation=self.dilation_schedule[i],
            )
        #
        self.dilation_schedule = get_dilation_schedule(config)
        #
        conv_layers = [
            first_conv,
            *[
                ConvLayer(
                    hidden_size=config.hidden_size,
                    kernel_size=config.kernel_size,
                    dilation=self.dilation_schedule[i],
                )
                for i in range(config.n_layers - 1)
            ]
        ]
        self.encoder = nn.Sequential(*conv_layers)
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return BaseModelOutput(last_hidden_state=x)


class MyConvNetForMaskedLM(MyConvNetPreTrainedModel):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.config = config
        self.model = MyConvNetModel(config)
        self.cls = MLMHead(config)
        self.post_init()

    def forward(self, input_ids=None, labels=None, loss_weight=None, **kwargs):
        hidden_state = self.model(input_ids=input_ids, **kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = None
        b, s = input_ids.shape
        if labels is not None:
            if loss_weight is not None:
                loss_fct = CrossEntropyLoss(reduction="none")
                labels = labels.view(-1)
                loss = loss_fct(logits.view(-1, self.config.vocab_size), labels)
                loss_weight = loss_weight.view(-1)
                loss_weight[labels==-100] = 0.0
                loss = (loss * loss_weight / loss_weight.sum()).sum()
            else:
                if self.config.mlm_loss == "default":
                    loss_fct = CrossEntropyLoss()
                elif self.config.mlm_loss == "focal":
                    loss_fct = FocalLoss(gamma=self.config.mlm_loss_focal_gamma)
                    logits = torch.softmax(logits, dim=-1)
                #
                loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
                
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )

class ConvNetForSequenceClassification(ConvNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = ConvNetModel(config)
        self.classifier = ClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor]]:
        r"""
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


AutoConfig.register("ConvNet", ConvNetConfig)
AutoModel.register(ConvNetConfig, ConvNetModel)
AutoModelForMaskedLM.register(ConvNetConfig, ConvNetForMaskedLM)
AutoModelForSequenceClassification.register(ConvNetConfig, ConvNetForSequenceClassification)

AutoConfig.register("MyConvNet", MyConvNetConfig)
AutoModel.register(MyConvNetConfig, MyConvNetModel)
AutoModelForMaskedLM.register(MyConvNetConfig, MyConvNetForMaskedLM)



from transformers import RoFormerConfig, RoFormerModel, RoFormerForMaskedLM
from transformers.models.roformer.modeling_roformer import RoFormerEncoder, RoFormerOnlyMLMHead, RoFormerSinusoidalPositionalEmbedding


class GPNRoFormerConfig(RoFormerConfig):
    model_type = "GPNRoFormer"

    def __init__(
        self,
        n_aux_features=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_aux_features = n_aux_features


class GPNRoFormerPreTrainedModel(PreTrainedModel):
    config_class = GPNRoFormerConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RoFormerSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, RoFormerEncoder):
            module.gradient_checkpointing = value


class GPNRoFormerModel(GPNRoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = GPNEmbedding(
            config.vocab_size, config.n_aux_features, config.hidden_size,
        )
        self.encoder = RoFormerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, input_probs=None, aux_features=None, **kwargs):
        x = self.embedding(input_ids=input_ids, input_probs=input_probs, aux_features=aux_features)
        x = self.encoder(x, **kwargs)
        return x


class GPNRoFormerForMaskedLM(GPNRoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = GPNRoFormerModel(config)
        self.cls = RoFormerOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        labels=None,
        output_probs=None,
        loss_weight=None,
        **kwargs
    ):
        hidden_state = self.model(**kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = None

        if labels is not None and loss_weight is None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
        elif labels is not None and loss_weight is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            labels = labels.view(-1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels)  # what if we first exclude the ones with -100??
            loss_weight = loss_weight.view(-1)
            loss_weight[labels==-100] = 0.0
            loss = (loss * loss_weight / loss_weight.sum()).sum()
        elif output_probs is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            output_probs = output_probs.view(-1, self.config.vocab_size)
            exclude = (output_probs == 0.0).all(dim=-1)
            output_probs = output_probs[~exclude]
            logits = logits.view(-1, self.config.vocab_size)[~exclude]
            loss_weight = loss_weight.view(-1)[~exclude]
            loss = loss_fct(logits, output_probs)
            loss = (loss * loss_weight / loss_weight.sum()).sum()
 
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )


AutoConfig.register("GPNRoFormer", GPNRoFormerConfig)
AutoModel.register(GPNRoFormerConfig, GPNRoFormerModel)
AutoModelForMaskedLM.register(GPNRoFormerConfig, GPNRoFormerForMaskedLM)
