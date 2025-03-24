import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions 
import typing as t
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput




def register_model(model_name, config_class):
    def decorator(cls):
        # This would be the actual registration logic in production code
        return cls
    return decorator


class LSTMConfig(PretrainedConfig):
    model_type = "lstm"

    def __init__(self, vocab_size=30522, embedding_dim=256, hidden_size=512, num_layers=2, dropout=0.1, **kwargs):
        super().__init__(vocab_size=vocab_size, **kwargs)  # Ensure correct argument passing

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


@register_model("lstm_lm", LSTMConfig)
class LSTMConfig(PretrainedConfig):
    model_type = "lstm"

    def __init__(self, vocab_size=30522, embedding_dim=None, hidden_size=512, num_layers=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim  # Placeholder (None if not set)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

class LSTMForLanguageModeling(PreTrainedModel):
    config_class = LSTMConfig

    def __init__(self, config):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim or 256  # Default to 256 if None
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        self.output = nn.Linear(self.hidden_size, self.vocab_size)

        self.init_weights()

    def forward(self, input_ids, labels=None, **kwargs):
        # Ignore unwanted kwargs like attention_mask
        if "attention_mask" in kwargs:
            kwargs.pop("attention_mask")

        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        logits = self.output(lstm_out)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )
