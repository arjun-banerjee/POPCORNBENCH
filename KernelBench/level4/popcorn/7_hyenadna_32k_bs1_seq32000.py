import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

class Model(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=self.config, trust_remote_code=True
        )

    def forward(self, x):
        return self.model(x).logits

model_name = "LongSafari/hyenadna-small-32k-seqlen-hf"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.num_labels = 2
vocab_size = config.vocab_size
sequence_length = 32000
batch_size = 1

def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]

def get_init_inputs():
    return [model_name, config]