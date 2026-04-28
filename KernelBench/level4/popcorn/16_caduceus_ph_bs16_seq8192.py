import torch
from transformers import AutoModelForMaskedLM, AutoConfig

class Model(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, config=self.config, trust_remote_code=True
        )

    def forward(self, x):
        return self.model(x).logits

model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
vocab_size = config.vocab_size
sequence_length = 8192
batch_size = 16

def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]

def get_init_inputs():
    return [model_name, config]