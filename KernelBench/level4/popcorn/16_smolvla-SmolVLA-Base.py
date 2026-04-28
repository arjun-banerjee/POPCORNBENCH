import torch
from transformers import AutoConfig, AutoModelForVision2Seq


class Model(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name, config=self.config)

    def forward(self, input_ids, pixel_values):
        return self.model(input_ids=input_ids, pixel_values=pixel_values).logits


model_name = "smolvla/SmolVLA-Base"
config = AutoConfig.from_pretrained(model_name)
vocab_size = config.vocab_size
sequence_length = 128
batch_size = 1
image_size = 224


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    return [input_ids, pixel_values]


def get_init_inputs():
    return [model_name, config]
