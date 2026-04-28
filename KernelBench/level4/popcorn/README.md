# Level 4 Popcorn: Whole-Model Wrappers

These are full-model references in the same style as `level4/original`.
Each file loads a pretrained model and runs a full forward pass.

## Included domains

- LLMs: Llama, Mistral, Mixtral, Gemma, Falcon, StarCoder2, Phi, Pythia, GPT-2, OPT, DeepSeek, Qwen
- VLMs/VLAs: LLaVA, Idefics2, OpenVLA, SmolVLA, XVLA
- Diffusion: SDXL UNet
- Protein: ESM2
- Physics: ClimaX

All files expose `Model`, `get_inputs()`, and `get_init_inputs()`.
