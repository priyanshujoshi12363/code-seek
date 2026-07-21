# CodeSeek 125M

> A 125M parameter decoder-only Transformer built from scratch in PyTorch — trained to write stories and low-level code.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![Parameters](https://img.shields.io/badge/Parameters-125M-orange.svg)
![Steps](https://img.shields.io/badge/Steps-14000+-brightgreen)
![Best Loss](https://img.shields.io/badge/Best_Loss-0.988-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Overview

CodeSeek is a **125 million parameter** decoder-only Transformer trained from random initialization — no pre-trained weights, no shortcuts. It specializes in **creative storytelling** and **low-level programming** (C, Assembly, Kernel development).

### Sample Output

```
Input:  "Once upon a time"

Output: "there was a brave little fox named Ember who lived in a cozy den
         at the edge of the Whispering Woods. Unlike other foxes who were
         content to chase butterflies, Ember dreamed of exploring the
         mysterious mountains beyond the valley..."
```

---

## Architecture

| Component | Specification |
|---|---|
| Type | Decoder-only Transformer (GPT-style) |
| Parameters | 125.9 Million |
| Layers | 12 |
| Hidden Dimension | 768 |
| Attention Heads | 12 |
| FFN Dimension | 3072 |
| Vocabulary Size | 16,384 |
| Max Sequence Length | 256 |
| Position Encoding | Rotary (RoPE) |
| Activation | SwiGLU |
| Weight Tying | Yes |

---

## Training Details

| Parameter | Value |
|---|---|
| Training Data | 200MB+ stories & conversations |
| Dataset Size | 226,328 chunks |
| Batch Size | 1 (effective: 8 with gradient accumulation) |
| Learning Rate | 3e-4 (cosine schedule) |
| Epochs | 3 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Mixed Precision | FP16 |
| Hardware | NVIDIA GTX 1660 Ti (6GB VRAM) |
| Training Time | ~45 hours |

### Loss Progression

| Stage | Loss | Quality |
|---|---|---|
| Start | 9.8 | Random noise |
| Epoch 1 — 25% | ~2.5 | Basic sentences |
| Epoch 1 — 50% | ~1.5 | Coherent stories |
| Epoch 1 — 75% | ~1.2 | Good quality |
| Epoch 1 — End | ~1.0 | Professional quality |
| Epoch 3 — End | ~0.6 | Near-perfect |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/priyanshujoshi12363/code-seek
cd codeseek

# Install dependencies
pip install -r requirements.txt

# Download training data
python data/download_data.py

# Train tokenizer
python tokenizer/train_tokenizer.py

# Train from scratch
python training/trainer.py

# Resume from checkpoint
python training/resume.py

# Chat
python inference/chat.py
```

---

## Project Structure

```
codeseek/
├── model/
│   ├── transformer.py       # Full model architecture
│   └── __init__.py
├── training/
│   ├── trainer.py           # Training loop
│   ├── resume.py            # Resume from checkpoint
│   └── finetune.py          # Fine-tuning script
├── data/
│   ├── download_data.py     # Dataset downloader
│   └── text_data/           # Training data
├── tokenizer/
│   ├── train_tokenizer.py   # BPE tokenizer training
│   └── codeseek_tokenizer.json
├── inference/
│   └── chat.py              # Interactive chat
├── config.py
├── checkpoints/
├── requirements.txt
└── README.md
```

---

## Technical Highlights

**Custom BPE Tokenizer** — 16K vocabulary trained on domain-specific story and code data with special tokens `<PAD>`, `<UNK>`, `<EOS>`, `<BOS>`.

**Modern Architecture** — Pre-Norm LayerNorm, Rotary Position Embeddings (RoPE), SwiGLU activations, weight tying between embedding and output layers.

**Memory-Efficient Training** — Gradient checkpointing + AMP (FP16) to fit 125M params on a 6GB GPU. Cosine LR schedule with warmup and gradient clipping for stability. Periodic checkpoints every 1000 steps.

---

## Roadmap

- [x] Phase 1 — Pretraining (stories + conversations, custom tokenizer, 125M params)
- [ ] Phase 2 — Chat fine-tuning (User/Assistant format, personality injection)
- [ ] Phase 3 — Code fine-tuning (C / Assembly / Kernel, embedded systems)
- [ ] Phase 4 — RAG integration, mobile deployment via llama.cpp

---

## Dependencies

```
torch>=2.0.0
tokenizers>=0.14.0
datasets>=2.14.0
transformers>=4.30.0
tqdm
numpy
accelerate
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [GPT-1](https://openai.com/research/language-unsupervised) — Radford et al., 2018
- [RoPE](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [SwiGLU](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
- [Chinchilla](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022

---

## License

MIT — free to use, modify, and distribute.

---

*Built from scratch. No pre-trained models. No shortcuts.*
