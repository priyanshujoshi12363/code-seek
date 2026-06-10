```markdown
# CodeSeek 125M — A Custom Transformer for Storytelling & Low-Level Coding

<div align="center">

🚀 **A 125M parameter decoder-only Transformer built from scratch in PyTorch**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Parameters](https://img.shields.io/badge/Parameters-125M-orange.svg)]()

</div>

---

## Overview

CodeSeek is a 125 million parameter decoder-only Transformer trained from scratch. It specializes in storytelling and low-level programming including C, Assembly, and Kernel development. The entire model is built in PyTorch with zero pre-trained dependencies.

---

## Key Features

- 125M Parameters at GPT-1 scale, trained from random initialization
- Rotary Position Embeddings for modern position encoding
- SwiGLU Activation outperforming GELU for language tasks
- Multi-Head Self-Attention with 12 heads and 768 hidden dimension
- Custom BPE Tokenizer with 16K vocabulary trained on story and code data
- Mixed Precision Training with FP16 and gradient scaling
- Cosine Learning Rate Schedule with warmup
- Gradient Checkpointing for memory efficient training

---

## Architecture

| Component | Specification |
|-----------|--------------|
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

## Project Structure

```
codeseek/
├── model/
│   ├── transformer.py
│   └── __init__.py
├── training/
│   ├── trainer.py
│   ├── resume.py
│   ├── finetune.py
│   └── code_pretrain.py
├── data/
│   ├── download_data.py
│   ├── text_data/
│   └── code_data/
├── tokenizer/
│   ├── train_tokenizer.py
│   ├── update_tokenizer.py
│   └── codeseek_tokenizer.json
├── inference/
│   └── chat.py
├── config.py
├── checkpoints/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/priyanshujoshi12363/code-seek.git
cd codeseek
pip install -r requirements.txt
```

### Download Training Data

```bash
python data/download_data.py
```

### Train Tokenizer

```bash
python tokenizer/train_tokenizer.py
```

### Train Model

```bash
python training/trainer.py
```

### Resume From Checkpoint

```bash
python training/resume.py
```

### Code Pretraining

```bash
python training/code_pretrain.py
```

### Chat With CodeSeek

```bash
python inference/chat.py
```

---

## Training Details

### Story Pretraining

| Parameter | Value |
|-----------|-------|
| Data | 200MB+ stories and conversations |
| Chunks | 226,328 |
| Batch Size | 1 (effective 8 with gradient accumulation) |
| Learning Rate | 3e-4 with cosine schedule |
| Epochs | 3 (1 completed) |
| Optimizer | AdamW (beta1=0.9, beta2=0.95) |
| Hardware | NVIDIA GTX 1660 Ti (6GB VRAM) |

### Code Pretraining

| Parameter | Value |
|-----------|-------|
| Data | 493MB embedded code and documentation |
| Chunks | 1,082,601 |
| Batch Size | 1 (effective 8 with gradient accumulation) |
| Learning Rate | 1e-4 with cosine schedule |
| Epochs | 2 (in progress) |
| Hardware | NVIDIA GTX 1660 Ti (6GB VRAM) |

---

## Performance

### Story Pretraining Loss Progression

| Milestone | Loss | Quality |
|-----------|------|---------|
| Start | 9.8 | Random initialization |
| 25 Percent | 2.5 | Basic sentence structure |
| 50 Percent | 1.5 | Good story quality |
| 75 Percent | 1.2 | Great storytelling |
| Epoch 1 End | 1.0 | Professional quality |

### Code Pretraining Loss Progression

| Milestone | Loss | Quality |
|-----------|------|---------|
| Start | 7.3 | Story model baseline |
| 1000 Steps | 1.5 | Basic C patterns emerging |
| 4000 Steps | 0.2 | Real API names appearing |
| 10000 Steps | TBD | In progress |

### Sample Story Output

```
Input: Once upon a time
Output: there was a brave little fox named Ember who lived in a 
cozy den at the edge of the Whispering Woods. Unlike other foxes 
who were content to chase butterflies, Ember dreamed of exploring 
the mysterious mountains beyond the valley...
```

### Sample Code Output

```
Input: void setup() {
Output: pinMode(LED_BUILTIN, OUTPUT);
        Serial.begin(9600);
        esp_wifi_init(&cfg);
        xTaskCreate(task, "task", 2048, NULL, 1, NULL);
```

---

## Technical Features

### Custom Tokenizer

- Byte-Pair Encoding with 16,384 vocabulary size
- Special tokens for PAD, UNK, EOS, and BOS
- Trained on domain-specific story and code data
- Handles C syntax, embedded APIs, and natural language

### Modern Architecture Choices

- Pre-Norm with Layer Normalization for training stability
- Rotary Position Embeddings for better sequence handling
- SwiGLU activation in Feed-Forward networks
- Weight Tying between embedding and output layers
- Causal self-attention with triangular masking

### Training Optimizations

- Automatic Mixed Precision for faster training
- Gradient Checkpointing to reduce memory usage
- Cosine Learning Rate Schedule with warmup period
- Gradient Clipping to prevent training instability
- Periodic Checkpointing every 1000 steps
- Best model tracking and saving

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Story Pretraining | Complete | 200MB stories, 1 epoch completed |
| Code Pretraining | In Progress | 493MB embedded code, training |
| Chat Fine-Tuning | Planned | User/Assistant format training |
| Code Instruction | Planned | Code-specific instruction tuning |
| RAG Integration | Planned | Vector database for documentation |
| Mobile Deployment | Planned | llama.cpp quantization |
| Vector Memory | Planned | Long-term conversation memory |

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

## Live Training Stats

- Current Phase: Code Pretraining
- Steps Completed: 4,000+
- Best Code Loss: 0.204
- Total Code Chunks: 1,082,601
- Model Size: 125.9M parameters
- Status: Training

---

## License

MIT License. Feel free to use, modify, and distribute.

---

## Acknowledgments

- Attention Is All You Need by Vaswani et al., 2017
- GPT-1 by Radford et al., 2018
- RoPE by Su et al., 2021
- SwiGLU by Shazeer, 2020
- Chinchilla by Hoffmann et al., 2022

---

<div align="center">

**Built from scratch. No pre-trained models. No shortcuts.**

</div>
```
