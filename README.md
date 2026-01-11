# Transformer Architecture Experiments

This repository contains PyTorch implementations of Transformer architectures to demonstrate their core mechanisms: **Autoencoding (Masked Language Modeling)** and **Autoregression (Seq2Seq Generation)**.

## 📂 Project Structure

```
transformer-experiments/
├── transformer-encoder-autoencoding/  # Experiment 1: Encoder-only (BERT-style)
│   ├── encoder.py                     # Self-Attention & Feed-Forward blocks
│   ├── train_mlm.py                   # Training script for Masked Language Model
│   ├── visualize_attention.py         # Attention heatmap generation
│   └── results/                       # Saved models and visualization plots
│
└── transformer-seq2seq/               # Experiment 2: Encoder-Decoder (Translation-style)
    ├── transformer.py                 # Full Seq2Seq Transformer
    ├── train.py                       # Training script for Text Generation
    ├── inference.py                   # Script to generate text from prompts
    └── samples/                       # Generated outputs and attention plots
```

---

## 🔬 Experiment 1: Transformer Encoder (Autoencoding)

**Objective:** Understand Self-Attention and how models like BERT learn context by reconstructing masked words.

### Tasks Implemented
- **Masked Language Modeling (MLM):** The model predicts hidden words (e.g., `Input: "Transformers use [MASK] attention"` → `Output: "self"`).
- **Self-Attention Visualization:** Visualizing how words attend to each other.

### Usage
```bash
# 1. Train the model
python3 transformer-encoder-autoencoding/train_mlm.py

# 2. Visualize Attention Heatmaps
python3 transformer-encoder-autoencoding/visualize_attention.py
```

<p align="center">
  <img src="transformer-encoder-autoencoding/results/attention_heatmap.png" width="400" alt="Attention Heatmap">
  <br>
  <em>Figure 1: Self-Attention weights visualization.</em>
</p>

---

## 🔬 Experiment 2: Transformer Decoder (Seq2Seq)

**Objective:** Understand Autoregression, Causal Masking, and Encoder-Decoder interaction for text generation.

### Tasks Implemented
- **Seq2Seq Generation:** Training a model to paraphrase or answer questions (e.g., `Input: "AI improves healthcare"` → `Output: "AI enhances medical diagnosis and treatment"`).
- **Causal Masking:** Ensuring the model predicts tokens one step at a time without seeing the future.

### Usage
```bash
# 1. Train the Seq2Seq model
python3 transformer-seq2seq/train.py

# 2. Run Inference (Generate Text)
python3 transformer-seq2seq/inference.py

# 3. Visualize Decoder Attention
python3 transformer-seq2seq/visualize_attention.py
```

---

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rkrishnaprasath1/transformer-experiments.git
   cd transformer-experiments
   ```

2. **Install Dependencies:**
   ```bash
   pip install torch matplotlib numpy
   ```
   *(Note: If you don't have a GPU, install the CPU version of PyTorch).*

## 📊 Results Summary
- **MLM:** Successfully reconstructs masked terms in educational sentences with low loss (< 0.005).
- **Seq2Seq:** accurately paraphrases inputs and generates answers to set questions after ~160 epochs.

---
*Created by Krishna*
