# LLM Architecture: Text to Output

## Overview

This is an explanation of the end-to-end data flow of input text in an LLM, from tokenization to output generation. It includes tokenization, embeddings, neural network layers (weights, biases), and attention mechanisms.

---

## 1. Tokenization

**Purpose**: Convert raw text into tokens (smaller units such as words, subwords, or characters) that the model can process.

### Common Tokenization Algorithms

* **BPE (Byte-Pair Encoding)**: Splits text into frequent subword units.
* **WordPiece**: BPE-like but used in BERT. Optimizes for balancing vocabulary size and sequence length.
* **SentencePiece**: Trains subword units on raw text without pre-tokenization.
* **Tools**: `tiktoken` (OpenAI), Hugging Face `tokenizers`

### Special Tokens

* `[CLS]`: Classification token (used in BERT-like models)
* `[SEP]`: Separator token for multiple inputs
* `[PAD]`: Padding token for equal-length sequences

**Example**:

```text
Input: "ChatGPT is powerful."
```
Tokens: ["Chat", "G", "PT", "is", "power", "ful"]
```
---

## 2. Embedding Layer

**Purpose**: Convert token IDs to dense vectors that capture meaning.

### How It Works

* Each token ID is mapped to a vector (e.g., 768 dimensions).
* This layer is learned during training.

**Analogy**: Think of a dictionary: token ID → vector.

```python
embedding = torch.nn.Embedding(vocab_size, embedding_dim)
vector = embedding(token_id)
```

---

## 3. Weights, Biases, and Linear Layers

**Purpose**: Perform transformations on embeddings with parameters learned during training.

### Components

* **Weights**: Matrices that scale and transform data.
* **Biases**: Values added to shift output.
* **Linear Layer**: Transformation `output = input @ W + b` is applied

### In LLMs

There are a few linear layers in every transformer block for computing queries (Q), keys (K), and values (V) in the attention mechanism.

---
## 4. Attention Mechanism (Self-Attention)

**Purpose**: Allows the model to focus on the important sections of the sequence.

### How It Works

* Compute **Q**, **K**, **V** for each token:

```text
Q = Embedding * Wq
K = Embedding * Wk
V = Embedding * Wv
```

* Compute attention scores: `Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V`

### Output

Attention-weighted sum of values (V) gives context-aware representations.

**Types**:

* **Scaled Dot-Product Attention**
* **Multi-Head Attention**: Multiple applications of attention in parallel

---

## 5. Transformer Block

Each block is composed of:

1. **Multi-Head Attention**
2. **Add & Norm** (residual connections + layer normalization)
3. **Feedforward Neural Network** (MLP layer)
4. **Add & Norm** again

These blocks are repeated numerous times (e.g., 12, 24, 96) in large models.

---  
## 6. Output Layer

**Purpose**: Convert the final vector back to a probability distribution over the vocabulary.

### Steps

* Final vector is passed through a linear layer (projection to vocab size)
* Apply **softmax** to get probabilities

**Example**:

```text
"ChatGPT is pow." → [0.01, 0.02, ., 0.95 for "powerful"]
```

---  
## 7. Complete Flow Summary

```text
Text → Tokenizer → Token IDs → Embedding Layer → Transformer Blocks (with Attention) → Output Probabilities → Selected Token
```

---

## Tools & Libraries Used

* **Tokenization**: `tiktoken`, Hugging Face `tokenizers`
* **Embedding**: `torch.nn.Embedding`
* **Architecture**: PyTorch `nn.Module`, attention layers
* **Vector DB (optional)**: FAISS, ChromaDB for retrieval-augmented generation

---

## Sample Output

Input:

```text
"The capital of France is"
```

Output:

```text
"Paris"
```

With top prediction probability = 0.97

---

