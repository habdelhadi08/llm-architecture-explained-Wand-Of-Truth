# 🧠 LLM Architecture: From Text to Output

This guide explains how Large Language Models (LLMs) like GPT process text, from raw input to generated output. We’ll walk through each step — tokenization, embedding, attention, and final prediction — using simplified examples and clear terms.

---

## 1. 🔤 Tokenization

**Goal**: Break down raw text into smaller chunks (tokens) that a model can understand.

These tokens might be:

* **Words** (`"ChatGPT"`)
* **Subwords** (`"power" + "ful"`)
* **Characters**

### Popular Tokenization Algorithms

| Algorithm         | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| **BPE**           | Byte-Pair Encoding: merges frequent pairs of characters or subwords. |
| **WordPiece**     | Similar to BPE, used in BERT.                                        |
| **SentencePiece** | Does not require pre-tokenization; trains on raw text.               |

### Special Tokens

These are added to handle structure:

* `[CLS]`: Beginning of input (used for classification)
* `[SEP]`: Separator between parts (e.g., question and context)
* `[PAD]`: Padding for uniform sequence length

### Example

```text
Input: "ChatGPT is powerful."
Tokenized: ["Chat", "GPT", " is", " powerful", "."]
Token IDs: [2011, 4536, 318, 1745, 13]
```

---

## 2. 📦 Embedding Layer

**Goal**: Convert each token ID into a dense vector (a list of numbers) that represents its meaning.

### How It Works

* Each token ID is mapped to a high-dimensional vector (e.g., 768 floats).
* These vectors are **learned during training**.

### Analogy

Think of a token as a word, and an embedding as a numerical fingerprint that captures its context.

```python
embedding = torch.nn.Embedding(vocab_size, embedding_dim)
vector = embedding(token_id)  # Outputs a vector like [0.32, -0.12, ..., 0.99]
```

---

## 3. 🔀 Linear Layers: Weights & Biases

**Goal**: Transform input vectors using learned mathematical operations.

### Components

* **Weights (W)**: Matrix that scales/rotates the input.
* **Bias (b)**: Added to shift the output.
* **Linear Transformation**:

  $$
  \text{Output} = \text{Input} \times W + b
  $$

These are used throughout the model, especially in computing attention.

---

## 4. 🎯 Attention Mechanism (Self-Attention)

**Goal**: Let the model decide which words to focus on in a sentence — even words far apart.

### Steps

1. Compute three vectors for each token:

   * **Q** (Query): What this word wants to know.
   * **K** (Key): What other words offer.
   * **V** (Value): The actual information.

   $$
   Q = \text{Embedding} \times W_q,\quad K = \text{Embedding} \times W_k,\quad V = \text{Embedding} \times W_v
   $$

2. Calculate attention scores:

   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \times V
   $$

### Output

Each token now holds a **context-aware representation** — it “knows” what else is in the sentence.

### Types of Attention

* **Scaled Dot-Product Attention**: Basic form.
* **Multi-Head Attention**: Multiple attentions in parallel to learn different types of relationships.

---

## 5. 🔄 Transformer Block

Each transformer block contains:

1. **Multi-Head Attention**
2. **Add & Normalize** (residual + layer norm)
3. **Feedforward Layer (MLP)**
4. **Add & Normalize** (again)

These blocks are **stacked** — e.g., GPT-3 uses 96 such blocks!

---

## 6. 🧼 Output Layer

**Goal**: Predict the next token by generating probabilities over the entire vocabulary.

### Steps

1. Final vector from transformer → Linear Layer → Vocabulary Size
2. Apply **softmax** → Converts to probability distribution

### Example

```text
Input: "ChatGPT is pow"
Model Output (softmax):
"powerful": 0.95  
"powered": 0.02  
"power": 0.01  
→ Selected token = "powerful"
```

---

## 7. ♻️ End-to-End Flow

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embedding Layer
   ↓
Transformer Blocks (with Self-Attention)
   ↓
Final Vector
   ↓
Linear Layer + Softmax
   ↓
Next Token Prediction
```

---

## 🛠️ Tools and Libraries

| Task                 | Tool                                  |
| -------------------- | ------------------------------------- |
| Tokenization         | `tiktoken`, Hugging Face `tokenizers` |
| Embedding Layer      | `torch.nn.Embedding`                  |
| Transformer          | `nn.Transformer`, `nn.Module`         |
| Vector DB (optional) | FAISS, ChromaDB (for RAG)             |

---

## ✅ Sample Output

**Input**:

```text
"The capital of France is"
```

**Model Prediction**:

```text
"Paris"
```

→ **Confidence**: 97%


