# Building a Small Language Model (SLM) from Scratch

A step-by-step, heavily commented PyTorch notebook that builds a GPT-style small language model from the ground up, trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

**Notebook:** `main_commented.ipynb`

| | |
|---|---|
| **Dataset** | TinyStories (`roneneldan/TinyStories`) |
| **Tokenizer** | GPT-2 BPE via `tiktoken` (vocab size 50,257) |
| **Framework** | PyTorch |
| **Hidden size** | 768 (matches GPT-2 embeddings) |
| **Max context** | 1024 tokens (limited by GPT-2's positional embeddings) |

---

## Project Status

This is a work in progress. The notebook follows a roadmap, and the later steps are not yet implemented.

| Step | Description | Status |
|------|-------------|--------|
| 1 | Load the TinyStories dataset | Done |
| 2 | Tokenize stories and save to `train.bin` / `validation.bin` | Done |
| 3 | Build training batches (inputs + targets) and input embeddings | Done |
| 4 | LayerNorm | Done (in progress) |
| 5 | Masked multi-head causal attention | Done (in progress) |
| 6 | Feed-forward network (MLP) | To do |
| 7 | Transformer block, full model, output head | To do |
| 8 | Loss function and training loop | To do |

---

## What the Notebook Covers

### 1. Loading the data
Loads TinyStories from Hugging Face with the `datasets` library. It has two splits, `train` and `validation`.

### 2. Tokenization
- Encodes text with the GPT-2 tokenizer (`tiktoken`).
- Appends the `<|endoftext|>` token (ID `50256`) to every story so the model can tell where stories end.
- Shows how token IDs map to embedding vectors using GPT-2's pretrained embedding table.

### 2b. Saving tokens to disk
Tokenizing the full dataset is slow, so it is done once and cached:
- All stories are concatenated into one flat 1-D stream of token IDs.
- Stored as `uint16` (2 bytes per token, since 50,257 < 65,536).
- Processed in chunks of 2,048 stories to keep RAM usage low.
- Output: `train.bin` (roughly 474M tokens) and `validation.bin`.
- Verified by reading the file back with `np.memmap` and decoding a slice to text.

### 3. Training batches
`get_batch(split, batch_size, context_length)` samples random windows from the memory-mapped files. Targets are the inputs **shifted by one token**, which sets up next-token prediction:

```
x = [ A  B  C  D ]
y = [ B  C  D  E ]
```

### 3b. Input embeddings
Input to the transformer = token embedding (`wte`) + positional embedding (`wpe`).
This project reuses GPT-2's pretrained tables and **freezes** them, which means:
- hidden size must be 768
- max context length is 1024
- the embedding tables account for roughly 38M parameters

### 4. LayerNorm
A thin wrapper around `nn.LayerNorm(768)`. The architecture follows GPT-2's **pre-LN** layout (normalize before attention and before the MLP).

### 5. Masked multi-head causal attention
A from-scratch implementation with:
- separate `q_proj`, `k_proj`, `v_proj` linear layers
- 12 heads of dimension 64 (768 / 12)
- a causal (upper-triangular) mask so tokens cannot see the future
- softmax over attention scores, then a weighted sum of values
- head merging followed by an output projection (`o_proj`)

Tensor shapes are annotated in the code comments for the `(4, 4, 768)` example batch.

---

## Requirements

- Python 3.9+
- `torch`
- `transformers`
- `datasets`
- `tiktoken`
- `numpy`
- `jupyter` (or any environment that runs `.ipynb` files)

```bash
pip install torch transformers datasets tiktoken numpy jupyter
```

## Usage

```bash
jupyter notebook main_commented.ipynb
```

Run the cells in order. Notes:
- The first run downloads the dataset and GPT-2 weights from Hugging Face, so an internet connection is required.
- Step 2b creates `train.bin` and `validation.bin` in the working directory. This takes a while but only needs to be done once; later steps read from these files.
- `context_length` and `batch_size` are set to `4` in the demo cells so the output is easy to inspect. Increase them for real training.

---

## Known Issues

- **Attention scaling bug (Step 5):** the scores are scaled with floor division:
  ```python
  attention_scores = attention_scores // (self.head_dim ** 0.5)
  ```
  This should be true division (`/`). Floor division rounds the scores to whole numbers and distorts the attention weights.

---

## Dataset Credit

TinyStories: *"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"* by Ronen Eldan and Yuanzhi Li.
