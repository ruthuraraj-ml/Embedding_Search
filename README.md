# Word2Vec Embedding Explorer
### Skip-Gram with Negative Sampling · WikiText-2 · Live Interactive Demo

> **Train a model. Then build something around it.**  
> This project implements Word2Vec from scratch in PyTorch and deploys the trained embeddings as a live, browser-based interactive explorer — letting you watch semantic structure emerge across training epochs.

---

## 🔗 Live Demo

**[ruthuraraj-ml.github.io/Embedding_Search/](https://ruthuraraj-ml.github.io/Embedding_Search/)**

No installation needed. Runs entirely in the browser.

---

## What This Project Does

Most Word2Vec tutorials end at the training loop. This one goes further.

The notebook trains a full Skip-Gram model with Negative Sampling on the WikiText-2 corpus, saves embedding checkpoints at **epoch 2, 5, and 10**, exports them to structured JSON, and serves them through a static HTML page deployed on GitHub Pages. Every computation in the demo — cosine similarity, nearest neighbour search, vector arithmetic, PCA projection — runs client-side in JavaScript with no backend.

---

## Repository Structure

```
.
├── Notebook                          # Full training notebook (Google Colab)
├── Report                            # Technical Specifications, step by step implementation and inferences
├── word2vec_demo_epochs.html         # Interactive demo (single-file, no dependencies)
├── embeddings_epoch2.json            # Checkpoint: epoch 2  (5000 words, 100-dim, L2-normalised + PCA-2D)
├── embeddings_epoch5.json            # Checkpoint: epoch 5
├── embeddings_epoch10.json           # Checkpoint: epoch 10
└── README.md
```

---

## The Notebook — Pipeline Overview

The notebook is structured in five phases. All code is written from scratch using standard Python libraries and PyTorch.

### Phase 1 · Dataset Processing

| Step | Detail |
|------|--------|
| Dataset | `dlwh/wikitext_2_detokenized` via HuggingFace Datasets |
| Split sizes | Train: 23,767 · Validation: 2,461 · Test: 2,891 |
| Combined | 29,119 sentences concatenated into a single corpus |
| Heading removal | Lines beginning with `=` (WikiText section markers) removed |
| Lowercasing | All text lowercased |
| Hyphen cleaning | Hyphens replaced with spaces (`str.maketrans`) |
| Post-cleaning | 21,580 usable sentences |

### Phase 2 · Tokenization & Vocabulary

| Step | Detail |
|------|--------|
| Tokenization | Whitespace split — no subword, no punctuation stripping |
| Raw vocabulary | 145,192 unique tokens |
| Frequency filter | Words appearing < 5 times removed |
| Final vocabulary | **29,780 unique tokens** (min count: 5, e.g. `senjō`) |
| Word index | Sorted by descending frequency: `word2idx` / `idx2word` |
| Top 5 words | the (159,625) · of (69,265) · and (60,523) · in (53,869) · to (48,198) |

### Phase 3 · Subsampling (Token Reduction)

Frequent words like "the" dominate context windows without adding semantic signal. Following Mikolov et al. 2013, each word is discarded with probability:

```
P_discard(w) = max(0, 1 − √(t / f(w)))     where t = 1e-5
```

| Metric | Value |
|--------|-------|
| Total tokens before subsampling | 1,935,519 |
| Frequency of "the" before | ~159,625 occurrences |
| Frequency of "the" after | ~1,800 occurrences |
| Total tokens after subsampling | 568,118 |
| Sentences retained (≥ 2 tokens) | 20,475 of 21,580 |
| Random seed | 42 (reproducible) |

### Phase 4 · Training Architecture

**Sliding Window — Pair Generation**

```python
window_size = 3   # context window on each side
# Total (center, context) pairs generated: 3,164,394
```

**Negative Sampling Distribution**

Negatives are sampled from the pre-subsampling frequency distribution raised to the 0.75 power (as in the original paper), preserving the relative importance of frequent words while giving rare words a better chance:

```python
P_neg(w) ∝ f(w) ** 0.75
k = 5   # negatives per positive pair
```

Negatives use `updated_word_count` (post min-freq filter, pre-subsampling) — sampling from the subsampled distribution would artificially underrepresent common words.

**SkipGram Model**

```python
class SkipGram(nn.Module):
    # Two separate Embedding tables: input and output
    # input_embeddings:  shape (V, 100) — used at inference
    # output_embeddings: shape (V, 100) — used only during training
    # Init: N(0, 0.01)
```

Loss per batch (Negative Sampling objective):

```
L = -[ log σ(v_c · v_pos) + Σ_k log σ(−v_c · v_neg_k) ]
```

**Hyperparameters**

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 100 |
| Window size | 3 |
| Negative samples (k) | 5 |
| Batch size | 256 |
| Epochs | 10 |
| Optimiser | Adam (lr = 0.001) |
| Device | CUDA (Google Colab T4 GPU) |

**Training Loss**

| Epoch | Avg Loss |
|-------|----------|
| 1 | 2.7062 |
| 2 | 2.3077 |
| 3 | 2.0196 |
| 4 | 1.8137 |
| 5 | 1.6730 |
| 6 | 1.5796 |
| 7 | 1.5151 |
| 8–10 | converging |

### Phase 5 · Export & Evaluation

**POS Filtering for Export**

Rather than exporting all 29,780 vocabulary words, NLTK's averaged perceptron tagger filters to the top 5,000 **nouns and adjectives** (POS tags: NN, NNS, NNP, NNPS, JJ, JJR, JJS) sorted by frequency. This gives the demo a semantically rich, browsable vocabulary without bloating the JSON files.

**JSON Schema (per checkpoint)**

```json
{
  "epoch": 10,
  "vocab_size": 5000,
  "embed_dim": 100,
  "pca_variance": [0.042, 0.031],
  "words": ["science", "music", ...],
  "embeddings": { "science": [0.23, -0.87, ..., 0.64] },
  "coords_2d":  { "science": [-0.41, 1.23] }
}
```

All vectors are **L2-normalised** before export so that cosine similarity reduces to a dot product in the browser.

**Evaluation Results (Epoch 10)**

```python
get_similarity('player', 'game')  # → 0.4444

get_nearest('gaming')    # → ['sequel', 'famitsu', 'visuals', 'esports', 'dota', ...]
get_nearest('monarchs')  # → ['parthian', 'rulers', 'constantinople', 'greeks', ...]
get_nearest('music')     # → ['awards', 'recordings', 'musical', 'mccartney', ...]

analogy('sony', 'shimomura', 'video')  # → ['video', 'sony', 'rockstar', 'online', ...]
```

**PCA Visualisation**

A 2D PCA scatter plot is produced for three semantic clusters (gaming, monarchs, music) with their nearest neighbours, with dashed cluster boundary circles overlaid using `matplotlib.patches.Circle`.

---

## The Interactive Demo

The demo is a **single HTML file** with no external dependencies beyond Plotly.js (CDN). All inference runs client-side.

### Features

| Tab | What It Does |
|-----|-------------|
| 🗺 Explore Space | Interactive Plotly 2D scatter plot of 5,000 word embeddings, coloured by semantic category, toggleable clusters |
| ≈ Similarity | Cosine similarity between any two words, with cross-epoch comparison bars |
| ⬡ Neighbors | Top-k nearest neighbor search for any word |
| ⊕ Analogy | Vector arithmetic solver: A − B + C = ? |
| 📐 How It Works | Educational breakdown of all pipeline stages |

### Epoch Switcher

A sticky bar at the top lets users switch between Epoch 2, Epoch 5, and Epoch 10. All five panels update instantly — no page reload. This makes the learning dynamics visible: early epochs show scattered, noisy neighbors; later epochs show tighter, more meaningful groupings.

### Client-Side Inference

```javascript
// Cosine similarity (vectors are pre-normalised, so this is just a dot product)
function cosine(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return Math.max(-1, Math.min(1, dot));
}
```

Nearest neighbor search, vector arithmetic for analogies, and PCA scatter rendering all run in pure JavaScript, making the demo fully deployable as static files.

---

## Key Insight: What WikiText-2 Teaches

The nearest neighbors of `"science"` at Epoch 10 are `fiction`, `magazines`, `stories`, `technology` — not `physics` or `quantum`. This isn't a failure. WikiText-2 is Wikipedia text, and Wikipedia treats science as a publishing and coverage domain, not a technical one. The embeddings faithfully reflect their training distribution.

This is one of the most valuable things visualising embeddings reveals: **geometry mirrors corpus statistics, not the world as you imagine it**. Changing the corpus changes what the model learns.

---

## How to Run the Notebook

The notebook is designed for **Google Colab** with GPU runtime (T4 recommended).

```bash
# 1. Open in Colab
# 2. Runtime → Change runtime type → GPU
# 3. Run all cells in order
# 4. Embeddings saved as: embeddings_epoch2.json, embeddings_epoch5.json, embeddings_epoch10.json
```

Dependencies (auto-installed in Colab):
```
torch · datasets · nltk · scikit-learn · numpy · matplotlib
```

---

## How to Deploy the Demo

```bash
# 1. Clone or fork this repo
# 2. Place all three JSON files in the same folder as the HTML file
# 3. Push to GitHub
# 4. Settings → Pages → Deploy from main branch
# 5. Access at: https://<username>.github.io/<repo-name>/word2vec_demo_epochs.html
```

---

## References

- Mikolov et al. (2013) — [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Mikolov et al. (2013) — [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546)
- WikiText-2 dataset — [dlwh/wikitext_2_detokenized](https://huggingface.co/datasets/dlwh/wikitext_2_detokenized)

---

*Built by Ruthuraraj R · February 2026*
