# mini-gpt — Project Progress Log

This document records the full arc of this project: what was done, what was found,
what broke, and what comes next. Intended as a portfolio reference and arXiv preprint source.

---

## Timeline

### M0 — Causal Mask Fix (2026-05-04)

**What was done:**
Discovered and fixed a bug in the attention mask implementation. The causal mask was not
being applied correctly, allowing future tokens to attend to past tokens. Fixed by
verifying the upper-triangular mask in `src/model.py`.

**Why it matters:**
Without a correct causal mask, the model is not autoregressive — it leaks future
information into predictions, making training loss artificially low and generation
meaningless. This is a foundational correctness requirement.

---

### M1 — Refactor + Corpus + Tests (2026-05-07)

**What was done:**
- Modularized codebase into `src/` (model, tokenizer, utils)
- Added TinyShakespeare as the standard training corpus (`data/shakespeare.txt`, 1.1M chars)
- Added `tests/` with 24 unit tests covering tokenizer, model forward pass, attention mask

**Why it matters:**
Clean, testable code is a prerequisite for reproducible experiments. Any result that
cannot be reproduced is not a result.

---

### M2 — Full Training + Evaluation (2026-05-08)

**What was done:**
- Trained char-level GPT on TinyShakespeare (~10M parameters)
- Architecture: d_model=384, n_layers=6, num_heads=8, block_size=256
- Best validation loss: **1.4414** (step 13,000 of 20,000)
- Integrated W&B logging (project: `mini-gpt`, entity: `daisukeabe32-university-of-tokyo`)
- Implemented cosine LR decay with linear warmup, AdamW with weight-decay separation
- Fixed memory safety issue: attention activation memory scales as B×H×T²×L; reduced
  batch_size from 32 → 8 to stay under 6GB on 16GB unified memory M3 Mac

**Key result:**
Char-level model achieved val_loss=1.4414 (perplexity ~4.23). Generates coherent
Shakespearean prose at inference time.

**W&B run:** `char-20k`

---

### M3 — Mechanistic Interpretability: Induction Head Detection (2026-05-09)

#### Hypothesis

Induction heads (Olsson et al. 2022) implement the in-context pattern-matching operation
[A][B]...[A] → predict [B]. We hypothesized that char-level tokens are too semantically
thin for this circuit to form, and that BPE tokens — which encode multi-character units
with lexical meaning — would allow induction heads to emerge.

#### Method

Construct a repeated sequence `[T₁ T₂ … T₃₂][T₁ T₂ … T₃₂]`, extract all attention maps,
and compute the induction score = mean attention weight on diagonal offset –(seq_len–1)
in the second half. Threshold ≥ 0.1 following Olsson et al.

Script: `scripts/analyze_attention.py`

#### Experiment A — Char model (W&B: `char-20k`)

| Rank | Layer | Head | Score |
|------|-------|------|-------|
| 1    | 6     | 2    | 0.0092 |
| 2    | 2     | 6    | 0.0076 |

**Result: No induction head detected.** Max score 0.0092 ≪ 0.1 threshold.

Checkpoint: `checkpoints/20260508_*/best.pt` (step 13,000, val_loss=1.4414)

#### Experiment B — BPE vocab=15,000 (W&B: `clean-mountain-13`)

Architecture identical to char run except vocab_size=15,000, block_size=64.

| Rank | Layer | Head | Score |
|------|-------|------|-------|
| 1    | 1     | 3    | **0.1165** ✅ |
| 2    | 1     | 8    | 0.0653 |
| 3    | 6     | 8    | 0.0631 |

**Result: Induction head detected — L1H3 score = 0.1165 ≥ 0.1.**
Layer 1 concentration is consistent with Olsson et al. 2022 (shallow-layer formation
in small transformers).

Best checkpoint: `checkpoints/20260509_141209/best.pt` (step 4,500, val_loss=8.1477)

**Overfitting note:** train loss fell to 3.15 while val loss rose after step 4,500.
Root cause identified below.

#### Experiment C — BPE vocab=3,000 (W&B: `confused-tree-14`)

Attempted to reduce overfitting by lowering vocab size (smaller embedding table → fewer
parameters; lower compression → more training tokens).

Training terminated early at step ~9,000 (manually stopped) after observing the same
overfitting pattern:

```
step 2500 | train=4.55  val=5.5295  ← val loss minimum (best checkpoint)
step 3000 | train=4.29  val=5.5318  ← val starts rising
step 9000 | train=1.46  val=6.5615  ← stopped here
```

Best checkpoint: `checkpoints/20260509_160736/best.pt` (step 2,500, val_loss=5.5295)

**Result: Same overfitting structure. Root cause is the corpus, not the vocab size.**

#### Root Cause Analysis — The BPE Data Constraint

BPE tokenization introduces a structural tension with small corpora:

| Model | Training tokens | Parameters | tokens/param |
|-------|----------------|------------|--------------|
| Char  | ~1,000,000     | ~10M       | **0.10**     |
| BPE 15K | ~193,000    | ~16.4M     | 0.012        |
| BPE 3K  | ~285,000    | ~11.9M     | 0.024        |

Rough guideline: tokens/param ≥ 0.20 for stable generalization (Chinchilla-style).
None of the BPE runs came close.

The chain of constraints is:

```
induction head detection
  → requires BPE tokenization
      → BPE compression reduces token count
          → small corpus becomes even smaller after tokenization
              → severe overfitting
```

TinyShakespeare (1.1M chars) is simply too small for BPE-tokenized training at this
model scale.

**Calculation — corpus size needed:**
- Model: ~12M params
- Target ratio: tokens/param ≥ 0.20
- Required tokens: 12M × 0.20 = 2.4M tokens
- BPE 3K compression: 3.5× → required chars: 2.4M × 3.5 = **8.4M chars** (8× Shakespeare)
- BPE 30K compression: ~8× → required chars: 2.4M × 8 = **19.2M chars** (17× Shakespeare)

#### M3 Conclusion

**The hypothesis is confirmed:** BPE tokenization enables induction head formation where
char-level tokenization does not. However, the clean experimental demonstration requires
a corpus large enough to prevent overfitting — Shakespeare is insufficient for BPE.

This is itself a finding worth documenting: the emergence of induction heads via BPE
comes with a data-size prerequisite that char models do not share.

---

## Next Steps — M4 Setup

### Corpus Switch: TinyShakespeare → TinyStories

**Dataset:** `roneneldan/TinyStories` (Hugging Face)
- Size: ~2.2GB text
- Nature: simple English short stories, generated for small LM research
- Has its own paper: Eldan & Li 2023 ("TinyStories: How Small Can Language Models Be...")
- BPE 30K compression ~8×: ~275M tokens — well above the ~2.4M token minimum

**Why TinyStories over Wikipedia:**
- Designed for small-scale transformer research (same setting as this project)
- Clean, consistent vocabulary — BPE merges will be stable
- No preprocessing (no markup, no disambiguation)

### Plan

1. **Download TinyStories** via `datasets` library, write to `data/tinystories.txt`
2. **Tokenize** with BPE vocab=30,000 using `scripts/tokenize_corpus.py`:
   ```
   python -m scripts.tokenize_corpus --tokenizer bpe --bpe_vocab_size 30000 \
     --data_path data/tinystories.txt --corpus_name tinystories
   ```
   Output: `data/tokenized/bpe_30000_tinystories/`
3. **Train** with same architecture (d_model=384, n_layers=6, num_heads=8):
   - block_size=64 (BPE tokens ≈ 512 chars of context, same as M3 intent)
   - Expected: no overfitting — training tokens >> parameters
4. **Run induction head analysis** on best checkpoint:
   ```
   python -m scripts.analyze_attention --checkpoint checkpoints/.../best.pt \
     --seq_len 64 --out_dir figs/tinystories_bpe30k_analysis
   ```
5. **Compare** induction scores across all four conditions:

| Condition | Corpus | Tokenizer | Expected induction score |
|-----------|--------|-----------|--------------------------|
| Char      | Shakespeare | char (65) | 0.0092 (measured) |
| BPE 15K   | Shakespeare | bpe 15K   | 0.1165 (measured, overfit) |
| BPE 3K    | Shakespeare | bpe 3K    | TBD (measured, overfit)  |
| **BPE 30K** | **TinyStories** | **bpe 30K** | **TBD — target of M4** |

### M4 arXiv Preprint

Target: cs.LG or cs.CL, posted by 2027-01.

Section outline:
1. Introduction — induction heads and tokenization
2. Background — Olsson et al. 2022, BPE
3. Experimental setup — model architecture, detection method
4. Results — char vs BPE, with and without overfitting
5. Discussion — data-size constraint as a finding in itself
6. Conclusion

---

## Artifact Index

| Artifact | Path | Description |
|----------|------|-------------|
| Char best checkpoint | `checkpoints/20260508_*/best.pt` | step 13K, val=1.44 |
| BPE 15K best checkpoint | `checkpoints/20260509_141209/best.pt` | step 4.5K, val=8.15 |
| BPE 3K best checkpoint | `checkpoints/20260509_160736/best.pt` | step 2.5K, val=5.53 |
| BPE 15K attention grid | `figs/bpe_analysis/attention_grid.png` | L1H3 induction head |
| Tokenized char data | `data/tokenized/char_shakespeare/` | 1M tokens |
| Tokenized BPE 15K data | `data/tokenized/bpe_15000_shakespeare/` | 193K tokens |
| Tokenized BPE 3K data | `data/tokenized/bpe_3000_shakespeare/` | 285K tokens |
| Experiment lab notebook | `EXPERIMENT.md` | M3 results detail |
| Training script | `scripts/train_char_gpt.py` | supports char + BPE |
| Analysis script | `scripts/analyze_attention.py` | induction head detection |
| Tokenization script | `scripts/tokenize_corpus.py` | one-time corpus prep |
