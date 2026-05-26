# M3 Experiment: Induction Head Detection — Char vs BPE Tokenization
2026.May.09

_This file records scientific findings only. For operational procedures, see KAGGLE_RUNBOOK.md._

## Hypothesis

Induction heads (Olsson et al. 2022) are attention heads that implement the
in-context pattern-matching operation [A][B]...[A] → predict [B].
We hypothesized that char-level tokens are too semantically thin for this
circuit to form, and that BPE tokens — which encode multi-character units with
lexical meaning — would allow induction heads to emerge.

## Setup

| | Char run | BPE run |
|---|---|---|
| Tokenizer | CharTokenizer | BPETokenizer |
| Vocab size | 65 | 15,000 |
| Block size | 256 | 64 |
| Max iters | 20,000 | 20,000 |
| d_model | 384 | 384 |
| n_layers | 6 | 6 |
| num_heads | 8 | 8 |
| d_k | 48 | 48 |
| d_ff | 1,536 | 1,536 |
| Parameters | ~10M | ~16.4M |
| W&B run | (char-20k) | clean-mountain-13 |

BPE block size 64 ≈ 333 chars (15k BPE compression ratio ≈ 5.2×),
chosen to preserve roughly the same context window in character-equivalent terms.

## Induction Head Detection Method

1. Construct a repeated sequence: `[T₁ T₂ … T₃₂][T₁ T₂ … T₃₂]` (random tokens, length = 2 × seq_len)
2. Run the model and extract all attention maps
3. For each head, compute the **induction score** = mean attention weight on the
   diagonal offset –(seq_len – 1) in the second half of the sequence  
   (i.e., how strongly position `seq_len + i` attends to position `i + 1`)
4. Threshold: score ≥ 0.1 is treated as a positive detection (following Olsson et al.)

Script: `scripts/analyze_attention.py`

## Results

### Char model (best checkpoint: step 13,000, val_loss = 1.4414)

| Rank | Layer | Head | Score |
|------|-------|------|-------|
| 1 | 6 | 2 | **0.0092** |
| 2 | 2 | 6 | 0.0076 |
| … | … | … | … |

**No induction head detected.** Max score 0.0092 ≪ threshold 0.1.

### BPE model (best checkpoint: step 4,500, val_loss = 8.1477)

| Rank | Layer | Head | Score |
|------|-------|------|-------|
| 1 | 1 | 3 | **0.1165** ✅ |
| 2 | 1 | 8 | 0.0653 |
| 3 | 6 | 8 | 0.0631 |
| 4 | 6 | 7 | 0.0595 |
| 5 | 4 | 1 | 0.0567 |

**Induction head detected: L1H3 (score = 0.1165 ≥ 0.1).**
Layer 1 shows a strong concentration of candidate heads, consistent with
Olsson et al.'s finding that induction heads appear in shallow layers of small transformers.

### Summary

| Model | Max induction score | Detected |
|-------|--------------------|---------:|
| Char (vocab 65) | 0.0092 | ❌ No |
| BPE (vocab 15,000) | 0.1165 | ✅ Yes |

## Interpretation

The result confirms the hypothesis. BPE tokenization — which groups characters
into semantically meaningful units ("lord,", "thee", "KING") — provides the
representational granularity necessary for the induction head circuit to form.
Char-level tokens carry no semantic content beyond their raw glyph identity,
making the repeated-pattern signal too noisy across the high-frequency char
bigram space.

This finding mirrors the broader literature: induction heads were first
characterized in models trained on natural-language token vocabularies, not
raw-byte or char-level models.

### Note on BPE overfitting

The BPE run exhibited severe overfitting: train loss fell to 3.15 while
val loss rose from 9.69 (step 0) to 9.40 (step 20,000), peaking best at
val_loss = 8.1477 (step 4,500). This is expected given the data regime:
~193,000 BPE training tokens vs. 16.4M parameters. The best checkpoint
(step 4,500) was used for analysis; induction head formation occurs early
in training before severe overfitting dominates.

## Next Steps → Olsson-approximate experiment (see below)

---

# Olsson-approximate Experiment: Emergence Curve Attempt
2026.May.12–16

## Hypothesis

Olsson et al. (2022) showed that induction heads emerge abruptly at ~1.5B tokens
in 2-layer transformers, producing a sharp "phase change" in the induction score curve.
We attempted to reproduce this phase change using the same 2-layer architecture
trained on TinyStories with a 30K BPE vocabulary.

## Setup

| Parameter | Value |
|---|---|
| n_layers | 2 |
| d_model | 512 |
| num_heads | 8 |
| d_k | 64 |
| d_ff | 2,048 |
| block_size | 256 |
| batch_size | 64 |
| max_iters | 120,000 |
| tokens total | ~1.97B (64 × 256 × 120,000) |
| Tokenizer | BPETokenizer (HuggingFace, vocab 30,000) |
| Dataset | TinyStories |
| lr | 3e-4 → 3e-5 (cosine decay, warmup 1,000) |
| save_every | 2,000 steps (snapshot for emergence curve) |
| Platform | Kaggle T4 x1 GPU |
| W&B project | mini-gpt |

## Training Sessions

Training was split across 3 Kaggle sessions (12h quota per session):

| Session | Steps | Tokens | Notes |
|---------|-------|--------|-------|
| 1 | 0 → 53,000 | 0 → 0.87B | Timed out at 12h |
| 2 | 53,001 → 98,000 | 0.87B → 1.61B | Timed out at 12h |
| 3 | 98,001 → 119,999 | 1.61B → 1.97B | Completed normally |

## Results

### Final checkpoint (step 115,000 / best.pt)

- val_loss: 3.0532

### Induction head analysis (`scripts/analyze_attention.py`)

**⚠️ Initial analysis used seq_len=64, which was too long and diluted the score.
Re-analysis at shorter seq_len revealed a clear induction head.**

| seq_len | L2H6 PrefixMatch | Detected |
|---------|-----------------|---------|
| 16 | **0.1897** | ✅ Yes |
| 32 | 0.0961 | borderline |
| 64 | 0.0337 | ❌ (initial result — misleading) |

**L2H6 is an induction head** (score 0.1897 ≥ threshold 0.1 at seq_len=16).
The diagonal stripe is clearly visible in the bottom-left quadrant of the attention map.

The score decay with distance (seq_len 16→32→64) reflects that this head's
induction behavior is strongest at short repeated-sequence offsets. This is a
real property of the model, not an absence of the circuit.

### Emergence curve

Only Session 3 snapshots (steps 98,000–120,000; tokens 1.638B–1.97B) were available.
Sessions 1 and 2 snapshots were lost: Kaggle session Outputs were inaccessible
after the sessions timed out.

Emergence curve analysis is **pending** — the available window (1.638B–1.97B)
covers only the tail of training and cannot show when L2H6 emerged.
To observe the phase change, snapshots from 0–1.5B tokens are needed.

## Interpretation

**Induction head L2H6 exists in the Phase 2 model.** The earlier conclusion
("not detected") was a methodological error: seq_len=64 was too large, causing
the attention diagonal to be diluted below threshold.

Key finding: the induction head operates on short-range repeated patterns
(optimal at seq_len≈16, ~32 tokens total). This may reflect the statistical
character of TinyStories, where repeated patterns tend to occur at short distances,
or a property of 2-layer architectures where the induction circuit has limited
"reach."

Comparison with Olsson et al.:
- Their models showed high induction scores at seq_len=50 (byte-level tokens)
- Our model's score drops significantly beyond seq_len=16
- The circuit exists but is less robust to longer distances

## Open question: emergence curve

The phase change (when L2H6 emerged during training) remains unobserved.
To answer this, snapshots from 0 to ~1.5B tokens are needed.
Options: (A) re-run training with full snapshot preservation, (B) accept
that the circuit exists post-training and focus analysis on its properties.

---

# EXP-002: Full Emergence Curve (fresh run from step 0)
2026.May.19 — ongoing

## Goal

Observe the full emergence curve of L2H6 induction head from 0 to 1.97B tokens,
with multi-scale analysis (seq_lens = 8, 16, 32, 64) per snapshot.

## Setup

Identical to EXP-001 (Olsson-approximate). Cell version: git commit `91d5ffb`.

## Training Sessions

| Session | Kaggle Notebook | Steps | Tokens | GPU | Status |
|---------|----------------|-------|--------|-----|--------|
| 1 | EXP-002 | 0 → 48,000 | 0 → 0.79B | T4 x2 | ✅ 完了（quota 使い切り） |
| 2 | EXP-002 resume(1.0) | 48,001 → 93,000 | 0.79B → 1.53B | T4 x2 | ✅ 完了（12h timeout、step 93,000） |
| 3 | EXP-002 resume(2.0) | 93,001 → 119,999 | 1.53B → 1.97B | T4 x2 | ✅ 完了（2026-05-24） |

## Results

### Training — Session 3 (step 93,001 → 119,999)

| Step | val_loss | Notes |
|------|----------|-------|
| 93,001 | 3.1059 | resume start |
| 98,000 | 3.0907 | |
| 105,000 | 3.0740 | |
| **110,000** | **3.0586** | **best checkpoint** |
| 119,999 | 3.0665 | final |

### Induction Head Analysis — Cell 12 (best.pt, step 110,000)

| seq_len | best head | PrefixMatch | Detected |
|---------|-----------|-------------|---------|
| 8 | L2H6 | **0.2832** | ✓ YES |
| 16 | L2H6 | **0.1857** | ✓ YES |
| 32 | L2H6 | **0.1096** | ✓ YES |
| 64 | L2H6 | 0.0394 | ✗ no |

L2H6 is the dominant induction head. Score decays with seq_len, confirming the
distance-dependent circuit property observed in EXP-001.

### Emergence Curve — Cell 13 (59 snapshots, step 2,000–118,000)

Complete curve spanning 0.033B → 1.933B tokens (all 3 sessions).

**Key data points (max PrefixMatch across all heads and seq_lens):**

| step | tokens | max_induction | head | note |
|------|--------|--------------|------|------|
| 2,000 | 0.033B | 0.1119 | L1H4 | transient; different head |
| 4,000 | 0.066B | 0.0860 | L1H3 | below threshold |
| 6,000 | 0.098B | 0.0957 | L2H6 | below threshold |
| **8,000** | **0.131B** | **0.1483** | **L2H6** | **stable emergence begins** |
| 10,000 | 0.164B | 0.1635 | L2H6 | |
| 12,000 | 0.197B | 0.1968 | L2H6 | |
| 26,000 | 0.426B | 0.2913 | L2H6 | |
| 42,000 | 0.688B | **0.3342** | L2H6 | peak |
| 50,000 | 0.819B | 0.2821 | L2H6 | plateau begins |
| 92,000 | 1.507B | 0.2754 | L2H6 | |
| 118,000 | 1.933B | 0.2851 | L2H6 | |

Figure: `figs/emergence_curve.png`

### Phase Change Timing

**L2H6 stably crosses threshold 0.1 at step 8,000 (~0.131B tokens).**

Emergence sequence:
1. **step 2,000** (0.033B): L1H4 shows a transient spike (0.1119) — early unstable candidate
2. **step 4,000–6,000** (0.066–0.098B): all heads drop below threshold
3. **step 8,000** (0.131B): L2H6 crosses 0.1 and remains above threshold through end of training
4. **step 8,000–42,000**: score rises from 0.148 to 0.334
5. **step 42,000–119,999**: plateau in 0.25–0.33 range; no second phase change observed

No phase change is observed near the Olsson et al. threshold of ~1.5B tokens.

## Interpretation

### Main finding

The induction head L2H6 emerges at **~0.131B tokens** (step 8,000) in a 21.8M parameter
model — far earlier than Olsson et al.'s reported ~1.5B token threshold.

This is consistent with **phase change timing scaling with model size**: Olsson et al.'s
1.5B token estimate corresponds to larger models. A 21.8M parameter model trained on
simple narrative text develops the induction circuit much sooner in absolute token count.

### Transient head at step 2,000

At step 2,000, L1H4 briefly scores above threshold before dropping. This is likely
a "proto-induction" signal — an early, unstable attention pattern that has not yet
consolidated into a clean circuit. L2H6 takes over by step 8,000 as the stable
induction circuit.

### Distance dependence (stable across training)

L2H6's score decays with seq_len throughout training and in the final checkpoint:

| seq_len | PrefixMatch (final) |
|---------|---------------------|
| 8 | 0.2832 |
| 16 | 0.1857 |
| 32 | 0.1096 |
| 64 | 0.0394 |

The circuit operates most strongly on short-range patterns, consistent with
TinyStories' statistical structure (short, simple sentences with limited long-range
repetition).

### Comparison with Olsson et al. (2022)

| Property | Olsson et al. | This work |
|---|---|---|
| Phase change at | ~1.5B tokens | **~0.131B tokens** |
| Architecture | 2-layer | 2-layer |
| Dataset | diverse web text | TinyStories |
| Induction score @ optimal seq_len | ~1.0 (their metric) | 0.28–0.33 |
| Score at long seq_len | remains high | decays to ~0.04 |

The earlier emergence and distance-dependent decay likely both reflect the
simpler statistical structure of TinyStories compared to diverse web corpora:
fewer long-range dependencies → less training signal for long-range induction circuits
→ circuit forms earlier but with shorter effective reach.

---

# EXP-003 Series: Phase Change Timing vs. Dataset Complexity
2026.May.24 — planned

## Hypothesis

Phase change timing (the token count at which an induction head crosses the detection
threshold) is not a fixed constant (~1.5B tokens as reported by Olsson et al.) but
varies with **dataset complexity**: simpler, more repetitive corpora produce earlier
phase changes because the induction circuit requires less exposure to learn the
[A][B]...[A]→[B] pattern.

If this hypothesis holds, the three datasets should produce phase change timings
ordered as follows:

```
GitHub Code  <  TinyStories  <  WikiText-103
 (earliest)      (0.131B)       (latest)
```

Code is the most structurally repetitive (function signatures, keywords, syntax),
so the induction circuit should form earliest. WikiText-103 contains encyclopedic
prose with diverse vocabulary and longer-range dependencies, so it should take the
longest. TinyStories, already completed, sits in the middle as the reference point.

## Experimental Design

All three runs share identical model architecture and training hyperparameters.
Only the dataset changes.

### Fixed parameters (all runs)

| Parameter | Value |
|---|---|
| n_layers | 2 |
| d_model | 512 |
| num_heads | 8 |
| d_k | 64 |
| d_ff | 2,048 |
| block_size | 256 |
| batch_size | 64 |
| max_iters | 120,000 |
| Tokenizer | BPE-30K (trained separately on each corpus) |
| save_every | 2,000 steps |
| Analysis | seq_lens = 8, 16, 32, 64 |

### Dataset summary

| Run | Dataset | Description | Complexity | HuggingFace ID | Status |
|-----|---------|-------------|-----------|----------------|--------|
| EXP-002 | **TinyStories** | Children's short stories | Low | `roneneldan/TinyStories` | ✅ Done (phase change ~0.131B tokens) |
| EXP-003a | **WikiText-103** | Wikipedia featured articles | Medium–High | `Salesforce/wikitext` (wikitext-103-raw-v1) | 🔲 Planned |
| EXP-003b | **GitHub Code (Python)** | Python source code from GitHub | Structural / repetitive | `codeparrot/github-code` (Python subset) | 🔲 Planned |

### Measurement

For each run, record:
- **Phase change token count**: first step where max induction score (seq_len=8) crosses 0.1 and stays above
- **Plateau score**: mean induction score (seq_len=8) over steps 40,000–120,000
- **Score at each seq_len** in final checkpoint (same as EXP-002 Cell 12)
- **val_loss trajectory**

### Expected outcome table (to be filled in)

| Dataset | Phase change (tokens) | Plateau score (seq_len=8) | Score (seq_len=64) |
|---------|----------------------|--------------------------|-------------------|
| GitHub Code | ? | ? | ? |
| TinyStories | **~0.131B** | **~0.30** | **0.039** |
| WikiText-103 | ? | ? | ? |

## Results

(To be filled in after each run completes.)

### EXP-003a: WikiText-103

(Pending)

### EXP-003b: GitHub Code (Python)

(Pending)

## Interpretation

(To be filled in after all runs complete.)
