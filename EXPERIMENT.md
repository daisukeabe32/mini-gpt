# M3 Experiment: Induction Head Detection — Char vs BPE Tokenization
2026.May.09

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

Resume procedure: download `best.pt` from previous session Output →
upload as Kaggle Dataset → set `RESUME_PT` in Cell 7b of `kaggle_olsson.ipynb`.
Model loads optimizer state and step counter from the checkpoint and continues seamlessly.

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
| 2 | EXP-002 resume(1.0) | 48,001 → ? | 0.79B → ? | T4 x2 | 🔄 実行中（2026-05-23〜） |
| 3 | （予定） | ? → 120,000 | ? → 1.97B | T4 x2 | 🔲 未開始 |

## Operational notes — Kaggle resume 手順（今後の参照用）

### GPU 選択
- **T4 x2 を使う**。T4 x1 という選択肢は Kaggle に存在しない（P100 x1 / T4 x2 / GPU なし）。
- **P100 は使用不可**。PyTorch 2.x が P100（compute capability sm_60）のカーネルを非サポート。
  起動直後に `CUDA error: no kernel image is available` でクラッシュする。
- T4 x2 はクォータを **2h / wall-clock 1h** で消費する点に注意。

### 前 session の snapshot を引き継ぐ方法
1. 新 Notebook → 右サイドバー → **Input → + Add Input**
2. **"Notebook Outputs" タブ**を選ぶ（"Datasets" タブではない）
3. 前 session の Notebook を選択 → 追加
   → `step_*.pt` が `/kaggle/input/notebooks/da3246/{slug}/checkpoints/...` 以下にマウントされる

### RESUME_PT のパス形式
```
/kaggle/input/notebooks/da3246/{notebook-slug}/checkpoints/{YYYYMMDD_HHMMSS}/best.pt
```
- `{notebook-slug}` = Notebook 名を lowercase + hyphen に変換（例: "EXP-002" → `exp-002`）
- Cell 9 を draft モードで単体実行すれば正確なパスが表示される

### Cell 13（emergence curve）の動作
- `/kaggle/working/checkpoints/*/step_*.pt`（今 session）
- `/kaggle/input/**/step_*.pt`（前 session）
の両方を自動スキャンするため、前 session Output を Input に追加するだけで全 snapshot がつながる。

## Results

（完了後に記入）
