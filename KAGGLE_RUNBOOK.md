# Kaggle Runbook — mini-gpt Experiments

_This file covers operational procedures only. For experiment findings, see EXPERIMENT.md._

## 原則

- **1実験 = 1 Notebook**（Output が混ざらないよう）
- 必ず **Save Version（commit mode）** で実行（draft は session 終了時にファイルが消える）
- クラッシュ・クォータ切れは想定内。新 Notebook を立ち上げて resume する

---

## 新しい Notebook を立ち上げる手順

### 1. Notebook 作成

1. Kaggle → **Code** → **New Notebook**
2. 右上 **Settings** → Accelerator: `GPU T4 x1`、Internet: `ON`
3. 右サイドバーが閉じている場合は**右端の `›` ボタン**をクリックして開く

### 2. W&B API Key を紐づける

1. 右サイドバー → **Add-ons** → **Secrets**
2. `WANDB_API_KEY` が表示されている → **Attach to notebook** をオン
3. （初回のみ）`WANDB_API_KEY` がない場合 → **Add a new secret** → W&B の API キーを入力

### 3. Dataset をマウントする

1. 右サイドバー → **Input** → **+ Add Input**
2. **Your Datasets** タブ → `mini-gpt-tinystories` → `+` ボタンで追加
3. resume する場合はさらに **Your Datasets** → 前回の `best.pt` を含む dataset も追加

---

## Cell 一覧

各 Cell の内容は `kaggle_olsson.ipynb`（リポジトリ）からコピーする。

| Cell | 役割 | 毎回実行 |
|---|---|---|
| Cell 1 | GPU 確認 | ✅ |
| Cell 2 | pip install（tokenizers, wandb） | ✅ |
| Cell 3 | git clone / pull（mini-gpt リポジトリ） | ✅ |
| Cell 4 | W&B ログイン（Kaggle Secret 経由） | ✅ |
| Cell 5 | tokenized data を `/kaggle/input` → working にコピー | ✅ |
| Cell 6 | data 検証（token 数確認） | ✅ |
| Cell 7 | checkpoint 出力ディレクトリ作成 | ✅ |
| **Cell 7b** | **RESUME_PT 設定（ここだけ書き換える）** | ✅ |
| Cell 8 | 訓練（fresh run、RESUME_PT が空のときのみ実行） | ✅ |
| Cell 9 | `/kaggle/input` 以下の `.pt` ファイルを列挙（resume パス確認用） | ✅ |
| Cell 10 | 訓練（resume、RESUME_PT が設定されているときのみ実行） | ✅ |
| Cell 11 | 訓練ログ確認 | ✅ |
| Cell 12 | Induction head 分析（訓練完了後に単体実行） | 完了後 |
| Cell 13 | Emergence curve（全 session の snapshots をスキャン） | 完了後 |

---

## Fresh Run の手順

1. Cell 7b の `RESUME_PT` を空にする：
   ```python
   RESUME_PT = ''
   ```
2. 右上 **Save Version** → **Save & Run All** → 実行開始

---

## Resume の手順（クラッシュ・クォータ切れ後）

### Step 1: 前 session の Output を Input として追加

新しい Notebook（次 session）を開いた後：

1. 右サイドバー → **Input** → **+ Add Input**
2. **Notebook Outputs** タブ → 前 session の Notebook → 最新 version を選択 → 追加

> ⚠️ **「Datasets」タブではなく「Notebook Outputs」タブ**を使う。
> これにより `step_*.pt`（snapshot）も含めて `/kaggle/input/` 以下にすべてマウントされる。
> `best.pt` だけを Dataset としてアップロードする旧手順は不要。

### Step 2: Cell 9 でパスを確認

Cell 9 を実行 → `best.pt` のパスを確認してコピー

### Step 3: Cell 7b を書き換える

```python
RESUME_PT = '/kaggle/input/{前 session の notebook slug}/checkpoints/YYYYMMDD_HHMMSS/best.pt'
```

### Step 4: Save Version で実行

- Cell 8 は自動スキップ（RESUME_PT が設定されているため）
- Cell 10 が resume から訓練を継続
- 前 session の `step_*.pt` は `/kaggle/input/` 以下に残るので Cell 13 が自動的に全 snapshot をスキャンする

---

## Emergence Curve を完成させる手順（全 session 分）

各 session の Output には `step_XXXXXX.pt` が含まれる。
Cell 13 は `/kaggle/working/` と `/kaggle/input/` 両方をスキャンするため、
**過去 session の Output を Input として追加**すれば自動的に全 snapshot を使う。

1. 最終 session の Notebook → **+ Add Input** → 過去の全 session の Output を追加
2. Cell 13 を実行 → `figs/emergence_curve.png` が全期間をカバーした curve として出力される

---

## チェックポイント・クォータ管理

| 項目 | 値 |
|---|---|
| Kaggle GPU クォータ | 30h / 週 |
| 1 session 上限 | 12h |
| ペース（T4 x1, d_model=512） | ~15 min / 1000 steps |
| save_every | 2000 steps（~32M tokens ごと） |
| 12h で到達できる steps | ~46,000 steps |
| phase change 閾値 | ~91,000 steps（1.5B tokens） |

→ fresh run から phase change まで約 **2 session（24h）** 必要。

---

## GPU 選択

- **T4 x2 を使う**。T4 x1 という選択肢は Kaggle に存在しない（P100 x1 / T4 x2 / GPU なし）。
- **P100 は使用不可**。PyTorch 2.x が P100（compute capability sm_60）のカーネルを非サポート。
  起動直後に `CUDA error: no kernel image is available` でクラッシュする。
- T4 x2 はクォータを **2h / wall-clock 1h** で消費する点に注意。

---

## RESUME_PT のパス形式

```
/kaggle/input/{notebook-slug}/checkpoints/{YYYYMMDD_HHMMSS}/best.pt
```

- `{notebook-slug}` = Notebook 名を lowercase + hyphen に変換（例: "EXP-002" → `exp-002`）
- Cell 9 を draft モードで単体実行すれば正確なパスが表示される

---

## Cell 13（emergence curve）の動作

Cell 13 は以下の両方を自動スキャンする：

- `/kaggle/working/checkpoints/*/step_*.pt`（今 session）
- `/kaggle/input/**/step_*.pt`（前 session）

前 session Output を Input に追加するだけで全 snapshot がつながり、`figs/emergence_curve.png` が全期間をカバーした curve として出力される。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `train_ids.pt not found` | Dataset がマウントされていない | `+ Add Input` → `mini-gpt-tinystories` を追加 |
| `RESUME_PT not found` | Dataset がマウントされていない / パスが間違い | Cell 9 でパスを確認し Cell 7b を修正 |
| `zsh: command not found: kaggle` | PATH が通っていない | `~/miniconda3/envs/ml/bin/kaggle` でフルパス実行 |
| Output の `best.pt` が 0 bytes | ダウンロード失敗 | ブラウザから直接ダウンロード |
| セッションが step X で止まる | 12h タイムアウト | 次 session で前 Output を Input に追加 → resume |
| GPU が選択できない | 電話番号未確認 | kaggle.com/settings で確認 |
| Cell 9 に step_*.pt が出ない | 前 session Output を Input 追加し忘れ | Notebook Outputs タブから前 session を追加 |

---

## Notebook Registry

実験ごとの Kaggle Notebook・使用 Cell バージョン・セッション記録。
Cell の内容は git で管理（`kaggle_olsson.ipynb`）。ここでは「何をした実験か」と「操作ログ」を記録する。

---

### EXP-001: Olsson-approximate（2-layer emergence 初回試行）

| 項目 | 値 |
|---|---|
| 実験 ID | EXP-001 |
| Kaggle Notebook 名 | （記録なし） |
| 実験意図 | 2-layer / BPE 30K / TinyStories で Olsson et al. の phase change を再現する。1.97B tokens まで訓練し emergence curve を得る |
| Cell バージョン | git commit `a50cf13`（multi-seq_len 対応前） |
| 開始日 | 2026-05-12 |
| 終了日 | 2026-05-16 |
| ステータス | ✅ 完了（3 session） |

**Session ログ**

| Session | Steps | Tokens | 終了理由 | best.pt | step_*.pt |
|---------|-------|--------|---------|---------|-----------|
| 1 | 0 → 53,000 | 0 → 0.87B | 12h timeout | 回収済み | ❌ 喪失 |
| 2 | 53,001 → 98,000 | 0.87B → 1.61B | 12h timeout | 回収済み | ❌ 喪失 |
| 3 | 98,001 → 119,999 | 1.61B → 1.97B | 正常終了 | 回収済み（`best (3).pt`） | ✅ 一部保全 |

**結果サマリー**
- val_loss: 3.0532（step 115,000）
- Induction head: L2H6 ✅（score 0.1897 @ seq_len=16）
- 初回分析は seq_len=64 → score=0.034 と誤判定。seq_len=16 で再分析して検出
- Emergence curve: Session 3 分（1.638B〜1.97B tokens）のみ。phase change タイミング未観測

---

### EXP-002: Emergence Curve（full snapshot 保全版）

| 項目 | 値 |
|---|---|
| 実験 ID | EXP-002 |
| Kaggle Notebook 名 | （開始時に記入） |
| 実験意図 | EXP-001 で induction head L2H6 の**存在**は確認済み。今回は 0 tokens からの全 snapshot を保全し、L2H6 が**いつ・どのように出現したか**（phase change の有無と発生タイミング）を観測する。multi-scale 分析（seq_len = 8, 16, 32, 64）で距離依存性の時間発展も見る |
| Cell バージョン | git commit `91d5ffb`（multi-seq_len 対応済み） |
| 開始日 | （開始時に記入） |
| 終了日 | （完了時に記入） |
| ステータス | 🔲 未開始 |

**Session ログ**

| Session | Steps | Tokens | 終了理由 | best.pt | step_*.pt | 次 session への引き継ぎ |
|---------|-------|--------|---------|---------|-----------|----------------------|
| 1 | 0 → 48,000 | 0 → 0.79B | quota 使い切り | 回収済み | ✅ 保全 | Output を Input に追加 |
| 2 | 48,001 → 93,000 | 0.79B → 1.53B | 12h timeout | 回収済み | ✅ 保全 | Output を Input に追加 |
| 3 | 93,001 → 120,000 | 1.53B → 1.97B | 🔲 未開始 | — | — | — |

**結果サマリー**
- Session 1: 完了（step 48,000 / 0.79B tokens）
- Session 2: 完了（step 93,000 / 1.53B tokens、12h timeout）— phase change 閾値（~1.5B tokens = ~91,000 steps）は Session 2 で通過済み
- Session 3: 未開始（93,001 → 120,000 steps）
- Emergence curve: `figs/emergence_curve.png`（Session 3 完了後）
- Phase change 観測: （完了後に記入）
- Phase change タイミング: （記入）
