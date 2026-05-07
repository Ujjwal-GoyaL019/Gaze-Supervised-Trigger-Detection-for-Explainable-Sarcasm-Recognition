# Gaze-Guided Sarcasm Detection with Trigger Localization

A BERT-based sarcasm detection model that uses **eye-tracking (gaze) data** as weak supervision to learn *where* sarcasm happens — not just *whether* it does. The model predicts a soft probability distribution over tokens, trained to focus on the same words human readers fixate on when processing ironic language.

---

## Overview

Sarcasm is defined by a local mismatch: a positive word in a negative context (or vice versa). Most models classify the sentence as sarcastic but can't say which token is the trigger. This project trains a **dual-output BERT model** that jointly:

1. **Classifies** whether a sentence is sarcastic (standard cross-entropy loss)
2. **Localizes** the sarcasm trigger (KL-divergence against soft gaze-derived targets)

The trigger head is supervised by human fixation durations — words readers stare at longer are more likely to be the sarcasm trigger.

---

## Features

- **Gaze-guided trigger supervision** — fixation durations converted to soft token-level targets
- **Local contradiction features** — VADER-based sentiment gap and sign-flip detection per word
- **Data augmentation** — synonym-swap augmentation on sarcastic samples only (no leakage)
- **Lambda sweep** — automatic hyperparameter search for the KL loss weight (λ)
- **6-condition ablation study** — isolates contribution of each component
- **Full evaluation suite** — Precision@k, Spearman ρ, Token IoU@k, entropy, and contradiction hit rate
- **Qualitative analysis** — per-sentence trigger probability visualizations with VADER scores

---

## Model Architecture

```
Input Text
    │
    ▼
BERT (bert-base-uncased)
    │
    ├──► [CLS] pooler output ──► Dropout ──► Linear(768, 2) ──► Sarcasm logits
    │
    └──► Token hidden states [H_i]
              │
              + Contradiction features (5-dim per token)
              │
              ▼
         Dropout ──► Linear ──► GELU ──► Linear(1) ──► Trigger logits
```

**Loss function:**

```
L = CE(sarcasm, class_weighted) + λ × KL(softmax(trigger), gaze_target)
```

KL is computed only on sarcastic rows with non-zero gaze targets.

---

## Local Contradiction Features (5-dim per token)

| Feature | Description |
|---|---|
| `local_gap` | `\|word_sentiment − mean_neighbor_sentiment\|` |
| `sign_flip` | 1 if word and neighbors have opposite polarity |
| `\|word_sent\|` | Absolute VADER compound score of the word |
| `\|neigh_sent\|` | Absolute mean VADER score of neighbors |
| `is_polar` | 1 if `\|word_sent\|` > 0.05 |

Motivated by Riloff et al. (2013): positive verbs in negative situations are the canonical sarcasm trigger pattern.

---

## Dataset

The notebook expects two CSV files:

| File | Required Columns |
|---|---|
| `text_and_annorations.csv` | `Text_ID`, `Text`, `Sarcasm` (Yes/No) |
| `Fixation_sequence.csv` | `Text_ID`, `Word_ID`, `Word`, `Fixation_Duration`, `Participant_ID` |

Configured for Kaggle at `/kaggle/input/datasets/ujjwalgoyalsvnit/filess/`. Update `CONFIG['text_path']` and `CONFIG['fix_path']` for local use.

---

## Ablation Study

Six conditions are trained and compared end-to-end:

| Condition | CE | KL (Gaze) | Contradiction | λ | Purpose |
|---|---|---|---|---|---|
| A | ✓ | ✗ | ✗ | — | Pure BERT baseline |
| B | ✓ | ✓ | ✗ | 0.05 | Gaze, weak λ |
| C | ✓ | ✓ | ✗ | best | Gaze, best λ |
| D | ✓ | ✓ | ✓ | 0.05 | Full model, weak λ |
| **E** | **✓** | **✓** | **✓** | **best** | **Full model, best λ (ours)** |
| F | ✓ | ✓ | ✓ | 0.30 | Full model, strong λ |

---

## Evaluation Metrics

| Metric | What It Measures |
|---|---|
| Macro-F1 | Sarcasm classification balance across both classes |
| Precision@k | Top-k predicted tokens overlap with gaze ground truth |
| Spearman ρ | Rank correlation between predicted probs and gaze scores |
| Token IoU@k | Intersection-over-union of top-k token sets |
| Trigger Entropy | Distribution sharpness (lower = more confident localization) |
| Contradiction Hit Rate | Fraction of top-k predictions with local sentiment sign-flip |

---

## Requirements

```
pip install vaderSentiment transformers torch scikit-learn matplotlib seaborn scipy
```

| Package | Purpose |
|---|---|
| `transformers` | BERT model and tokenizer |
| `torch` | Training, loss, DataLoader |
| `vaderSentiment` | Word-level sentiment scoring |
| `scikit-learn` | Metrics, train/test split |
| `scipy` | Spearman correlation |
| `matplotlib` / `seaborn` | Visualization |

---

## Configuration

All hyperparameters are defined in a single `CONFIG` dict:

```python
CONFIG = {
    'model_name'      : 'bert-base-uncased',
    'max_len'         : 128,
    'batch_size'      : 16,
    'epochs'          : 8,
    'lr'              : 2e-5,
    'dropout'         : 0.2,
    'lambda_trigger'  : 0.15,   # updated after sweep
    'gaze_z_threshold': 0.5,
    'local_window'    : 2,
    'topk'            : 3,
    'val_size'        : 0.10,
    'test_size'       : 0.15,
    'patience'        : 3,
    'aug_factor'      : 2,
    'entropy_collapse_threshold': 0.85,
}
```

---

## Notebook Structure

| Section | Description |
|---|---|
| 1. Imports & Config | Dependencies, reproducibility seeds, CONFIG dict |
| 2. Load & Validate Data | CSV loading, label encoding, sanity checks |
| 3. Dataset Augmentation | Synonym-swap augmentation on sarcastic training rows only |
| 4. Gaze Dictionary & Alignment | Fixation durations → normalized per-word gaze scores |
| 5. Local Contradiction Features | VADER-based 5-dim feature computation per word |
| 6. Soft Trigger Labels | Gaze + contradiction → soft token-level target distributions |
| 7. Token Alignment & Dataset | Subword alignment, `SarcasmDataset`, weighted sampler |
| 8. Model Architecture | `SarcasmTriggerModel` definition (BERT + dual heads) |
| 9. Loss + Sharpness Monitor | Weighted CE + KL loss; entropy collapse detection |
| 10. Lambda Sweep | Grid search over λ ∈ {0.05, 0.10, 0.15, 0.20, 0.30} |
| 11. Full Training | Best-λ training with early stopping and LR schedule |
| 12. Ablation Study | All 6 conditions trained and checkpointed |
| 13. Trigger Evaluation Suite | P@k, Spearman ρ, IoU, entropy, contradiction hit rate |
| 14. Qualitative Analysis | Per-sentence trigger visualizations with VADER bars |
| 15. Visualization Dashboard | Training curves, ablation bar charts, gaze heatmaps |

---

## Outputs

All artifacts are saved to `./outputs/`:

```
outputs/
├── config.json           # Full CONFIG dump
├── best_A_CE_only.pt     # Model checkpoint — Condition A
├── best_B_Gaze_weakLam.pt
├── best_C_Gaze_bestLam.pt
├── best_D_Full_weakLam.pt
├── best_E_Full_bestLam.pt
├── best_F_Full_strongLam.pt
├── lambda_sweep.png      # λ sweep: F1, P@k, entropy, KL
├── fig1_curves.png       # Training curves (A vs C vs E)
├── fig2_ablation.png     # Full 6-condition ablation bar chart
└── fig3_heatmap.png      # Gaze GT vs model trigger heatmap
```

---

## Training Details

- **BERT frozen** for epoch 1, unfrozen from epoch 2 (warm-up strategy)
- **Layer-wise LR**: BERT at `2e-5`, task heads at `1e-4`
- **Linear warmup + decay** schedule (10% warmup)
- **Weighted random sampler** to handle class imbalance
- **Early stopping** with patience=3 on combined score `0.7×F1 + 0.3×P@k`
- **Gradient clipping** at norm 1.0

---

## Reference

Riloff, E., Qadir, A., Surve, P., De Silva, L., Gilbert, N., & Huang, R. (2013). Sarcasm as contrast between a positive sentiment and negative situation. *EMNLP 2013*.
