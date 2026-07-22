# Bangla Hate Speech Fragility Research

**Paper Title**: Beyond Accuracy: Evaluating Temporal Drift, Dialectal Bias, and Adversarial Transliteration Fragility in Bangla Hate Speech Models

**Conference**: COMPAS 2026

## Project Overview

This repository contains the complete technical implementation for evaluating robustness of Bangla hate speech detection models across three critical failure modes:

1. **Axis A - Temporal Robustness**: Continual learning across chronologically ordered datasets
2. **Axis B - Dialectal Fairness**: Per-dialect performance audit and fairness metrics
3. **Axis C - Adversarial Transliteration**: Robustness to Romanized Bangla and code-mixed attacks

## Repository Structure

```
bangla-hatespeech-fragility/
├── data/
│   ├── raw/                      # Original datasets (not tracked)
│   ├── interim/                  # Intermediate processing
│   └── processed/                # Final unified master CSV
├── notebooks/                    # Exploratory notebooks only
├── src/
│   ├── data/                     # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py             # Load all datasets
│   │   └── preprocessor.py       # Text normalization
│   ├── models/                   # Baseline model training
│   │   ├── __init__.py
│   │   ├── baseline.py           # TF-IDF + LR, BanglaBERT, XLM-R
│   │   └── trainer.py            # Training loop utilities
│   ├── attacks/                  # Transliteration attacks (Axis C)
│   │   ├── __init__.py
│   │   └── transliteration.py    # Attack generation & evaluation
│   ├── fairness/                 # Dialect fairness (Axis B)
│   │   ├── __init__.py
│   │   └── audit.py              # Per-dialect metrics
│   ├── continual/                # Continual learning (Axis A)
│   │   ├── __init__.py
│   │   └── strategies.py         # EWC, Replay, LoRA
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # F1, confusion matrix, etc.
│       └── config.py             # Global config management
├── configs/                      # YAML configs for experiments
├── scripts/                      # Executable Python scripts
│   ├── 00_prepare_data.py        # M1: Master CSV pipeline
│   ├── 01_train_baseline.py      # M2: TF-IDF + Transformer baselines
│   ├── 02_run_transliteration.py # M3: Robustness evaluation
│   ├── 03_run_fairness_audit.py  # M4: Dialect fairness
│   └── 04_run_continual_learning.py # M5: Continual learning
├── results/
│   ├── tables/                   # CSV outputs for each experiment
│   ├── figures/                  # PNG/PDF plots
│   └── logs/                     # Training logs
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone and Setup Environment

```bash
cd bangla-hatespeech-fragility
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 2. Download Datasets

Download the following datasets and place them in `data/raw/`:

- **BIDWESH** (2025) - Dialectal Bangla hate speech
- **BOISHOMMO** (2025) - Multi-label Bangla hate speech  
- **BanTH** (2024) - Transliterated Bangla hate speech
- **Karim et al. / BD-SHS** (2020) - Older Bangla hate speech

Detailed dataset info in `configs/datasets_metadata.csv`

### 3. Run Milestone Sequence

```bash
# M1: Prepare unified master CSV
python scripts/00_prepare_data.py

# M2: Train baseline models
python scripts/01_train_baseline.py

# M3: Transliteration robustness (Axis C)
python scripts/02_run_transliteration.py

# M4: Dialect fairness audit (Axis B)
python scripts/03_run_fairness_audit.py

# M5: Continual learning (Axis A)
python scripts/04_run_continual_learning.py
```

## Datasets Used

| Dataset | Size | Script | Use Case | Download |
|---------|------|--------|----------|----------|
| BIDWESH | 9,183 | Bangla | Axis B (dialectal fairness) | Mendeley |
| BOISHOMMO | 2,499 | Bangla | Axis A Phase 3 + Axis B | Mendeley |
| BanTH | 37,300 | Romanized | Axis C (transliteration) | HuggingFace |
| Karim 2020 | ~10,000 | Bangla | Axis A Phase 1 (temporal) | UCI ML |

**Total: ~80,000 samples across 4 primary datasets**

## Key Contributions

1. **First unified three-axis fragility benchmark** for Bangla hate speech
2. **First fairness audit** across regional dialect groups using BIDWESH
3. **First adversarial transliteration attack study** using BanTH
4. **First continual learning comparison** on chronological Bangla corpora
5. **Public reproducibility package** (code + configs) on GitHub

## Reproducibility

All experiments use:
- Fixed random seeds (42) for full reproducibility
- Experiment tracking via WandB
- Config-driven execution (all hyperparameters in `configs/`)
- Version-controlled dependencies (`requirements.txt`)

## Citation

```bibtex
@inproceedings{bangla-fragility-2026,
  title={Beyond Accuracy: Evaluating Temporal Drift, Dialectal Bias, and Adversarial Transliteration Fragility in Bangla Hate Speech Models},
  author={Author Name},
  booktitle={Proceedings of COMPAS 2026},
  year={2026}
}
```

## License

MIT License

## Contact

For questions, please open an issue on GitHub.
