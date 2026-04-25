# Verification Debt: Trust and Reliability of AI-Generated Code

Reproducible figures for the research paper *"Verification Debt: Trust and Reliability of AI-Generated Code"*, which introduces **verification debt** — the accumulated risk from insufficient verification of AI-generated code.

The paper analyzes 1,000 human–AI interactions across six LLMs (ChatGPT-3.5, GPT-4, Claude, Gemini, Llama, Mistral) and twelve query categories, examining the calibration gap between AI confidence and actual accuracy, user trust patterns, and verification behavior.

## Key Findings

| Metric | Value |
|---|---|
| Mean AI Confidence | 72.01% |
| Mean Answer Accuracy | 69.98% |
| Mean Overconfidence Gap | 2.02% |
| Mean User Trust | 7.91 / 10 |
| Users Who Never Verify | 37.2% |
| Trust Calibration Valid | 72.7% |

## Repository Structure

```
├── README.md
├── plot_figures.py              # Script to generate all 9 figures
├── data/
│   └── ai_skepticism_dataset.csv   # Source dataset (1,000 records, 23 columns)
├── output_figures/              # Generated figures (after running the script)
│   ├── fig1_trust_by_model.png
│   ├── fig2_confidence_vs_accuracy.png
│   ├── fig3_verification_behavior.png
│   ├── fig4_calibration_by_category.png
│   ├── fig5_debt_heatmap.png
│   ├── fig6_factcheck_methods.png
│   ├── fig7_education_trust.png
│   ├── fig8_importance_verification.png
│   └── fig9_framework.png
└── verification_debt_paper_overleaf/
    ├── main.tex                 # LaTeX source of the paper
    └── figures/                 # Paper-ready figures (PDF + PNG)
```

## Figures

| # | Figure | Description |
|---|--------|-------------|
| 1 | Trust by Model | Box plot of trust score distributions across six AI models |
| 2 | Confidence vs Accuracy | Scatter plot with regression line (R²) and perfect calibration diagonal |
| 3 | Verification Behavior | Verification rate and duration by user skepticism level |
| 4 | Calibration by Category | Trust calibration rate across 12 query categories |
| 5 | Debt Heatmap | Overconfidence gap heatmap (models × categories) |
| 6 | Fact-Check Methods | Distribution of verification methods used by participants |
| 7 | Education & Trust | Trust scores and verification rates by education level |
| 8 | Decision Importance | Verification rate and duration by decision stakes |
| 9 | Framework | Conceptual framework diagram for verification debt |

## Quick Start

### Prerequisites

- Python 3.6+
- `matplotlib`
- `numpy`

### Install Dependencies

```bash
pip install matplotlib numpy
```

### Download the Dataset

The dataset is sourced from Kaggle: [AI Skepticism Dataset](https://www.kaggle.com/datasets/ayeshaseherr/ai-dataset) (CC0 Public Domain).

```bash
pip install kaggle
kaggle datasets download -d ayeshaseherr/ai-dataset --unzip -p ./data
```

Or download manually from the link above and place `ai_skepticism_dataset.csv` in the `data/` directory.

### Generate Figures

```bash
python plot_figures.py
```

All 9 figures will be saved to `output_figures/` at 300 DPI.

## Dataset

The [AI Skepticism Dataset](https://www.kaggle.com/datasets/ayeshaseherr/ai-dataset) contains 1,000 records with 23 columns covering:

- **AI attributes**: model name, confidence %, response length, citations, disclaimers, hedging
- **User demographics**: age bracket, education level, digital literacy, AI familiarity
- **Interaction context**: query category, decision importance, urgency, belief alignment
- **Outcomes**: trust score (0–10), answer accuracy %, fact-check behavior, verification duration, trust calibration validity, skepticism category

## Citation

If you use this code or reference the paper, please cite:

```bibtex
@inproceedings{verificationdebt2026,
  title     = {Verification Debt: Trust and Reliability of AI-Generated Code},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of IEEE Conference},
  year      = {2026}
}
```

## License

- **Code**: MIT License
- **Dataset**: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
