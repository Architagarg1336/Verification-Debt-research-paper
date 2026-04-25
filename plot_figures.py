"""
Reproduce all 9 figures from the Verification Debt paper using matplotlib.
Data loaded from the actual AI Skepticism Dataset (Kaggle).
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = "output_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
DATA_PATH = "data/ai_skepticism_dataset.csv"

def load_data():
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Convert numeric fields
    for r in rows:
        r['ai_confidence_percentage'] = float(r['ai_confidence_percentage'])
        r['answer_accuracy_percentage'] = float(r['answer_accuracy_percentage'])
        r['trust_score_out_of_10'] = float(r['trust_score_out_of_10'])
        r['verification_duration_mins'] = float(r['verification_duration_mins'])
        r['performed_fact_check'] = r['performed_fact_check'].strip().upper() == 'TRUE'
        r['trust_calibration_valid'] = r['trust_calibration_valid'].strip().upper() == 'TRUE'
    return rows

data = load_data()

# ── Helper ────────────────────────────────────────────────────────────────────
def group_by(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return groups

CATEGORY_LABELS = {
    'current_events': 'Current Events',
    'opinion_based': 'Opinion Based',
    'math_calculation': 'Math Calculation',
    'scientific_facts': 'Scientific Facts',
    'recipe_cooking': 'Recipe Cooking',
    'general_knowledge': 'General Knowledge',
    'financial_advice': 'Financial Advice',
    'legal_advice': 'Legal Advice',
    'technical_coding': 'Technical Coding',
    'medical_advice': 'Medical Advice',
    'creative_writing': 'Creative Writing',
    'factual_historical': 'Factual Historical',
}


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Trust Score Distribution by AI Model (box plot)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_trust_by_model():
    models = ['ChatGPT-3.5', 'GPT-4', 'Claude', 'Gemini', 'Llama', 'Mistral']
    by_model = group_by(data, 'ai_model_name')
    box_data = [[r['trust_score_out_of_10'] for r in by_model[m]] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(box_data, labels=models, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel('Trust Score (0\u201310)')
    ax.set_title('Trust Score Distribution Across AI Models')
    ax.set_ylim(0, 10.5)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig1_trust_by_model.png')
    plt.close(fig)
    print("\u2713 Figure 1 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – AI Confidence vs. Actual Accuracy (scatter + regression)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_confidence_vs_accuracy():
    conf = np.array([r['ai_confidence_percentage'] for r in data])
    acc = np.array([r['answer_accuracy_percentage'] for r in data])

    coeffs = np.polyfit(conf, acc, 1)
    x_line = np.linspace(conf.min(), conf.max(), 100)
    y_line = np.polyval(coeffs, x_line)

    # Compute R^2
    y_pred = np.polyval(coeffs, conf)
    ss_res = np.sum((acc - y_pred) ** 2)
    ss_tot = np.sum((acc - np.mean(acc)) ** 2)
    r_sq = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(conf, acc, alpha=0.15, s=12, color='#4C72B0', edgecolors='none')
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression ($R^2$ = {r_sq:.3f})')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1.2, alpha=0.6, label='Perfect Calibration')

    ax.set_xlabel('AI Confidence (%)')
    ax.set_ylabel('Actual Accuracy (%)')
    ax.set_title('AI Confidence vs. Actual Accuracy')
    ax.set_xlim(35, 100)
    ax.set_ylim(15, 105)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig2_confidence_vs_accuracy.png')
    plt.close(fig)
    print("\u2713 Figure 2 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Verification Behavior by Skepticism Level (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_verification_behavior():
    order = ['Blind Trust', 'Moderate Trust', 'Skeptical', 'Highly Skeptical']
    by_skep = group_by(data, 'user_skepticism_category')

    verification_rate = []
    avg_duration = []
    labels = []
    for cat in order:
        rows = by_skep[cat]
        n = len(rows)
        n_checked = sum(1 for r in rows if r['performed_fact_check'])
        rate = (n_checked / n * 100) if n else 0
        dur = np.mean([r['verification_duration_mins'] for r in rows]) if n else 0
        pct = n / len(data) * 100
        verification_rate.append(rate)
        avg_duration.append(dur)
        labels.append(f'{cat}\n({pct:.1f}%)')

    x = np.arange(len(order))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width/2, verification_rate, width, label='Verification Rate (%)',
            color='#4C72B0', alpha=0.8)
    ax1.set_ylabel('Verification Rate (%)', color='#4C72B0')
    ax1.set_ylim(0, max(verification_rate) * 1.15)
    ax1.tick_params(axis='y', labelcolor='#4C72B0')

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, avg_duration, width, label='Avg Duration (min)',
            color='#C44E52', alpha=0.8)
    ax2.set_ylabel('Average Verification Duration (min)', color='#C44E52')
    ax2.set_ylim(0, max(avg_duration) * 1.25)
    ax2.tick_params(axis='y', labelcolor='#C44E52')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('Verification Rate and Duration by User Skepticism Level')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig3_verification_behavior.png')
    plt.close(fig)
    print("\u2713 Figure 3 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Trust Calibration Rate by Query Category (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_calibration_by_category():
    by_cat = group_by(data, 'query_category')
    cats = []
    cal_rates = []
    for raw_cat, rows in by_cat.items():
        n = len(rows)
        n_valid = sum(1 for r in rows if r['trust_calibration_valid'])
        rate = (n_valid / n * 100) if n else 0
        cats.append(CATEGORY_LABELS.get(raw_cat, raw_cat))
        cal_rates.append(rate)

    # Sort by calibration rate descending
    paired = sorted(zip(cal_rates, cats), reverse=True)
    cal_rates, cats = zip(*paired)

    overall_mean = sum(1 for r in data if r['trust_calibration_valid']) / len(data) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#55A868' if c >= 50 else '#C44E52' for c in cal_rates]
    bars = ax.barh(cats, cal_rates, color=colors, alpha=0.8, height=0.6)
    ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% Threshold')
    ax.axvline(x=overall_mean, color='blue', linestyle=':', linewidth=1.2, alpha=0.6,
               label=f'Overall Mean ({overall_mean:.1f}%)')

    ax.set_xlabel('Trust Calibration Rate (%)')
    ax.set_title('Trust Calibration Rate by Query Category')
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, cal_rates):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%',
                va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig4_calibration_by_category.png')
    plt.close(fig)
    print("\u2713 Figure 4 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Verification Debt Heatmap (models × categories)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_debt_heatmap():
    models = ['ChatGPT-3.5', 'GPT-4', 'Claude', 'Gemini', 'Llama', 'Mistral']
    raw_cats = sorted(CATEGORY_LABELS.keys())
    cat_labels = [CATEGORY_LABELS[c] for c in raw_cats]

    matrix = np.zeros((len(models), len(raw_cats)))
    by_model = group_by(data, 'ai_model_name')

    for i, model in enumerate(models):
        model_rows = by_model[model]
        by_cat = group_by(model_rows, 'query_category')
        for j, cat in enumerate(raw_cats):
            rows = by_cat.get(cat, [])
            if rows:
                mean_conf = np.mean([r['ai_confidence_percentage'] for r in rows])
                mean_acc = np.mean([r['answer_accuracy_percentage'] for r in rows])
                matrix[i, j] = mean_conf - mean_acc
            else:
                matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = LinearSegmentedColormap.from_list('debt', ['#2166AC', '#F7F7F7', '#B2182B'])
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(cat_labels)))
    ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(raw_cats)):
            color = 'white' if abs(matrix[i, j]) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center',
                    fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Overconfidence Gap (%)')
    ax.set_title('Verification Debt Heatmap: Overconfidence Gap by Model and Category')
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig5_debt_heatmap.png')
    plt.close(fig)
    print("\u2713 Figure 5 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 – Fact-Check Methods Distribution (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig6_factcheck_methods():
    checked = [r for r in data if r['performed_fact_check']]
    method_counts = defaultdict(int)
    for r in checked:
        method = r['fact_check_method_used'].strip()
        if method:
            method_counts[method] += 1

    total = sum(method_counts.values())
    methods = sorted(method_counts.keys(), key=lambda m: method_counts[m])
    percentages = [method_counts[m] / total * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.8, 0.35, len(methods)))
    bars = ax.barh(methods, percentages, color=colors, height=0.6, edgecolor='white')

    ax.set_xlabel('Percentage of Fact-Checking Users (%)')
    ax.set_title('Distribution of Verification Methods Used')
    ax.set_xlim(0, max(percentages) * 1.15)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, percentages):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig6_factcheck_methods.png')
    plt.close(fig)
    print("\u2713 Figure 6 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 – Education Level vs Trust & Verification (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig7_education_trust():
    edu_order = ['High School', 'Bachelors', 'Masters', 'PhD', 'Professional']
    by_edu = group_by(data, 'education_level')

    trust_scores = []
    verification_rates = []
    for edu in edu_order:
        rows = by_edu.get(edu, [])
        n = len(rows)
        trust_scores.append(np.mean([r['trust_score_out_of_10'] for r in rows]) if n else 0)
        n_checked = sum(1 for r in rows if r['performed_fact_check'])
        verification_rates.append((n_checked / n * 100) if n else 0)

    x = np.arange(len(edu_order))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width/2, trust_scores, width, label='Trust Score (0\u201310)',
            color='#4C72B0', alpha=0.8)
    ax1.set_ylabel('Trust Score (0\u201310)', color='#4C72B0')
    ax1.set_ylim(0, 10)
    ax1.tick_params(axis='y', labelcolor='#4C72B0')

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, verification_rates, width,
            label='Verification Rate (%)', color='#55A868', alpha=0.8)
    ax2.set_ylabel('Verification Rate (%)', color='#55A868')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='#55A868')

    ax1.set_xticks(x)
    ax1.set_xticklabels(edu_order)
    ax1.set_title('Trust Scores and Verification Rates by Education Level')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig7_education_trust.png')
    plt.close(fig)
    print("\u2713 Figure 7 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 – Decision Importance vs Verification (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig8_importance_verification():
    imp_order = ['Low', 'Medium', 'High', 'Critical']
    by_imp = group_by(data, 'decision_importance')

    verification_rate = []
    avg_duration = []
    for imp in imp_order:
        rows = by_imp.get(imp, [])
        n = len(rows)
        n_checked = sum(1 for r in rows if r['performed_fact_check'])
        verification_rate.append((n_checked / n * 100) if n else 0)
        avg_duration.append(np.mean([r['verification_duration_mins'] for r in rows]) if n else 0)

    x = np.arange(len(imp_order))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.bar(x - width/2, verification_rate, width,
            label='Verification Rate (%)', color='#4C72B0', alpha=0.8)
    ax1.set_ylabel('Verification Rate (%)', color='#4C72B0')
    ax1.set_ylim(0, max(verification_rate) * 1.15)
    ax1.tick_params(axis='y', labelcolor='#4C72B0')

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, avg_duration, width,
            label='Avg Duration (min)', color='#C44E52', alpha=0.8)
    ax2.set_ylabel('Average Verification Duration (min)', color='#C44E52')
    ax2.set_ylim(0, max(avg_duration) * 1.25)
    ax2.tick_params(axis='y', labelcolor='#C44E52')

    ax1.set_xticks(x)
    ax1.set_xticklabels(imp_order)
    ax1.set_xlabel('Decision Importance')
    ax1.set_title('Verification Rate and Duration by Decision Importance')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig8_importance_verification.png')
    plt.close(fig)
    print("\u2713 Figure 8 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 – Conceptual Framework Diagram
# ══════════════════════════════════════════════════════════════════════════════
def fig9_framework():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.5', facecolor='#D6EAF8', edgecolor='#2C3E50', linewidth=1.5)
    gap_style = dict(boxstyle='round,pad=0.5', facecolor='#FADBD8', edgecolor='#C0392B', linewidth=2)
    debt_style = dict(boxstyle='round,pad=0.5', facecolor='#F9E79F', edgecolor='#B7950B', linewidth=2)
    factor_style = dict(boxstyle='round,pad=0.4', facecolor='#E8F8F5', edgecolor='#1ABC9C', linewidth=1.2)

    ax.text(2, 5.5, 'AI Confidence\nSignals', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=box_style)
    ax.text(8, 5.5, 'Human Verification\nEffort', ha='center', va='center',
            fontsize=11, fontweight='bold', bbox=box_style)
    ax.text(5, 3.5, 'Trust Calibration\nGap', ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=gap_style)
    ax.text(5, 1.2, 'Verification Debt\nAccumulation', ha='center', va='center',
            fontsize=12, fontweight='bold', bbox=debt_style)

    ax.text(1, 3.5, 'Domain\nRisk', ha='center', va='center', fontsize=9, bbox=factor_style)
    ax.text(9, 3.5, 'User\nSkepticism', ha='center', va='center', fontsize=9, bbox=factor_style)
    ax.text(1.5, 1.2, 'Decision\nImportance', ha='center', va='center', fontsize=9, bbox=factor_style)
    ax.text(8.5, 1.2, 'Education\nLevel', ha='center', va='center', fontsize=9, bbox=factor_style)

    arrow_kw = dict(arrowstyle='->', color='#2C3E50', lw=1.8, connectionstyle='arc3,rad=0.1')
    ax.annotate('', xy=(4, 4.1), xytext=(2.8, 5.0), arrowprops=arrow_kw)
    ax.annotate('', xy=(6, 4.1), xytext=(7.2, 5.0), arrowprops=arrow_kw)
    ax.annotate('', xy=(5, 2.7), xytext=(5, 3.0),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2.2))

    thin_arrow = dict(arrowstyle='->', color='#1ABC9C', lw=1.2, connectionstyle='arc3,rad=0.15')
    ax.annotate('', xy=(3.8, 3.5), xytext=(1.8, 3.5), arrowprops=thin_arrow)
    ax.annotate('', xy=(6.2, 3.5), xytext=(8.2, 3.5), arrowprops=thin_arrow)
    ax.annotate('', xy=(3.8, 1.2), xytext=(2.3, 1.2), arrowprops=thin_arrow)
    ax.annotate('', xy=(6.2, 1.2), xytext=(7.7, 1.2), arrowprops=thin_arrow)

    ax.set_title('Conceptual Framework for Verification Debt', fontsize=13,
                 fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig9_framework.png')
    plt.close(fig)
    print("\u2713 Figure 9 saved")


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"Loaded {len(data)} rows from {DATA_PATH}\n")
    print("Generating all 9 figures...\n")
    fig1_trust_by_model()
    fig2_confidence_vs_accuracy()
    fig3_verification_behavior()
    fig4_calibration_by_category()
    fig5_debt_heatmap()
    fig6_factcheck_methods()
    fig7_education_trust()
    fig8_importance_verification()
    fig9_framework()
    print(f"\nAll figures saved to '{OUTPUT_DIR}/' directory.")
