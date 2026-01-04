"""
Visualization Script for Model Comparison Results
Generates publication-quality plots for the final report

Authors: Tugan Başaran, Zehra Sağın, Mert Korkmaz
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Load results
comparison_df = pd.read_csv('model_comparison_results.csv')
cv_results_df = pd.read_csv('kfold_cv_detailed_results.csv')

print("📊 Generating visualizations...")

# =====================================
# FIGURE 1: Model Comparison Bar Chart
# =====================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison: SMOTE-NC vs No SMOTE-NC', fontsize=16, fontweight='bold')

metrics = ['F1-Score', 'Recall', 'Precision', 'ROC-AUC']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    # Prepare data
    models = comparison_df['Model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    without_smote = []
    with_smote = []
    
    for model in models:
        without_val = float(comparison_df[(comparison_df['Model'] == model) & 
                                          (comparison_df['Condition'] == 'Without SMOTE-NC')][metric].values[0])
        with_val = float(comparison_df[(comparison_df['Model'] == model) & 
                                      (comparison_df['Condition'] == 'With SMOTE-NC')][metric].values[0])
        without_smote.append(without_val)
        with_smote.append(with_val)
    
    # Create bars
    bars1 = ax.bar(x - width/2, without_smote, width, label='Without SMOTE-NC', 
                   color='skyblue', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, with_smote, width, label='With SMOTE-NC', 
                   color='lightcoral', edgecolor='black', linewidth=1.2)
    
    # Customize
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure1_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure1_model_comparison.png")
plt.close()

# =====================================
# FIGURE 2: K-Fold CV Box Plots
# =====================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('K-Fold Cross-Validation Results Distribution', fontsize=16, fontweight='bold')

metrics_cv = ['F1', 'Recall', 'Precision', 'ROC_AUC']
metric_names = ['F1-Score', 'Recall', 'Precision', 'ROC-AUC']

for idx, (metric, metric_name) in enumerate(zip(metrics_cv, metric_names)):
    ax = axes[idx // 2, idx % 2]
    
    # Prepare data for box plot
    data_to_plot = []
    labels = []
    
    models = cv_results_df['Model'].unique()
    
    for model in models:
        # Without SMOTE-NC
        without_data = cv_results_df[(cv_results_df['Model'] == model) & 
                                    (cv_results_df['Condition'] == 'Without SMOTE-NC')][metric].values
        data_to_plot.append(without_data)
        labels.append(f'{model}\n(No SMOTE)')
        
        # With SMOTE-NC
        with_data = cv_results_df[(cv_results_df['Model'] == model) & 
                                 (cv_results_df['Condition'] == 'With SMOTE-NC')][metric].values
        data_to_plot.append(with_data)
        labels.append(f'{model}\n(SMOTE-NC)')
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    notch=True, showmeans=True)
    
    # Color boxes
    colors = ['skyblue', 'lightcoral'] * len(models)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    ax.set_ylabel(metric_name, fontweight='bold')
    ax.set_title(f'{metric_name} - 10-Fold CV Distribution', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_kfold_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure2_kfold_boxplots.png")
plt.close()

# =====================================
# FIGURE 3: Heatmap of Model Performance
# =====================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Performance Heatmap', fontsize=16, fontweight='bold')

models = comparison_df['Model'].unique()
metrics_list = ['F1-Score', 'Recall', 'Precision', 'ROC-AUC']

# Without SMOTE-NC
data_without = []
for model in models:
    row = []
    for metric in metrics_list:
        val = float(comparison_df[(comparison_df['Model'] == model) & 
                                 (comparison_df['Condition'] == 'Without SMOTE-NC')][metric].values[0])
        row.append(val)
    data_without.append(row)

df_without = pd.DataFrame(data_without, columns=metrics_list, index=models)
sns.heatmap(df_without, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0], 
            cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='black')
axes[0].set_title('Without SMOTE-NC', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Model', fontweight='bold')

# With SMOTE-NC
data_with = []
for model in models:
    row = []
    for metric in metrics_list:
        val = float(comparison_df[(comparison_df['Model'] == model) & 
                                 (comparison_df['Condition'] == 'With SMOTE-NC')][metric].values[0])
        row.append(val)
    data_with.append(row)

df_with = pd.DataFrame(data_with, columns=metrics_list, index=models)
sns.heatmap(df_with, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1], 
            cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='black')
axes[1].set_title('With SMOTE-NC', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('figure3_performance_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure3_performance_heatmap.png")
plt.close()

# =====================================
# FIGURE 4: F1-Score Comparison Line Plot
# =====================================

fig, ax = plt.subplots(figsize=(12, 7))

for model in models:
    # Get fold-by-fold F1 scores
    without_f1 = cv_results_df[(cv_results_df['Model'] == model) & 
                              (cv_results_df['Condition'] == 'Without SMOTE-NC')]['F1'].values
    with_f1 = cv_results_df[(cv_results_df['Model'] == model) & 
                           (cv_results_df['Condition'] == 'With SMOTE-NC')]['F1'].values
    
    folds = range(1, len(without_f1) + 1)
    
    # Plot
    ax.plot(folds, without_f1, marker='o', linestyle='--', linewidth=2, 
            label=f'{model} (No SMOTE)', alpha=0.7)
    ax.plot(folds, with_f1, marker='s', linestyle='-', linewidth=2, 
            label=f'{model} (SMOTE-NC)', alpha=0.7)

ax.set_xlabel('Fold Number', fontweight='bold', fontsize=12)
ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
ax.set_title('F1-Score Across K-Fold Cross-Validation Folds', fontweight='bold', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig('figure4_f1_across_folds.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure4_f1_across_folds.png")
plt.close()

# =====================================
# FIGURE 5: Model Improvement with SMOTE-NC
# =====================================

fig, ax = plt.subplots(figsize=(12, 7))

improvements = []
for model in models:
    without_f1 = float(comparison_df[(comparison_df['Model'] == model) & 
                                    (comparison_df['Condition'] == 'Without SMOTE-NC')]['F1-Score'].values[0])
    with_f1 = float(comparison_df[(comparison_df['Model'] == model) & 
                                 (comparison_df['Condition'] == 'With SMOTE-NC')]['F1-Score'].values[0])
    improvement = ((with_f1 - without_f1) / without_f1) * 100 if without_f1 > 0 else 0
    improvements.append(improvement)

colors = ['green' if imp > 0 else 'red' for imp in improvements]
bars = ax.barh(models, improvements, color=colors, edgecolor='black', linewidth=1.2, alpha=0.7)

ax.set_xlabel('F1-Score Improvement (%)', fontweight='bold', fontsize=12)
ax.set_title('Percentage Improvement with SMOTE-NC', fontweight='bold', fontsize=14)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    width = bar.get_width()
    label_x = width + (1 if width > 0 else -1)
    ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{imp:+.1f}%',
            ha='left' if width > 0 else 'right', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figure5_smote_improvement.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure5_smote_improvement.png")
plt.close()

# =====================================
# FIGURE 6: Summary Statistics Table
# =====================================

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Calculate summary statistics
summary_data = []
for model in models:
    for condition in ['Without SMOTE-NC', 'With SMOTE-NC']:
        cv_data = cv_results_df[(cv_results_df['Model'] == model) & 
                               (cv_results_df['Condition'] == condition)]
        
        test_data = comparison_df[(comparison_df['Model'] == model) & 
                                 (comparison_df['Condition'] == condition)]
        
        row = [
            model,
            condition.replace('SMOTE-NC', 'SMOTE'),
            f"{cv_data['F1'].mean():.4f} ± {cv_data['F1'].std():.4f}",
            f"{cv_data['Recall'].mean():.4f} ± {cv_data['Recall'].std():.4f}",
            test_data['F1-Score'].values[0],
            test_data['Recall'].values[0],
            test_data['ROC-AUC'].values[0]
        ]
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data, columns=[
    'Model', 'Condition', 'CV F1 (Mean±Std)', 'CV Recall (Mean±Std)', 
    'Test F1', 'Test Recall', 'Test AUC'
])

table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_df) + 1):
    if i % 2 == 0:
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor('#f0f0f0')

plt.title('Summary Statistics - All Models', fontweight='bold', fontsize=14, pad=20)
plt.savefig('figure6_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure6_summary_table.png")
plt.close()

print("\n" + "=" * 60)
print("✅ All visualizations generated successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  📊 figure1_model_comparison.png")
print("  📊 figure2_kfold_boxplots.png")
print("  📊 figure3_performance_heatmap.png")
print("  📊 figure4_f1_across_folds.png")
print("  📊 figure5_smote_improvement.png")
print("  📊 figure6_summary_table.png")
print("\nThese figures are ready to include in your final report!")
