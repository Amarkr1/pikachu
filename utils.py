import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
import json
import os

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Compute per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {
        'precision': precision_per_class.tolist(),
        'recall': recall_per_class.tolist(),
        'f1': f1_per_class.tolist(),
        'support': support.tolist()
    }
    
    # Compute AUC if probabilities provided
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:  # Multi-class
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            metrics['auc'] = auc
        except Exception as e:
            print(f"Could not compute AUC: {e}")
            metrics['auc'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    if 'auc' in metrics and metrics['auc'] is not None:
        print(f"AUC:       {metrics['auc']:.4f}")
    
    if 'per_class' in metrics:
        print(f"\nPer-Class Metrics:")
        for i, (p, r, f, s) in enumerate(zip(
            metrics['per_class']['precision'],
            metrics['per_class']['recall'],
            metrics['per_class']['f1'],
            metrics['per_class']['support']
        )):
            print(f"  Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_prob, class_names, save_path=None):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    
    if y_prob.shape[1] == 2:  # Binary
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    else:  # Multi-class
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            auc = roc_auc_score(y_true_binary, y_prob[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results: Dict, save_path: str):
    """Save results to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {save_path}")


def compare_results(baseline_results: Dict, icl_results: Dict, save_dir: str):
    """
    Compare baseline and ICL results
    
    Args:
        baseline_results: Dictionary of baseline results per model
        icl_results: Dictionary of ICL results per model
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics for comparison - only models present in both results
    models = [m for m in baseline_results.keys() if m in icl_results]
    
    if not models:
        print("Warning: No common models found between baseline and ICL results")
        return None
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create comparison dataframe
    comparison_data = []
    for model in models:
        for metric in metrics_to_compare:
            baseline_val = baseline_results[model].get(metric, 0)
            icl_val = icl_results[model].get(metric, 0)
            improvement = ((icl_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            
            comparison_data.append({
                'Model': model,
                'Metric': metric.capitalize(),
                'Baseline': baseline_val,
                'ICL': icl_val,
                'Improvement (%)': improvement
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    df.to_csv(os.path.join(save_dir, 'comparison_table.csv'), index=False)
    print(f"\nComparison table saved to: {os.path.join(save_dir, 'comparison_table.csv')}")
    
    # Plot comparison for each metric
    for metric in metrics_to_compare:
        metric_df = df[df['Metric'] == metric.capitalize()]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        
        baseline_vals = metric_df['Baseline'].values
        icl_vals = metric_df['ICL'].values
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width/2, icl_vals, width, label='ICL', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison: Baseline vs ICL')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot improvement percentages
    fig, ax = plt.subplots(figsize=(14, 6))
    
    improvement_data = []
    for model in models:
        model_improvements = []
        for metric in metrics_to_compare:
            metric_df = df[(df['Model'] == model) & (df['Metric'] == metric.capitalize())]
            if not metric_df.empty:
                model_improvements.append(metric_df['Improvement (%)'].values[0])
        improvement_data.append(model_improvements)
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_compare):
        values = [improvement_data[j][i] for j in range(len(models))]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Performance Improvement with ICL')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'improvement_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {save_dir}")
    
    return df