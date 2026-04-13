"""
Main Experiment Runner
Runs all baseline and ICL experiments and generates comparison
"""

import os
import json
import argparse
from config import ExperimentConfig
from utils import compare_results

def run_baseline_experiments(config):
    """Run all baseline experiments"""
    print("\n" + "="*80)
    print(" BASELINE EXPERIMENTS ".center(80, "="))
    print("="*80 + "\n")
    
    baseline_results = {}
    
    # Run CLIP baseline
    print("\n[1/3] Running CLIP baseline...")
    try:
        from baseline_clip import main as run_clip_baseline
        metrics = run_clip_baseline()
        baseline_results['CLIP'] = metrics
        print("✓ CLIP baseline completed")
    except Exception as e:
        print(f"✗ CLIP baseline failed: {e}")
    
    # Run DINOv2 baseline
    print("\n[2/3] Running DINOv2 baseline...")
    try:
        from baseline_dinov2 import main as run_dinov2_baseline
        metrics = run_dinov2_baseline()
        baseline_results['DINOv2'] = metrics
        print("✓ DINOv2 baseline completed")
    except Exception as e:
        print(f"✗ DINOv2 baseline failed: {e}")
    
    # Run ViT baseline
    print("\n[3/3] Running ViT baseline...")
    try:
        from baseline_vit import main as run_vit_baseline
        metrics = run_vit_baseline()
        baseline_results['ViT'] = metrics
        print("✓ ViT baseline completed")
    except Exception as e:
        print(f"✗ ViT baseline failed: {e}")
    
    # Save combined baseline results
    results_path = os.path.join(config.results_dir, "baseline", "all_baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump(baseline_results, f, indent=4)
    
    print(f"\n✓ All baseline results saved to: {results_path}")
    
    return baseline_results


def run_icl_experiments(config):
    """Run all ICL experiments"""
    print("\n" + "="*80)
    print(" ICL-ENHANCED EXPERIMENTS ".center(80, "="))
    print("="*80 + "\n")
    
    icl_results = {}
    
    # Run ICL-CLIP
    print("\n[1/3] Running ICL-enhanced CLIP...")
    try:
        from icl_clip import main as run_clip_icl
        metrics = run_clip_icl()
        icl_results['CLIP'] = metrics
        print("✓ ICL-CLIP completed")
    except Exception as e:
        print(f"✗ ICL-CLIP failed: {e}")
    
    # Run ICL-DINOv2
    print("\n[2/3] Running ICL-enhanced DINOv2...")
    try:
        from icl_dinov2 import main as run_dinov2_icl
        metrics = run_dinov2_icl()
        icl_results['DINOv2'] = metrics
        print("✓ ICL-DINOv2 completed")
    except Exception as e:
        print(f"✗ ICL-DINOv2 failed: {e}")
    
    # Run ICL-ViT
    print("\n[3/3] Running ICL-enhanced ViT...")
    try:
        from icl_vit import main as run_vit_icl
        metrics = run_vit_icl()
        icl_results['ViT'] = metrics
        print("✓ ICL-ViT completed")
    except Exception as e:
        print(f"✗ ICL-ViT failed: {e}")
    
    # Save combined ICL results
    results_path = os.path.join(config.results_dir, "icl", "all_icl_results.json")
    with open(results_path, 'w') as f:
        json.dump(icl_results, f, indent=4)
    
    print(f"\n✓ All ICL results saved to: {results_path}")
    
    return icl_results


def generate_comparison(baseline_results, icl_results, config):
    """Generate comparison plots and tables"""
    print("\n" + "="*80)
    print(" GENERATING COMPARISON ".center(80, "="))
    print("="*80 + "\n")
    
    comparison_dir = os.path.join(config.results_dir, "comparison")
    
    comparison_df = compare_results(baseline_results, icl_results, comparison_dir)
    
    print("\n" + "="*80)
    print(" RESULTS SUMMARY ".center(80, "="))
    print("="*80)
    
    print("\n📊 Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*80)
    print(f"\n✓ Comparison plots saved to: {comparison_dir}")
    print(f"  - comparison_table.csv")
    print(f"  - accuracy_comparison.png")
    print(f"  - precision_comparison.png")
    print(f"  - recall_comparison.png")
    print(f"  - f1_comparison.png")
    print(f"  - improvement_comparison.png")
    
    return comparison_df


def print_experiment_summary(baseline_results, icl_results):
    """Print experiment summary"""
    print("\n" + "="*80)
    print(" EXPERIMENT SUMMARY ".center(80, "="))
    print("="*80)
    
    if baseline_results:
        print("\n📈 Baseline Performance:")
        for model, metrics in baseline_results.items():
            print(f"\n  {model}:")
            print(f"    Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"    Precision: {metrics.get('precision', 0):.4f}")
            print(f"    Recall:    {metrics.get('recall', 0):.4f}")
            print(f"    F1 Score:  {metrics.get('f1', 0):.4f}")
    else:
        print("\n⚠️  No baseline results available")
    
    if icl_results:
        print("\n🚀 ICL-Enhanced Performance:")
        for model, metrics in icl_results.items():
            print(f"\n  {model}:")
            print(f"    Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"    Precision: {metrics.get('precision', 0):.4f}")
            print(f"    Recall:    {metrics.get('recall', 0):.4f}")
            print(f"    F1 Score:  {metrics.get('f1', 0):.4f}")
            if 'mean_icl_weight' in metrics:
                print(f"    ICL Weight: {metrics['mean_icl_weight']:.4f}")
    else:
        print("\n⚠️  No ICL results available")
    
    if baseline_results and icl_results:
        print("\n📊 Improvements:")
        common_models = [m for m in baseline_results.keys() if m in icl_results]
        
        if not common_models:
            print("\n⚠️  No common models to compare")
        else:
            for model in common_models:
                baseline_acc = baseline_results[model].get('accuracy', 0)
                icl_acc = icl_results[model].get('accuracy', 0)
                if baseline_acc > 0:
                    improvement = ((icl_acc - baseline_acc) / baseline_acc * 100)
                    print(f"\n  {model}:")
                    print(f"    Accuracy Improvement: {improvement:+.2f}%")
                    print(f"    Baseline:    {baseline_acc:.4f}")
                    print(f"    ICL-Enhanced: {icl_acc:.4f}")
    
    print("\n" + "="*80)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run ICL Medical Imaging Experiments")
    parser.add_argument('--mode', type=str, choices=['all', 'baseline', 'icl', 'compare'],
                       default='all', help='Experiment mode')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--k_shot', type=int, default=5,
                       help='Number of support examples per class')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = ExperimentConfig()
    config.data.data_root = args.data_root
    config.icl.num_support_shots = args.k_shot
    config.device = args.device
    
    print("\n" + "="*80)
    print(" IN-CONTEXT LEARNING FOR MEDICAL IMAGE CLASSIFICATION ".center(80))
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data Root: {config.data.data_root}")
    print(f"  K-Shot: {config.icl.num_support_shots}")
    print(f"  Device: {config.device}")
    print(f"  Models: CLIP, DINOv2, ViT")
    print("="*80 + "\n")
    
    baseline_results = {}
    icl_results = {}
    
    # Run experiments based on mode
    if args.mode in ['all', 'baseline']:
        baseline_results = run_baseline_experiments(config)
    
    if args.mode in ['all', 'icl']:
        icl_results = run_icl_experiments(config)
    
    # Generate comparison if we have both results
    if args.mode == 'compare':
        # Load existing results
        baseline_path = os.path.join(config.results_dir, "baseline", "all_baseline_results.json")
        icl_path = os.path.join(config.results_dir, "icl", "all_icl_results.json")
        
        if os.path.exists(baseline_path) and os.path.exists(icl_path):
            with open(baseline_path, 'r') as f:
                baseline_results = json.load(f)
            with open(icl_path, 'r') as f:
                icl_results = json.load(f)
        else:
            print("Error: Results files not found. Run baseline and ICL experiments first.")
            return
    
    if baseline_results and icl_results:
        generate_comparison(baseline_results, icl_results, config)
        print_experiment_summary(baseline_results, icl_results)
    elif baseline_results:
        print("\n⚠️  Only baseline results available - ICL experiments needed for comparison")
        print_experiment_summary(baseline_results, {})
    elif icl_results:
        print("\n⚠️  Only ICL results available - baseline experiments needed for comparison")
        print_experiment_summary({}, icl_results)
    else:
        print("\n⚠️  No results available")
    
    print("\n✓ Experiments completed!")
    print(f"\nResults directory: {config.results_dir}")
    print(f"Checkpoints directory: {config.checkpoints_dir}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()