#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_scores(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} sample pairs")
    return df


def create_similarity_matrix(df, feature, max_samples=50):
    score_col = f'{feature}_score'
    if score_col not in df.columns:
        return None
    
    apk_list = df['apk1'].unique()[:max_samples]
    n = len(apk_list)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                apk_i, apk_j = apk_list[i], apk_list[j]
                mask = ((df['apk1'] == apk_i) & (df['apk2'] == apk_j)) | \
                       ((df['apk1'] == apk_j) & (df['apk2'] == apk_i))
                matches = df[mask]
                
                if len(matches) > 0:
                    matrix[i, j] = matches.iloc[0][score_col]
    
    return matrix


def plot_heatmap(matrix, feature, output_path, cmap='YlOrRd'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap=cmap, vmin=0, vmax=1, cbar_kws={'label': 'Similarity'})
    plt.title(f'{feature.upper()} Similarity Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_heatmap(matrices, output_path):
    n_features = len(matrices)
    fig, axes = plt.subplots(1, n_features, figsize=(16, 4))
    
    for idx, (feature, matrix) in enumerate(matrices.items()):
        ax = axes[idx] if n_features > 1 else axes
        sns.heatmap(matrix, cmap='YlOrRd', vmin=0, vmax=1, ax=ax)
        ax.set_title(f'{feature.upper()}', fontweight='bold')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Sample')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined: {output_path}")


def compute_statistics(matrix, feature):
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    values = matrix[mask]
    
    return {
        'feature': feature,
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75)
    }


def print_statistics_table(all_stats):
    print("\n" + "="*80)
    print("SIMILARITY STATISTICS")
    print("="*80)
    print(f"{'Feature':<15} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*80)
    for stats in all_stats:
        print(f"{stats['feature']:<15} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
              f"{stats['std']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores-csv', required=True)
    parser.add_argument('--output', default='heatmaps')
    parser.add_argument('--features', default='icon,so,smaliopcode,apicall')
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--combined', action='store_true')
    
    args = parser.parse_args()
    features = [f.strip() for f in args.features.split(',')]
    
    os.makedirs(args.output, exist_ok=True)
    df = load_scores(args.scores_csv)
    
    matrices = {}
    all_stats = []
    
    for feature in features:
        matrix = create_similarity_matrix(df, feature, args.max_samples)
        if matrix is None:
            continue
        
        matrices[feature] = matrix
        stats = compute_statistics(matrix, feature)
        all_stats.append(stats)
        
        output_path = os.path.join(args.output, f'{feature}_heatmap.png')
        plot_heatmap(matrix, feature, output_path)
    
    if args.combined and len(matrices) > 1:
        combined_path = os.path.join(args.output, 'combined_heatmap.png')
        plot_combined_heatmap(matrices, combined_path)
    
    if all_stats:
        print_statistics_table(all_stats)
        df_stats = pd.DataFrame(all_stats)
        stats_path = os.path.join(args.output, 'statistics.csv')
        df_stats.to_csv(stats_path, index=False)


if __name__ == '__main__':
    main()
