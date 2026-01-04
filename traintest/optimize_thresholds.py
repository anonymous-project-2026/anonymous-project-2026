import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import argparse
import sys
import os

# Add current directory to path to allow imports if needed (though we use standalone logic here)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # Ensure labels are integers
    df['label'] = df['label'].astype(int)
    return df

def generate_threshold_pairs(start=0.0, end=1.0, step=0.1):
    """
    Generate (low, high) pairs where low <= high.
    """
    thresholds = np.arange(start, end + 1e-9, step)
    pairs = []
    for low in thresholds:
        for high in thresholds:
            if low <= high:
                pairs.append((round(low, 2), round(high, 2)))
    return pairs

def generate_single_thresholds(start=0.5, end=1.0, step=0.05):
    return [round(x, 2) for x in np.arange(start, end + 1e-9, step)]

def apply_pipeline(df, icon_thr, so_thr, omm_thr, sfcg_thr):
    """
    Apply the 4-stage pipeline.
    Order: Icon -> SO -> Smali Opcode (OMM) -> SFCG
    
    icon_thr: (low, high)
    so_thr: (low, high)
    omm_thr: (low, high)
    sfcg_thr: single float
    
    Returns: predictions (numpy array)
    """
    n = len(df)
    preds = np.full(n, -1, dtype=int) # -1 means undetermined
    
    # Extract columns as numpy arrays for speed
    icon_sim = df['icon_sim'].values
    so_sim = df['so_sim'].values
    omm_sim = df['omm_sim'].values
    sfcg_sim = df['sfcg_sim'].values
    
    # --- Stage 1: Icon ---
    low, high = icon_thr
    mask_high = (icon_sim >= high) & (icon_sim != -1)
    preds[mask_high] = 1
    
    mask_low = (icon_sim < low) & (icon_sim != -1)
    preds[mask_low] = 0
    
    # --- Stage 2: SO ---
    undetermined = (preds == -1)
    if not np.any(undetermined):
        return preds
        
    low, high = so_thr
    curr_sim = so_sim
    
    mask_high = undetermined & (curr_sim >= high) & (curr_sim != -1)
    preds[mask_high] = 1
    
    mask_low = undetermined & (curr_sim < low) & (curr_sim != -1)
    preds[mask_low] = 0
    
    # --- Stage 3: Smali Opcode (OMM) ---
    undetermined = (preds == -1)
    if not np.any(undetermined):
        return preds
        
    low, high = omm_thr
    curr_sim = omm_sim
    
    mask_high = undetermined & (curr_sim >= high) & (curr_sim != -1)
    preds[mask_high] = 1
    
    mask_low = undetermined & (curr_sim < low) & (curr_sim != -1)
    preds[mask_low] = 0
    
    # --- Stage 4: SFCG ---
    undetermined = (preds == -1)
    if not np.any(undetermined):
        return preds
        
    curr_sim = sfcg_sim
    threshold = sfcg_thr
    
    mask_pos = undetermined & (curr_sim >= threshold)
    preds[mask_pos] = 1
    
    mask_neg = undetermined & (curr_sim < threshold)
    preds[mask_neg] = 0
    
    preds[preds == -1] = 0
    
    return preds

def analyze_pipeline_performance(df, icon_thr, so_thr, omm_thr, sfcg_thr):
    stats = []
    n_total = len(df)
    labels = df['label'].values
    
    # Initialize predictions as undetermined
    preds = np.full(n_total, -1, dtype=int)
    
    # Extract features
    icon_sim = df['icon_sim'].values
    so_sim = df['so_sim'].values
    omm_sim = df['omm_sim'].values
    sfcg_sim = df['sfcg_sim'].values
    
    stages = [
        ('Icon', icon_thr, icon_sim, True),
        ('SO', so_thr, so_sim, True),
        ('Smali Opcode', omm_thr, omm_sim, True),
        ('SFCG', sfcg_thr, sfcg_sim, False)
    ]
    
    current_undetermined_mask = np.ones(n_total, dtype=bool)
    
    print("\nDetailed Layer Analysis:")
    print(f"{'Stage':<10} | {'Input':<8} | {'Decided':<8} | {'Passed':<8} | {'Filt%':<8} | {'Cum_F1':<8} | {'Cum_Pre':<8} | {'Cum_Rec':<8}")
    print("-" * 90)
    
    header = "Stage,Input,Decided,Passed,Filtered_Pct,Cum_Precision,Cum_Recall,Cum_F1,Cum_Accuracy\n"
    csv_rows = []
    
    for name, thr, data, is_double in stages:
        n_input = np.sum(current_undetermined_mask)
        
        # Identify newly decided samples
        newly_decided_mask = np.zeros(n_total, dtype=bool)
        
        description = ""
        
        if is_double:
            low, high = thr
            # Check if it's effectively single threshold (Icon special case)
            if low <= 0.0:
                description = f"High Threshold Only (High={high})"
            else:
                description = f"Double Threshold (Low={low}, High={high})"
                
            # Valid data check (data != -1)
            valid_data_mask = (data != -1)
            
            # High -> 1
            mask_high = current_undetermined_mask & valid_data_mask & (data >= high)
            preds[mask_high] = 1
            
            # Low -> 0
            mask_low = current_undetermined_mask & valid_data_mask & (data < low)
            preds[mask_low] = 0
            
            newly_decided_mask = mask_high | mask_low
            
        else:
            # Single threshold (SFCG)
            threshold = thr
            description = f"Final Threshold (T={threshold})"
            # SFCG missing (-1) -> 0
            # If missing, it is < threshold
            
            mask_pos = current_undetermined_mask & (data >= threshold)
            preds[mask_pos] = 1
            
            mask_neg = current_undetermined_mask & (data < threshold)
            preds[mask_neg] = 0
            
            newly_decided_mask = mask_pos | mask_neg
        
        # Update undetermined mask
        current_undetermined_mask = current_undetermined_mask & (~newly_decided_mask)
        
        # Stats for this stage
        n_decided = np.sum(newly_decided_mask)
        n_passed = np.sum(current_undetermined_mask)
        percent_filtered = (n_decided / n_total) * 100
        
        # Cumulative Performance (all decided so far)
        decided_indices = np.where(preds != -1)[0]
        
        if len(decided_indices) > 0:
            curr_preds = preds[decided_indices]
            curr_labels = labels[decided_indices]
            p = precision_score(curr_labels, curr_preds, zero_division=0)
            r = recall_score(curr_labels, curr_preds, zero_division=0)
            f1 = f1_score(curr_labels, curr_preds, zero_division=0)
            acc = accuracy_score(curr_labels, curr_preds)
        else:
            p, r, f1, acc = 0, 0, 0, 0
            
        print(f"{name:<10} | {n_input:<8} | {n_decided:<8} | {n_passed:<8} | {percent_filtered:<8.1f} | {f1:<8.4f} | {p:<8.4f} | {r:<8.4f}")
        
        stats.append({
            'Stage': name,
            'Input': n_input,
            'Decided': n_decided,
            'Passed': n_passed,
            'Filtered_Pct': percent_filtered,
            'Cum_Precision': p,
            'Cum_Recall': r,
            'Cum_F1': f1,
            'Cum_Accuracy': acc,
            'Description': description
        })
        
        csv_rows.append(f"{name},{n_input},{n_decided},{n_passed},{percent_filtered:.2f},{p:.4f},{r:.4f},{f1:.4f},{acc:.4f}")
        
    return stats, header + "\n".join(csv_rows)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_sensitivity(df, best_config, output_dir):
    """
    Generate sensitivity plots for each layer around the best configuration.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    labels = df['label'].values
    icon_p, so_p, omm_p, sfcg_t = best_config
    
    # Define ranges for plotting
    plot_step = 0.05
    thresholds = np.arange(0.1, 1.0 + 1e-9, plot_step)
    
    print("\nGenerating sensitivity plots...")
    
    def plot_layer_metrics(layer_name, matrix_p, matrix_r, matrix_f1, thresholds, filename):
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        # Precision
        sns.heatmap(matrix_p, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="viridis", annot=True, fmt=".2f", ax=axes[0])
        axes[0].set_title(f"{layer_name} - Precision")
        axes[0].set_xlabel("High Threshold")
        axes[0].set_ylabel("Low Threshold")
        
        # Recall
        sns.heatmap(matrix_r, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="viridis", annot=True, fmt=".2f", ax=axes[1])
        axes[1].set_title(f"{layer_name} - Recall")
        axes[1].set_xlabel("High Threshold")
        axes[1].set_ylabel("Low Threshold")
        
        # F1
        sns.heatmap(matrix_f1, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="viridis", annot=True, fmt=".2f", ax=axes[2])
        axes[2].set_title(f"{layer_name} - F1 Score")
        axes[2].set_xlabel("High Threshold")
        axes[2].set_ylabel("Low Threshold")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    # --- 1. Icon Heatmap ---
    print("Plotting Icon sensitivity...")
    n = len(thresholds)
    p_matrix = np.zeros((n, n))
    r_matrix = np.zeros((n, n))
    f1_matrix = np.zeros((n, n))
    
    for i, low in enumerate(thresholds):
        for j, high in enumerate(thresholds):
            if low <= high:
                preds = apply_pipeline(df, (low, high), so_p, omm_p, sfcg_t)
                p_matrix[i, j] = precision_score(labels, preds, zero_division=0)
                r_matrix[i, j] = recall_score(labels, preds, zero_division=0)
                f1_matrix[i, j] = f1_score(labels, preds, zero_division=0)
            else:
                p_matrix[i, j] = np.nan
                r_matrix[i, j] = np.nan
                f1_matrix[i, j] = np.nan
                
    plot_layer_metrics("Icon Layer", p_matrix, r_matrix, f1_matrix, thresholds, "sensitivity_icon_metrics.png")

    # --- 2. SO Heatmap ---
    print("Plotting SO sensitivity...")
    p_matrix = np.zeros((n, n))
    r_matrix = np.zeros((n, n))
    f1_matrix = np.zeros((n, n))
    
    for i, low in enumerate(thresholds):
        for j, high in enumerate(thresholds):
            if low <= high:
                preds = apply_pipeline(df, icon_p, (low, high), omm_p, sfcg_t)
                p_matrix[i, j] = precision_score(labels, preds, zero_division=0)
                r_matrix[i, j] = recall_score(labels, preds, zero_division=0)
                f1_matrix[i, j] = f1_score(labels, preds, zero_division=0)
            else:
                p_matrix[i, j] = np.nan
                r_matrix[i, j] = np.nan
                f1_matrix[i, j] = np.nan
                
    plot_layer_metrics("SO Layer", p_matrix, r_matrix, f1_matrix, thresholds, "sensitivity_so_metrics.png")

    # --- 3. Smali Opcode Heatmap ---
    print("Plotting Smali Opcode sensitivity...")
    p_matrix = np.zeros((n, n))
    r_matrix = np.zeros((n, n))
    f1_matrix = np.zeros((n, n))
    
    for i, low in enumerate(thresholds):
        for j, high in enumerate(thresholds):
            if low <= high:
                preds = apply_pipeline(df, icon_p, so_p, (low, high), sfcg_t)
                p_matrix[i, j] = precision_score(labels, preds, zero_division=0)
                r_matrix[i, j] = recall_score(labels, preds, zero_division=0)
                f1_matrix[i, j] = f1_score(labels, preds, zero_division=0)
            else:
                p_matrix[i, j] = np.nan
                r_matrix[i, j] = np.nan
                f1_matrix[i, j] = np.nan
                
    plot_layer_metrics("Smali Opcode Layer", p_matrix, r_matrix, f1_matrix, thresholds, "sensitivity_smali_opcode_metrics.png")

    # --- 4. SFCG Line Plot ---
    print("Plotting SFCG sensitivity...")
    sfcg_thresholds = np.arange(0.5, 1.0 + 1e-9, 0.02) # Finer for line plot
    f1_scores = []
    p_scores = []
    r_scores = []
    
    for thr in sfcg_thresholds:
        preds = apply_pipeline(df, icon_p, so_p, omm_p, thr)
        f1_scores.append(f1_score(labels, preds, zero_division=0))
        p_scores.append(precision_score(labels, preds, zero_division=0))
        r_scores.append(recall_score(labels, preds, zero_division=0))
        
    plt.figure(figsize=(10, 6))
    plt.plot(sfcg_thresholds, f1_scores, marker='o', label='F1 Score')
    plt.plot(sfcg_thresholds, p_scores, marker='s', linestyle='--', label='Precision')
    plt.plot(sfcg_thresholds, r_scores, marker='^', linestyle=':', label='Recall')
    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("SFCG Layer Sensitivity (Precision, Recall, F1)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "sensitivity_sfcg_metrics.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def plot_local_safety(df, output_dir):
    """
    Generate 'Local Safety' plots for each layer independently.
    Focus on:
    1. High Threshold Precision (Are we safe to say 'Malicious'?)
    2. Low Threshold NPV (Are we safe to say 'Benign'?)
    3. Coverage (How much data do we filter?)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    labels = df['label'].values
    
    # Define thresholds
    step = 0.05
    thresholds = np.arange(0.1, 1.0 + 1e-9, step)
    n = len(thresholds)
    
    layers = [
        ('Icon', df['icon_sim'].values),
        ('SO', df['so_sim'].values),
        ('Smali Opcode', df['omm_sim'].values)
    ]
    
    print("\nGenerating Local Safety plots (Independent Layer Analysis)...")
    
    for layer_name, data in layers:
        print(f"Plotting {layer_name} safety...")
        
        # Valid data mask (ignore -1)
        valid_mask = (data != -1)
        valid_data = data[valid_mask]
        valid_labels = labels[valid_mask]
        
        prec_high_matrix = np.zeros((n, n))
        npv_low_matrix = np.zeros((n, n))
        coverage_matrix = np.zeros((n, n))
        
        for i, low in enumerate(thresholds):
            for j, high in enumerate(thresholds):
                if low <= high:
                    # High Threshold Analysis (Positive Prediction)
                    # Predict 1 if sim >= high
                    mask_high = (valid_data >= high)
                    if np.sum(mask_high) > 0:
                        # Precision = TP / (TP + FP)
                        # TP: label is 1 and pred is 1
                        tp = np.sum((valid_labels[mask_high] == 1))
                        fp = np.sum((valid_labels[mask_high] == 0))
                        prec_high = tp / (tp + fp) if (tp + fp) > 0 else 0
                    else:
                        prec_high = 1.0 # No risky decisions made
                        
                    # Low Threshold Analysis (Negative Prediction)
                    # Predict 0 if sim < low
                    mask_low = (valid_data < low)
                    if np.sum(mask_low) > 0:
                        # NPV = TN / (TN + FN)
                        # TN: label is 0 and pred is 0
                        tn = np.sum((valid_labels[mask_low] == 0))
                        fn = np.sum((valid_labels[mask_low] == 1))
                        npv_low = tn / (tn + fn) if (tn + fn) > 0 else 0
                    else:
                        npv_low = 1.0 # No risky decisions made
                        
                    # Coverage
                    n_decided = np.sum(mask_high) + np.sum(mask_low)
                    coverage = n_decided / len(labels) # Use total N for global impact
                    
                    prec_high_matrix[i, j] = prec_high
                    npv_low_matrix[i, j] = npv_low
                    coverage_matrix[i, j] = coverage
                    
                else:
                    prec_high_matrix[i, j] = np.nan
                    npv_low_matrix[i, j] = np.nan
                    coverage_matrix[i, j] = np.nan
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        # 1. High Threshold Precision
        sns.heatmap(prec_high_matrix, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="RdYlGn", annot=True, fmt=".2f", ax=axes[0], vmin=0.8, vmax=1.0)
        axes[0].set_title(f"{layer_name} - High Threshold Precision\n(Safety of predicting 'Malicious')")
        axes[0].set_xlabel("High Threshold")
        axes[0].set_ylabel("Low Threshold")
        
        # 2. Low Threshold NPV
        sns.heatmap(npv_low_matrix, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="RdYlGn", annot=True, fmt=".2f", ax=axes[1], vmin=0.8, vmax=1.0)
        axes[1].set_title(f"{layer_name} - Low Threshold NPV\n(Safety of predicting 'Benign')")
        axes[1].set_xlabel("High Threshold")
        axes[1].set_ylabel("Low Threshold")
        
        # 3. Coverage
        sns.heatmap(coverage_matrix, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="Blues", annot=True, fmt=".2f", ax=axes[2])
        axes[2].set_title(f"{layer_name} - Filter Coverage %\n(Fraction of total data decided)")
        axes[2].set_xlabel("High Threshold")
        axes[2].set_ylabel("Low Threshold")
        
        plt.tight_layout()
        filename = f"safety_{layer_name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def analyze_threshold_confidence(df, output_dir):
    """
    Generate confidence curves for picking High/Low thresholds independently.
    For each threshold T, calculate:
    1. Precision if we set High = T (samples >= T)
    2. NPV if we set Low = T (samples < T)
    """
    print("\nGenerating Threshold Confidence Analysis (Precision/NPV Curves)...")
    
    layers = [
        ('Icon', 'icon_sim'),
        ('SO', 'so_sim'),
        ('Smali Opcode', 'omm_sim')
    ]
    
    # Use a wider range to cover potential negative similarities
    thresholds = np.arange(-0.2, 1.01, 0.01)
    labels = df['label'].values
    
    stats_all = []
    
    for name, col in layers:
        sim_scores = df[col].values
        
        # 1. Calculate Missing Ratio
        n_total = len(sim_scores)
        n_missing = np.sum(sim_scores == -1)
        missing_ratio = n_missing / n_total * 100
        
        # 2. Filter Valid Data
        valid_mask = (sim_scores != -1)
        valid_scores = sim_scores[valid_mask]
        valid_labels = labels[valid_mask]
        
        if len(valid_scores) == 0:
            print(f"Skipping {name} (No valid data)")
            continue
            
        min_val = valid_scores.min()
        max_val = valid_scores.max()
        
        print(f"Layer {name}: Valid Range [{min_val:.4f}, {max_val:.4f}], Missing: {missing_ratio:.1f}%")
        
        precisions = []
        npvs = []
        counts_high = []
        counts_low = []
        
        for t in thresholds:
            # High Threshold Logic (>= t)
            mask_high = (valid_scores >= t)
            n_high = np.sum(mask_high)
            
            if n_high > 0:
                # Precision = TP / (TP + FP)
                tp = np.sum(valid_labels[mask_high] == 1)
                p = tp / n_high
            else:
                p = 1.0 # Default to safe (no mispredictions)
            
            precisions.append(p)
            counts_high.append(n_high)
            
            # Low Threshold Logic (< t)
            mask_low = (valid_scores < t)
            n_low = np.sum(mask_low)
            
            if n_low > 0:
                # NPV = TN / (TN + FN)
                tn = np.sum(valid_labels[mask_low] == 0)
                npv = tn / n_low
            else:
                npv = 1.0 # Default to safe
            
            npvs.append(npv)
            counts_low.append(n_low)
            
            stats_all.append({
                'Layer': name,
                'Threshold': t,
                'Precision_Above': p,
                'NPV_Below': npv,
                'Count_Above': n_high,
                'Count_Below': n_low
            })
            
        # Plot
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Plot valid range background
        ax1.axvspan(min_val, max_val, color='gray', alpha=0.1, label='Valid Data Range')
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Confidence Score (Precision / NPV)', color='black')
        
        # Plot Precision (High Conf) - Green
        line1 = ax1.plot(thresholds, precisions, color='green', label='Precision (if High=T)', linewidth=2.5)
        
        # Plot NPV (Low Conf) - Red
        line2 = ax1.plot(thresholds, npvs, color='red', label='NPV (if Low=T)', linewidth=2.5)
        
        # Add Reference Lines
        ax1.axhline(0.95, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(0.98, color='gray', linestyle='--', alpha=0.5)
        ax1.text(thresholds[0], 0.955, '95% Conf.', color='gray', fontsize=8)
        
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0.0, 1.05) # Full range to see bad performance too
        ax1.grid(True, which='both', linestyle=':', alpha=0.6)
        
        # Secondary Axis for Data Distribution (Histogram-like)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sample Count (Cumulative)', color='blue')
        
        # Use fill_between for better visibility of counts
        # High count decreases as T increases
        line3 = ax2.plot(thresholds, counts_high, color='green', linestyle=':', alpha=0.3, label='Count >= T')
        ax2.fill_between(thresholds, counts_high, 0, color='green', alpha=0.05)
        
        # Low count increases as T increases
        line4 = ax2.plot(thresholds, counts_low, color='red', linestyle=':', alpha=0.3, label='Count < T')
        ax2.fill_between(thresholds, counts_low, 0, color='red', alpha=0.05)
        
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, len(valid_scores) * 1.1)
        
        # Title with stats
        plt.title(f"{name} Layer Analysis\nValid Data: {len(valid_scores)} ({100-missing_ratio:.1f}%) | Missing: {n_missing} ({missing_ratio:.1f}%)")
        
        # Combined Legend
        lines = line1 + line2 + [line3[0], line4[0]]
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc='center right', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confidence_curve_{name.lower().replace(' ', '_')}.png"))
        plt.close()
        
    # Save CSV
    pd.DataFrame(stats_all).to_csv(os.path.join(output_dir, 'layer_threshold_confidence.csv'), index=False)
    print(f"Confidence stats saved to {os.path.join(output_dir, 'layer_threshold_confidence.csv')}")

def main():
    parser = argparse.ArgumentParser(description="Optimize Multi-Stage Detection Thresholds")
    parser.add_argument('--csv', type=str, required=True, help='Path to results csv')
    parser.add_argument('--output', type=str, default='threshold_optimization_report.txt', help='Path to save the report')
    parser.add_argument('--plot_output', type=str, default=None, help='Directory to save plots (optional)')
    args = parser.parse_args()
    
    # If no specific plot output is given but user wants plots, default to same dir as report
    if args.plot_output is None:
        args.plot_output = os.path.dirname(os.path.abspath(args.output))

    
    print(f"Loading data from {args.csv}...")
    df = load_data(args.csv)
    labels = df['label'].values
    print(f"Loaded {len(df)} pairs. Positives: {sum(labels)}, Negatives: {len(labels)-sum(labels)}")
    
    # Generate search space
    # To keep it feasible:
    # Icon: Coarse steps 0.1
    # OMM/SO: Coarse steps 0.2? or 0.1
    # SFCG: Coarse steps 0.05
    
    
    # Coordinate Descent Optimization
    print("Starting Coordinate Descent Optimization...")
    
    # Initial guess
    current_config = {
        'icon': (0.2, 0.8),
        'omm': (0.2, 0.8),
        'so': (0.2, 0.8),
        'sfcg': 0.8
    }
    
    best_f1_global = -1.0
    
    # Define search spaces (finer grid)
    # Start from 0.1 to avoid 0.0 (no filtering) lower bounds, as per user concern about false positives
    
    # MODIFIED for User Request: Icon Single Threshold (High Only). 
    # Low is fixed to -0.1 so valid data (0.0-1.0) is NEVER rejected by Low Threshold.
    print("Configuring Icon layer as Single Threshold (High Only)...")
    icon_space = [(-0.1, round(h, 2)) for h in np.arange(0.1, 1.0 + 1e-9, 0.05)]
    
    omm_space = generate_threshold_pairs(0.4, 1.0, 0.05)
    so_space = generate_threshold_pairs(0.4, 1.0, 0.05)
    sfcg_space = generate_single_thresholds(0.5, 0.95, 0.05)
    
    max_cycles = 10
    no_change_count = 0
    
    for cycle in range(max_cycles):
        print(f"Cycle {cycle+1}...")
        changed = False
        
        # 1. Optimize Icon
        best_local_f1 = -1
        best_local_val = current_config['icon']
        for val in icon_space:
            preds = apply_pipeline(df, val, current_config['so'], current_config['omm'], current_config['sfcg'])
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_local_f1:
                best_local_f1 = f1
                best_local_val = val
        
        if best_local_val != current_config['icon']:
            print(f"  Icon updated: {current_config['icon']} -> {best_local_val} (F1: {best_local_f1:.4f})")
            current_config['icon'] = best_local_val
            changed = True

        # 2. Optimize SO
        best_local_f1 = -1
        best_local_val = current_config['so']
        for val in so_space:
            preds = apply_pipeline(df, current_config['icon'], val, current_config['omm'], current_config['sfcg'])
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_local_f1:
                best_local_f1 = f1
                best_local_val = val
        
        if best_local_val != current_config['so']:
            print(f"  SO updated: {current_config['so']} -> {best_local_val} (F1: {best_local_f1:.4f})")
            current_config['so'] = best_local_val
            changed = True
            
        # 3. Optimize Smali Opcode (OMM)
        best_local_f1 = -1
        best_local_val = current_config['omm']
        for val in omm_space:
            preds = apply_pipeline(df, current_config['icon'], current_config['so'], val, current_config['sfcg'])
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_local_f1:
                best_local_f1 = f1
                best_local_val = val
        
        if best_local_val != current_config['omm']:
            print(f"  Smali Opcode (OMM) updated: {current_config['omm']} -> {best_local_val} (F1: {best_local_f1:.4f})")
            current_config['omm'] = best_local_val
            changed = True

        # 4. Optimize SFCG
        best_local_f1 = -1
        best_local_val = current_config['sfcg']
        for val in sfcg_space:
            preds = apply_pipeline(df, current_config['icon'], current_config['so'], current_config['omm'], val)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_local_f1:
                best_local_f1 = f1
                best_local_val = val
        
        if best_local_val != current_config['sfcg']:
            print(f"  SFCG updated: {current_config['sfcg']} -> {best_local_val} (F1: {best_local_f1:.4f})")
            current_config['sfcg'] = best_local_val
            changed = True
            
        best_f1_global = max(best_f1_global, best_local_f1)
        
        if not changed:
            print("Converged.")
            break
            
    # Final eval
    preds = apply_pipeline(df, current_config['icon'], current_config['so'], current_config['omm'], current_config['sfcg'])
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    best_config = (current_config['icon'], current_config['so'], current_config['omm'], current_config['sfcg'])
    best_metrics = (p, r, f1)

    print("\n" + "="*50)
    print("Optimization Complete")
    print("="*50)
    
    icon_p, so_p, omm_p, sfcg_t = best_config
    p, r, f1 = best_metrics
    print(f"Best F1 Score: {f1:.4f}")
    print(f"Precision:     {p:.4f}")
    print(f"Recall:        {r:.4f}")
    print("-" * 30)
    print("Best Threshold Configuration:")
    print(f"Stage 1 (Icon): Low={icon_p[0]}, High={icon_p[1]}")
    print(f"Stage 2 (SO):   Low={so_p[0]}, High={so_p[1]}")
    print(f"Stage 3 (Smali Opcode): Low={omm_p[0]}, High={omm_p[1]}")
    print(f"Stage 4 (SFCG): Threshold={sfcg_t}")
    print("-" * 30)
    
    # Analyze and Save Report
    stats, csv_content = analyze_pipeline_performance(df, icon_p, so_p, omm_p, sfcg_t)
    
    # Calculate detailed metrics
    cm = confusion_matrix(labels, preds)
    cr = classification_report(labels, preds, digits=4)
    
    tn, fp, fn, tp = cm.ravel()
    
    print("\nVerification - Confusion Matrix:")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")
    print("\nClassification Report:")
    print(cr)
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write("Optimization Report\n")
        f.write("="*30 + "\n")
        f.write(f"Best F1 Score: {f1:.4f}\n")
        f.write(f"Precision:     {p:.4f}\n")
        f.write(f"Recall:        {r:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("Best Threshold Configuration:\n")
        f.write(f"Stage 1 (Icon): Low={icon_p[0]}, High={icon_p[1]}\n")
        f.write(f"Stage 2 (SO):   Low={so_p[0]}, High={so_p[1]}\n")
        f.write(f"Stage 3 (Smali Opcode): Low={omm_p[0]}, High={omm_p[1]}\n")
        f.write(f"Stage 4 (SFCG): Threshold={sfcg_t}\n")
        f.write("-" * 30 + "\n\n")
        
        f.write("Verification - Confusion Matrix:\n")
        f.write(f"TP: {tp} | FP: {fp}\n")
        f.write(f"FN: {fn} | TN: {tn}\n\n")
        f.write("Classification Report:\n")
        f.write(cr + "\n")
        
        f.write("Detailed Layer Analysis:\n")
        f.write(csv_content)
        
    print(f"\nReport saved to {args.output}")

    # Save Excel Report as requested
    excel_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), 'pipeline_step_metrics.xlsx')
    
    # Create DataFrame from stats
    df_stats = pd.DataFrame(stats)
    
    # Rename columns to match user's requested format (bilingual friendly)
    df_stats = df_stats[['Stage', 'Input', 'Decided', 'Passed', 'Filtered_Pct', 'Cum_F1', 'Cum_Precision', 'Cum_Recall', 'Description']]
    df_stats.columns = [
        '层级 (Stage)', 
        '输入数量 (Input)', 
        '处理掉的数量 (Decided)', 
        '剩余数量 (Passed)', 
        '过滤比例% (Filtered)', 
        '当前累计F1 (Cum F1)', 
        '当前累计Precision (Cum Pre)', 
        '当前累计Recall (Cum Rec)',
        '说明 (Description)'
    ]
    
    df_stats.to_excel(excel_path, index=False)
    print(f"Excel report saved to {excel_path}")
    
    # Generate Plots
    plot_sensitivity(df, best_config, args.plot_output)
    
    # Generate Local Safety Plots
    plot_local_safety(df, args.plot_output)
    
    # Generate Confidence Curves (New)
    analyze_threshold_confidence(df, args.plot_output)

if __name__ == '__main__':
    main()
