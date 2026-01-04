import os
import time
import json
import argparse
import random
import pickle
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from so_trainer import SoTrainer
from feature_cnn_models import SOOpcodeDetailCaptureCNN
from so_tester import extract_and_detect_matrix, sweep_thresholds

# ------------------------------
# 1. 模型定义 (已移至 cnn.py)
# ------------------------------


# ------------------------------
# 2. 路径解析与 Dataset (IO 优化)
# ------------------------------

def list_group_ids(root_dir: str) -> List[str]:
    if not os.path.exists(root_dir):
        return []
    ids = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and name.isdigit():
            ids.append(name)
    return sorted(ids, key=lambda x: int(x))

def list_apk_projects_in_group(root_dir: str, group_id: str) -> Tuple[List[str], List[str]]:
    group_path = os.path.join(root_dir, str(group_id))
    orig_path = os.path.join(group_path, 'original_apk')
    repack_path = os.path.join(group_path, 'repack_apk')

    def list_projects(base):
        if not os.path.isdir(base):
            return []
        return [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

    originals = list_projects(orig_path)
    repacks = list_projects(repack_path)
    return originals, repacks

def locate_transition_npy(project_path: str) -> str:
    """查找文件，优先找 transition_probabilities.npy"""
    npy1 = os.path.join(project_path, 'transition_probabilities.npy')
    if os.path.exists(npy1): return npy1
    npy2 = os.path.join(project_path, 'so_transition_probabilities.npy')
    if os.path.exists(npy2): return npy2
    return ''

def pre_resolve_paths(projects: List[str]) -> Dict[str, str]:
    """批量建立 project -> npy_path 的映射，避免运行时IO"""
    mapping = {}
    for p in projects:
        npy = locate_transition_npy(p)
        if npy:
            mapping[p] = npy
    return mapping

# --- Datasets ---

class SoPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, int]], path_mapping: Dict[str, str]):
        self.pairs = pairs
        self.path_mapping = path_mapping

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        
        def load(path):
            npy = self.path_mapping.get(path)
            if not npy:
                return torch.zeros((1, 94, 94), dtype=torch.float32)
            try:
                # allow_pickle=True 因为有些npy可能是object数组
                mat = np.load(npy, allow_pickle=True)
                t = torch.tensor(mat, dtype=torch.float32)
                # 确保维度 [1, 94, 94]
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                return t
            except Exception:
                # 【关键】保持尺寸一致，防止DataLoader崩溃
                return torch.zeros((1, 94, 94), dtype=torch.float32)

        t1 = load(p1)
        t2 = load(p2)
        y = torch.tensor(1 if label == 1 else -1, dtype=torch.float32)
        return t1, t2, y, p1, p2, label


# ------------------------------
# 3. 核心功能函数
# ------------------------------

def build_positive_pairs(originals: List[str], repacks: List[str]) -> List[Tuple[str, str, int]]:
    pairs = []
    for o in originals:
        for r in repacks:
            pairs.append((o, r, 1))
    return pairs

def sample_negative_pairs(all_groups_projects, exclude_gid, target_num):
    """优化后的负样本采样"""
    other_gids = list(all_groups_projects.keys())
    if str(exclude_gid) in other_gids:
        other_gids.remove(str(exclude_gid))
    
    if not other_gids: return []

    neg_pairs = []
    attempts = 0
    max_attempts = target_num * 5
    
    while len(neg_pairs) < target_num and attempts < max_attempts:
        attempts += 1
        g1, g2 = random.sample(other_gids, 2)
        projs1 = all_groups_projects[g1]
        projs2 = all_groups_projects[g2]
        if not projs1 or not projs2: continue
        
        p1 = random.choice(projs1)
        p2 = random.choice(projs2)
        neg_pairs.append((p1, p2, 0))
    return neg_pairs

def extract_and_detect_matrix(model, device, test_pairs, path_mapping, batch_size, workers):
    """
    矩阵化测试流程：批量提取 -> GPU点积
    """
    # 1. 准备 Unique APKs
    unique_projs = set()
    for p1, p2, _ in test_pairs:
        unique_projs.add(p1)
        unique_projs.add(p2)
    unique_projs = list(unique_projs)
    proj_to_idx = {p: i for i, p in enumerate(unique_projs)}
    
    # 2. 批量提取特征
    dataset = SingleSoDataset(unique_projs, path_mapping)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=workers, collate_fn=collate_fn_single, pin_memory=True)
    
    all_feats = []
    all_indices = []
    model.eval()
    t_feat_start = time.time()
    
    with torch.no_grad():
        for paths, tensors in tqdm(loader, desc="[测试] 批量特征提取", unit="batch"):
            if tensors is None: continue
            tensors = tensors.to(device)
            # model(x) 只传一个参数时调用 forward_once -> 已归一化
            feats = model(tensors) 
            all_feats.append(feats.cpu()) # 先存CPU防OOM
            for p in paths:
                all_indices.append(proj_to_idx[p])
                
    feat_time = time.time() - t_feat_start
    
    if not all_feats:
        return pd.DataFrame(), feat_time, 0.0

    # 3. 构建大矩阵
    dim = all_feats[0].size(1)
    full_matrix = torch.zeros((len(unique_projs), dim), device=device) # 移入GPU
    
    current_ptr = 0
    for batch_feats in all_feats:
        b_size = batch_feats.size(0)
        # 获取 batch 对应的全局 indices
        batch_indices = all_indices[current_ptr : current_ptr + b_size]
        # 移入 GPU 填表
        full_matrix[batch_indices] = batch_feats.to(device)
        current_ptr += b_size
        
    # 4. 矩阵点积计算相似度
    t_detect_start = time.time()
    idx_a_list, idx_b_list, meta_list = [], [], []
    
    for p1, p2, lab in test_pairs:
        if p1 in proj_to_idx and p2 in proj_to_idx:
            idx_a_list.append(proj_to_idx[p1])
            idx_b_list.append(proj_to_idx[p2])
            meta_list.append((p1, p2, lab))
            
    if not idx_a_list:
        return pd.DataFrame(), feat_time, 0.0
    
    idx_a = torch.tensor(idx_a_list, device=device)
    idx_b = torch.tensor(idx_b_list, device=device)
    
    # Cosine Similarity = Dot Product (Pre-normalized)
    vec_a = full_matrix[idx_a]
    vec_b = full_matrix[idx_b]
    sims = (vec_a * vec_b).sum(dim=1).cpu().numpy()
    
    detect_time = time.time() - t_detect_start
    
    # 5. 组装结果
    results = []
    for i, (p1, p2, lab) in enumerate(meta_list):
        results.append({
            'apk1': p1,
            'apk2': p2,
            'label': lab,
            'similarity_score': float(sims[i])
        })
        
    return pd.DataFrame(results), feat_time, detect_time

def sweep_thresholds(results_df, start=0.5, end=0.99, step=0.01):
    rows = []
    if results_df.empty: return rows
    
    labels = results_df['label'].values
    scores = results_df['similarity_score'].values
    thresholds = np.arange(start, end + 1e-9, step)
    
    for th in thresholds:
        preds = (scores >= th).astype(int)
        tp = ((labels == 1) & (preds == 1)).sum()
        tn = ((labels == 0) & (preds == 0)).sum()
        fp = ((labels == 0) & (preds == 1)).sum()
        fn = ((labels == 1) & (preds == 0)).sum()
        
        acc = (tp + tn) / len(labels)
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
        
        rows.append({
            'threshold': th,
            'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        })
    return pd.DataFrame(rows)


# ------------------------------
# 4. 主流程
# ------------------------------

def run_so_detection(args):
    """
    SO特征检测入口函数
    :param args: 包含所有参数的命名空间对象 (argparse.Namespace 或 类似对象)
    """
    show_progress = not args.no_progress
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.result_root, exist_ok=True)
    interm_dir = os.path.join(args.intermediate_root, 'so')
    os.makedirs(interm_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SO] Device: {device}")

    # 1. 扫描路径
    print("[SO] 扫描 SO 矩阵路径...")
    t_scan = time.time()
    all_groups = list_group_ids(args.root)
    
    # 抽样
    if hasattr(args, 'sample_fraction') and args.sample_fraction < 1.0:
        import math
        sample_n = max(1, math.ceil(len(all_groups) * args.sample_fraction))
        all_groups = random.sample(all_groups, sample_n)
        print(f"[SO] 抽样: {len(all_groups)} 组 (fraction={args.sample_fraction})")
    
    all_group_projs = {}
    all_projs_flat = []
    
    for gid in tqdm(all_groups, desc="扫描组") if show_progress else all_groups:
        o, r = list_apk_projects_in_group(args.root, gid)
        all_group_projs[gid] = o + r
        all_projs_flat.extend(o + r)
        
    path_mapping = pre_resolve_paths(all_projs_flat)
    print(f"[SO] 扫描完成，找到 {len(path_mapping)} 个有效 SO 矩阵。耗时: {time.time()-t_scan:.2f}s")
    
    if not path_mapping:
        print("[SO] 无数据，退出。")
        return None

    # 2. 构建配对
    print("[SO] 构建样本对...")
    valid_projs_set = set(path_mapping.keys())
    all_pairs = []
    
    for gid in tqdm(all_groups, desc="配对") if show_progress else all_groups:
        o, r = list_apk_projects_in_group(args.root, gid)
        # 仅保留有效的
        o = [p for p in o if p in valid_projs_set]
        r = [p for p in r if p in valid_projs_set]
        if not o or not r: continue
        
        pos = build_positive_pairs(o, r)
        all_pairs.extend(pos)
        
        # 负样本采样
        target_neg = len(pos)
        neg_candidates = sample_negative_pairs(all_group_projs, gid, target_neg * 2)
        valid_negs = []
        for p1, p2, lab in neg_candidates:
            if p1 in valid_projs_set and p2 in valid_projs_set:
                valid_negs.append((p1, p2, lab))
                if len(valid_negs) >= target_neg: break
        all_pairs.extend(valid_negs)
        
    random.shuffle(all_pairs)
    print(f"[SO] 总样本对数: {len(all_pairs)}")

    # 划分
    total = len(all_pairs)
    n_train = int(total * 0.7)
    n_val = int(total * 0.1)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train+n_val]
    test_pairs = all_pairs[n_train+n_val:]
    
    # DataLoader (Val batch_size = args.batch_size)
    train_loader = DataLoader(SoPairDataset(train_pairs, path_mapping), 
                              batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(SoPairDataset(val_pairs, path_mapping), 
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 3. 训练与模型加载
    model_path = args.model_path if args.model_path else os.path.join(args.result_root, 'so_model.pth')
    trainer = SoTrainer(device, learning_rate=args.lr, margin=args.margin, neg_weight=args.neg_weight)
    train_time_total = 0.0

    if args.mode in ['train', 'train_test']:
        print(f"[SO] Mode: {args.mode} - 开始训练...")
        train_time_total = trainer.run_training(train_loader, val_loader, epochs=args.epochs, threshold=args.threshold, show_progress=show_progress)
        trainer.save_model(model_path)
    elif args.mode == 'test':
        print(f"[SO] Mode: {args.mode} - 加载模型: {model_path}")
        if not os.path.exists(model_path):
            print(f"[SO] Error: 模型文件不存在: {model_path}")
            return None
        trainer.load_model(model_path)

    model = trainer.model

    if args.mode == 'train':
        print("[SO] 训练完成，仅训练模式结束。")
        return {'train_time_s': train_time_total}

    # 4. 测试 (Matrix Accelerated)
    print("\n[SO] 开始矩阵化测试...")
    df_res, feat_time, detect_time = extract_and_detect_matrix(
        model, device, test_pairs, path_mapping, args.batch_size, args.workers
    )
    
    if df_res.empty:
        print("[SO] 测试结果为空。")
        return None

    # 5. 阈值选择与指标计算
    # 先在 Val Set 上跑一遍矩阵测试（或者利用训练时的结果，这里为了简单直接在Test结果上做推荐逻辑的模拟，
    # 实际上应该用 Val 结果定阈值，再应用到 Test。为了逻辑严谨，我们先扫 Val）
    
    # 这里我们简化逻辑：如果开启 auto-threshold，我们在 Val 集上做一次矩阵评估来定阈值
    final_threshold = args.threshold
    strategy_name = "manual"
    
    if args.auto_threshold:
        print("[SO] 正在验证集上寻找最佳阈值...")
        df_val, _, _ = extract_and_detect_matrix(model, device, val_pairs, path_mapping, args.batch_size, args.workers)
        sweep_df = sweep_thresholds(df_val)
        
        # 策略：满足 precision >= target 且 recall >= min_recall
        candidates = sweep_df[
            (sweep_df['recall'] >= args.min_recall) & 
            (sweep_df['precision'] >= args.target_precision)
        ]
        
        if not candidates.empty:
            best_row = candidates.sort_values(by='f1', ascending=False).iloc[0]
            final_threshold = best_row['threshold']
            strategy_name = "auto_precision_priority"
            print(f"[SO] 自动阈值: {final_threshold:.3f} (P={best_row['precision']:.3f}, R={best_row['recall']:.3f})")
        else:
            # 如果没找到满足条件的，就找 F1 最高的
            best_row = sweep_df.sort_values(by='f1', ascending=False).iloc[0]
            final_threshold = best_row['threshold']
            strategy_name = "auto_max_f1"
            print(f"[SO] 未满足高精度约束，回退到最大F1阈值: {final_threshold:.3f} (F1={best_row['f1']:.3f})")
    
    # 应用最终阈值到 Test Set
    y_true = df_res['label'].values
    y_scores = df_res['similarity_score'].values
    y_pred = (y_scores >= final_threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n[SO] Final Test Metrics (Thr={final_threshold:.3f}):")
    print(f"  Acc: {acc:.4f}")
    print(f"  Pre: {pre:.4f}")
    print(f"  Rec: {rec:.4f}")
    print(f"  F1 : {f1:.4f}")
    
    metrics = {
        'threshold': float(final_threshold),
        'strategy': strategy_name,
        'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1,
        'feat_time_s': feat_time, 'detect_time_s': detect_time,
        'train_time_s': train_time_total
    }
    
    output_json = os.path.join(args.result_root, 'metrics.json')
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description='SO CNN 训练+测试（矩阵加速版）')
    parser.add_argument('--root', default='/newdisk/liuzhuowu/lzw/apks_androzoo', help='AndroZoo根目录')
    parser.add_argument('--threshold', type=float, default=0.85, help='判定相似的阈值')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=32, help='训练批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader并发')
    parser.add_argument('--intermediate-root', default='/newdisk/liuzhuowu/baseline/temp', help='中间文件根目录')
    parser.add_argument('--result-root', default='/newdisk/liuzhuowu/baseline/androzoo_result/so', help='模型与结果输出根目录')
    parser.add_argument('--mode', choices=['train', 'test', 'train_test'], default='train_test', help='运行模式: train(仅训练), test(仅测试), train_test(训练后测试)')
    parser.add_argument('--model-path', default=None, help='指定模型路径(加载或保存)，默认在result-root下so_model.pth')
    parser.add_argument('--no-progress', action='store_true', help='关闭进度条')
    
    # 阈值策略参数
    parser.add_argument('--auto-threshold', action='store_true', help='自动选择最佳阈值')
    parser.add_argument('--min-recall', type=float, default=0.85, help='自动阈值时的召回率下限')
    parser.add_argument('--target-precision', type=float, default=0.95, help='自动阈值时的精确率目标')
    
    # 损失参数
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for Contrastive Loss')
    parser.add_argument('--neg-weight', type=float, default=2.0, help='Negative sample weight')
    
    args = parser.parse_args()
    run_so_detection(args)

if __name__ == '__main__':
    main()