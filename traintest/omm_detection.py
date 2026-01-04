import os
import sys
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

# 允许导入同级目录下的feature_cnn_models模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from feature_cnn_models import SmaliOpcodeDetailCaptureCNN
except ImportError:
    # 作为一个fallback，防止用户没有放feature_cnn_models.py导致报错，实际运行时需要有feature_cnn_models.py
    class SmaliOpcodeDetailCaptureCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 16, 3)
            self.fc = nn.Linear(100, 128)
        def forward(self, x):
            return torch.randn(x.size(0), 128).to(x.device)
    print("Warning: feature_cnn_models.py not found, using dummy model for syntax check.")

# ------------------------------------------------------------------------
# 1. 辅助函数：路径扫描与解析 (优化IO)
# ------------------------------------------------------------------------

def list_group_ids(root_dir: str) -> List[str]:
    groups = []
    if not os.path.exists(root_dir):
        return []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and name.isdigit():
            groups.append(name)
    return sorted(groups, key=lambda x: int(x))

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

def locate_omm_npy(project_path: str) -> str:
    """查找 .npy 文件，返回绝对路径，找不到返回空字符串"""
    npy1 = os.path.join(project_path, 'dalvik.npy')
    if os.path.exists(npy1): return npy1
    npy2 = os.path.join(project_path, 'omm', 'dalvik.npy')
    if os.path.exists(npy2): return npy2
    return ''

def pre_resolve_paths(projects: List[str]) -> Dict[str, str]:
    """批量解析路径，建立 project_path -> npy_path 的映射"""
    mapping = {}
    for p in projects:
        npy = locate_omm_npy(p)
        if npy:
            mapping[p] = npy
    return mapping

# ------------------------------------------------------------------------
# 2. 样本对构建与 Dataset
# ------------------------------------------------------------------------

def build_positive_pairs(originals: List[str], repacks: List[str]) -> List[Tuple[str, str, int]]:
    pairs = []
    for o in originals:
        for r in repacks:
            pairs.append((o, r, 1))
    return pairs

def sample_negative_pairs(all_groups_projects, exclude_gid, target_num):
    """优化后的负样本采样：直接从预加载的列表里采"""
    other_gids = list(all_groups_projects.keys())
    if str(exclude_gid) in other_gids:
        other_gids.remove(str(exclude_gid))
    
    if not other_gids:
        return []

    neg_pairs = []
    attempts = 0
    max_attempts = target_num * 5
    
    while len(neg_pairs) < target_num and attempts < max_attempts:
        attempts += 1
        g1, g2 = random.sample(other_gids, 2)
        
        # 获取该组的所有项目
        projs1 = all_groups_projects[g1]
        projs2 = all_groups_projects[g2]
        
        if not projs1 or not projs2: continue
        
        p1 = random.choice(projs1)
        p2 = random.choice(projs2)
        
        neg_pairs.append((p1, p2, 0))
        
    return neg_pairs

class OmmPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, int]], path_mapping: Dict[str, str]):
        """
        pairs: [(proj_a, proj_b, label), ...]
        path_mapping: {proj_path: npy_absolute_path}
        """
        self.pairs = pairs
        self.path_mapping = path_mapping

    def __len__(self):
        return len(self.pairs)

    def _load_npy(self, project_path: str):
        # 直接查表，避免IO check
        npy_path = self.path_mapping.get(project_path)
        if not npy_path:
            # 理论上不该发生，因为构建pair时过滤过了
            return torch.zeros((1, 64, 64), dtype=torch.float32) 
        
        try:
            mat = np.load(npy_path)
            # 确保数据是 float32
            t = torch.from_numpy(mat).float()
            # 如果是二维 [H, W]，增加通道维 -> [1, H, W]
            if t.dim() == 2:
                t = t.unsqueeze(0)
            return t
        except Exception:
            return torch.zeros((1, 64, 64), dtype=torch.float32)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        t1 = self._load_npy(p1)
        t2 = self._load_npy(p2)
        y = torch.tensor(1 if label == 1 else -1, dtype=torch.float32)
        return t1, t2, y, p1, p2, label

class SingleIconDataset(Dataset):
    """用于测试阶段批量提取特征的 Dataset"""
    def __init__(self, project_paths, path_mapping):
        self.project_paths = project_paths
        self.path_mapping = path_mapping

    def __len__(self):
        return len(self.project_paths)

    def __getitem__(self, idx):
        proj = self.project_paths[idx]
        npy_path = self.path_mapping.get(proj)
        if not npy_path:
            return proj, None
        try:
            mat = np.load(npy_path)
            t = torch.from_numpy(mat).float()
            if t.dim() == 2:
                t = t.unsqueeze(0)
            return proj, t
        except:
            return proj, None

def collate_fn_single(batch):
    batch = [b for b in batch if b[1] is not None]
    if not batch:
        return [], None
    paths, tensors = zip(*batch)
    return paths, torch.stack(tensors)

# ------------------------------------------------------------------------
# 3. 核心优化：矩阵化测试 (取代逐个 evaluate_cached)
# ------------------------------------------------------------------------

def extract_and_detect_matrix(model, device, test_pairs, path_mapping, batch_size, workers):
    """
    优化版测试流程：
    1. 提取 Unique APKs
    2. 批量提取特征 -> 预归一化 -> 存入 GPU Tensor
    3. 矩阵点积计算相似度
    """
    # A. 准备 unique projects
    unique_projs = set()
    for p1, p2, _ in test_pairs:
        unique_projs.add(p1)
        unique_projs.add(p2)
    unique_projs = list(unique_projs)
    proj_to_idx = {p: i for i, p in enumerate(unique_projs)}
    num_projs = len(unique_projs)
    
    print(f"测试集涉及唯一项目数: {num_projs}")

    # B. 批量特征提取
    dataset = SingleIconDataset(unique_projs, path_mapping)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=workers, collate_fn=collate_fn_single, pin_memory=True)
    
    # 假设特征维度未知，先用 list 存，提取第一批后确定维度
    # 通常 CNN output 是 flattened vector
    feature_matrix = None 
    valid_indices = [] # 记录成功加载的 index
    
    model.eval()
    t_feat_start = time.time()
    
    all_feats_list = []
    all_indices_list = []
    
    with torch.no_grad():
        for paths, tensors in tqdm(loader, desc="[测试] 批量提取特征", unit="batch"):
            if tensors is None: continue
            
            tensors = tensors.to(device)
            feats = model(tensors) # [B, Dim]
            
            # 展平
            feats = feats.view(feats.size(0), -1)
            
            # 预归一化 (Pre-normalize)
            feats = F.normalize(feats, p=2, dim=1)
            
            # 转 CPU 暂存 (或者直接在 GPU 拼接，取决于显存)
            # CNN 特征通常很小 (如 128维)，全放 GPU 没问题
            all_feats_list.append(feats)
            
            for p in paths:
                all_indices_list.append(proj_to_idx[p])

    if not all_feats_list:
        return pd.DataFrame(), 0, 0

    # 拼成大矩阵 [N, Dim]
    # 这里我们只存了有效的部分，需要构建一个全量的矩阵或者映射表
    # 为了方便，我们构建一个 full_matrix
    
    # 获取维度
    dim = all_feats_list[0].size(1)
    # 初始化全零矩阵 (放在 GPU)
    full_matrix = torch.zeros((num_projs, dim), device=device)
    
    # 填充
    current_ptr = 0
    for batch_feats in all_feats_list:
        b_size = batch_feats.size(0)
        # 获取对应的 indices
        batch_indices = all_indices_list[current_ptr : current_ptr + b_size]
        full_matrix[batch_indices] = batch_feats
        current_ptr += b_size
        
    feat_time = time.time() - t_feat_start
    
    # C. 矩阵计算相似度
    t_detect_start = time.time()
    
    idx_a_list = []
    idx_b_list = []
    meta_list = []
    
    # 只有当两个都在 unique_projs 且成功提取了特征 (非0向量) 才计算
    # 这里简化处理：如果在 full_matrix 里是 0 向量，点积自然是 0
    
    for p1, p2, lab in test_pairs:
        if p1 in proj_to_idx and p2 in proj_to_idx:
            idx_a_list.append(proj_to_idx[p1])
            idx_b_list.append(proj_to_idx[p2])
            meta_list.append((p1, p2, lab))
            
    if not idx_a_list:
        return pd.DataFrame(), feat_time, 0
    
    # 转 Tensor
    idx_a = torch.tensor(idx_a_list, device=device)
    idx_b = torch.tensor(idx_b_list, device=device)
    
    # 计算 (Batch processing)
    # 如果对数太多(>100w)，可以分块，这里假设显存够用
    # Dot Product
    # vec_a: [NumPairs, Dim], vec_b: [NumPairs, Dim]
    vec_a = full_matrix[idx_a]
    vec_b = full_matrix[idx_b]
    
    # sum(a * b, dim=1)
    sims = (vec_a * vec_b).sum(dim=1).cpu().numpy()
    
    detect_time = time.time() - t_detect_start
    
    # 组装结果
    results = []
    for i, (p1, p2, lab) in enumerate(meta_list):
        score = float(sims[i])
        pred = 1 if score >= 0.85 else 0 # 临时阈值，后面还会重算
        
        # 简单统计 tp/fp 用于 debug
        is_tp = 1 if (lab == 1 and pred == 1) else 0
        is_fp = 1 if (lab == 0 and pred == 1) else 0
        is_tn = 1 if (lab == 0 and pred == 0) else 0
        is_fn = 1 if (lab == 1 and pred == 0) else 0
        
        results.append({
            'apk1': p1,
            'apk2': p2,
            'label': lab,
            'similarity_score': score,
            'tp': is_tp, 'fp': is_fp, 'tn': is_tn, 'fn': is_fn
        })
        
    return pd.DataFrame(results), feat_time, detect_time

# ------------------------------------------------------------------------
# 4. 主流程
# ------------------------------------------------------------------------

def run_omm_detection(args):
    """
    OMM特征检测入口函数
    :param args: 包含所有参数的命名空间对象
    """
    show_progress = not args.no_progress
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.result_root, exist_ok=True)
    interm_dir = os.path.join(args.intermediate_root, 'omm')
    os.makedirs(interm_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[OMM] Device: {device}")

    # 1. 扫描与路径映射 (Pre-scan)
    print("[OMM] 正在扫描所有项目的 OMM 特征路径...")
    t0 = time.time()
    all_groups = list_group_ids(args.root)
    
    # 抽样
    if hasattr(args, 'sample_fraction') and args.sample_fraction < 1.0:
        import math
        sample_n = max(1, math.ceil(len(all_groups) * args.sample_fraction))
        all_groups = random.sample(all_groups, sample_n)
        print(f"[OMM] 抽样: {len(all_groups)} 组 (fraction={args.sample_fraction})")

    all_group_projs = {} # gid -> [proj_paths]
    all_proj_paths_flat = []
    
    for gid in tqdm(all_groups, desc="扫描组") if show_progress else all_groups:
        o, r = list_apk_projects_in_group(args.root, gid)
        all_group_projs[gid] = o + r
        all_proj_paths_flat.extend(o + r)
        
    # 建立路径映射表 (Pre-resolve)
    path_mapping = pre_resolve_paths(all_proj_paths_flat)
    print(f"[OMM] 扫描完成，耗时 {time.time()-t0:.2f}s。找到 {len(path_mapping)} 个有效的 OMM 特征文件。")
    
    if not path_mapping:
        print("[OMM] 未找到任何 .npy 文件，请检查路径。")
        return None

    # 2. 构建样本对
    print("[OMM] 构建正负样本对...")
    all_pairs = []
    
    # 仅使用存在 feature 的项目
    valid_projs_set = set(path_mapping.keys())
    
    for gid in tqdm(all_groups, desc="配对") if show_progress else all_groups:
        o, r = list_apk_projects_in_group(args.root, gid)
        # 过滤
        o = [p for p in o if p in valid_projs_set]
        r = [p for p in r if p in valid_projs_set]
        if not o or not r: continue
        
        # 正样本
        pos = build_positive_pairs(o, r)
        all_pairs.extend(pos)
        
        # 负样本
        target_neg = len(pos)
        neg_candidates = sample_negative_pairs(all_group_projs, gid, target_neg * 2) # 多采一点备用
        
        valid_negs = []
        for p1, p2, lab in neg_candidates:
            if p1 in valid_projs_set and p2 in valid_projs_set:
                valid_negs.append((p1, p2, lab))
                if len(valid_negs) >= target_neg:
                    break
        all_pairs.extend(valid_negs)

    if not all_pairs:
        print("[OMM] 无有效样本对。")
        return None
        
    random.shuffle(all_pairs)
    print(f"[OMM] 总样本对数: {len(all_pairs)}")

    # 划分
    total = len(all_pairs)
    n_train = int(total * 0.7)
    n_val = int(total * 0.1)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train+n_val]
    test_pairs = all_pairs[n_train+n_val:]

    # DataLoader (注意 batch_size 和 num_workers)
    # 验证集 batch_size 也设为 args.batch_size 提高速度
    train_ds = OmmPairDataset(train_pairs, path_mapping)
    val_ds = OmmPairDataset(val_pairs, path_mapping)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    # 模型与训练
    model = SmaliOpcodeDetailCaptureCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CosineEmbeddingLoss(margin=0.0)

    train_time_total = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', unit='batch') if show_progress else train_loader
        for x1, x2, y, _, _, _ in pbar:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            optimizer.zero_grad()
            emb1 = model(x1) # [B, Dim]
            emb2 = model(x2)
            
            # 确保是 flat 的
            emb1 = emb1.view(emb1.size(0), -1)
            emb2 = emb2.view(emb2.size(0), -1)
            
            loss = criterion(emb1, emb2, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_dt = time.time() - t_epoch
        train_time_total += epoch_dt
        avg_loss = running_loss / len(train_loader)
        
        # 验证 (使用 Batch 加速)
        model.eval()
        val_correct = 0
        val_total = 0
        val_t0 = time.time()
        with torch.no_grad():
            for x1, x2, y, _, _, _ in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                e1 = model(x1).view(x1.size(0), -1)
                e2 = model(x2).view(x2.size(0), -1)
                sim = F.cosine_similarity(e1, e2)
                pred = (sim >= args.threshold).float()
                # y is 1 or -1. Convert y to 0/1 for comparison
                label_01 = (y > 0).float()
                val_correct += (pred == label_01).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_time = time.time() - val_t0
        print(f"[OMM] Epoch {epoch} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val Time: {val_time:.2f}s")

    # 3. 测试阶段 (矩阵加速)
    print("\n[OMM] 开始测试阶段 (Matrix Accelerated)...")
    
    df, feat_time, detect_time = extract_and_detect_matrix(
        model, device, test_pairs, path_mapping, args.batch_size, args.workers
    )
    
    if df.empty:
        print("[OMM] 测试结果为空。")
        return None

    # 重新应用阈值计算指标 (防止硬编码)
    df['pred'] = (df['similarity_score'] >= args.threshold).astype(int)
    
    tp = int(((df['label'] == 1) & (df['pred'] == 1)).sum())
    fp = int(((df['label'] == 0) & (df['pred'] == 1)).sum())
    tn = int(((df['label'] == 0) & (df['pred'] == 0)).sum())
    fn = int(((df['label'] == 1) & (df['pred'] == 0)).sum())
    
    acc = (tp + tn) / len(df)
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    
    total_pairs = len(df)
    print(f"\n[OMM] 测试完成: Pairs={total_pairs}")
    print(f"Accuracy: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"特征提取耗时: {feat_time:.2f}s")
    print(f"相似度检测耗时: {detect_time:.4f}s (平均 {detect_time/total_pairs*1000:.4f} ms/pair)")

    # 保存结果
    model_path = os.path.join(args.result_root, 'omm_cnn_trained.pth')
    torch.save(model.state_dict(), model_path)
    
    results_csv = os.path.join(args.result_root, 'omm_cnn_test_results.csv')
    df.to_csv(results_csv, index=False)
    
    metrics = {
        'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1,
        'train_time_s': train_time_total,
        'feat_time_s': feat_time,
        'detect_time_s': detect_time,
        'detect_avg_ms': detect_time/total_pairs*1000
    }
    
    output_json = os.path.join(args.result_root, 'metrics.json')
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    return metrics

def main():
    parser = argparse.ArgumentParser(description='OMM CNN 训练+测试（矩阵加速版）')
    parser.add_argument('--root', default='/newdisk/liuzhuowu/lzw/apks_androzoo', help='AndroZoo根目录')
    parser.add_argument('--threshold', type=float, default=0.85, help='判定相似的阈值')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=32, help='训练批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader并发数(建议4-8)')
    parser.add_argument('--intermediate-root', default='/newdisk/liuzhuowu/baseline/temp', help='中间文件根目录')
    parser.add_argument('--result-root', default='/newdisk/liuzhuowu/baseline/androzoo_result/omm', help='结果根目录')
    parser.add_argument('--no-progress', action='store_true', help='关闭进度条')
    args = parser.parse_args()
    run_omm_detection(args)

if __name__ == '__main__':
    main()