import time
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SingleSoDataset(Dataset):
    """用于测试阶段批量提取特征"""
    def __init__(self, project_paths, path_mapping):
        self.project_paths = project_paths
        self.path_mapping = path_mapping

    def __len__(self):
        return len(self.project_paths)

    def __getitem__(self, idx):
        path = self.project_paths[idx]
        npy = self.path_mapping.get(path)
        if not npy:
            return path, None
        try:
            mat = np.load(npy, allow_pickle=True)
            t = torch.tensor(mat, dtype=torch.float32)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            return path, t
        except:
            return path, None

def collate_fn_single(batch):
    batch = [b for b in batch if b[1] is not None]
    if not batch: return [], None
    paths, tensors = zip(*batch)
    return paths, torch.stack(tensors)

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
