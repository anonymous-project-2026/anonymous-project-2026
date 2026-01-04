import os
import time
import json
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19, VGG19_Weights

# ------------------------------
# 1. 模型与数据集定义
# ------------------------------

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layer_names):
        super(VGG19FeatureExtractor, self).__init__()
        weights = VGG19_Weights.DEFAULT
        base_model = vgg19(weights=weights)
        self.features = nn.Sequential(*list(base_model.features.children()))
        self.avgpool = base_model.avgpool
        self.fc = nn.Sequential(*list(base_model.classifier.children())[:-1])
        self.layer_names = layer_names

    def forward(self, x):
        feature_maps = {}
        # VGG features part
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in self.layer_names:
                feature_maps[name] = x
        
        # Classifier part (for fc2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if 'fc2' in self.layer_names:
            feature_maps['fc2'] = x
        return feature_maps

class IconDataset(Dataset):
    def __init__(self, icon_paths, transform):
        self.icon_paths = icon_paths
        self.transform = transform

    def __len__(self):
        return len(self.icon_paths)

    def __getitem__(self, idx):
        path = self.icon_paths[idx]
        try:
            image = Image.open(path)
            if image.mode != 'RGB':
                image = convert_rgba_to_rgb(image)
            image = image.resize((224, 224))
            if self.transform:
                image = self.transform(image)
            return path, image
        except Exception:
            # 返回 None 表示加载失败
            return path, None

def collate_fn(batch):
    # 过滤掉加载失败的图片 (None)
    batch = [item for item in batch if item[1] is not None]
    if not batch:
        return [], torch.Tensor()
    paths, images = zip(*batch)
    return paths, torch.stack(images)

def convert_rgba_to_rgb(image):
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    else:
        return image.convert('RGB')

# ------------------------------
# 2. 辅助函数：文件扫描与配对
# ------------------------------

def find_icon_path(apk_project_path):
    images_dir = os.path.join(apk_project_path, 'images')
    if not os.path.isdir(images_dir):
        return None
    candidate_paths = []
    for root, dirs, files in os.walk(images_dir):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                candidate_paths.append(os.path.join(root, file_name))
    if not candidate_paths:
        return None
    
    def score(p):
        name = os.path.basename(p).lower()
        name_score = 1 if ('launcher' in name or 'icon' in name) else 0
        try:
            size = os.path.getsize(p)
        except Exception:
            size = 0
        return (name_score, size)
    
    candidate_paths.sort(key=score, reverse=True)
    return candidate_paths[0]

def list_group_ids(root_dir):
    ids = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and name.isdigit():
            ids.append(name)
    return sorted(ids, key=lambda x: int(x))

def list_apk_projects_in_group(root_dir, group_id):
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

def build_positive_pairs(originals, repacks):
    pairs = []
    for o in originals:
        for r in repacks:
            pairs.append((o, r, 1))
    return pairs

def sample_negative_pairs_from_cache(all_group_projects, current_group_id, target_num):
    other_gids = [gid for gid in all_group_projects if gid != str(current_group_id)]
    if not other_gids:
        return []
    
    neg_pairs = []
    attempts = 0
    max_attempts = target_num * 5
    
    while len(neg_pairs) < target_num and attempts < max_attempts:
        attempts += 1
        g1, g2 = random.sample(other_gids, 2)
        projs1 = all_group_projects[g1][0] + all_group_projects[g1][1]
        projs2 = all_group_projects[g2][0] + all_group_projects[g2][1]
        
        if not projs1 or not projs2:
            continue
            
        p1 = random.choice(projs1)
        p2 = random.choice(projs2)
        neg_pairs.append((p1, p2, 0))
        
    return neg_pairs

# ------------------------------
# 3. 核心优化：批量特征提取与矩阵检测
# ------------------------------

def extract_and_detect_optimized(model, device, icons_map, all_pairs, batch_size, workers):
    """
    1. 提取所有唯一图标的特征，预归一化，存入大矩阵 (CPU/GPU)
    2. 将 pair 转换为 index pair
    3. 使用矩阵点积批量计算相似度
    """
    
    # --- A. 准备阶段 ---
    unique_paths = list(set(icons_map.values()))
    path_to_idx = {p: i for i, p in enumerate(unique_paths)}
    num_icons = len(unique_paths)
    
    print(f"唯一图标总数: {num_icons}")
    
    # 转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = IconDataset(unique_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=workers, collate_fn=collate_fn, pin_memory=True)

    # --- B. 特征提取阶段 (Feature Extraction) ---
    # 定义维度: fc2=4096, style(conv5_4)=512*512=262144
    content_dim = 4096
    style_dim = 512 * 512
    
    # 使用 Float16 节省内存，先存在 CPU 上以防显存不足
    all_content = torch.zeros((num_icons, content_dim), dtype=torch.float16)
    all_style = torch.zeros((num_icons, style_dim), dtype=torch.float16)
    
    # 记录成功加载的索引
    loaded_indices = set()
    
    t_feat_start = time.time()
    model.eval()
    
    with torch.no_grad():
        for paths, images in tqdm(loader, desc="[1/2] 批量特征提取", unit="batch"):
            if len(paths) == 0:
                continue
            
            # 移入 GPU
            images = images.to(device)
            
            # Forward
            feats = model(images)
            
            # 1. Content Feature (fc2)
            f_content = feats['fc2'] # [B, 4096]
            # L2 Normalize immediately
            f_content = torch.nn.functional.normalize(f_content, p=2, dim=1)
            
            # 2. Style Feature (Gram Matrix)
            f_style = feats['28'] # [B, 512, H, W]
            b, ch, h, w = f_style.size()
            f_style = f_style.view(b, ch, h * w)
            # Gram calculation: (B, C, HW) x (B, HW, C) -> (B, C, C)
            grams = torch.bmm(f_style, f_style.transpose(1, 2)) / (ch * h * w)
            grams = grams.view(b, -1) # Flatten -> [B, 262144]
            # L2 Normalize immediately
            grams = torch.nn.functional.normalize(grams, p=2, dim=1)
            
            # 转回 CPU 并转为半精度存储
            f_content_cpu = f_content.cpu().half()
            grams_cpu = grams.cpu().half()
            
            # 填入大表
            for i, p in enumerate(paths):
                idx = path_to_idx[p]
                all_content[idx] = f_content_cpu[i]
                all_style[idx] = grams_cpu[i]
                loaded_indices.add(idx)

    feature_time = time.time() - t_feat_start

    # --- C. 相似度检测阶段 (Detection) ---
    t_detect_start = time.time()
    
    # 1. 筛选有效对并将路径转换为索引
    valid_pairs_indices = []
    valid_pairs_meta = [] # (gid, apk1, apk2, label)
    
    for gid, a, b, lab in all_pairs:
        # 必须两个图标路径都在，且都成功加载了特征
        if a in icons_map and b in icons_map:
            path_a = icons_map[a]
            path_b = icons_map[b]
            idx_a = path_to_idx[path_a]
            idx_b = path_to_idx[path_b]
            
            if idx_a in loaded_indices and idx_b in loaded_indices:
                valid_pairs_indices.append((idx_a, idx_b))
                valid_pairs_meta.append((gid, a, b, lab))
    
    if not valid_pairs_indices:
        return pd.DataFrame(), feature_time, 0.0

    # 2. 准备批量计算
    pairs_tensor = torch.tensor(valid_pairs_indices, dtype=torch.long)
    idx_A = pairs_tensor[:, 0]
    idx_B = pairs_tensor[:, 1]
    
    num_pairs = len(idx_A)
    sims_content = []
    sims_style = []
    
    # 根据显存大小调整 batch
    # 5000 pairs * 26w dims * 2 (float16) * 2 (A&B) ≈ 5GB data per batch if fully expanded
    # 但我们只取 index，实际运算在 GPU
    calc_batch_size = 5000 
    
    # 尝试将大矩阵移入 GPU (如果显存 > 8GB 且图标数 < 10000 可能放得下)
    # Style Matrix: 10000 * 262144 * 2 bytes = 5 GB
    # 为求稳妥，默认放在 CPU，每个 batch 拷贝需要的数据进 GPU
    # 如果你显存很大(24G+)，可以把 all_style.to(device)
    
    # all_content 比较小，可以常驻 GPU
    all_content = all_content.to(device)
    # all_style 保持在 CPU 或根据显存情况决定
    # all_style = all_style.to(device) 
    
    print(f"开始批量比对 {num_pairs} 对样本...")
    
    for i in tqdm(range(0, num_pairs, calc_batch_size), desc="[2/2] GPU加速比对", unit="blk"):
        end = min(i + calc_batch_size, num_pairs)
        
        # 获取当前 batch 的索引
        batch_idx_a = idx_A[i:end].to(device)
        batch_idx_b = idx_B[i:end].to(device)
        
        # --- Content Sim ---
        # 索引获取特征 (已经在 GPU)
        c_a = all_content[batch_idx_a]
        c_b = all_content[batch_idx_b]
        # Dot Product (因为已经归一化，点积==余弦相似度)
        sim_c = (c_a * c_b).sum(dim=1)
        sims_content.append(sim_c.cpu().float()) # 存回 float32
        
        # --- Style Sim ---
        # 索引获取特征 (从 CPU 拷贝到 GPU)
        # 注意：这里需要先把 indices 转回 CPU 才能去 slice cpu tensor (如果 all_style 在 cpu)
        b_idx_a_cpu = idx_A[i:end]
        b_idx_b_cpu = idx_B[i:end]
        
        s_a = all_style[b_idx_a_cpu].to(device)
        s_b = all_style[b_idx_b_cpu].to(device)
        
        sim_s = (s_a * s_b).sum(dim=1)
        sims_style.append(sim_s.cpu().float())
        
        # 显存清理 (非必须，但保险)
        del s_a, s_b, c_a, c_b
    
    # 拼接结果
    sims_content = torch.cat(sims_content).numpy()
    sims_style = torch.cat(sims_style).numpy()
    sims_overall = 0.6 * sims_content + 0.4 * sims_style
    
    detect_time = time.time() - t_detect_start
    
    # --- D. 组装结果 ---
    results = []
    for i, (gid, a, b, lab) in enumerate(valid_pairs_meta):
        results.append({
            'group_id': gid,
            'apk1': a,
            'apk2': b,
            'label': lab,
            'content_similarity': float(sims_content[i]),
            'style_similarity': float(sims_style[i]),
            'overall_similarity': float(sims_overall[i])
        })
        
    return pd.DataFrame(results), feature_time, detect_time

# ------------------------------
# 4. 主流程
# ------------------------------

def run_icon_detection(args):
    """
    图标特征检测入口函数
    :param args: 包含所有参数的命名空间对象
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 目录准备
    interm_dir = os.path.join(args.intermediate_root, 'image', 'all_groups')
    result_dir = os.path.join(args.result_root, 'image')
    os.makedirs(interm_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Icon] Running on {device}")

    # 模型加载
    layer_names = ['28', 'fc2']
    vgg_model = VGG19FeatureExtractor(layer_names).to(device).eval()

    # 1. 扫描组
    all_ids = list_group_ids(args.root)
    if not all_ids:
        print('[Icon] 根目录下未找到数字序号文件夹。')
        return None
    
    print(f"[Icon] 正在预扫描所有 {len(all_ids)} 个组的 APK 项目...")
    all_group_projects = {}
    for gid in tqdm(all_ids, desc='扫描组结构'):
        all_group_projects[gid] = list_apk_projects_in_group(args.root, gid)
        
    sample_count = max(1, int(round(len(all_ids) * args.sample_fraction)))
    sampled_ids = random.sample(all_ids, sample_count)
    print(f"[Icon] 已抽样 {len(sampled_ids)} 个组。")

    # 2. 构建正负样本对
    pos_pairs_by_group = {}
    neg_pairs_by_group = {}
    
    for gid in tqdm(sampled_ids, desc='构建样本对'):
        originals, repacks = all_group_projects[gid]
        pos_pairs = build_positive_pairs(originals, repacks)
        pos_pairs_by_group[gid] = pos_pairs
        
        # 负样本采样
        neg_pairs = sample_negative_pairs_from_cache(all_group_projects, gid, target_num=len(pos_pairs))
        neg_pairs_by_group[gid] = neg_pairs

    # 记录负样本对日志
    neg_log_path = os.path.join(result_dir, f'negative_pairs_all_sampled.csv')
    neg_rows = []
    for gid in neg_pairs_by_group:
        for a, b, _ in neg_pairs_by_group[gid]:
            neg_rows.append({'group_id': gid, 'apk1': a, 'apk2': b})
    pd.DataFrame(neg_rows).to_csv(neg_log_path, index=False, encoding='utf-8-sig')

    # 3. 收集所有需要的图标路径
    all_pairs_list = [] # (gid, a, b, lab)
    all_projects = set()
    
    for gid in sampled_ids:
        for a, b, lab in pos_pairs_by_group[gid]:
            all_pairs_list.append((gid, a, b, lab))
            all_projects.add(a)
            all_projects.add(b)
        for a, b, lab in neg_pairs_by_group[gid]:
            all_pairs_list.append((gid, a, b, lab))
            all_projects.add(a)
            all_projects.add(b)

    print("[Icon] 解析图标路径...")
    icons_map = {}
    for proj in all_projects:
        icon = find_icon_path(proj)
        if icon:
            icons_map[proj] = icon
    
    if not icons_map:
        print("[Icon] 未找到任何图标，程序结束。")
        return None

    # 4. 执行核心优化流程：提取 -> 矩阵计算
    df, feat_time, detect_time = extract_and_detect_optimized(
        model=vgg_model,
        device=device,
        icons_map=icons_map,
        all_pairs=all_pairs_list,
        batch_size=args.batch_size,
        workers=args.workers
    )

    if df.empty:
        print('[Icon] 无有效结果。')
        return None

    # 5. 指标计算与保存
    print("[Icon] 计算评估指标...")
    df['pred'] = (df['overall_similarity'] >= args.threshold).astype(int)
    
    tp = int(((df['label'] == 1) & (df['pred'] == 1)).sum())
    tn = int(((df['label'] == 0) & (df['pred'] == 0)).sum())
    fp = int(((df['label'] == 0) & (df['pred'] == 1)).sum())
    fn = int(((df['label'] == 1) & (df['pred'] == 0)).sum())
    
    accuracy = (tp + tn) / len(df)
    recall = tp / max(1, int((df['label'] == 1).sum()))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    total_pairs = len(df)
    feat_avg = feat_time / total_pairs if total_pairs > 0 else 0
    detect_avg = detect_time / total_pairs if total_pairs > 0 else 0
    total_avg = feat_avg + detect_avg

    # 分组指标
    group_metrics_rows = []
    for gid, gdf in df.groupby('group_id'):
        tp_g = int(((gdf['label'] == 1) & (gdf['pred'] == 1)).sum())
        fp_g = int(((gdf['label'] == 0) & (gdf['pred'] == 1)).sum())
        rec_g = tp_g / max(1, int((gdf['label'] == 1).sum()))
        pre_g = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0.0
        f1_g = (2 * pre_g * rec_g / (pre_g + rec_g)) if (pre_g + rec_g) > 0 else 0.0
        
        group_metrics_rows.append({
            'group_id': gid,
            'precision': pre_g,
            'recall': rec_g,
            'f1': f1_g,
            'total_pairs': len(gdf),
        })

    output_path = os.path.join(result_dir, f'image_all_groups_sampled.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='pairs', index=False)
        metrics_overall = pd.DataFrame({
            'metric': ['accuracy', 'precision', 'recall', 'f1', 'feat_time_s', 'detect_time_s', 'detect_avg_s'],
            'value': [accuracy, precision, recall, f1, feat_time, detect_time, detect_avg]
        })
        metrics_overall.to_excel(writer, sheet_name='metrics_overall', index=False)
        pd.DataFrame(group_metrics_rows).to_excel(writer, sheet_name='metrics_by_group', index=False)

    print(f"[Icon] 完成。总耗时: {feat_time + detect_time:.2f}s")
    print(f"[Icon] 检测平均耗时/对: {detect_avg:.6f}s")
    print(f"[Icon] 结果保存至: {output_path}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'feat_time_s': feat_time,
        'detect_time_s': detect_time
    }

def main():
    parser = argparse.ArgumentParser(description='AndroZoo全量抽样图标相似度检测（GPU加速版）')
    parser.add_argument('--root', default='/newdisk/liuzhuowu/lzw/apks_androzoo', help='AndroZoo根目录')
    parser.add_argument('--sample-fraction', type=float, default=0.25, help='抽样的组比例')
    parser.add_argument('--threshold', type=float, default=0.6, help='判定为相似的阈值')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader线程数')
    parser.add_argument('--batch-size', type=int, default=64, help='特征提取批大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--intermediate-root', default='/newdisk/liuzhuowu/baseline/suidroid', help='中间文件根目录')
    parser.add_argument('--result-root', default='/newdisk/liuzhuowu/baseline/suidroid_result', help='结果根目录')
    args = parser.parse_args()
    run_icon_detection(args)

if __name__ == '__main__':
    main()