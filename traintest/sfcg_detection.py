import os
import sys
import time
import json
import math
import random
import argparse
import itertools
import pandas as pd
from tqdm import tqdm

# 直接调用 traintest 目录下的 sfcg_ot_utils.py
sys.path.append(os.path.dirname(__file__))
from sfcg_ot_utils import SFCG_OT_ThresholdAnalyzer

def list_groups(root_dir):
    return [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

def list_apks_in_group(root_dir, group_id, gexf_name='community_processed_graph.gexf', apk_subdirs=None):
    group_path = os.path.join(root_dir, group_id)
    apk_paths = []
    apk_names = []
    if not os.path.isdir(group_path):
        return apk_paths, apk_names
    # 若指定子目录（例如 original_apk、repack_apk），优先在子目录中查找
    if apk_subdirs:
        for subdir in apk_subdirs:
            sub_path = os.path.join(group_path, subdir)
            if not os.path.isdir(sub_path):
                continue
            for apk_folder in os.listdir(sub_path):
                apk_path = os.path.join(sub_path, apk_folder)
                gexf_file = os.path.join(apk_path, gexf_name)
                if os.path.isdir(apk_path) and os.path.exists(gexf_file):
                    apk_paths.append(apk_path)
                    apk_names.append(apk_folder)
    # 兼容：如果未指定子目录或子目录未匹配，则尝试直接在组目录下一层查找
    if not apk_paths:
        for apk_folder in os.listdir(group_path):
            apk_path = os.path.join(group_path, apk_folder)
            gexf_file = os.path.join(apk_path, gexf_name)
            if os.path.isdir(apk_path) and os.path.exists(gexf_file):
                apk_paths.append(apk_path)
                apk_names.append(apk_folder)
    return apk_paths, apk_names

def build_positive_pairs(root_dir, selected_groups, gexf_name='community_processed_graph.gexf', apk_subdirs=None):
    pairs = []
    for gid in selected_groups:
        # 如果指定了两个子目录，则构造笛卡尔积（original_apk x repack_apk）作为正样本
        if apk_subdirs and len(apk_subdirs) >= 2:
            originals, _ = list_apks_in_group(root_dir, gid, gexf_name=gexf_name, apk_subdirs=[apk_subdirs[0]])
            repacks, _ = list_apks_in_group(root_dir, gid, gexf_name=gexf_name, apk_subdirs=[apk_subdirs[1]])
            for op in originals:
                for rp in repacks:
                    pairs.append((op, rp, 1, gid))
            # 如果某组只有一个子目录有数据，退化为组内两两组合
            if not originals or not repacks:
                apk_paths, _ = list_apks_in_group(root_dir, gid, gexf_name=gexf_name, apk_subdirs=apk_subdirs)
                for i, j in itertools.combinations(range(len(apk_paths)), 2):
                    pairs.append((apk_paths[i], apk_paths[j], 1, gid))
        else:
            apk_paths, _ = list_apks_in_group(root_dir, gid, gexf_name=gexf_name, apk_subdirs=apk_subdirs)
            for i, j in itertools.combinations(range(len(apk_paths)), 2):
                pairs.append((apk_paths[i], apk_paths[j], 1, gid))
    return pairs

def build_negative_pairs(root_dir, selected_groups, num_needed, rng, gexf_name='community_processed_graph.gexf', apk_subdirs=None):
    # 聚合所选组的所有APK，并提供各自的组ID，随机采样跨组对
    all_apks = []
    for gid in selected_groups:
        paths, _ = list_apks_in_group(root_dir, gid, gexf_name=gexf_name, apk_subdirs=apk_subdirs)
        for p in paths:
            all_apks.append((gid, p))
    neg_pairs = []
    if len(all_apks) < 2:
        return neg_pairs
    attempts = 0
    max_attempts = num_needed * 20
    while len(neg_pairs) < num_needed and attempts < max_attempts:
        attempts += 1
        (g1, p1), (g2, p2) = rng.choice(all_apks), rng.choice(all_apks)
        if g1 == g2:
            continue
        neg_pairs.append((p1, p2, 0, f"{g1}|{g2}"))
    return neg_pairs

def evaluate_pairs(analyzer, pairs, threshold=0.85, show_progress=True, sinkhorn_reg=None, prefilter_margin=0.0):
    tp = tn = fp = fn = 0
    total_time = 0.0
    results = []
    it = tqdm(pairs, desc='SFCG配对检测', unit='pair') if show_progress else pairs
    for p1, p2, label, group_info in it:
        t0 = time.time()
        # 两阶段预筛选：若启用且近似与阈值差距超过 margin，则直接判定
        f1 = analyzer.features_cache.get(p1)
        f2 = analyzer.features_cache.get(p2)
        do_full_ot = True
        if prefilter_margin is not None and prefilter_margin > 0 and f1 is not None and f2 is not None:
            approx_sim = analyzer.quick_prefilter_similarity(f1, f2, metric='euclidean')
            if approx_sim >= (threshold + prefilter_margin) or approx_sim <= (threshold - prefilter_margin):
                sim = approx_sim
                do_full_ot = False
            else:
                sim = analyzer.calculate_ot_similarity(p1, p2, metric='euclidean', sinkhorn_reg=sinkhorn_reg)
        else:
            sim = analyzer.calculate_ot_similarity(p1, p2, metric='euclidean', sinkhorn_reg=sinkhorn_reg)
        dt = time.time() - t0
        total_time += dt
        pred = 1 if sim >= threshold else 0
        pair_tp = pair_fp = pair_tn = pair_fn = 0
        if label == 1 and pred == 1:
            tp += 1
            pair_tp = 1
        elif label == 0 and pred == 0:
            tn += 1
            pair_tn = 1
        elif label == 0 and pred == 1:
            fp += 1
            pair_fp = 1
        elif label == 1 and pred == 0:
            fn += 1
            pair_fn = 1
        results.append({
            'apk1': p1,
            'apk2': p2,
            'apk1_name': os.path.basename(p1),
            'apk2_name': os.path.basename(p2),
            'group_info': group_info,
            'label': label,
            'similarity_score': sim,
            'detect_time_s': dt,
            'used_full_ot': 1 if do_full_ot else 0,
            'pred': pred,
            'tp': pair_tp,
            'fp': pair_fp,
            'tn': pair_tn,
            'fn': pair_fn,
        })
    if isinstance(it, tqdm):
        it.close()
    total_pairs = len(results)
    accuracy = (tp + tn) / max(1, total_pairs)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_time = total_time / max(1, total_pairs)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'f1': f1,
        'total_time_s': total_time,
        'avg_time_per_pair_s': avg_time,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'results': results,
        'total_pairs': total_pairs,
    }

def run_sfcg_detection(args):
    """
    SFCG特征检测入口函数
    :param args: 包含所有参数的命名空间对象
    """
    # 输出路径位于 ./test/ 目录
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, 'sfcg_pairwise_results.csv')
    metrics_excel = os.path.join(out_dir, 'sfcg_pairwise_metrics.xlsx')
    metrics_json = os.path.join(out_dir, 'sfcg_pairwise_metrics.json')

    # 初始化分析器
    analyzer = SFCG_OT_ThresholdAnalyzer(args.root, gexf_name=args.gexf_name)

    # 组列表与抽样
    try:
        all_groups = list_groups(args.root)
    except FileNotFoundError:
        print(f"[SFCG] 错误：基础目录 {args.root} 不存在。请确保路径正确。")
        return None
    if not all_groups:
        print("[SFCG] 未在基础目录中找到任何组。")
        return None

    rng = random.Random(args.seed)
    rng.shuffle(all_groups)
    sample_n = max(1, math.ceil(len(all_groups) * args.sample_fraction))
    selected_groups = all_groups[:sample_n]
    print(f"[SFCG] 总组数: {len(all_groups)}，抽样: {sample_n} 组。")

    # 解析子目录参数
    apk_subdirs = [s.strip() for s in args.apk_subdirs.split(',') if s.strip()] if args.apk_subdirs else None
    # 构建正样本（组内两两配对或 original x repack）
    pos_pairs = build_positive_pairs(args.root, selected_groups, gexf_name=args.gexf_name, apk_subdirs=apk_subdirs)
    print(f"[SFCG] 正样本对数量: {len(pos_pairs)}")

    # 构建负样本（跨组随机配对），数量与正样本相同
    neg_pairs = build_negative_pairs(args.root, selected_groups, len(pos_pairs), rng, gexf_name=args.gexf_name, apk_subdirs=apk_subdirs)
    print(f"[SFCG] 负样本对数量: {len(neg_pairs)}")

    all_pairs = pos_pairs + neg_pairs
    if len(all_pairs) == 0:
        print("[SFCG] 没有可检测的样本对，退出。")
        return None

    # 预加载特征缓存（可选节点子采样提速）
    apk_set = set([p for p, _, _, _ in pos_pairs] + [q for _, q, _, _ in pos_pairs] +
                  [p for p, _, _, _ in neg_pairs] + [q for _, q, _, _ in neg_pairs])
    loaded = analyzer.preload_apk_features(list(apk_set), max_nodes=args.max_nodes if args.max_nodes and args.max_nodes > 0 else None)
    print(f"[SFCG] 预加载特征数量: {loaded} APK")

    show_progress = not args.no_progress
    eval_res = evaluate_pairs(
        analyzer,
        all_pairs,
        threshold=args.threshold,
        show_progress=show_progress,
        sinkhorn_reg=(args.sinkhorn_reg if args.sinkhorn_reg and args.sinkhorn_reg > 0 else None),
        prefilter_margin=(args.prefilter_margin if args.prefilter_margin and args.prefilter_margin > 0 else 0.0)
    )

    # 保存结果CSV
    results_df = pd.DataFrame(eval_res['results'])
    results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')

    # 汇总指标（包含两个关键值：召回与误报率）
    metrics = {
        'threshold': args.threshold,
        'gexf_name': args.gexf_name,
        'apk_subdirs': apk_subdirs if apk_subdirs else [],
        'num_groups_total': len(all_groups),
        'num_groups_sampled': len(selected_groups),
        'num_pairs_total': eval_res['total_pairs'],
        'num_pairs_positive': len(pos_pairs),
        'num_pairs_negative': len(neg_pairs),
        'num_pairs_full_ot': int(pd.DataFrame(eval_res['results'])['used_full_ot'].sum()) if eval_res['results'] else 0,
        'num_pairs_prefilter_only': int(eval_res['total_pairs']) - int(pd.DataFrame(eval_res['results'])['used_full_ot'].sum()) if eval_res['results'] else 0,
        'test_recall': eval_res['recall'],
        'test_false_positive_rate': eval_res['fpr'],
        'test_precision': eval_res['precision'],
        'test_f1': eval_res['f1'],
        'test_accuracy': eval_res['accuracy'],
        'confusion_tp': eval_res['tp'],
        'confusion_fp': eval_res['fp'],
        'confusion_tn': eval_res['tn'],
        'confusion_fn': eval_res['fn'],
        'detect_total_s': eval_res['total_time_s'],
        'detect_avg_per_pair_s': eval_res['avg_time_per_pair_s'],
        'results_csv': results_csv,
    }

    # 保存指标到 Excel 与 JSON
    with pd.ExcelWriter(metrics_excel, engine='openpyxl') as writer:
        pd.DataFrame([metrics]).to_excel(writer, sheet_name='metrics_overall', index=False)
        results_df.to_excel(writer, sheet_name='pairs', index=False)
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 控制台输出关键指标
    print(f"\n=== SFCG配对阈值评估完成 ===")
    print(f"阈值: {args.threshold}")
    print(f"召回率: {metrics['test_recall']:.4f}")
    print(f"误报率: {metrics['test_false_positive_rate']:.4f}")
    print(f"精确率: {metrics['test_precision']:.4f}")
    print(f"F1: {metrics['test_f1']:.4f}")
    print(f"准确率: {metrics['test_accuracy']:.4f}")
    print(f"总样本对: {metrics['num_pairs_total']}")
    print(f"检测总时长: {metrics['detect_total_s']:.2f} s, 平均每对: {metrics['detect_avg_per_pair_s']:.4f} s")
    if args.sinkhorn_reg and args.sinkhorn_reg > 0:
        print(f"启用Sinkhorn加速，reg={args.sinkhorn_reg}")
    if args.max_nodes and args.max_nodes > 0:
        print(f"启用节点子采样，上限={args.max_nodes}")
    if args.prefilter_margin and args.prefilter_margin > 0:
        print(f"启用两阶段预筛选，margin={args.prefilter_margin}")
    print(f"结果CSV: {results_csv}")
    print(f"指标: {metrics_excel} / {metrics_json}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='SFCG配对阈值检测（抽样1/4组，无需训练）')
    parser.add_argument('--root', type=str, required=True, help='AndroZoo解包数据根目录')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), ''), help='输出目录，默认脚本所在test目录')
    parser.add_argument('--threshold', type=float, default=0.85, help='相似度阈值，用于召回/误报评估')
    parser.add_argument('--sample_fraction', type=float, default=0.25, help='抽样组的比例，默认1/4')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，保证可复现抽样')
    parser.add_argument('--no_progress', action='store_true', help='禁用进度条')
    parser.add_argument('--gexf_name', type=str, default='community_processed_graph.gexf', help='SFCG特征文件名（仅文件名变化时使用）')
    parser.add_argument('--apk_subdirs', type=str, default='', help='组内子目录名（逗号分隔），如 original_apk,repack_apk')
    parser.add_argument('--sinkhorn_reg', type=float, default=0.0, help='Sinkhorn熵正则系数，>0 启用加速近似')
    parser.add_argument('--max_nodes', type=int, default=0, help='每APK节点最大数，>0 启用随机子采样以提速')
    parser.add_argument('--prefilter_margin', type=float, default=0.0, help='两阶段预筛选margin，>0 启用近似快速判定')
    args = parser.parse_args()
    run_sfcg_detection(args)

if __name__ == '__main__':
    main()