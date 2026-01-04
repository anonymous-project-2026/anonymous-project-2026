import argparse
import os
import sys
import json
import pandas as pd
from datetime import datetime

# 导入各特征检测模块的入口函数
# 确保当前目录在 sys.path 中，以便能导入同目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from so_detection import run_so_detection
    from icon_detection import run_icon_detection
    from omm_detection import run_omm_detection
    from sfcg_detection import run_sfcg_detection
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保 so_detection.py, icon_detection.py, omm_detection.py, sfcg_detection.py 在同一目录下。")
    sys.exit(1)

def get_arg_parser():
    parser = argparse.ArgumentParser(description="多特征融合检测入口脚本")
    
    # 全局通用参数
    parser.add_argument('--root', type=str, default='/newdisk/liuzhuowu/lzw/apks_androzoo', help='包含所有APK家族子文件夹的根目录')
    parser.add_argument('--output_dir', type=str, default='/newdisk/liuzhuowu/baseline/androzoo_result', help='结果输出根目录')
    parser.add_argument('--intermediate_dir', type=str, default='/newdisk/liuzhuowu/baseline/temp', help='中间文件目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--workers', type=int, default=4, help='全局默认数据加载线程数 (若各模块未指定则使用此值)')
    parser.add_argument('--batch_size', type=int, default=32, help='全局默认批处理大小 (若各模块未指定则使用此值)')
    parser.add_argument('--no_progress', action='store_true', help='关闭进度条显示')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='采样比例 (0.0-1.0]')
    
    # 模块开关
    parser.add_argument('--skip_so', action='store_true', help='跳过 SO 特征检测')
    parser.add_argument('--skip_icon', action='store_true', help='跳过 Icon 特征检测')
    parser.add_argument('--skip_omm', action='store_true', help='跳过 OMM 特征检测')
    parser.add_argument('--skip_sfcg', action='store_true', help='跳过 SFCG 特征检测')
    
    # 各模块特定参数
    # SO
    parser.add_argument('--so_epochs', type=int, default=5, help='SO Epochs')
    parser.add_argument('--so_batch_size', type=int, default=32, help='SO Batch Size')
    parser.add_argument('--so_lr', type=float, default=1e-4, help='SO Learning Rate')
    parser.add_argument('--so_threshold', type=float, default=0.85, help='SO Threshold')
    parser.add_argument('--so_workers', type=int, default=0, help='SO Workers')
    
    # Icon
    parser.add_argument('--icon_sample_fraction', type=float, default=0.25, help='Icon Sample Fraction')
    parser.add_argument('--icon_threshold', type=float, default=0.6, help='Icon Threshold')
    parser.add_argument('--icon_workers', type=int, default=8, help='Icon Workers')
    
    # OMM
    parser.add_argument('--omm_epochs', type=int, default=5, help='OMM Epochs')
    parser.add_argument('--omm_batch_size', type=int, default=64, help='OMM Batch Size')
    parser.add_argument('--omm_lr', type=float, default=0.001, help='OMM Learning Rate')
    parser.add_argument('--omm_threshold', type=float, default=0.85, help='OMM Threshold')
    parser.add_argument('--omm_workers', type=int, default=4, help='OMM Workers')

    # SFCG
    parser.add_argument('--sfcg_gexf_name', type=str, default='community_processed_graph.gexf', help='SFCG GEXF Name')
    parser.add_argument('--sfcg_apk_subdirs', type=str, default='original_apk,repack_apk', help='SFCG APK Subdirs')
    parser.add_argument('--sfcg_sinkhorn_reg', type=float, default=0.1, help='SFCG Sinkhorn Reg')
    parser.add_argument('--sfcg_max_nodes', type=int, default=2000, help='SFCG Max Nodes')
    parser.add_argument('--sfcg_prefilter_margin', type=float, default=0.05, help='SFCG Prefilter Margin')
    parser.add_argument('--sfcg_threshold', type=float, default=0.85, help='SFCG Threshold')
    parser.add_argument('--sfcg_output_dir', type=str, default='./test', help='SFCG Output Dir')
    
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # 准备总输出目录 (使用时间戳，避免覆盖)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 如果用户没有指定 --output_dir 为默认值，则使用用户指定的；否则使用带有时间戳的目录以防冲突，或者直接使用用户指定的目录
    # 这里我们直接使用用户指定的 output_dir 作为 base，然后里面分 so_results, omm_results 等
    # 但为了不混淆多次运行，建议还是加一层 timestamp 或者 run_id，或者直接覆盖
    # 根据用户习惯，可能是直接覆盖或追加。为了安全，我们还是在 output_dir 下建个子目录，或者直接用 output_dir
    # 鉴于用户给的是 /newdisk/.../androzoo_result，这像是一个总目录。
    # 我们这里保持原有逻辑：在 output_dir 下创建一个 run_xxx 目录，或者直接输出到 output_dir/feature_name
    # 用户之前的脚本是直接输出到 androzoo_result/so, androzoo_result/omm 等。
    # 我们尽量贴合这个逻辑。
    
    # 既然是集成脚本，我们优先使用 args.output_dir 作为 base
    # 但为了区分不同次运行，通常会有 run_id。
    # 如果用户希望完全复刻原来的行为（覆盖式），则不需要 run_id。
    # 暂时保持 run_timestamp 逻辑，但 base 是用户指定的。
    
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    summary_metrics = {}
    
    print(f"=== 开始多特征检测任务 ===")
    print(f"根目录: {args.root}")
    print(f"输出目录: {run_output_dir}")
    print("=" * 30)

    # 1. SO 特征检测
    if not args.skip_so:
        print("\n>>> 正在运行 SO 特征检测...")
        try:
            # 构造 SO 模块需要的参数
            # 注意：so_detection.py 需要 root, result_root, intermediate_root 等
            # 使用 args.so_* 参数覆盖
            so_args = argparse.Namespace(
                root=args.root,
                result_root=os.path.join(run_output_dir, 'so'),
                intermediate_root=os.path.join(args.intermediate_dir, 'so') if args.intermediate_dir else os.path.join(run_output_dir, 'intermediate', 'so'),
                batch_size=args.so_batch_size,
                workers=args.so_workers,
                lr=args.so_lr, 
                epochs=args.so_epochs, 
                seed=args.seed,
                no_progress=args.no_progress,
                mode='train_test', 
                model_path=None,   
                threshold=args.so_threshold,
                margin=0.5,        
                neg_weight=2.0,    
                auto_threshold=False,
                min_recall=0.85,
                target_precision=0.95
            )
            metrics = run_so_detection(so_args)
            summary_metrics['so'] = metrics
            print(f"[SO] 完成。Metrics: {metrics}")
        except Exception as e:
            print(f"[SO] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            summary_metrics['so'] = {'error': str(e)}
    else:
        print("\n>>> 跳过 SO 特征检测")

    # 2. Icon 特征检测
    if not args.skip_icon:
        print("\n>>> 正在运行 Icon 特征检测...")
        try:
            # 构造 Icon 模块需要的参数
            icon_args = argparse.Namespace(
                root=args.root,
                result_root=run_output_dir, # Icon 脚本内部会追加 /image
                intermediate_root=args.intermediate_dir if args.intermediate_dir else os.path.join(run_output_dir, 'intermediate'),
                batch_size=args.batch_size, # Icon 模块没有指定 batch_size 参数，沿用全局 default
                workers=args.icon_workers,
                seed=args.seed,
                threshold=args.icon_threshold,
                no_progress=args.no_progress,
                sample_fraction=args.icon_sample_fraction
            )
            metrics = run_icon_detection(icon_args)
            summary_metrics['icon'] = metrics
            print(f"[Icon] 完成。Metrics: {metrics}")
        except Exception as e:
            print(f"[Icon] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            summary_metrics['icon'] = {'error': str(e)}
    else:
        print("\n>>> 跳过 Icon 特征检测")

    # 3. OMM 特征检测
    if not args.skip_omm:
        print("\n>>> 正在运行 OMM 特征检测...")
        try:
            # 构造 OMM 模块需要的参数
            omm_args = argparse.Namespace(
                root=args.root,
                result_root=os.path.join(run_output_dir, 'omm'),
                intermediate_root=os.path.join(args.intermediate_dir, 'omm') if args.intermediate_dir else os.path.join(run_output_dir, 'intermediate', 'omm'),
                batch_size=args.omm_batch_size,
                workers=args.omm_workers,
                lr=args.omm_lr,
                epochs=args.omm_epochs,
                seed=args.seed,
                no_progress=args.no_progress,
                sample_fraction=1.0 # OMM 模块未指定 sample_fraction，默认全量或使用 args.sample_fraction? 鉴于用户未指定 omm 的 sample-fraction，这里设为 1.0 或沿用 global
            )
            # 用户之前 OMM 命令没指定 sample_fraction，所以默认全量。
            # 这里我把 sample_fraction 加上去，防止 omm_detection 报错（如果我之前改了需要这个参数的话）
            if hasattr(args, 'sample_fraction'):
                 omm_args.sample_fraction = args.sample_fraction # 复用全局的，或者默认为1.0
            
            # 修正：用户命令没指定 omm 的 sample_fraction，但指定了 icon 的。
            # 全局 args.sample_fraction 默认为 1.0。
            # 如果想让 OMM 也不采样，就用 1.0。
            omm_args.sample_fraction = 1.0 

            metrics = run_omm_detection(omm_args)
            summary_metrics['omm'] = metrics
            print(f"[OMM] 完成。Metrics: {metrics}")
        except Exception as e:
            print(f"[OMM] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            summary_metrics['omm'] = {'error': str(e)}
    else:
        print("\n>>> 跳过 OMM 特征检测")

    # 4. SFCG 特征检测
    if not args.skip_sfcg:
        print("\n>>> 正在运行 SFCG 特征检测...")
        try:
            # 构造 SFCG 模块需要的参数
            sfcg_args = argparse.Namespace(
                root=args.root,
                output_dir=args.sfcg_output_dir if args.sfcg_output_dir else os.path.join(run_output_dir, 'sfcg_results'),
                gexf_name=args.sfcg_gexf_name, 
                seed=args.seed,
                sample_fraction=args.sample_fraction,
                apk_subdirs=args.sfcg_apk_subdirs,
                max_nodes=args.sfcg_max_nodes,
                threshold=args.sfcg_threshold,
                sinkhorn_reg=args.sfcg_sinkhorn_reg,
                prefilter_margin=args.sfcg_prefilter_margin,
                no_progress=args.no_progress
            )
            metrics = run_sfcg_detection(sfcg_args)
            summary_metrics['sfcg'] = metrics
            print(f"[SFCG] 完成。Metrics: {metrics}")
        except Exception as e:
            print(f"[SFCG] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            summary_metrics['sfcg'] = {'error': str(e)}
    else:
        print("\n>>> 跳过 SFCG 特征检测")

    # 汇总报告
    print("\n" + "=" * 30)
    print("=== 多特征检测任务汇总 ===")
    
    summary_file = os.path.join(run_output_dir, 'summary.json')
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_metrics, f, indent=4, ensure_ascii=False)
        print(f"汇总报告已保存至: {summary_file}")
    except Exception as e:
        print(f"保存汇总报告失败: {e}")

    # 简单的控制台输出
    for feature, result in summary_metrics.items():
        print(f"\n[{feature.upper()}] Results:")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {result}")

if __name__ == '__main__':
    main()
