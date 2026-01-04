import os
import sys
import shutil
import subprocess
import argparse
import re
import csv
import difflib
import time
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入本地特征模块
import icon_extractor
import omm_extractor
import sfcg_extractor
import sfcg_enhance
import so_preprocessor
import so_extractor
from feature_config import FILTER_LIBRARIES

def decompile_apk(apk_path, output_dir):
    """使用apktool反编译APK文件"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建apktool命令
        apk_name = os.path.splitext(os.path.basename(apk_path))[0]
        decompiled_path = os.path.join(output_dir, apk_name)
        
        # 若已存在反编译目录，直接复用以支持断点续跑
        if os.path.exists(decompiled_path):
            print(f"已存在反编译目录，跳过反编译: {decompiled_path}")
            return decompiled_path
        
        # 执行apktool反编译
        cmd = ["apktool", "d", apk_path, "-o", decompiled_path, "-f"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"APK反编译成功: {apk_path}")
            
            # 预处理：删除第三方库smali文件
            _preprocess_remove_libs(decompiled_path)
            
            return decompiled_path
        else:
            print(f"APK反编译失败: {apk_path}, 错误信息: {result.stderr}")
            return None
    except Exception as e:
        print(f"反编译APK时出错 {apk_path}: {e}")
        return None

def _preprocess_remove_libs(decompiled_path):
    """
    预处理：删除第三方库的 smali 文件
    """
    # 加载过滤规则
    libs = [lib.replace('.', os.sep) for lib in FILTER_LIBRARIES]
    
    if not libs:
        return

    print("  正在执行Smali预过滤(删除第三方库)...")
    count = 0
    
    # 遍历所有 smali* 目录
    for root_dir in os.listdir(decompiled_path):
        if not root_dir.startswith('smali'):
            continue
            
        full_root = os.path.join(decompiled_path, root_dir)
        if not os.path.isdir(full_root):
            continue
            
        # 检查每个库
        for lib_path in libs:
            target_dir = os.path.join(full_root, lib_path)
            if os.path.exists(target_dir):
                try:
                    shutil.rmtree(target_dir)
                    count += 1
                except:
                    pass
    
    if count > 0:
        print(f"  已移除 {count} 个第三方库目录")

def build_all_smali(decompiled_path):
    all_smali_dir = os.path.join(decompiled_path, 'all_smali')
    if os.path.exists(all_smali_dir):
        try:
            shutil.rmtree(all_smali_dir)
        except Exception:
            pass
    os.makedirs(all_smali_dir, exist_ok=True)
    libs = [lib.replace('\\', '/').strip('/') for lib in FILTER_LIBRARIES]
    sources = [d for d in os.listdir(decompiled_path) if d.startswith('smali') and os.path.isdir(os.path.join(decompiled_path, d))]
    copied = 0
    for src in sources:
        src_root = os.path.join(decompiled_path, src)
        for root, _, files in os.walk(src_root):
            for f in files:
                if not f.endswith('.smali'):
                    continue
                abs_path = os.path.join(root, f)
                rel = os.path.relpath(abs_path, src_root).replace('\\', '/')
                skip = False
                for lib in libs:
                    if rel.startswith(lib):
                        skip = True
                        break
                if skip:
                    continue
                dest_path = os.path.join(all_smali_dir, rel)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                try:
                    shutil.copy2(abs_path, dest_path)
                    copied += 1
                except Exception:
                    continue
    return all_smali_dir, copied

def process_icon_feature(apk_folder, output_folder):
    """处理图标特征"""
    try:
        # 创建输出目录
        icon_output_dir = os.path.join(output_folder, "icon")
        os.makedirs(icon_output_dir, exist_ok=True)
        
        # 调用图标处理函数
        icon_file = icon_extractor.collect_apk_icons(apk_folder)
        if icon_file:
            # 复制图标文件到输出目录
            dest_path = os.path.join(icon_output_dir, os.path.basename(icon_file))
            shutil.copy2(icon_file, dest_path)
            print(f"图标特征处理完成: {apk_folder}")
            return True
        else:
            print(f"未找到图标: {apk_folder}")
            return False
    except Exception as e:
        print(f"处理图标特征时出错 {apk_folder}: {e}")
        return False

def process_omm_feature(apk_folder, output_folder):
    """处理OMM特征"""
    try:
        # 创建输出目录
        omm_output_dir = os.path.join(output_folder, "omm")
        os.makedirs(omm_output_dir, exist_ok=True)
        
        # OMM处理需要的文件路径
        excel_path = os.path.join(current_dir, "res", "omm_opcode.xlsx")
        smali_directory = os.path.join(apk_folder, "all_smali")
        
        # 如果没有all_smali目录，使用默认的smali目录
        if not os.path.exists(smali_directory):
            smali_directory = os.path.join(apk_folder, "smali")
        
        # 检查smali目录是否存在
        if not os.path.exists(smali_directory):
            print(f"Smali目录不存在: {smali_directory}")
            return False
            
        # 创建处理器实例并处理
        # 注意：OMMProcess 接受 excel_path
        analyzer = omm_extractor.OMMProcess(excel_path)
        # process_directory 接受 smali_directory 和 output_directory
        analyzer.process_directory(smali_directory, omm_output_dir)
        
        # 检查是否生成了文件
        generated_file = os.path.join(omm_output_dir, "dalvik.npy")
        if os.path.exists(generated_file):
            print(f"OMM特征处理完成: {apk_folder}")
            return True
        else:
            print(f"OMM特征文件未生成: {apk_folder}")
            return False
    except Exception as e:
        print(f"处理OMM特征时出错 {apk_folder}: {e}")
        return False

def process_sfcg_feature(apk_folder, output_folder):
    """处理SFCG特征"""
    try:
        # 创建输出目录
        sfcg_output_dir = os.path.join(output_folder, "sfcg")
        os.makedirs(sfcg_output_dir, exist_ok=True)
        
        # 加载必要的文件
        entities_txt = os.path.join(current_dir, "res", "sfcg_entities.txt")
        entity_embedding_pkl = os.path.join(current_dir, "res", "sfcg_embeddings.pkl")
        
        # 检查必要文件是否存在
        required_files = [entities_txt, entity_embedding_pkl]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"缺少必要文件: {file_path}")
                return False
        
        # 加载数据
        s_nodes_txt = sfcg_enhance.read_nodes_from_txt(entities_txt)
        s_nodes_with_vectors = sfcg_enhance.load_nodes_with_vectors(entity_embedding_pkl)
        
        # 创建处理器实例
        processor = sfcg_extractor.SfcgProcessor(
            directory=apk_folder,
            s_nodes_txt=s_nodes_txt,
            s_nodes_with_vectors=s_nodes_with_vectors,
            nodes_txt=s_nodes_txt,
            nodes_with_vectors=s_nodes_with_vectors
        )
        
        # 运行处理
        processor.run(sfcg_output_dir)
        
        print(f"SFCG特征处理完成: {apk_folder}")
        return True
    except Exception as e:
        print(f"处理SFCG特征时出错 {apk_folder}: {e}")
        return False

def process_so_feature(apk_folder, output_folder):
    """处理SO特征"""
    try:
        # 创建输出目录
        so_output_dir = os.path.join(output_folder, "so")
        os.makedirs(so_output_dir, exist_ok=True)
        
        # 首先提取SO文件
        if not so_preprocessor.extract_so_files(apk_folder):
            print(f"提取SO文件失败: {apk_folder}")
            return False
            
        # SO处理需要的文件路径
        excel_path = os.path.join(current_dir, "res", "so_arm_opcode.xlsx")
        
        # 创建处理器实例并处理单个APK
        analyzer = so_extractor.SoOMMProcess(excel_path, "")
        success = analyzer.process_single_apk(apk_folder)
        
        if success:
            # 将生成的文件移动到输出目录
            generated_file = os.path.join(apk_folder, "transition_probabilities.npy")
            if os.path.exists(generated_file):
                dest_path = os.path.join(so_output_dir, "transition_probabilities.npy")
                shutil.move(generated_file, dest_path)
            print(f"SO特征处理完成: {apk_folder}")
            return True
        else:
            print(f"SO特征处理失败: {apk_folder}")
            return False
    except Exception as e:
        print(f"处理SO特征时出错 {apk_folder}: {e}")
        return False

def process_single_apk(apk_file, output_base_dir, decompiled_base_dir, enable_timing=False, include_size=False):
    """处理单个APK文件，生成所有特征"""
    try:
        # 计时数据初始化
        timing_data = {}
        start_total = time.time()
        
        # 获取APK文件名（不含扩展名）
        apk_name = os.path.splitext(os.path.basename(apk_file))[0]
        timing_data['apk_name'] = apk_name
        
        # 统计APK大小（如果启用）
        if include_size:
            try:
                size_mb = os.path.getsize(apk_file) / (1024 * 1024)
                timing_data['apk_size_mb'] = f"{size_mb:.2f}"
            except Exception:
                timing_data['apk_size_mb'] = '/'

        output_folder = os.path.join(output_base_dir, apk_name)
        
        # 反编译APK
        t0 = time.time()
        decompiled_folder = decompile_apk(apk_file, decompiled_base_dir)
        decompile_duration = time.time() - t0
        
        if decompiled_folder:
            timing_data['decompile_time'] = decompile_duration
        else:
            timing_data['decompile_time'] = '/'
            # 如果反编译失败，后续步骤无法进行，全部标记为/
            timing_data['icon_time'] = '/'
            timing_data['filter_smali_time'] = '/'
            timing_data['omm_matrix_time'] = '/'
            timing_data['sfcg_graph_time'] = '/'
            timing_data['so_disasm_time'] = '/'
            timing_data['so_matrix_time'] = '/'
            timing_data['so_time'] = '/'
            timing_data['total_time'] = time.time() - start_total
            if enable_timing:
                return timing_data
            else:
                return False

        results = []
        
        # Icon特征
        t0 = time.time()
        # process_icon_feature 内部会处理缺失情况：如果未找到图标，返回False
        if process_icon_feature(decompiled_folder, output_folder):
            timing_data['icon_time'] = time.time() - t0
            results.append(True)
        else:
            # 缺失或处理失败，标记为失败，时间记为/
            timing_data['icon_time'] = '/'
            results.append(False)

        # Smali过滤与聚合
        t0 = time.time()
        all_smali_dir, _ = build_all_smali(decompiled_folder)
        timing_data['filter_smali_time'] = time.time() - t0

        # OMM特征
        t0 = time.time()
        omm_out = os.path.join(output_folder, "omm", "dalvik.npy")
        if os.path.exists(omm_out):
            print(f"OMM矩阵已存在，跳过OMM特征: {apk_name}")
            timing_data['omm_matrix_time'] = 0
            # 如果是跳过，通常意味着成功（之前成功了）。但这里是计时。
            # 如果跳过，时间几乎为0。保留为0即可。
            results.append(True)
        else:
            omm_output_dir = os.path.join(output_folder, "omm")
            os.makedirs(omm_output_dir, exist_ok=True)
            excel_path = os.path.join(current_dir, "res", "omm_opcode.xlsx")
            smali_directory = all_smali_dir
            if os.path.exists(smali_directory):
                analyzer = omm_extractor.OMMProcess(excel_path)
                analyzer.process_directory(smali_directory, omm_output_dir)
                timing_data['omm_matrix_time'] = time.time() - t0
                results.append(True)
            else:
                timing_data['omm_matrix_time'] = '/'
                results.append(False)

        # SFCG特征
        t0 = time.time()
        sfcg_output_dir = os.path.join(output_folder, "sfcg")
        os.makedirs(sfcg_output_dir, exist_ok=True)
        sfcg_gexf = os.path.join(sfcg_output_dir, "community_processed_graph.gexf")
        entities_txt = os.path.join(current_dir, "res", "sfcg_entities.txt")
        entity_embedding_pkl = os.path.join(current_dir, "res", "sfcg_embeddings.pkl")
        if os.path.exists(sfcg_gexf):
             print(f"SFCG图已存在，跳过SFCG特征: {apk_name}")
             timing_data['sfcg_graph_time'] = 0
             results.append(True)
        elif os.path.exists(entities_txt) and os.path.exists(entity_embedding_pkl):
             s_nodes_txt = sfcg_enhance.read_nodes_from_txt(entities_txt)
             s_nodes_with_vectors = sfcg_enhance.load_nodes_with_vectors(entity_embedding_pkl)
             processor = sfcg_extractor.SfcgProcessor(
                 directory=all_smali_dir,
                 s_nodes_txt=s_nodes_txt,
                 s_nodes_with_vectors=s_nodes_with_vectors,
                 nodes_txt=s_nodes_txt,
                 nodes_with_vectors=s_nodes_with_vectors
             )
             processor.run(sfcg_output_dir)
             timing_data['sfcg_graph_time'] = time.time() - t0
             results.append(True)
        else:
             timing_data['sfcg_graph_time'] = '/'
             results.append(False)

        # SO特征
        t0 = time.time()
        so_out = os.path.join(output_folder, "so", "transition_probabilities.npy")
        if os.path.exists(so_out):
            print(f"SO矩阵已存在，跳过SO特征: {apk_name}")
            timing_data['so_disasm_time'] = 0
            timing_data['so_matrix_time'] = 0
            timing_data['so_time'] = 0
            results.append(True)
        else:
            so_output_dir = os.path.join(output_folder, "so")
            os.makedirs(so_output_dir, exist_ok=True)
            t_dis = time.time()
            if so_preprocessor.extract_so_files(decompiled_folder):
                excel_path = os.path.join(current_dir, "res", "so_arm_opcode.xlsx")
                analyzer = so_extractor.SoOMMProcess(excel_path, "")
                so_directory = os.path.join(decompiled_folder, "so_files")
                txt_directory = os.path.join(decompiled_folder, "so_txt")
                if os.path.exists(txt_directory):
                    try:
                        shutil.rmtree(txt_directory)
                    except Exception:
                        pass
                os.makedirs(txt_directory, exist_ok=True)
                try:
                    analyzer.extract_disassemblies_from_folder(so_directory, txt_directory)
                    timing_data['so_disasm_time'] = time.time() - t_dis
                    t_mat = time.time()
                    analyzer.analyze_directory(txt_directory)
                    probs = analyzer.calculate_transition_probabilities()
                    analyzer.save_transition_probabilities(probs, so_output_dir)
                    timing_data['so_matrix_time'] = time.time() - t_mat
                    timing_data['so_time'] = time.time() - t0
                    results.append(True)
                except Exception:
                    timing_data['so_disasm_time'] = '/'
                    timing_data['so_matrix_time'] = '/'
                    timing_data['so_time'] = '/'
                    results.append(False)
            else:
                timing_data['so_disasm_time'] = '/'
                timing_data['so_matrix_time'] = '/'
                timing_data['so_time'] = '/'
                results.append(False)
        
        success_count = sum(results)
        # print(f"APK {apk_name} 处理完成 ({success_count}/{len(results)} 特征成功)")
        
        timing_data['total_time'] = time.time() - start_total
        
        if enable_timing:
            return timing_data
        else:
            return True
    except Exception as e:
        print(f"处理APK时出错 {apk_file}: {e}")
        return False if not enable_timing else None

def main(input_apk_dir, output_dir, decompiled_dir, max_workers=4, enable_timing=False, include_size=False):
    """
    主函数：处理APK文件，为每个APK反编译并生成四个特征文件
    
    参数:
    input_apk_dir: 包含APK文件的输入目录
    output_dir: 输出目录，将按照APK分别存放特征文件
    decompiled_dir: 反编译文件的存储目录
    max_workers: 最大并发处理数
    enable_timing: 是否启用计时并保存结果
    include_size: 是否统计APK大小
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_apk_dir):
        print(f"输入目录不存在: {input_apk_dir}")
        return
    
    # 创建输出目录和反编译目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(decompiled_dir, exist_ok=True)
    
    # 获取所有APK文件路径（默认全量）
    apk_files = []
    for item in os.listdir(input_apk_dir):
        item_path = os.path.join(input_apk_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith('.apk'):
            apk_files.append(item_path)
    print(f"找到 {len(apk_files)} 个APK文件（默认全量）")
    
    # 断点续跑预过滤：若目标输出目录下OMM与SO矩阵均存在，则跳过该APK
    apk_files_to_process = []
    skipped_count = 0
    for apk_path in apk_files:
        apk_name = os.path.splitext(os.path.basename(apk_path))[0]
        out_apk_dir = os.path.join(output_dir, apk_name)
        omm_done = os.path.exists(os.path.join(out_apk_dir, "omm", "dalvik.npy"))
        so_done = os.path.exists(os.path.join(out_apk_dir, "so", "transition_probabilities.npy"))
        if omm_done and so_done:
            skipped_count += 1
        else:
            apk_files_to_process.append(apk_path)
    print(f"断点续跑过滤：将处理 {len(apk_files_to_process)} 个APK，跳过 {skipped_count} 个已完成APK")
    
    # 用于收集计时结果
    timing_results = []
    
    # 使用线程池并发处理APK
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_apk = {
            executor.submit(process_single_apk, apk_file, output_dir, decompiled_dir, enable_timing, include_size): apk_file 
            for apk_file in apk_files_to_process
        }
        
        # 处理结果
        if tqdm:
            iterator = tqdm(as_completed(future_to_apk), total=len(apk_files_to_process), desc="处理APK进度")
        else:
            iterator = as_completed(future_to_apk)
            
        for future in iterator:
            apk_file = future_to_apk[future]
            try:
                result = future.result()
                if enable_timing and isinstance(result, dict):
                    timing_results.append(result)
            except Exception as e:
                print(f"APK处理生成异常 {apk_file}: {e}")
                
    # 如果启用了计时，保存结果到CSV
    if enable_timing and timing_results:
        csv_file = os.path.join(output_dir, "timing_stats.csv")
        try:
            fieldnames = ['apk_name', 'decompile_time', 'filter_smali_time', 'icon_time', 'omm_matrix_time', 'sfcg_graph_time', 'so_disasm_time', 'so_matrix_time', 'so_time', 'total_time']
            if include_size:
                fieldnames.insert(1, 'apk_size_mb')
                
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(timing_results)
            print(f"计时统计已保存至: {csv_file}")
        except Exception as e:
            print(f"保存计时统计失败: {e}")
        
        # 等待任务完成
        # 注意：上面的 iterator 已经遍历了所有 future，所以不需要再次遍历
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APK特征提取工具")
    parser.add_argument("input_dir", help="输入APK文件夹路径")
    parser.add_argument("output_dir", help="输出特征文件夹路径")
    parser.add_argument("--decom_dir", default="decom", help="反编译文件夹路径 (默认: decom)")
    parser.add_argument("--workers", type=int, default=4, help="最大线程数 (默认: 4)")
    parser.add_argument("--time", action="store_true", help="是否开启计时统计 (默认: True)")
    parser.add_argument("--size", action="store_true", help="是否统计APK大小 (默认: False, 需配合 --time 使用)")
    
    args = parser.parse_args()
    
    # 处理相对路径
    decom_dir = args.decom_dir
    if not os.path.isabs(decom_dir):
        decom_dir = os.path.join(args.output_dir, decom_dir)
        
    main(args.input_dir, args.output_dir, decom_dir, args.workers, enable_timing=True, include_size=args.size)
