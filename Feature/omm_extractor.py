import os
import pandas as pd
from collections import defaultdict
import numpy as np
import concurrent.futures
from feature_config import FILTER_LIBRARIES

class OMMProcess:
    def __init__(self, excel_path):
        """
        初始化 OMMProcess 处理器
        :param excel_path: Dalvik操作码Excel文件路径
        """
        self.excel_path = excel_path
        
        # 加载配置数据
        self.syntax_to_index = self._load_excel_config()
        self.third_party_libs = FILTER_LIBRARIES
        
    def _load_excel_config(self):
        """加载Excel配置并返回映射字典 (Syntax -> Index)"""
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel文件未找到: {self.excel_path}")
            
        # 读取 Opcode 列为字符串，防止 pandas 自动推断类型
        # converters 参数指定列类型，确保 Opcode 始终是字符串
        simplified_df = pd.read_excel(self.excel_path, converters={'Opcode': str})
        
        # 建立 Syntax -> Opcode(Index) 的直接映射
        # 既然已经强制转为字符串，直接按十六进制解析即可
        
        syntax_to_index = {}
        for _, row in simplified_df.iterrows():
            syntax = row['Simplified_Syntax']
            opcode_str = row['Opcode']
            
            try:
                # 统一按十六进制解析
                opcode_int = int(opcode_str, 16)
                syntax_to_index[syntax] = opcode_int
            except ValueError:
                print(f"Warning: Could not parse opcode '{opcode_str}' for syntax '{syntax}'")
                
        return syntax_to_index

    def _should_filter(self, rel_path):
        """
        判断相对路径是否匹配第三方库前缀。
        会自动去除顶层的 smali* 目录前缀 (如 smali, smali_classes2)。
        """
        if not self.third_party_libs:
            return False
            
        # 统一路径分隔符，使用正斜杠便于前缀匹配
        norm = rel_path.replace('\\', '/')
        
        # 去除顶层 smali 目录前缀 (smali, smali_classes2 等)
        parts = norm.split('/')
        if parts and parts[0].startswith('smali'):
            if len(parts) > 1:
                norm = '/'.join(parts[1:])
            else:
                # 只有 smali 这一层，忽略
                pass
        
        for lib in self.third_party_libs:
            lib_norm = lib.replace('\\', '/').strip('/')
            if norm.startswith(lib_norm):
                return True
        return False

    @staticmethod
    def _analyze_file(smali_file_path, syntax_to_index):
        """
        分析单个Smali文件，返回局部统计结果
        :param smali_file_path: smali 文件路径
        :param syntax_to_index: 语法到索引的映射表
        :return: (local_matrix, has_error)
                 local_matrix: np.ndarray (256x256)
        """
        try:
            with open(smali_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 快速解析：获取每一行的第一个单词
            # 使用列表推导式比循环更快
            tokens = []
            for line in content.splitlines():
                line = line.strip()
                if not line: continue
                token = line.split(None, 1)[0]
                if token == 'nop': continue
                tokens.append(token)
            
            if not tokens:
                return None, False

            # 映射到索引
            indices = []
            for t in tokens:
                idx = syntax_to_index.get(t)
                if idx is not None:
                    indices.append(idx)
            
            if len(indices) < 2:
                return None, False
            
            # 使用 numpy 进行向量化计算
            # 转换为 int32 数组
            arr = np.array(indices, dtype=np.int32)
            
            # 构造前后对 (prev, curr)
            prev = arr[:-1]
            curr = arr[1:]
            
            # 将二维坐标展平为一维索引: index = prev * 256 + curr
            flat_indices = prev * 256 + curr
            
            # 使用 bincount 快速统计频率
            # minlength=65536 确保结果覆盖所有可能的 (0-255, 0-255) 组合
            counts = np.bincount(flat_indices, minlength=65536)
            
            # 重塑为 256x256 矩阵
            local_matrix = counts.reshape(256, 256)
            
            return local_matrix, False
            
        except Exception as e:
            print(f"Error analyzing file {smali_file_path}: {e}")
            return None, True

    def _calculate_transition_probabilities(self, transition_matrix):
        """计算转移概率矩阵"""
        # 避免除以零
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        # 只有 row_sum > 0 的行才进行除法，否则保持 0
        # 使用 np.divide 的 where 参数可以安全处理
        probabilities = np.divide(
            transition_matrix, 
            row_sums, 
            out=np.zeros_like(transition_matrix, dtype=float), 
            where=row_sums!=0
        )
        return probabilities

    def _save_results(self, transition_matrix, output_directory):
        """保存结果"""
        transition_probabilities = self._calculate_transition_probabilities(transition_matrix)
        npy_path = os.path.join(output_directory, 'dalvik.npy')
        np.save(npy_path, transition_probabilities)

    def process_directory(self, smali_directory, output_directory):
        """
        处理指定的 Smali 目录
        :param smali_directory: smali 源码目录
        :param output_directory: 结果输出目录
        """
        os.makedirs(output_directory, exist_ok=True)
        
        # 最终的转移矩阵 (256x256)
        final_transition_matrix = np.zeros((256, 256), dtype=int)
        
        if not os.path.exists(smali_directory):
            print(f"Directory not found: {smali_directory}")
            self._save_results(final_transition_matrix, output_directory)
            return

        # 收集文件
        smali_files = []
        for root, _, files in os.walk(smali_directory):
            for file in files:
                if file.endswith('.smali'):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, smali_directory)
                    
                    if self._should_filter(rel_path):
                        continue
                        
                    smali_files.append(abs_path)

        if not smali_files:
            print(f"No smali files found in {smali_directory} (after filtering)")
            self._save_results(final_transition_matrix, output_directory)
            return

        # 并发分析 (Map)
        # 使用 max_workers 默认值 (通常是 CPU 核心数 * 5)
        # 对于 IO 密集型任务，可以适当增加，但 smali 解析也是计算密集的
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交任务
            future_to_file = {
                executor.submit(self._analyze_file, file, self.syntax_to_index): file 
                for file in smali_files
            }
            
            # 收集结果 (Reduce)
            for future in concurrent.futures.as_completed(future_to_file):
                local_matrix, has_error = future.result()
                if has_error or local_matrix is None:
                    continue
                
                # 聚合到主矩阵
                # 直接累加 numpy 矩阵，底层 C 实现，速度极快
                final_transition_matrix += local_matrix
            
        # 保存结果
        self._save_results(final_transition_matrix, output_directory)


    # Example usage
    if __name__ == "__main__":
        # 统一路径处理，确保跨平台兼容性 (Windows/Linux)
        # 使用脚本所在目录作为基准
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 配置文件路径 (使用 os.path.join 自动处理分隔符)
        excel_path = os.path.join(script_dir, 'res', 'omm_opcode.xlsx')
        
        # 定位 tpl.txt
        tpl_path = os.path.join(script_dir, 'res', 'omm_filter_tpl.txt')
        
        # 数据集根目录 (Linux路径示例)
        # 如果在 Windows 上运行但访问的是 Linux 风格路径，建议保持原样或根据环境修改
        decom_base_directory = "/newdisk/liuzhuowu/analysis/data/decom"
        
        # 统一规范化路径分隔符
        decom_base_directory = os.path.normpath(decom_base_directory)
    
        print(f"Configuring Analyzer...")
        print(f"Excel: {excel_path}")
        print(f"Filter List: {tpl_path}")
        print(f"Data Directory: {decom_base_directory}")
    
        try:
            # 实例化一次，加载配置
            analyzer = OMMProcess(excel_path, tpl_path)
            print(f"Loaded {len(analyzer.third_party_libs)} filter rules.")
    
            all_apk_paths = []
            if os.path.exists(decom_base_directory):
                for group_folder_name in os.listdir(decom_base_directory):
                    group_folder_path = os.path.join(decom_base_directory, group_folder_name)
                    if os.path.isdir(group_folder_path):
                        for apk_folder_name in os.listdir(group_folder_path):
                            apk_folder_path = os.path.join(group_folder_path, apk_folder_name)
                            if os.path.isdir(apk_folder_path):
                                all_apk_paths.append(apk_folder_path)
                all_apk_paths = sorted(all_apk_paths)
    
                print(f"Found {len(all_apk_paths)} APK folders to process")
    
                # 串行处理每个 APK (如果需要并行处理 APK，可以在这里使用 ThreadPoolExecutor)
                # 由于 process_directory 现在是无状态且线程安全的，外部并行调用也是安全的
                for apk_path in all_apk_paths:
                    try:
                        print(f"Processing: {os.path.basename(apk_path)}")
                        smali_directory = os.path.join(apk_path, 'all_smali')
                        output_directory = apk_path # 输出到APK目录本身
                        
                        analyzer.process_directory(smali_directory, output_directory)
                        
                    except Exception as e:
                        print(f"Error processing APK folder {apk_path}: {e}")
            else:
                print(f"Base directory not found: {decom_base_directory}")
                print("Tip: If running on Windows with a different path, please update 'decom_base_directory'.")
    
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Initialization failed: {e}")
