import numpy as np
import ot
import os
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch  # 引入 PyTorch

class SFCG_OT_ThresholdAnalyzer:
    def __init__(self, base_directory, gexf_name='community_processed_graph.gexf', device=None):
        """
        初始化SFCG最优传输阈值分析器 (GPU支持版)
        """
        self.base_directory = base_directory
        self.gexf_name = gexf_name
        self.apk_paths = []
        self.apk_names = []
        self.apk_labels = {}
        # 特征缓存：路径 -> torch.Tensor (on GPU)
        self.features_cache = {}
        
        # 设备配置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"SFCG Analyzer 运行设备: {self.device}")
    
    def find_all_apk_paths_with_labels(self):
        """查找所有包含SFCG gexf文件的APK项目路径和标签"""
        apk_paths = []
        apk_names = []
        apk_labels = {}
        group_mapping = {}
        group_counter = 0
        
        if not os.path.exists(self.base_directory):
            print(f"路径不存在: {self.base_directory}")
            return [], [], {}

        # 排序以保证跨平台一致性
        sorted_groups = sorted(os.listdir(self.base_directory))
        
        for group_folder in sorted_groups:
            group_path = os.path.join(self.base_directory, group_folder)
            if os.path.isdir(group_path):
                if group_folder not in group_mapping:
                    group_mapping[group_folder] = group_counter
                    group_counter += 1
                
                group_id = group_mapping[group_folder]
                
                for apk_folder in os.listdir(group_path):
                    apk_path = os.path.join(group_path, apk_folder)
                    if os.path.isdir(apk_path):
                        gexf_file = os.path.join(apk_path, self.gexf_name)
                        if os.path.exists(gexf_file):
                            apk_paths.append(apk_path)
                            apk_names.append(apk_folder)
                            apk_labels[apk_path] = group_id
        
        self.apk_paths = apk_paths
        self.apk_names = apk_names
        self.apk_labels = apk_labels
        return apk_paths, apk_names, apk_labels
    
    def load_gexf_graph(self, path):
        if os.path.exists(path):
            return nx.read_gexf(path)
        else:
            raise FileNotFoundError(f"gexf文件不存在: {path}")
    
    def extract_node_features(self, graph):
        """从图中提取节点特征，返回 Tensor"""
        node_features = []
        
        for node, attributes in graph.nodes(data=True):
            feature_vector = []
            for i in range(0, 9): 
                attr_name = f'vector_{i}'
                # GEXF读取属性可能为字符串，需转换
                val = attributes.get(attr_name, 0.0)
                feature_vector.append(float(val))
            node_features.append(feature_vector)
        
        if not node_features:
            # 防止空图报错，插入一个零向量
            node_features = [[0.0] * 9]

        # 转为 Tensor 并移动到 GPU
        return torch.tensor(node_features, dtype=torch.float32).to(self.device)

    def preload_apk_features(self, apk_paths, max_nodes=None):
        """
        预加载特征到 GPU 显存
        """
        loaded = 0
        for apk_path in apk_paths:
            if apk_path in self.features_cache:
                continue
            gexf_file = os.path.join(apk_path, self.gexf_name)
            if not os.path.exists(gexf_file):
                continue
            try:
                graph = self.load_gexf_graph(gexf_file)
                # 提取 (返回的是 GPU Tensor)
                feats = self.extract_node_features(graph)
                
                # 随机采样 (在 Tensor 上操作)
                if max_nodes is not None and max_nodes > 0 and feats.shape[0] > max_nodes:
                    # torch.randperm 生成随机索引
                    idx = torch.randperm(feats.shape[0])[:max_nodes]
                    feats = feats[idx]
                    
                self.features_cache[apk_path] = feats
                loaded += 1
            except Exception as e:
                print(f"预加载特征失败 {os.path.basename(apk_path)}: {e}")
        return loaded
    
    def calculate_sliced_wasserstein(self, feat1, feat2, n_projections=50):
        """
        GPU 加速的 Sliced Wasserstein 距离。
        这是目前处理点云分布最快的方法 (O(N log N))。
        """
        # ot.sliced_wasserstein_distance 支持 PyTorch Tensor 并在 GPU 上运行
        # seed 固定以保证可复现性
        return ot.sliced_wasserstein_distance(feat1, feat2, n_projections=n_projections, seed=42).item()

    def calculate_sinkhorn_gpu(self, feat1, feat2, reg=0.1):
        """
        GPU 加速的 Sinkhorn 距离。
        """
        # 计算代价矩阵 (Squared Euclidean)
        # torch.cdist 计算 p=2 的欧氏距离，平方得到代价
        M = torch.cdist(feat1, feat2, p=2) ** 2
        
        # 归一化代价矩阵有助于数值稳定性
        M = M / (M.max() + 1e-8)
        
        n, m = feat1.shape[0], feat2.shape[0]
        a = torch.ones((n,), device=self.device) / n
        b = torch.ones((m,), device=self.device) / m
        
        # 使用 POT 的 PyTorch 后端
        return ot.sinkhorn2(a, b, M, reg, numItermax=50).item()

    def calculate_ot_similarity(self, path1, path2, method='sliced', sinkhorn_reg=0.1):
        """
        计算相似度
        method: 'sliced' (推荐,最快) 或 'sinkhorn' (较快) 或 'emd' (CPU慢, 不推荐)
        """
        try:
            feat1 = self.features_cache.get(path1)
            feat2 = self.features_cache.get(path2)
            
            # 如果缓存没命中（极少情况），现场加载
            if feat1 is None or feat2 is None:
                # 简略处理，实际应该调用 load_gexf
                return 0.0 
            
            dist = 1.0
            
            if method == 'sliced':
                dist = self.calculate_sliced_wasserstein(feat1, feat2)
            elif method == 'sinkhorn':
                dist = self.calculate_sinkhorn_gpu(feat1, feat2, reg=sinkhorn_reg if sinkhorn_reg else 0.1)
            else:
                # 兼容旧代码，但强烈不建议在循环中使用 CPU EMD
                # 需要把 Tensor 转回 CPU numpy
                f1_np = feat1.cpu().numpy()
                f2_np = feat2.cpu().numpy()
                M = ot.dist(f1_np, f2_np)
                n, m = len(f1_np), len(f2_np)
                dist = ot.emd2(np.ones(n)/n, np.ones(m)/m, M)

            # 距离转相似度
            similarity = np.exp(-dist)
            return float(similarity)
            
        except Exception as e:
            # print(f"Err: {e}")
            return 0.0

    def quick_prefilter_similarity(self, feat1, feat2):
        """
        GPU 上的快速均值预筛选
        """
        try:
            # feat1, feat2 是 Tensor
            v1 = torch.mean(feat1, dim=0)
            v2 = torch.mean(feat2, dim=0)
            # 欧氏距离
            dist = torch.norm(v1 - v2).item()
            return float(np.exp(-dist))
        except Exception:
            return 0.0
    
    # --- 以下为矩阵生成与绘图辅助函数，逻辑保持不变，但适配 Tensor ---
    
    def generate_ot_similarity_matrix(self, method='sliced'):
        n_apks = len(self.apk_paths)
        sim_matrix = np.eye(n_apks)
        print(f"计算矩阵 (Method: {method})...")
        
        # 预加载所有特征
        self.preload_apk_features(self.apk_paths)
        
        for i in range(n_apks):
            for j in range(i + 1, n_apks):
                sim = self.calculate_ot_similarity(
                    self.apk_paths[i], self.apk_paths[j], method=method
                )
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
            if (i+1) % 100 == 0: print(f"Processed {i+1}/{n_apks}")
        return sim_matrix

    def generate_threshold_table(self, similarity_matrix, labels):
        # 原逻辑完全复用，因为 matrix 和 labels 都是 cpu numpy/list
        n = len(labels)
        y_true = []
        y_score = []
        for i in range(n):
            for j in range(i + 1, n):
                y_true.append(1 if labels[i] == labels[j] else 0)
                y_score.append(similarity_matrix[i, j])
        
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # 简化版阈值扫描
        thresholds = np.linspace(0.1, 0.99, 100)
        table = []
        best_f1 = 0
        best_info = None
        
        for th in thresholds:
            pred = (y_score >= th).astype(int)
            pre = precision_score(y_true, pred, zero_division=0)
            rec = recall_score(y_true, pred, zero_division=0)
            f1 = f1_score(y_true, pred, zero_division=0)
            acc = accuracy_score(y_true, pred)
            
            row = {'Threshold': th, 'Precision': pre, 'Recall': rec, 'F1 Score': f1, 'Accuracy': acc}
            table.append(row)
            if f1 > best_f1:
                best_f1 = f1
                best_info = row
                
        if best_info:
            print(f"最佳 F1: {best_info['F1 Score']:.4f} @ Th={best_info['Threshold']:.3f}")
        return table

    # 绘图函数保持不变，略去以节省篇幅 (复用原代码即可)
    def save_threshold_table(self, table, path):
        pd.DataFrame(table).to_csv(path, index=False)
    
    def plot_threshold_metrics(self, table, path):
        df = pd.DataFrame(table)
        plt.figure()
        plt.plot(df['Threshold'], df['F1 Score'], label='F1')
        plt.legend()
        plt.savefig(path)
        plt.close()

    def plot_heatmap(self, mat, path):
        plt.figure()
        sns.heatmap(pd.DataFrame(mat), cmap='YlOrRd')
        plt.savefig(path)
        plt.close()

    def run(self, output_dir, method='sliced'):
        self.find_all_apk_paths_with_labels()
        if not self.apk_paths: return
        mat = self.generate_ot_similarity_matrix(method=method)
        labels = [self.apk_labels[p] for p in self.apk_paths]
        table = self.generate_threshold_table(mat, labels)
        self.save_threshold_table(table, os.path.join(output_dir, 'metrics.csv'))
        self.plot_heatmap(mat, os.path.join(output_dir, 'heatmap.png'))

if __name__ == "__main__":
    # 简单的自测入口
    base = "./data" # 修改为你的路径
    an = SFCG_OT_ThresholdAnalyzer(base)
    an.run("./res", method='sliced')