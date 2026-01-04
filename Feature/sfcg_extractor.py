import os
import pandas as pd
import networkx as nx
import subprocess
from sfcg_enhance import traverse_graph, read_nodes_from_txt, load_nodes_with_vectors
import sfcg_graph_builder
import numpy as np
import time
import community
from concurrent.futures import ThreadPoolExecutor


class SfcgProcessor:
    def __init__(self, directory, s_nodes_txt, s_nodes_with_vectors, nodes_txt, nodes_with_vectors):
        self.directory = directory
        self.nodes_txt = nodes_txt
        self.nodes_with_vectors = nodes_with_vectors
        self.s_nodes_txt = s_nodes_txt
        self.s_nodes_with_vectors = s_nodes_with_vectors
        self.fcg =  nx.DiGraph()  # 初始化图变量

    def read_gexf_file(self, file_path):
        """
        读取 GEXF 文件并返回 NetworkX 图对象。
        返回:nx.Graph: NetworkX 图对象
        """
        try:
            graph = nx.read_gexf(file_path)
            return graph
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the GEXF file: {e}")
            return None

    def get_top_k_subgraphs_by_sensitive_function_ratio(self, file_path, k):
        """
        保存非零向量节点比例高的前k个子图到一个图中，并将其保存为一个GEXF文件。
        参数:
            file_path (str): 保存子图的GEXF文件路径
            k (int): 保存前k个子图
        """
        if self.fcg is None:
            print("Graph not loaded.")
            return

        # 创建一个新的图对象，仅包含权重不为0的边
        filtered_graph = nx.Graph()
        for u, v, attributes in self.fcg.edges(data=True):
            if attributes.get('weight', 0) != 0:
                filtered_graph.add_edge(u, v, **attributes)

        # 添加节点属性
        for node, attributes in self.fcg.nodes(data=True):
            if node in filtered_graph:
                filtered_graph.nodes[node].update(attributes)

        # 找到所有的连通子图
        subgraphs = [filtered_graph.subgraph(c).copy() for c in nx.connected_components(filtered_graph)]

        # 计算每个子图的非零向量节点比例
        subgraph_ratios = []
        for subgraph in subgraphs:
            total_nodes = len(subgraph.nodes)
            non_zero_vector_nodes = sum(1 for _, attributes in subgraph.nodes(data=True) if np.any(
                [attributes.get(f'vector_{i}', 0.0) for i in range(1, 10)]))
            non_zero_vector_ratio = non_zero_vector_nodes / total_nodes if total_nodes != 0 else 0
            subgraph_ratios.append((subgraph, non_zero_vector_ratio))

        # 按照非零向量节点比例排序并选择前k个子图
        subgraph_ratios.sort(key=lambda x: x[1], reverse=True)
        top_k_subgraphs = [subgraph for subgraph, _ in subgraph_ratios[:k]]

        # 合并前k个子图
        top_k_graph = nx.Graph()
        for subgraph in top_k_subgraphs:
            top_k_graph.add_nodes_from(subgraph.nodes(data=True))
            top_k_graph.add_edges_from(subgraph.edges(data=True))

        # 创建保存路径的目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 将top_k_graph保存为GEXF文件
        nx.write_gexf(top_k_graph, os.path.join(file_path, "community_processed_graph.gexf"))

    def process_sensitive_node(self, sensitive_node):
        nodes_within_k = nx.single_source_shortest_path_length(self.fcg, sensitive_node, cutoff=self.k).keys()
        subgraph = self.fcg.subgraph(nodes_within_k)
        return subgraph

    def get_sensitive_node_subgraphs_by_path_length(self, file_path, k):
        """
        为每个敏感节点保留与它路径长度为k的节点，并将结果保存为一个GEXF文件。
        参数:
            file_path (str): 保存子图的GEXF文件路径
            k (int): 路径长度
        """
        if self.fcg is None:
            print("Graph not loaded.")
            return
        self.k = k
        sensitive_node_graph = nx.Graph()

        # 找到所有敏感节点
        sensitive_nodes = [node for node, attributes in self.fcg.nodes(data=True) if np.any(
            [attributes.get(f'vector_{i}', 0.0) for i in range(1, 10)])]

        with ThreadPoolExecutor() as executor:
            subgraphs = list(executor.map(self.process_sensitive_node, sensitive_nodes))

        # 合并所有子图
        for subgraph in subgraphs:
            sensitive_node_graph.add_nodes_from(subgraph.nodes(data=True))
            sensitive_node_graph.add_edges_from(subgraph.edges(data=True))

        # 去除自环
        self_loops = list(nx.selfloop_edges(sensitive_node_graph))
        sensitive_node_graph.remove_edges_from(self_loops)

        return sensitive_node_graph

    def save_processed_graph(self, communities, output_folder):
        """
        保存经过社区划分处理的图
        """
        processed_graph = self.fcg.copy()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 处理边的权重
        for u, v in self.fcg.edges():
            u_community = None
            v_community = None

            for idx, community in enumerate(communities):
                if u in community:
                    u_community = idx
                if v in community:
                    v_community = idx

            if u_community is not None and v_community is not None and u_community == v_community:
                processed_graph[u][v]['weight'] = 1
            else:
                processed_graph[u][v]['weight'] = 0

        # 对节点和边进行排序，确保确定性
        sorted_nodes = sorted(processed_graph.nodes(data=True))
        sorted_edges = sorted(processed_graph.edges(data=True))

        # 创建新的有序图对象
        ordered_graph = nx.Graph()
        ordered_graph.add_nodes_from(sorted_nodes)
        ordered_graph.add_edges_from(sorted_edges)

        return ordered_graph

    def detect_communities_louvain(self, resolution=1, random_state=42, weight="weight"):
        """
        使用Louvain算法检测社区，设置固定的随机种子以确保确定性
        """
        undirected_graph = self.fcg.to_undirected()
        partition = community.best_partition(undirected_graph, resolution=resolution, random_state=random_state,
                                             weight=weight)

        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        return list(communities.values())

    def run(self, output_path):
        """
        主运行逻辑，确保图的生成具有确定性
        """
        # 使用 sfcg_graph_builder 直接构建图，替代 subprocess 调用
        fcg_builder = sfcg_graph_builder.FunctionCallGraph()
        
        # 查找 smali 文件
        smali_files = [
            os.path.join(root, f)
            for root, dirs, files in os.walk(self.directory)
            for f in files if f.endswith(".smali")
        ]
        
        fcg_builder.build_graph(smali_files)
        fcg_builder.remove_none_nodes_edges()
        fcg_builder.remove_small_subgraphs(6)
        fcg_builder.visualize_graph('gexf', output_path)

        # 读取生成的 GEXF 文件
        self.fcg = self.read_gexf_file(os.path.join(output_path, "community_processed_graph.gexf"))

        # 处理敏感节点及其子图
        traverse_graph(self.fcg, self.s_nodes_txt, self.s_nodes_with_vectors)
        self.fcg = self.get_sensitive_node_subgraphs_by_path_length(output_path, 1)

        # Louvain社区划分，确保确定性
        communities = self.detect_communities_louvain()
        self.fcg = self.save_processed_graph(communities, output_path)

        # 再次遍历并保存结果
        traverse_graph(self.fcg, self.nodes_txt, self.nodes_with_vectors)
        nx.write_gexf(self.fcg, os.path.join(output_path, "community_processed_graph.gexf"))



if __name__ == "__main__":
    # 初始化
    input_pkl_file = r"./entity_embedding_TransE.pkl"
    input_txt_file = r"./entities.txt"
    nodes_txt = read_nodes_from_txt(input_txt_file)
    nodes_with_vectors = load_nodes_with_vectors(input_pkl_file)
    filename = r'./matched_result1.xlsx'
    df = pd.read_excel(filename)
    s_nodes_txt = df['Found Elements'].tolist()
    s_nodes_with_vectors = df.iloc[:, 1:].values.tolist()
    
    # 设置基础目录路径
    decom_base_directory = r"/newdisk/liuzhuowu/analysis/data/decom"
    
    # 查找所有APK项目的all_smali目录
    all_smali_paths = []
    for group_folder_name in os.listdir(decom_base_directory):
        group_folder_path = os.path.join(decom_base_directory, group_folder_name)
        if os.path.isdir(group_folder_path):
            for apk_folder_name in os.listdir(group_folder_path):
                apk_folder_path = os.path.join(group_folder_path, apk_folder_name)
                if os.path.isdir(apk_folder_path):
                    # 查找all_smali目录
                    all_smali_path = os.path.join(apk_folder_path, "all_smali")
                    if os.path.exists(all_smali_path) and os.path.isdir(all_smali_path):
                        all_smali_paths.append(all_smali_path)
    
    print(f"Found {len(all_smali_paths)} APK folders with all_smali directory to process")
    
    # 处理每个APK项目
    for i, smali_directory in enumerate(all_smali_paths):
        print(f"Processing APK {i+1}/{len(all_smali_paths)}: {smali_directory}")
        
        # 设置输出路径为APK文件夹
        apk_folder = os.path.dirname(smali_directory)
        output_path = apk_folder
        
        try:
            # 创建处理器实例并运行
            analyzer = SfcgProcessor(
                smali_directory, 
                s_nodes_txt, 
                s_nodes_with_vectors, 
                nodes_txt, 
                nodes_with_vectors
            )
            analyzer.run(output_path)
            print(f"Successfully processed: {smali_directory}")
            
        except Exception as e:
            print(f"Error processing {smali_directory}: {str(e)}")
            continue
    
    print("All APK processing completed!")