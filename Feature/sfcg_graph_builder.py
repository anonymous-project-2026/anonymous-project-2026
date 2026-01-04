import os
import re
import itertools
import networkx as nx
import argparse
import community
import concurrent.futures
import threading
from feature_config import FILTER_LIBRARIES, FILTER_FUNCTIONS


class FunctionCallGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.filter_functions = FILTER_FUNCTIONS
        self.filter_libraries = [lib.replace('/', '.') for lib in FILTER_LIBRARIES]
        self.function_index_map = {}
        self.function_index = 1  # 初始化全局函数索引
        self.lock = threading.Lock()  # 用于确保线程安全的锁

    def add_edge(self, caller, callee, edge_type='calls', weight=1):
        """添加边并设置边的类型（例如调用）"""
        self.graph.add_edge(caller, callee, type=edge_type, weight=weight)

    def add_node(self, node_id, node_type, lib_name, function_name=''):
        """添加节点并设置节点的类型、库名和函数名。"""
        self.graph.add_node(node_id, type=node_type, lib_name=lib_name, function_name=function_name)

    def build_graph(self, smali_files):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.parse_smali_file, smali_file) for smali_file in smali_files]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 确保所有线程都完成

    def parse_smali_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            caller = None
            current_lib = None

            for line in lines:
                # 获取当前类库名
                if line.startswith(".class"):
                    match = re.match(r".*?L([^;]+);", line)
                    if match:
                        current_lib = match.group(1).replace('/', '.')

                # 处理方法定义
                if line.startswith(".method"):
                    caller = self.extract_method_name(line)

                    if caller in self.filter_functions:
                        continue

                    if caller is None or self.is_junk_method(caller):
                        continue

                    # 添加方法节点
                    if caller not in self.function_index_map:
                        with self.lock:
                            self.function_index_map[caller] = self.function_index
                            self.add_node(self.function_index, 'method', lib_name=current_lib, function_name=caller)
                            self.function_index += 1

                # 处理调用指令
                elif line.startswith("    invoke"):
                    callee = self.extract_invoke_target(line)
                    lib = self.extract_lib(line) or current_lib

                    # 添加调用关系边
                    if self.function_index_map.get(caller) and callee and lib:
                        if not self.is_junk_method(callee) and callee not in self.filter_functions:
                            # 如果被调用方法不存在于图中，添加该方法节点
                            if callee not in self.function_index_map:
                                with self.lock:
                                    self.function_index_map[callee] = self.function_index
                                    self.add_node(self.function_index, 'method', lib_name=lib, function_name=callee)
                                    self.function_index += 1

                            # 添加调用关系边
                            self.add_edge(self.function_index_map.get(caller),
                                          self.function_index_map.get(callee),
                                          edge_type='calls')

    def is_junk_method(self, method_name):
        if method_name is None:
            return True

        # 检查常见无效方法名特征
        if "." in method_name:  # 如果包含点，可能是有效的方法引用
            return False

        if len(method_name) < 3:  # 太短的方法名
            return True

        # 检查特殊字符
        for char in method_name:
            if char in '<>$&1234567890':
                return True

        # 检查看起来像随机字符串的名称
        if len(method_name) >= 8:
            # 统计大写字母数量
            upper_count = sum(1 for c in method_name if c.isupper())
            upper_ratio = upper_count / len(method_name)

            # 统计连续大写字母的最大长度
            max_upper_seq = 0
            current_seq = 0
            for c in method_name:
                if c.isupper():
                    current_seq += 1
                    max_upper_seq = max(max_upper_seq, current_seq)
                else:
                    current_seq = 0

            # 检查驼峰式命名的有效性
            is_valid_camel_case = (
                    method_name[0].islower() and  # 首字母小写
                    upper_count >= 1 and  # 至少一个大写字母
                    max_upper_seq == 1 and  # 没有连续大写
                    upper_ratio < 0.3 # 大写字母比例不过高

            )

            # 检查常见前缀/后缀
            common_prefixes = ['get', 'set', 'is', 'has', 'on', 'do']
            has_common_prefix = any(method_name.startswith(prefix) for prefix in common_prefixes)

            # 检查常见后缀
            common_suffixes = ['able', 'tion', 'ment', 'ing', 'ed']
            has_common_suffix = any(method_name.lower().endswith(suffix) for suffix in common_suffixes)

            # 如果是有效的驼峰式命名或有常见前后缀，则保留
            if is_valid_camel_case or has_common_prefix or has_common_suffix:
                return False
            if upper_count >=  5:
                return True

            # 如果不符合上述条件，且满足以下任一条件，则认为是垃圾
            if (upper_ratio > 0.4) or (max_upper_seq >= 3):
                return True

            # 检查元音字母比例（随机字符串通常元音较少）
            vowel_count = sum(1 for c in method_name.lower() if c in 'aeiou')
            vowel_ratio = vowel_count / len(method_name)
            if vowel_ratio < 0.2:  # 元音比例低于20%
                return True

        return False

    def extract_lib(self, line):
        if line.startswith("    invoke"):
            match = re.search(r'L(.*?);', line)
            if match:
                library_name = match.group(1).replace('/', '.')
                return library_name if '.' in library_name else None
        return None

    def extract_method_name(self, line):
        match = re.match(r".*\.method\s+(?:\S+\s+)*([^\s(]+).*", line)
        if match:
            method_with_modifiers = match.group(0)
            method_name = method_with_modifiers.replace('.method', '').strip().split('(')[0]
            method_name_parts = method_name.split()
            return method_name_parts[-1] if method_name_parts else None
        return None

    def extract_invoke_target(self, line):
        match = re.match(r".*->(.*?)\(.*\)", line)
        if match:
            return match.group(1)
        return None

    def visualize_graph(self, output_format, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        sorted_nodes = sorted(self.graph.nodes(data=True))
        sorted_edges = sorted(self.graph.edges(data=True))

        ordered_graph = nx.DiGraph()
        ordered_graph.add_nodes_from(sorted_nodes)
        ordered_graph.add_edges_from(sorted_edges)

        if output_format == 'gexf':
            for u, v, data in ordered_graph.edges(data=True):
                data['label'] = f"{data.get('type', '')}"
            nx.write_gexf(ordered_graph, os.path.join(output_folder, 'community_processed_graph.gexf'))

        elif output_format == 'dot':
            nx.nx_pydot.write_dot(ordered_graph, os.path.join(output_folder, 'community_processed_graph.dot'))

    def remove_none_nodes_edges(self):
        none_nodes = [node for node in self.graph.nodes() if node is None]
        self.graph.remove_nodes_from(none_nodes)

        none_edges = [(u, v) for u, v in self.graph.edges() if u is None or v is None]
        self.graph.remove_edges_from(none_edges)

        isolated_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]
        self.graph.remove_nodes_from(isolated_nodes)

        self_loops = [node for node in self.graph.nodes() if
                      self.graph.degree(node) == 2 and (node, node) in self.graph.edges()]
        self.graph.remove_nodes_from(self_loops)

    def detect_communities_louvain(self, resolution=1, random_state=True, weight="weight"):
        undirected_graph = self.graph.to_undirected()
        partition = community.best_partition(undirected_graph, resolution=resolution, random_state=42, weight=weight)
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        return list(communities.values())

    def remove_small_subgraphs(self, threshold):
        small_subgraphs = [subgraph for subgraph in nx.connected_components(self.graph.to_undirected()) if
                           len(subgraph) < threshold]
        for subgraph_nodes in small_subgraphs:
            self.graph.remove_nodes_from(subgraph_nodes)

    def save_processed_graph(self, communities, output_folder):
        processed_graph = self.graph.copy()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for u, v in self.graph.edges():
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

        nx.write_gexf(processed_graph, os.path.join(output_folder, "community_processed_graph.gexf"))


def read_file_to_list(filename):
    lines = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.split(',')[0].strip()
                lines.append(clean_line)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except IOError:
        print(f"Error: An error occurred while reading the file {filename}.")
    return lines
