import pickle
import concurrent.futures
import networkx as nx

# 读取.pkl文件
def load_nodes_with_vectors(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data[1:]
    except Exception as e:
        print("Error reading .pkl file:", e)
        return None

# 读取.txt文件
def read_nodes_from_txt(file_path):
    with open(file_path, 'r') as f:
            lines = f.readlines()
    split_lines = [line.strip().split(',')[0] for line in lines]
    return split_lines

    
def read_graph_from_gexf(file_path):
    try:
        # 使用networkx库读取GEXF文件
        G = nx.read_gexf(file_path)
        return G
    except Exception as e:
        print("Error reading .gexf file:", e)
        return None
    
def traverse_graph(graph,nodes_txt,nodes_with_vectors):

    for node, attrs in graph.nodes(data=True):
        lib_name=attrs.get("lib_name", None)
        function_name=attrs.get("function_name", None)

        if lib_name and function_name:
            api_name = lib_name + "." + function_name
        else:
            # 如果其中一个为空，就是类节点
            api_name = lib_name
        if api_name in nodes_txt:
            index = nodes_txt.index(api_name)
            for i,vector in enumerate(nodes_with_vectors[index]):             
                graph.nodes[node][f"vector_{i}"] = vector

        else:
            for i in range(10):
                graph.nodes[node][f"vector_{i}"] = 0.0


            
    
    
 

