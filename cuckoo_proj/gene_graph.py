import networkx as nx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# # g=nx.Graph()#创建空的无向图
# g=nx.DiGraph()#创建空的有向图
# g.add_node(1)
# g.add_nodes_from([2,3,4]) # 列表为顶点的ID集合
# print(g._node)
# # print(g.node[1]) #g.node[1]['name']
# g.add_edges_from([(1,2),(1,3)])
# print(g.edges())

# A=np.array(nx.adjacency_matrix(g).todense())
# print(A)


def init_nodes_set(node_lst,node_num=374):
    # nodes_lst=test_lst
    nodes_set=set(node_lst)
    all_nodes=[i for i in range(node_num)]
    return nodes_set,all_nodes


def get_edge_lst_from_lst(api_lst):
    edge_lst=[]
    for i in range(len(api_lst)-1):
        current_item=api_lst[i]
        next_item=api_lst[i+1]
        edge_lst.append((current_item,next_item))
    # print(test_lst,len(test_lst))
    # print(edge_lst,len(edge_lst))
    return edge_lst

def gene_adj_matrix(api_lst):
    g=nx.DiGraph()#创建空的有向图
    nodes_lst,all_nodes_lst=init_nodes_set(api_lst)
    g.add_nodes_from(all_nodes_lst) # 列表为顶点的ID集合
    edges_lst=get_edge_lst_from_lst(api_lst)
    # print(g.node[1]) #g.node[1]['name']
    g.add_edges_from(edges_lst)
    # print(g.edges())
    adj_matrix=np.array(nx.adjacency_matrix(g).todense())
    return (g,adj_matrix)

def gene_X_matrix(api_lst,adj_matrix):
    x_width=len(init_nodes_set(api_lst)[1])
    x_column_nums=len(api_lst)
    X_matrix=np.zeros((x_width,x_column_nums))
    print(X_matrix.shape)
    for i in range(x_column_nums):
        current_item=api_lst[i]
        next_item=api_lst[i+1]

    # for i in range(x_width):
    #     current_item=api_lst[i]
    #     row_index=current_item
    #     column_index=
        


def draw_graph(g):
    # draw graph with labels
    # pos = nx.spring_layout(g)
    pos = nx.circular_layout(g)
    # pos = nx.shell_layout(g) 
    # pos = nx.spectral_layout(g)
    nx.draw(g, pos)
    node_labels = nx.get_node_attributes(g, 'desc')
    nx.draw_networkx_labels(g, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(g, 'name')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    plt.show()


test_lst=[117, 149, 141, 104, 104, 82, 82, 82, 82, 176, 219, 219, 219, 212, 212, 178, 107, 183, 203, 176, 238, 176, 104, 173, 172, 46, 172, 50, 172, 50, 172, 33, 46, 50, 33, 171, 172, 46, 50, 33, 173, 172, 178, 178, 107, 183, 203, 176, 238, 176, 46, 172, 41, 41, 46, 50, 50, 33, 103, 46, 33, 46, 33, 46, 172, 48, 41, 41, 41, 41, 46, 33, 33, 46, 50, 50, 33, 94, 176, 46, 46, 50, 33, 173, 172, 172, 172, 286, 286, 175, 175, 175, 171, 172, 171, 172, 171, 172, 172, 171]
g,adj_matrix=gene_adj_matrix(test_lst)
print(adj_matrix)
gene_X_matrix(test_lst)
draw_graph(g)

