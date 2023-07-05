import numpy as np
import networkx as nx

def face_graph(X, eps, kde):
    """
    Create a kD-tree from the dataset X
    Args:
        X:

    Returns:

    """
    G = nx.Graph()
    for i in range(len(X)):
        G.add_node(i)
    for i in range(len(X)):
        for j in range(len(X)):
            dist = np.linalg.norm(X[i] - X[j])
            if dist < eps:
                w = dist * kde.score([(X[i] + X[j]) / 2])
                G.add_edge(i, j, weight=w)
    return G

def djik(G, fact, c_fact):
    return nx.shortest_path(G, source=int(fact), target=int(c_fact))