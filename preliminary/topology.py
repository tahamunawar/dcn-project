import networkx as nx
import random
import math


def generate_topology(type="random_geometric", n=100, **kwargs):
    if type == "random_geometric":
        G = nx.random_geometric_graph(n, radius=0.125)
        # Make directed and add weights
        G = G.to_directed()
    elif type == "barabasi_albert":
        G = nx.barabasi_albert_graph(n, m=3)
        G = G.to_directed()
    elif type == "grid":
        dim = int(n**0.5)
        G = nx.grid_2d_graph(dim, dim)
        G = nx.DiGraph(G)
        # Rename nodes to integers
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    elif type == "fat_tree":
        # Simplified fat tree approx
        G = nx.balanced_tree(r=3, h=int(math.log(n, 3)))
        G = G.to_directed()
    else:
        raise ValueError("Unknown topology")

    # Add random weights
    for u, v in G.edges():
        G[u][v]["weight"] = random.randint(1, 1000)
        # random.uniform(1.0, 10.0) (Was previously this)

    return G
