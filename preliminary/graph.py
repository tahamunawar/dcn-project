import networkx as nx
import random


class NetworkGraph:
    def __init__(self, nx_graph=None):
        """
        Wrapper around NetworkX for routing simulations.
        """
        self.graph = nx_graph if nx_graph else nx.DiGraph()
        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()

    def get_neighbors(self, u):
        return list(self.graph.successors(u))

    def get_weight(self, u, v):
        return self.graph[u][v].get("weight", 1.0)

    def nodes(self):
        return list(self.graph.nodes())
