import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph import NetworkGraph
from dijkstra import DijkstraOSPF
from new_algorithm import BreakingSortingSSSP
from topology import generate_topology


class NetworkSimulation:
    def __init__(self, num_nodes=10000, topo_type="random_geometric"):
        print(f"Initializing {topo_type} topology with {num_nodes} nodes...")
        self.nx_graph = generate_topology(topo_type, num_nodes)
        self.graph = NetworkGraph(self.nx_graph)

        self.algos = {"Dijkstra": DijkstraOSPF(), "NewAlgo": BreakingSortingSSSP()}

    def run_comparison(self, source_node=0):
        results = {}

        for name, algo in self.algos.items():
            start_time = time.perf_counter()
            dist, pred, stats = algo.compute_shortest_paths(self.graph, source_node)
            end_time = time.perf_counter()

            results[name] = {
                "time_sec": end_time - start_time,
                "dist": dist,
                "pred": pred,  # <--- THIS WAS MISSING
                "stats": stats,
            }

        return results

    def verify_correctness(self, res_dijkstra, res_new):
        """Check if distances match"""
        dist_d = res_dijkstra["dist"]
        dist_n = res_new["dist"]

        mismatches = 0
        for n in dist_d:
            # Allow small float error
            d1 = dist_d[n]
            d2 = dist_n[n]

            # Handle infinity comparison
            if d1 == float("inf") and d2 == float("inf"):
                continue

            if abs(d1 - d2) > 1e-6:
                mismatches += 1

        return mismatches == 0, mismatches

    def visualize(self, paths_pred, title="Routing Tree"):
        pos = nx.spring_layout(self.nx_graph, seed=42)
        plt.figure(figsize=(10, 8))

        # Draw all edges
        nx.draw_networkx_edges(self.nx_graph, pos, alpha=0.2, arrows=True)
        nx.draw_networkx_nodes(self.nx_graph, pos, node_size=30, node_color="blue")

        # Highlight tree
        tree_edges = []
        if paths_pred:
            for v, u in paths_pred.items():
                if u is not None:
                    # Verify edge exists to avoid drawing errors on disconnected components
                    if self.nx_graph.has_edge(u, v):
                        tree_edges.append((u, v))

        nx.draw_networkx_edges(
            self.nx_graph,
            pos,
            edgelist=tree_edges,
            edge_color="red",
            width=2,
            arrows=True,
        )
        plt.title(title)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    # Ensure graph is connected enough to be interesting, or use a grid
    sim = NetworkSimulation(num_nodes=100000, topo_type="barabasi_albert")

    # Run from node 0
    res = sim.run_comparison(source_node=0)

    print("-" * 40)
    print("COMPARISON RESULTS")
    print("-" * 40)

    correct, errs = sim.verify_correctness(res["Dijkstra"], res["NewAlgo"])
    print(f"Correctness Check: {'PASS' if correct else f'FAIL ({errs} mismatches)'}")

    for name, data in res.items():
        print(f"\nAlgorithm: {name}")
        print(f"  Time: {data['time_sec']:.6f}s")
        print(f"  Relaxations (Edge Visits): {data['stats'].get('relaxations', 'N/A')}")
        print(f"  Heap Ops: {data['stats'].get('pq_ops', 'N/A')}")

    # Visualization
    # print("\nDisplaying visualization...")
    # sim.visualize(res["NewAlgo"]["pred"], "New Algorithm Shortest Path Tree")
