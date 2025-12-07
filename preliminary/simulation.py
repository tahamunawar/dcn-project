import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tracemalloc
import random
import sys
import gc
from collections import defaultdict, deque

from graph import NetworkGraph
from dijkstra import DijkstraOSPF
from new_algorithm import BreakingSortingSSSP
from topology import generate_topology


class NetworkSimulation:
    def __init__(self, num_nodes=1000, topo_type="random_geometric"):
        print(f"\n[Init] Generating {topo_type} topology with {num_nodes} nodes...")
        self.topo_type = topo_type
        self.num_nodes = num_nodes
        self.nx_graph = generate_topology(topo_type, num_nodes)
        self.graph = NetworkGraph(self.nx_graph)

        # Instantiate algorithms
        self.algos = {
            "Dijkstra": DijkstraOSPF(),
            "NewAlgo": BreakingSortingSSSP(),
        }

    def _profile_algo(self, algo_name, source_node):
        """
        Runs a single algorithm and captures Time and Memory.
        """
        algo = self.algos[algo_name]

        # Force garbage collection before start to get clean baseline
        gc.collect()

        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            dist, pred, stats = algo.compute_shortest_paths(self.graph, source_node)
        except Exception as e:
            tracemalloc.stop()
            print(f"Error running {algo_name}: {e}")
            # print stack trace for debugging
            import traceback
            traceback.print_exc()
            return None

        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration = end_time - start_time
        peak_mb = peak / (1024 * 1024)

        return {
            "time_sec": duration,
            "peak_memory_mb": peak_mb,
            "dist": dist,
            "pred": pred,
            "stats": stats,
        }

    def _analyze_paths(self, pred, source):
        """Compute average hop count and max depth."""
        children = defaultdict(list)
        for v, u in pred.items():
            if u is not None:
                children[u].append(v)

        # BFS to determine depths
        queue = deque([(source, 0)])
        total_hops = 0
        max_depth = 0
        reachable = 0
        visited = {source}

        while queue:
            u, depth = queue.popleft()
            # Only count if u is in children map or is a leaf (it's reachable)
            # Actually every node popped is reachable
            if u != source: # Don't count source in average hops usually, or distance 0
                total_hops += depth
                reachable += 1
            max_depth = max(max_depth, depth)

            for v in children[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append((v, depth + 1))

        avg_hops = total_hops / reachable if reachable > 0 else 0
        return avg_hops, max_depth, reachable

    def verify_correctness(self, res_dijkstra, res_new):
        """Check if distances match."""
        if not res_dijkstra or not res_new:
            return False, "Missing Results"

        dist_d = res_dijkstra["dist"]
        dist_n = res_new["dist"]

        mismatches = 0
        max_diff = 0.0

        for n in dist_d:
            d1 = dist_d[n]
            d2 = dist_n.get(n, float("inf"))

            if d1 == float("inf") and d2 == float("inf"):
                continue

            diff = abs(d1 - d2)
            if diff > 1e-6:
                mismatches += 1
                max_diff = max(max_diff, diff)

        return mismatches == 0, f"{mismatches} mismatches (Max diff: {max_diff})"

    def run_comprehensive_test(self, source_node=0):
        results = {}

        print(f"\n[Test] Running Static SSSP from Node {source_node}")
        print(
            f"{'Algorithm':<15} | {'Time (s)':<10} | {'Memory (MB)':<12} | {'Relaxations':<12} | {'Heap Ops':<12}"
        )
        print("-" * 80)

        base_results = {}

        for name in self.algos:
            res = self._profile_algo(name, source_node)
            if res:
                print(
                    f"{name:<15} | {res['time_sec']:<10.6f} | {res['peak_memory_mb']:<12.4f} | {res['stats'].get('relaxations', 0):<12} | {res['stats'].get('heap_ops', res['stats'].get('pq_ops', 0)):<12}"
                )
                base_results[name] = res
            else:
                print(f"{name:<15} | FAILED")

        # Correctness Check
        if "Dijkstra" in base_results and "NewAlgo" in base_results:
            is_correct, msg = self.verify_correctness(
                base_results["Dijkstra"], base_results["NewAlgo"]
            )
            print(f"\nCorrectness vs Dijkstra: {'PASS' if is_correct else 'FAIL'} [{msg}]")

            # Path Analysis
            print("\nPath Quality Metrics:")
            for name, res in base_results.items():
                avg, depth, count = self._analyze_paths(res["pred"], source_node)
                print(
                    f"  > {name:<10}: Avg Hops={avg:.2f}, Max Depth={depth}, Reachable Nodes={count}"
                )

        return base_results

    def run_dynamic_failure_test(self, source_node=0):
        print("\n" + "="*80)
        print("[Test] Running Dynamic Link Failure Scenario")
        print("="*80)

        # 1. Identify a critical edge to remove (one on the Dijkstra shortest path)
        # We need to run Dijkstra first to find the tree
        dijkstra = DijkstraOSPF()
        d, pred, _ = dijkstra.compute_shortest_paths(self.graph, source_node)

        # Pick a node reasonably far away
        target_candidates = [
            n for n, dist in d.items() if dist != float("inf") and dist > 0
        ]
        if not target_candidates:
            print("Graph too disconnected for dynamic test.")
            return

        # Trace back from a random target to find an edge
        path_edges = []
        # Try a few times to find a path with edges
        for _ in range(5):
             target = random.choice(target_candidates)
             curr = target
             temp_path = []
             while curr in pred and pred[curr] is not None:
                 u = pred[curr]
                 if self.nx_graph.has_edge(u, curr):
                     temp_path.append((u, curr))
                 curr = u
                 if curr == source_node:
                     break
             if temp_path:
                 path_edges = temp_path
                 break
        
        if not path_edges:
            print("Could not identify path edges to remove.")
            return

        # Remove a random edge from the active path
        u_rem, v_rem = random.choice(path_edges)
        print(f"Simulating failure of link {u_rem} -> {v_rem} (part of active shortest path)")
        print("Re-calculating routes...")

        old_weight = self.nx_graph[u_rem][v_rem]["weight"]
        self.nx_graph.remove_edge(u_rem, v_rem)

        print(f"\n{'Algorithm':<15} | {'Re-calc Time (s)':<15} | {'Memory (MB)':<12}")
        print("-" * 60)

        results = {}
        for name in self.algos:
            # Re-run from scratch (Simulation of Convergence Time)
            res = self._profile_algo(name, source_node)
            if res:
                print(
                    f"{name:<15} | {res['time_sec']:<15.6f} | {res['peak_memory_mb']:<12.4f}"
                )
                results[name] = res
            else:
                print(f"{name:<15} | FAILED")
        
        # Verify correctness again
        if "Dijkstra" in results and "NewAlgo" in results:
             is_correct, msg = self.verify_correctness(results["Dijkstra"], results["NewAlgo"])
             print(f"Correctness after failure: {'PASS' if is_correct else 'FAIL'} [{msg}]")

        # Restore edge for future tests
        self.nx_graph.add_edge(u_rem, v_rem, weight=old_weight)
        print("\nLink restored.")


if __name__ == "__main__":
    # Settings
    # Use a larger graph to make memory diff noticeable
    N = 2000000
    TOPO = "barabasi_albert"  # Options: random_geometric, barabasi_albert, grid, fat_tree

    sim = NetworkSimulation(num_nodes=N, topo_type=TOPO)

    # 1. Static Test
    sim.run_comprehensive_test(source_node=0)

    # 2. Dynamic Failure Test
    sim.run_dynamic_failure_test(source_node=0)
