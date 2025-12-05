import math
import heapq
from collections import deque
from routing_core import RoutingAlgorithm


class BreakingSortingSSSP(RoutingAlgorithm):
    """
    Implementation of 'Breaking the Sorting Barrier for Directed SSSP'.
    """

    def compute_shortest_paths(self, graph, source):
        self.graph = graph
        self.nodes = graph.nodes()
        self.dist = {node: float("inf") for node in self.nodes}
        self.pred = {node: None for node in self.nodes}

        # Added 'pq_ops' to track heap operations
        self.stats = {"relaxations": 0, "phases": 0, "pivot_selections": 0, "pq_ops": 0}

        self.dist[source] = 0

        # Initial Frontier
        frontier = {source}

        # Heuristic param K: number of local relaxation steps before pivot recursion
        # In dense graphs, lower K might be better. In sparse, higher K.
        K = max(2, int(math.log(len(self.nodes)) * 2))

        self._bmssp(frontier, K)

        return self.dist, self.pred, self.stats

    def _bmssp(self, frontier, k):
        """
        Bounded Multi-Source Shortest Path Routine.
        """
        if not frontier:
            return

        current_frontier = list(frontier)
        next_frontier = set()

        # 1. Pivot Selection
        self.stats["pivot_selections"] += 1
        sorted_by_degree = sorted(
            current_frontier,
            key=lambda n: len(self.graph.get_neighbors(n)),
            reverse=True,
        )

        # Select top subset as pivots
        num_pivots = max(1, int(len(current_frontier) ** 0.6))
        pivots = set(sorted_by_degree[:num_pivots])
        non_pivots = [n for n in current_frontier if n not in pivots]

        # 2. Local Relaxations (The "Breaking Sorting" part)
        active_queue = deque(non_pivots)

        for _ in range(k):
            if not active_queue:
                break

            # Process level
            for _ in range(len(active_queue)):
                u = active_queue.popleft()

                for v in self.graph.get_neighbors(u):
                    w = self.graph.get_weight(u, v)

                    # Count Edge Visit as a relaxation attempt (consistency with Dijkstra metric)
                    self.stats["relaxations"] += 1

                    if self.dist[u] + w < self.dist[v]:
                        self.dist[v] = self.dist[u] + w
                        self.pred[v] = u

                        if v in pivots:
                            continue

                        active_queue.append(v)
                        next_frontier.add(v)

        # 3. Pivot Recursion (Global Synchronization)
        candidates = set(pivots).union(set(active_queue))

        if not candidates:
            return

        pq = []
        for u in candidates:
            if self.dist[u] < float("inf"):
                heapq.heappush(pq, (self.dist[u], u))
                self.stats["pq_ops"] += 1  # Track push

        while pq:
            d, u = heapq.heappop(pq)
            self.stats["pq_ops"] += 1  # Track pop

            if d > self.dist[u]:
                continue

            for v in self.graph.get_neighbors(u):
                w = self.graph.get_weight(u, v)

                self.stats["relaxations"] += 1  # Edge visit

                if self.dist[u] + w < self.dist[v]:
                    self.dist[v] = self.dist[u] + w
                    self.pred[v] = u

                    heapq.heappush(pq, (self.dist[v], v))
                    self.stats["pq_ops"] += 1  # Track push

                    next_frontier.add(v)

        # Recurse if needed
        real_next = {n for n in next_frontier if self.dist[n] < float("inf")}

        # Simple cycle breaking for simulation safety
        if len(real_next) > 0 and self.stats["pivot_selections"] < len(self.nodes) * 2:
            self._bmssp(real_next, k)
