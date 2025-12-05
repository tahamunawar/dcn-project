import heapq
from routing_core import RoutingAlgorithm


class DijkstraOSPF(RoutingAlgorithm):
    def compute_shortest_paths(self, graph, source):
        dist = {node: float("inf") for node in graph.nodes()}
        pred = {node: None for node in graph.nodes()}
        dist[source] = 0

        # Priority Queue: (distance, node)
        pq = [(0, source)]

        stats = {"relaxations": 0, "pq_ops": 0}

        while pq:
            d, u = heapq.heappop(pq)
            stats["pq_ops"] += 1

            if d > dist[u]:
                continue

            for v in graph.get_neighbors(u):
                weight = graph.get_weight(u, v)
                new_dist = dist[u] + weight
                stats["relaxations"] += 1

                if new_dist < dist[v]:
                    dist[v] = new_dist
                    pred[v] = u
                    heapq.heappush(pq, (new_dist, v))
                    stats["pq_ops"] += 1

        return dist, pred, stats

