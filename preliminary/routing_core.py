from abc import ABC, abstractmethod


class RoutingAlgorithm(ABC):
    @abstractmethod
    def compute_shortest_paths(self, graph, source):
        """
        Computes SSSP from source.
        Returns:
            dist: dict {node: distance}
            pred: dict {node: predecessor}
            stats: dict {metric_name: value} (e.g., relaxations, heap_ops)
        """
        pass
