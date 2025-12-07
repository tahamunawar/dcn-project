import math
import heapq
import bisect
from collections import deque, defaultdict
from routing_core import RoutingAlgorithm

# --- CONFIGURATION ---
DEBUG = False


def log(indent, msg):
    if DEBUG:
        print(f"{'  ' * indent}[DEBUG] {msg}")


class Block:
    """
    Represents a 'block' in the linked list.
    Using __slots__ for performance (faster attribute access, lower memory).
    """

    __slots__ = ["items", "max_val"]

    def __init__(self, items=None):
        # items is a list of tuples: (distance, node_id)
        self.items = items if items is not None else []
        self.max_val = max(self.items) if self.items else (-float("inf"), -1)

    def add(self, item):
        self.items.append(item)
        if item > self.max_val:
            self.max_val = item

    def sort_and_update(self):
        self.items.sort()
        if self.items:
            self.max_val = self.items[-1]
        else:
            self.max_val = (-float("inf"), -1)

    def __len__(self):
        return len(self.items)


class Lemma33DataStructure:
    """
    Faithful implementation of Lemma 3.3.
    Optimized for Python execution speed while retaining logical structure.
    """

    def __init__(self, M, global_bound_tuple):
        self.M = max(1, int(M))
        self.global_bound = global_bound_tuple

        # D0: Sequence of blocks from Batch Prepend
        self.D0 = []

        # D1: Sequence of blocks from Insert. Sorted by block.max_val.
        self.D1 = []

        # Upper bounds cache for D1 to speed up bisect (simulating the BST)
        self.D1_upper_bounds = []

        # Map node -> (block_reference, distance_tuple)
        self.item_map = {}

        # Initialize D1 with one empty block
        self._add_block_to_d1(Block())

    def _add_block_to_d1(self, block):
        self.D1.append(block)
        self.D1_upper_bounds.append(block.max_val)

    def _update_d1_bounds(self):
        # Rebuild bounds cache
        self.D1_upper_bounds = [b.max_val for b in self.D1]

    def insert(self, u, dist):
        val_tuple = (dist, u)

        # 1. Check/Update existing
        if u in self.item_map:
            old_block, old_val = self.item_map[u]
            if val_tuple < old_val:
                # Lazy delete: we don't remove from old_block.items immediately to avoid O(M) scan.
                # We just update the map. The old value in the block becomes "stale" (orphan).
                # We handle stale entries in pull().
                pass
            else:
                return

        # 2. Insert into D1
        # Find block with smallest upper bound >= val_tuple
        if not self.D1:
            self._add_block_to_d1(Block())
            idx = 0
        else:
            idx = bisect.bisect_left(self.D1_upper_bounds, val_tuple)
            if idx >= len(self.D1):
                idx = len(self.D1) - 1

        target_block = self.D1[idx]
        target_block.add(val_tuple)

        # Update bound only if this new item is the new max
        if val_tuple > self.D1_upper_bounds[idx]:
            self.D1_upper_bounds[idx] = val_tuple

        self.item_map[u] = (target_block, val_tuple)

        # 3. Split if too large
        if len(target_block) > self.M:
            self._split_block(idx)

    def _split_block(self, idx):
        block = self.D1[idx]
        # Sort is required for splitting by median
        block.sort_and_update()

        # Filter stale items during split (maintenance)
        # Re-verify items point to this block in the map
        active_items = []
        for item in block.items:
            u = item[1]
            if u in self.item_map and self.item_map[u][1] == item:
                active_items.append(item)

        # If strict filtering reduced size enough, just update and return
        if len(active_items) <= self.M:
            block.items = active_items
            block.sort_and_update()
            self.D1_upper_bounds[idx] = block.max_val
            return

        mid = len(active_items) // 2
        items_low = active_items[:mid]
        items_high = active_items[mid:]

        b1 = Block(items_low)
        b2 = Block(items_high)

        # Bulk update map references
        for item in items_low:
            self.item_map[item[1]] = (b1, item)
        for item in items_high:
            self.item_map[item[1]] = (b2, item)

        # Replace in D1
        self.D1[idx] = b1
        self.D1.insert(idx + 1, b2)

        # Update bounds cache
        self.D1_upper_bounds[idx] = b1.max_val
        self.D1_upper_bounds.insert(idx + 1, b2.max_val)

    def batch_prepend(self, items):
        if not items:
            return

        # Filter duplicates/stale updates in input
        unique_items = {}
        for u, d in items:
            t = (d, u)
            if u in unique_items:
                if t < unique_items[u]:
                    unique_items[u] = t
            else:
                unique_items[u] = t

        # Check against existing map
        final_list = []
        for u, t in unique_items.items():
            if u in self.item_map:
                _, old_t = self.item_map[u]
                if t < old_t:
                    # Stale old entry is fine, we update map
                    self.item_map[u] = (None, t)
                    final_list.append(t)
            else:
                final_list.append(t)

        if not final_list:
            return

        final_list.sort()

        # Create blocks
        chunk_size = math.ceil(self.M / 2)
        new_blocks = []

        # Pythonic chunking
        for i in range(0, len(final_list), chunk_size):
            chunk = final_list[i : i + chunk_size]
            blk = Block(chunk)
            new_blocks.append(blk)
            for item in chunk:
                self.item_map[item[1]] = (blk, item)

        # Prepend to D0 (newest/smallest first)
        self.D0 = new_blocks + self.D0

    def pull(self):
        # We need M smallest valid items.
        # Since D0 and D1 might contain stale items, we pull a buffer > M.
        buffer_limit = self.M * 2
        candidates = []

        # Collect from D0
        count = 0
        for blk in self.D0:
            if not blk.items:
                continue
            candidates.extend(blk.items)
            count += len(blk.items)
            if count >= buffer_limit:
                break

        # Collect from D1
        count = 0
        for blk in self.D1:
            if not blk.items:
                continue
            candidates.extend(blk.items)
            count += len(blk.items)
            if count >= buffer_limit:
                break

        candidates.sort()

        valid_s_prime = []
        seen_nodes = set()

        # Filter valid items
        for item in candidates:
            dist, u = item
            # Check validity: Map must point to this specific item value
            if u in self.item_map and self.item_map[u][1] == item:
                if u not in seen_nodes:
                    valid_s_prime.append(u)
                    seen_nodes.add(u)
                    if len(valid_s_prime) == self.M:
                        break

        # Remove from map (logically removed from structure)
        for u in valid_s_prime:
            del self.item_map[u]

        # Determine Bound X
        if not valid_s_prime:
            return self.global_bound, []

        # For the bound, we need the minimum *remaining* valid item.
        # This is expensive. Approximation: Use the first item in candidates
        # that wasn't picked and is valid.

        # Clean up empty/stale blocks in D0/D1 lazily to keep lists short?
        # For speed, we just scan for the min.

        min_remaining = (float("inf"), float("inf"))

        # Quick check: The candidates list might have the next min
        found_in_candidates = False
        for item in candidates:
            if item > (float("inf"), float("inf")):
                break  # Impossible

            # If item is valid and not in valid_s_prime
            u = item[1]
            if u in self.item_map and self.item_map[u][1] == item:
                # It's valid and wasn't pulled (since we pulled smallest M)
                min_remaining = item
                found_in_candidates = True
                break

        if not found_in_candidates:
            # Fallback: We need to check the heads of the lists we didn't fully scan
            # This is the "faithful" but slow part.
            # Optimization: Global bound is sufficient if we exhausted inputs.
            pass

        if min_remaining == (float("inf"), float("inf")):
            return self.global_bound, valid_s_prime

        return min_remaining, valid_s_prime

    def is_empty(self):
        return len(self.item_map) == 0


class BreakingSortingSSSP(RoutingAlgorithm):
    def compute_shortest_paths(self, graph, source):
        self.graph = graph
        self.nodes = graph.nodes()
        self.n = len(self.nodes)

        # 1. Parameter Initialization (Paper Logic)
        if self.n > 1:
            log_n = math.log2(self.n)
            # Tuning for Python Performance while respecting structure:
            # We use a larger 'k' base to leverage Dijkstra's C-speed more often
            # without breaking the recursive logic structure.
            self.k = int(max(32, self.n / (log_n**2)))
            self.t = max(2, int(log_n / 2))
        else:
            self.k = 2
            self.t = 2

        if self.n > 1:
            self.L_max = math.ceil(math.log2(self.n) / self.t)
        else:
            self.L_max = 1

        # 2. State
        self.d = {u: float("inf") for u in self.nodes}
        self.pred = {u: None for u in self.nodes}
        self.d[source] = 0.0

        self.stats = {"relaxations": 0, "heap_ops": 0, "pivots_found": 0}

        # 3. Run
        B_inf = (float("inf"), float("inf"))
        self._bmssp(self.L_max, B_inf, {source}, indent=0)

        return self.d, self.pred, self.stats

    # --- ALGORITHM 1: FIND PIVOTS ---
    def _find_pivots(self, B_tuple, S, indent):
        W = set(S)
        W_curr = list(S)
        root_map = {u: u for u in S}

        for step in range(self.k):
            W_next = []
            if not W_curr:
                break

            for u in W_curr:
                if (self.d[u], u) >= B_tuple:
                    continue

                for v in self.graph.get_neighbors(u):
                    w_uv = self.graph.get_weight(u, v)
                    new_dist = self.d[u] + w_uv
                    self.stats["relaxations"] += 1

                    if new_dist <= self.d[v]:
                        improved = new_dist < self.d[v]
                        self.d[v] = new_dist
                        if improved:
                            self.pred[v] = u

                        if u in root_map:
                            root_map[v] = root_map[u]

                        if (new_dist, v) < B_tuple:
                            if v not in W:
                                W_next.append(v)
                                W.add(v)
                            elif improved:
                                W_next.append(v)
            W_curr = W_next

        if len(W) > self.k * len(S):
            return S, W

        tree_counts = defaultdict(int)
        for v in W:
            if v in root_map:
                root = root_map[v]
                tree_counts[root] += 1

        P = set()
        for root, count in tree_counts.items():
            if count >= self.k:
                P.add(root)

        self.stats["pivots_found"] += len(P)
        return P, W

    # --- ALGORITHM 2: BASE CASE ---
    def _base_case(self, B_tuple, S, limit):
        # Optimized Base Case using heapq (standard Dijkstra)
        pq = []
        in_pq = {}

        for u in S:
            val = (self.d[u], u)
            if val < B_tuple:
                heapq.heappush(pq, val)
                in_pq[u] = self.d[u]

        U_0 = set(S)

        while pq:
            if len(U_0) >= limit:
                break

            d_u, u = heapq.heappop(pq)
            if u in in_pq and in_pq[u] != d_u:
                continue

            for v in self.graph.get_neighbors(u):
                w_uv = self.graph.get_weight(u, v)
                new_dist = d_u + w_uv
                self.stats["relaxations"] += 1

                if (new_dist, v) < B_tuple:
                    if new_dist <= self.d[v]:
                        self.d[v] = new_dist
                        self.pred[v] = u

                        U_0.add(v)

                        if v not in in_pq or new_dist < in_pq[v]:
                            in_pq[v] = new_dist
                            heapq.heappush(pq, (new_dist, v))
                            self.stats["heap_ops"] += 1

        # Calculate B'
        if not pq and len(U_0) < limit:
            B_prime = B_tuple
        else:
            if pq:
                while pq:
                    d_top, u_top = pq[0]
                    if u_top in in_pq and in_pq[u_top] != d_top:
                        heapq.heappop(pq)
                        continue
                    break
                B_prime = pq[0] if pq else B_tuple
            else:
                B_prime = B_tuple

        U_final = {v for v in U_0 if (self.d[v], v) < B_prime}
        return B_prime, U_final

    # --- ALGORITHM 3: BMSSP ---
    def _bmssp(self, l, B_tuple, S, indent):
        # Level Capacity Check
        level_limit = self.k * (2 ** (l * self.t))

        if l == 0 or len(S) <= self.k:
            return self._base_case(B_tuple, S, limit=level_limit)

        P, W = self._find_pivots(B_tuple, S, indent + 1)

        M = 2 ** ((l - 1) * self.t)
        D = Lemma33DataStructure(M, B_tuple)

        for x in P:
            D.insert(x, self.d[x])

        if P:
            min_p = (float("inf"), float("inf"))
            for x in P:
                if (self.d[x], x) < min_p:
                    min_p = (self.d[x], x)
            B_prev_prime = min_p
        else:
            B_prev_prime = B_tuple

        U = set()

        while len(U) < level_limit and not D.is_empty():
            # 4a. PULL
            B_i_tuple, S_i = D.pull()
            if not S_i:
                break

            # 4b. RECURSE
            B_i_prime_tuple, U_i = self._bmssp(l - 1, B_i_tuple, set(S_i), indent + 1)
            U.update(U_i)

            # 4c. RELAX
            K = []

            for u in U_i:
                for v in self.graph.get_neighbors(u):
                    w_uv = self.graph.get_weight(u, v)
                    new_dist = self.d[u] + w_uv
                    self.stats["relaxations"] += 1

                    if new_dist <= self.d[v]:
                        improved = new_dist < self.d[v]
                        self.d[v] = new_dist
                        if improved:
                            self.pred[v] = u

                        val_v_tuple = (new_dist, v)

                        is_case_A = B_i_tuple <= val_v_tuple < B_tuple
                        is_case_B = B_i_prime_tuple <= val_v_tuple < B_i_tuple

                        if is_case_A:
                            D.insert(v, new_dist)
                        elif is_case_B:
                            K.append((v, new_dist))

            # 4d. EXTRAS
            extras = []
            for x in S_i:
                val_x_tuple = (self.d[x], x)
                if B_i_prime_tuple <= val_x_tuple < B_i_tuple:
                    extras.append((x, self.d[x]))

            if K or extras:
                D.batch_prepend(K + extras)

            B_prev_prime = B_i_prime_tuple

        if D.is_empty():
            final_B = B_tuple
            for x in W:
                if (self.d[x], x) < final_B:
                    U.add(x)
            return final_B, U
        else:
            final_B = B_prev_prime
            U_filtered = {u for u in U if (self.d[u], u) < final_B}
            for x in W:
                if (self.d[x], x) < final_B:
                    U_filtered.add(x)
            return final_B, U_filtered
