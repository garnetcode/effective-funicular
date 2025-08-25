import heapq

class OptionModel:
    """
    A simple model for an "option" which represents a transition between two nodes in the STAG.
    It learns the expected reward and duration of taking this option.
    """
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        self.total_reward = 0
        self.total_duration = 0
        self.count = 0

    def update(self, reward, duration):
        """Update the model with a new observation."""
        self.total_reward += reward
        self.total_duration += duration
        self.count += 1

    @property
    def expected_reward(self):
        return self.total_reward / self.count if self.count > 0 else 0

    @property
    def expected_duration(self):
        return self.total_duration / self.count if self.count > 0 else float('inf')


class GraphPlanner:
    """
    A planner that performs A* search over a STAG to find a high-level plan.
    """
    def __init__(self):
        pass

    def plan(self, stag_graph, option_models, start_node_id, goal_node_id):
        """
        Finds a path from start_node_id to goal_node_id using A* search.
        The "cost" of traversing an edge is the expected duration of the option,
        and we aim to maximize the total expected reward.
        """
        if start_node_id not in stag_graph['nodes'] or goal_node_id not in stag_graph['nodes']:
            return None # Start or goal not in graph

        # A* search algorithm
        open_set = [(0, start_node_id)]  # (priority, node_id)
        came_from = {}
        g_score = {node_id: float('inf') for node_id in stag_graph['nodes']}
        g_score[start_node_id] = 0

        # The score is the estimated total cost from start to goal
        f_score = {node_id: float('inf') for node_id in stag_graph['nodes']}
        f_score[start_node_id] = self._heuristic(start_node_id, goal_node_id, stag_graph)

        while open_set:
            current_f_score, current_node_id = heapq.heappop(open_set)

            # If we've already found a better path, skip
            if current_f_score > f_score[current_node_id]:
                continue

            if current_node_id == goal_node_id:
                return self._reconstruct_path(came_from, current_node_id)

            neighbors = self._get_neighbors(current_node_id, stag_graph)
            for neighbor_id in neighbors:
                option_key = (current_node_id, neighbor_id)
                if option_key not in option_models:
                    continue

                option = option_models[option_key]
                # Cost is duration
                tentative_g_score = g_score[current_node_id] + option.expected_duration

                if tentative_g_score < g_score[neighbor_id]:
                    came_from[neighbor_id] = current_node_id
                    g_score[neighbor_id] = tentative_g_score
                    f_score[neighbor_id] = tentative_g_score + self._heuristic(neighbor_id, goal_node_id, stag_graph)
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))

        return None # No path found

    def _heuristic(self, node1_id, node2_id, stag_graph):
        """
        A heuristic for the A* search. It's the Euclidean distance between the node weights.
        This is an admissible heuristic if the cost of traversing an edge is at least
        the distance between the nodes, which is not guaranteed here. But it's a
        common choice when no better heuristic is available.
        """
        node1 = stag_graph['nodes'][node1_id]
        node2 = stag_graph['nodes'][node2_id]
        return ((node1['weight'] - node2['weight'])**2).sum()**0.5

    def _get_neighbors(self, node_id, stag_graph):
        """Returns the neighbors of a node in the STAG graph."""
        neighbors = set()
        for edge in stag_graph['edges']:
            if edge[0] == node_id:
                neighbors.add(edge[1])
            elif edge[1] == node_id:
                neighbors.add(edge[0])
        return neighbors

    def _reconstruct_path(self, came_from, current_node_id):
        """Reconstructs the path from the came_from dictionary."""
        total_path = [current_node_id]
        while current_node_id in came_from:
            current_node_id = came_from[current_node_id]
            total_path.append(current_node_id)
        return total_path[::-1]
