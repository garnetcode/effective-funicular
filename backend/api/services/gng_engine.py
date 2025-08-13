# Implements the topological mapping layer (Cortical Layer).
# This is a single Growing Neural Gas (GNG) instance that learns the
# topology of the input data. This version includes advanced optimization
# features: edge utility and node merging.

import numpy as np
import itertools

class GNG_Engine:
    def __init__(self, dimensions, **kwargs):
        self.dimensions = dimensions

        # Standard GNG Hyperparameters
        self.winner_learning_rate = kwargs.get('winner_learning_rate', 0.05)
        self.neighbor_learning_rate = kwargs.get('neighbor_learning_rate', 0.001)
        self.max_edge_age = kwargs.get('max_edge_age', 50)
        self.n_iter_before_neuron_added = kwargs.get('n_iter_before_neuron_added', 100)
        self.error_decay_rate = kwargs.get('error_decay_rate', 0.0005)
        self.after_split_error_decay_rate = kwargs.get('after_split_error_decay_rate', 0.5)

        # Hyperparameters for advanced optimization
        self.utility_prune_threshold = kwargs.get('utility_prune_threshold', 0.05)
        self.n_iter_before_pruning = kwargs.get('n_iter_before_pruning', 400)
        self.merge_threshold = kwargs.get('merge_threshold', 0.1)
        self.n_iter_before_merging = kwargs.get('n_iter_before_merging', 500)

        self.nodes = {}
        self.edges = set() # Edges are tuples: (id1, id2, age, utility)
        self._next_node_id = 0
        self._iterations = 0

        self._add_node(np.random.rand(self.dimensions))
        self._add_node(np.random.rand(self.dimensions))

    def _add_node(self, weight_vector, error=0.0):
        node_id = self._next_node_id
        self.nodes[node_id] = {'weight': weight_vector, 'error': error}
        self._next_node_id += 1
        return node_id

    def process_input(self, input_vector):
        if len(self.nodes) < 2: self._add_node(input_vector)

        s1_id, s2_id = self._find_winners(input_vector)
        if s1_id is None: return

        self._update_winner_edges(s1_id, s2_id)
        self.nodes[s1_id]['error'] += np.linalg.norm(input_vector - self.nodes[s1_id]['weight'])**2
        self._adapt_weights(s1_id, input_vector)

        self._iterations += 1
        if self._iterations % self.n_iter_before_neuron_added == 0: self._insert_node()
        if self._iterations % self.n_iter_before_pruning == 0: self._prune_by_utility()
        if self._iterations % self.n_iter_before_merging == 0: self._merge_nodes()

        for node_id in self.nodes: self.nodes[node_id]['error'] *= (1 - self.error_decay_rate)

    def _find_winners(self, input_vector):
        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2: return node_ids[0] if node_ids else None, None
        weights = np.array([self.nodes[nid]['weight'] for nid in node_ids])
        distances = np.linalg.norm(weights - input_vector, axis=1)
        sorted_indices = np.argsort(distances)
        return node_ids[sorted_indices[0]], node_ids[sorted_indices[1]]

    def _update_winner_edges(self, s1_id, s2_id):
        edges_to_increment = {edge for edge in self.edges if s1_id in edge[:2]}
        for edge in edges_to_increment:
            self.edges.remove(edge)
            self.edges.add((edge[0], edge[1], edge[2] + 1, edge[3]))

        edge_tuple = tuple(sorted((s1_id, s2_id)))
        found_edge = next((e for e in self.edges if e[0] == edge_tuple[0] and e[1] == edge_tuple[1]), None)

        if found_edge:
            self.edges.remove(found_edge)
            self.edges.add((found_edge[0], found_edge[1], 0, found_edge[3] + 1))
        else:
            self.edges.add((edge_tuple[0], edge_tuple[1], 0, 1))

        self.edges = {edge for edge in self.edges if edge[2] <= self.max_edge_age}

    def _adapt_weights(self, s1_id, input_vector):
        self.nodes[s1_id]['weight'] += self.winner_learning_rate * (input_vector - self.nodes[s1_id]['weight'])
        neighbor_ids = {e[1] if e[0] == s1_id else e[0] for e in self.edges if s1_id in e[:2]}
        for nid in neighbor_ids:
            self.nodes[nid]['weight'] += self.neighbor_learning_rate * (input_vector - self.nodes[nid]['weight'])
            edge_tuple = tuple(sorted((s1_id, nid)))
            found_edge = next((e for e in self.edges if e[0] == edge_tuple[0] and e[1] == edge_tuple[1]), None)
            if found_edge:
                self.edges.remove(found_edge)
                self.edges.add((found_edge[0], found_edge[1], found_edge[2], found_edge[3] + 1))

    def _insert_node(self):
        if len(self.nodes) < 2: return
        q_id = max(self.nodes, key=lambda nid: self.nodes[nid]['error'])
        neighbor_ids = {e[1] if e[0] == q_id else e[0] for e in self.edges if q_id in e[:2]}
        if not neighbor_ids: return
        f_id = max(neighbor_ids, key=lambda nid: self.nodes[nid]['error'])
        q_node, f_node = self.nodes[q_id], self.nodes[f_id]
        r_id = self._add_node(0.5 * (q_node['weight'] + f_node['weight']))
        edge_tuple = tuple(sorted((q_id, f_id)))
        old_edge = next((e for e in self.edges if e[0] == edge_tuple[0] and e[1] == edge_tuple[1]), None)
        if old_edge: self.edges.remove(old_edge)
        new_utility = old_edge[3] / 2 if old_edge else 1
        self.edges.add((tuple(sorted((q_id, r_id)))[0], tuple(sorted((q_id, r_id)))[1], 0, new_utility))
        self.edges.add((tuple(sorted((f_id, r_id)))[0], tuple(sorted((f_id, r_id)))[1], 0, new_utility))
        q_node['error'] *= self.after_split_error_decay_rate
        f_node['error'] *= self.after_split_error_decay_rate
        self.nodes[r_id]['error'] = q_node['error']

    def _prune_by_utility(self):
        if not self.edges: return
        avg_utility = np.mean([edge[3] for edge in self.edges])
        self.edges = {edge for edge in self.edges if edge[3] >= avg_utility * self.utility_prune_threshold}
        self._remove_isolated_nodes()

    def _merge_nodes(self):
        nodes_to_merge = []
        node_ids = list(self.nodes.keys())
        for id1, id2 in itertools.combinations(node_ids, 2):
            if id1 in self.nodes and id2 in self.nodes:
                dist = np.linalg.norm(self.nodes[id1]['weight'] - self.nodes[id2]['weight'])
                if dist < self.merge_threshold:
                    nodes_to_merge.append((id1, id2))

        for id1, id2 in nodes_to_merge:
            if id1 not in self.nodes or id2 not in self.nodes: continue # Already merged

            # Create new merged node
            node1, node2 = self.nodes[id1], self.nodes[id2]
            new_weight = (node1['weight'] + node2['weight']) / 2
            new_error = node1['error'] + node2['error']
            new_id = self._add_node(new_weight, new_error)

            # Re-route edges
            edges_to_remove = set()
            edges_to_add = set()
            for edge in self.edges:
                if id1 in edge[:2] or id2 in edge[:2]:
                    edges_to_remove.add(edge)
                    other_node = edge[1] if edge[0] in [id1, id2] else edge[0]
                    if other_node not in [id1, id2]:
                        new_edge = tuple(sorted((new_id, other_node)))
                        # Keep the highest utility if multiple edges connect to the same node
                        existing = next((e for e in edges_to_add if e[0]==new_edge[0] and e[1]==new_edge[1]), None)
                        if existing:
                           edges_to_add.remove(existing)
                           edges_to_add.add((new_edge[0], new_edge[1], 0, max(existing[3], edge[3])))
                        else:
                           edges_to_add.add((new_edge[0], new_edge[1], 0, edge[3]))

            self.edges -= edges_to_remove
            self.edges.update(edges_to_add)

            # Delete old nodes
            del self.nodes[id1]
            del self.nodes[id2]

        self._remove_isolated_nodes()

    def _remove_isolated_nodes(self):
        if len(self.nodes) <= 2: return
        connected_nodes = {node for edge in self.edges for node in edge[:2]}
        isolated_nodes = set(self.nodes.keys()) - connected_nodes
        for node_id in isolated_nodes:
            del self.nodes[node_id]

    def get_state(self):
        return {
            'dimensions': self.dimensions,
            'nodes': {nid: {'weight': n['weight'].tolist(), 'error': n['error']} for nid, n in self.nodes.items()},
            'edges': list(self.edges),
            'next_node_id': self._next_node_id,
            'iterations': self._iterations,
        }

    @classmethod
    def from_state(cls, state_dict, **kwargs):
        gng = cls(state_dict['dimensions'], **kwargs)
        gng.nodes = {int(nid): {'weight': np.array(n['weight']), 'error': n['error']} for nid, n in state_dict['nodes'].items()}
        gng.edges = {tuple(e) for e in state_dict['edges']}
        gng._next_node_id = state_dict['next_node_id']
        gng._iterations = state_dict['iterations']
        return gng
