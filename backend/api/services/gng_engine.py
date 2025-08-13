# Implements the topological mapping layer (Cortical Layer).
# This is a single Growing Neural Gas (GNG) instance that learns the
# topology of the input data (stable attractors from the Hopfield Core).
# See Section 3.1 of the Project Chimera specification.

import numpy as np
from collections import namedtuple

GNGNode = namedtuple('GNGNode', ['id', 'weight_vector', 'error'])
GNGEdge = namedtuple('GNGEdge', ['source_node_id', 'target_node_id', 'age'])

class GNG_Engine:
    def __init__(self, dimensions, **kwargs):
        """
        Initializes the GNG Engine.

        Args:
            dimensions (int): The dimensionality of the input vectors.
            **kwargs: Hyperparameters for the GNG algorithm.
        """
        self.dimensions = dimensions

        # Hyperparameters from Section 3.1
        self.winner_learning_rate = kwargs.get('winner_learning_rate', 0.05)       # ε_w
        self.neighbor_learning_rate = kwargs.get('neighbor_learning_rate', 0.001)   # ε_n
        self.max_edge_age = kwargs.get('max_edge_age', 50)
        self.n_iter_before_neuron_added = kwargs.get('n_iter_before_neuron_added', 100)
        self.error_decay_rate = kwargs.get('error_decay_rate', 0.0005)              # β
        self.after_split_error_decay_rate = kwargs.get('after_split_error_decay_rate', 0.5) # α_split

        self.nodes = {}
        self.edges = set()
        self._next_node_id = 0
        self._iterations = 0

        # Initialize with two random nodes
        self._add_node(np.random.rand(self.dimensions))
        self._add_node(np.random.rand(self.dimensions))

    def _add_node(self, weight_vector):
        node_id = self._next_node_id
        self.nodes[node_id] = {'weight': weight_vector, 'error': 0.0}
        self._next_node_id += 1
        return node_id

    def process_input(self, input_vector):
        """
        Processes a single input vector, performing one full GNG learning iteration.
        """
        if not self.nodes:
            self._add_node(input_vector)
            self._add_node(input_vector)

        # 1. Find winner and second winner nodes
        s1_id, s2_id = self._find_winners(input_vector)

        if s1_id is None: return # Should not happen if nodes exist

        # 2. Edge Management & Error Accumulation
        self._update_winner_edges(s1_id, s2_id)
        self.nodes[s1_id]['error'] += np.linalg.norm(input_vector - self.nodes[s1_id]['weight'])**2

        # 3. Adaptation
        self._adapt_weights(s1_id, input_vector)

        # 4. Node Insertion (Growth)
        self._iterations += 1
        if self._iterations % self.n_iter_before_neuron_added == 0:
            self._insert_node()

        # 5. Global Damping
        for node_id in self.nodes:
            self.nodes[node_id]['error'] *= (1 - self.error_decay_rate)

    def _find_winners(self, input_vector):
        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2:
            return node_ids[0] if node_ids else None, None

        weights = np.array([self.nodes[nid]['weight'] for nid in node_ids])
        distances = np.linalg.norm(weights - input_vector, axis=1)

        sorted_indices = np.argsort(distances)
        s1_index, s2_index = sorted_indices[0], sorted_indices[1]

        return node_ids[s1_index], node_ids[s2_index]

    def _update_winner_edges(self, s1_id, s2_id):
        # Increment age of all edges connected to the winner
        edges_to_increment = {edge for edge in self.edges if s1_id in edge}
        for edge in edges_to_increment:
            self.edges.remove(edge)
            self.edges.add((edge[0], edge[1], edge[2] + 1))

        # Create or reset edge between s1 and s2
        edge_tuple = tuple(sorted((s1_id, s2_id)))
        found_edge = next((e for e in self.edges if e[0] == edge_tuple[0] and e[1] == edge_tuple[1]), None)

        if found_edge:
            self.edges.remove(found_edge)
        self.edges.add((edge_tuple[0], edge_tuple[1], 0))

        # Remove old edges
        self.edges = {edge for edge in self.edges if edge[2] <= self.max_edge_age}

        # Remove isolated nodes
        connected_nodes = {node for edge in self.edges for node in edge[:2]}
        isolated_nodes = set(self.nodes.keys()) - connected_nodes
        for node_id in isolated_nodes:
            # Do not remove nodes if we only have 2 left
            if len(self.nodes) > 2:
                del self.nodes[node_id]

    def _adapt_weights(self, s1_id, input_vector):
        # Update winner
        winner_node = self.nodes[s1_id]
        winner_node['weight'] += self.winner_learning_rate * (input_vector - winner_node['weight'])

        # Update direct neighbors
        neighbor_ids = {edge[1] if edge[0] == s1_id else edge[0] for edge in self.edges if s1_id in edge[:2]}
        for nid in neighbor_ids:
            neighbor_node = self.nodes[nid]
            neighbor_node['weight'] += self.neighbor_learning_rate * (input_vector - neighbor_node['weight'])

    def _insert_node(self):
        if len(self.nodes) < 2: return

        # Find node q with max error
        q_id = max(self.nodes, key=lambda nid: self.nodes[nid]['error'])

        # Find neighbor f of q with max error
        neighbor_ids = {e[1] if e[0] == q_id else e[0] for e in self.edges if q_id in e[:2]}
        if not neighbor_ids: return

        f_id = max(neighbor_ids, key=lambda nid: self.nodes[nid]['error'])

        # Insert new node r between q and f
        q_node = self.nodes[q_id]
        f_node = self.nodes[f_id]
        new_weight = 0.5 * (q_node['weight'] + f_node['weight'])
        r_id = self._add_node(new_weight)

        # Update edges
        edge_tuple = tuple(sorted((q_id, f_id)))
        old_edge = next((e for e in self.edges if e[0] == edge_tuple[0] and e[1] == edge_tuple[1]), None)
        if old_edge:
            self.edges.remove(old_edge)

        self.edges.add((tuple(sorted((q_id, r_id)))[0], tuple(sorted((q_id, r_id)))[1], 0))
        self.edges.add((tuple(sorted((f_id, r_id)))[0], tuple(sorted((f_id, r_id)))[1], 0))

        # Update errors
        q_node['error'] *= self.after_split_error_decay_rate
        f_node['error'] *= self.after_split_error_decay_rate
        self.nodes[r_id]['error'] = q_node['error']

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
