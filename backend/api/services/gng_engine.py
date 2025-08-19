# Implements the topological mapping layer (Cortical Layer).
# This is a standard Growing Neural Gas (GNG) implementation as described
# in Section 3.1 of the Project Chimera specification, with additions for
# dynamic learning rates to protect established memories.

import numpy as np

class GNG_Engine:
    def __init__(self, dimensions, **kwargs):
        """
        Initializes the GNG Engine.

        Args:
            dimensions (int): The dimensionality of the input vectors.
            **kwargs: Hyperparameters for the GNG.
        """
        self.dimensions = dimensions

        # Hyperparameters from spec (Section 3.1)
        self.winner_learning_rate = kwargs.get('gng_winner_learning_rate', 0.1) # eb
        self.neighbor_learning_rate = kwargs.get('gng_neighbor_learning_rate', 0.01) # en
        self.max_edge_age = kwargs.get('gng_max_edge_age', 50)
        self.n_iter_before_neuron_added = kwargs.get('gng_n_iter_before_neuron_added', 100) # lambda
        self.after_split_error_decay_rate = kwargs.get('gng_after_split_error_decay_rate', 0.5) # alpha
        self.error_decay_rate = kwargs.get('gng_error_decay_rate', 0.001) # beta
        # New hyperparameter for utility-based memory protection
        self.utility_decay_rate = kwargs.get('gng_utility_decay_rate', 0.0005) # New
        self.utility_gain = kwargs.get('gng_utility_gain', 1.0) # New

        # State representation (Section 3.1)
        # Node List: { node_id: {'weight': np.array, 'error': float, 'utility': float} }
        self.nodes = {}
        # Edge List: set of (node_id_1, node_id_2, age) tuples
        self.edges = set()

        self._next_node_id = 0
        self._iterations = 0

        # Initialize with two random nodes
        self._add_node(np.random.randn(self.dimensions))
        self._add_node(np.random.randn(self.dimensions))

    def _add_node(self, weight_vector, error=0.0, utility=1.0):
        node_id = self._next_node_id
        self.nodes[node_id] = {
            'weight': weight_vector.astype('float32'),
            'error': error,
            'utility': utility
        }
        self._next_node_id += 1
        return node_id

    def process_input(self, input_vector):
        """
        Processes a single input vector according to the GNG algorithm.
        This corresponds to the "Algorithm Deep Dive" in Section 3.1.
        """
        if len(self.nodes) < 2:
            return

        # 1. Find Winners
        s1_id, s2_id = self._find_winners(input_vector)

        # 2. Update Utility and Error Accumulation
        winner_dist_sq = np.sum((self.nodes[s1_id]['weight'] - input_vector) ** 2)
        self.nodes[s1_id]['error'] += winner_dist_sq
        self.nodes[s1_id]['utility'] += self.utility_gain

        # 3. Adaptation with Dynamic Learning Rates
        # Learning rate is inversely proportional to the log of utility.
        # This makes high-utility (established) nodes more resistant to change.
        winner_utility = self.nodes[s1_id]['utility']
        dynamic_winner_lr = self.winner_learning_rate / np.log1p(winner_utility)
        self.nodes[s1_id]['weight'] += dynamic_winner_lr * (input_vector - self.nodes[s1_id]['weight'])

        neighbor_ids = self._get_neighbors(s1_id)
        for n_id in neighbor_ids:
            neighbor_utility = self.nodes[n_id]['utility']
            dynamic_neighbor_lr = self.neighbor_learning_rate / np.log1p(neighbor_utility)
            self.nodes[n_id]['weight'] += dynamic_neighbor_lr * (input_vector - self.nodes[n_id]['weight'])

        # 4. Edge Management
        self._update_edges(s1_id, s2_id)

        # 5. Node Insertion (Growth)
        self._iterations += 1
        if self._iterations % self.n_iter_before_neuron_added == 0:
            self._insert_node()

        # 6. Global Damping (Error and Utility)
        for node_id in self.nodes:
            self.nodes[node_id]['error'] *= (1 - self.error_decay_rate)
            self.nodes[node_id]['utility'] *= (1 - self.utility_decay_rate)

    def _find_winners(self, input_vector):
        """Finds the two nodes closest to the input vector."""
        node_ids = list(self.nodes.keys())
        weights = np.array([self.nodes[nid]['weight'] for nid in node_ids])
        distances = np.linalg.norm(weights - input_vector, axis=1)
        sorted_indices = np.argsort(distances)
        return node_ids[sorted_indices[0]], node_ids[sorted_indices[1]]

    def _get_neighbors(self, node_id):
        """Returns the set of node IDs connected to the given node."""
        neighbors = set()
        for u, v, _ in self.edges:
            if u == node_id:
                neighbors.add(v)
            elif v == node_id:
                neighbors.add(u)
        return neighbors

    def _update_edges(self, s1_id, s2_id):
        """
        Updates edges according to step 4 of the GNG algorithm.
        - Resets age of edge between s1 and s2 (or creates it).
        - Increments age of all edges connected to s1.
        - Removes old edges and any resulting isolated nodes.
        """
        # Reset age or create edge between s1 and s2
        edge = tuple(sorted((s1_id, s2_id)))
        found_edge = next((e for e in self.edges if e[0] == edge[0] and e[1] == edge[1]), None)
        if found_edge:
            self.edges.remove(found_edge)
            self.edges.add((found_edge[0], found_edge[1], 0))
        else:
            self.edges.add((edge[0], edge[1], 0))

        # Increment age of all edges connected to the winner s1
        edges_to_update = {e for e in self.edges if s1_id in e[:2]}
        for e in edges_to_update:
            self.edges.remove(e)
            self.edges.add((e[0], e[1], e[2] + 1))

        # Remove edges older than max_edge_age
        self.edges = {e for e in self.edges if e[2] <= self.max_edge_age}

        # Remove isolated nodes
        connected_nodes = {node for edge in self.edges for node in edge[:2]}
        isolated_node_ids = [nid for nid in self.nodes if nid not in connected_nodes]
        if len(self.nodes) - len(isolated_node_ids) >= 2: # Ensure at least 2 nodes remain
            for nid in isolated_node_ids:
                if nid in self.nodes: # Check if not already deleted
                    del self.nodes[nid]

    def _insert_node(self):
        """Inserts a new node into the network (step 5)."""
        if len(self.nodes) < 2: return

        # Find node q with the largest error
        q_id = max(self.nodes, key=lambda nid: self.nodes[nid]['error'])

        # Find neighbor f of q with the largest error
        q_neighbors = self._get_neighbors(q_id)
        if not q_neighbors: return
        f_id = max(q_neighbors, key=lambda nid: self.nodes[nid]['error'])

        # Insert new node r halfway between q and f
        q_node, f_node = self.nodes[q_id], self.nodes[f_id]
        r_weight = 0.5 * (q_node['weight'] + f_node['weight'])
        # Inherit utility from parents
        r_utility = 0.5 * (q_node['utility'] + f_node['utility'])
        r_id = self._add_node(r_weight, utility=r_utility)

        # Remove original edge between q and f, add new edges
        edge_to_remove = tuple(sorted((q_id, f_id)))
        original_edge = next((e for e in self.edges if e[0] == edge_to_remove[0] and e[1] == edge_to_remove[1]), None)
        if original_edge:
            self.edges.remove(original_edge)

        self.edges.add((tuple(sorted((q_id, r_id)))[0], tuple(sorted((q_id, r_id)))[1], 0))
        self.edges.add((tuple(sorted((f_id, r_id)))[0], tuple(sorted((f_id, r_id)))[1], 0))

        # Decrease errors of q and f
        q_node['error'] *= self.after_split_error_decay_rate
        f_node['error'] *= self.after_split_error_decay_rate

        # Initialize error of new node r to the (now decreased) error of q
        self.nodes[r_id]['error'] = self.nodes[q_id]['error']

    def get_state(self):
        """Returns the serializable state of the GNG."""
        # Convert numpy arrays in nodes to lists for JSON serialization
        serializable_nodes = {
            nid: {
                'weight': node['weight'].tolist(),
                'error': node['error'],
                'utility': node.get('utility', 1.0) # Default for older states
            }
            for nid, node in self.nodes.items()
        }
        return {
            'dimensions': self.dimensions,
            'nodes': serializable_nodes,
            'edges': list(self.edges),
            'next_node_id': self._next_node_id,
            'iterations': self._iterations
        }

    @classmethod
    def from_state(cls, state_dict, **kwargs):
        """Creates a GNG instance from a state dictionary."""
        gng = cls(state_dict['dimensions'], **kwargs)
        # Convert lists back to numpy arrays
        gng.nodes = {
            int(nid): {
                'weight': np.array(node['weight']),
                'error': node['error'],
                'utility': node.get('utility', 1.0) # Default for older states
            }
            for nid, node in state_dict['nodes'].items()
        }
        gng.edges = {tuple(e) for e in state_dict['edges']}
        gng._next_node_id = state_dict['next_node_id']
        gng._iterations = state_dict['iterations']
        return gng
