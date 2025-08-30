# Implements the topological mapping layer (Cortical Layer).
# This is a standard Growing Neural Gas (GNG) implementation as described
# in Section 3.1 of the Project Chimera specification, with additions for
# dynamic learning rates and FAISS-based search for performance.

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None

def _safe_unit(x, eps=1e-8):
    """Safely normalizes a vector to unit length."""
    n = np.linalg.norm(x)
    if n > 0:
        return x / (n + eps)
    return x

class GNG_Engine:
    def __init__(self, dimensions, **kwargs):
        """
        Initializes the GNG Engine.

        Args:
            dimensions (int): The dimensionality of the input vectors.
            **kwargs: Hyperparameters for the GNG.
        """
        self.dimensions = dimensions

        # Hyperparameters
        # Tuning phase learning rates (using existing param names)
        self.winner_learning_rate = kwargs.get('gng_winner_learning_rate', 0.1)
        self.neighbor_learning_rate = kwargs.get('gng_neighbor_learning_rate', 0.01)
        # Adaptive learning rate params
        self.learning_rate_decay = kwargs.get('gng_learning_rate_decay', 0.9995)
        self.winner_learning_rate_initial = kwargs.get('gng_winner_learning_rate_initial', 0.8)
        self.neighbor_learning_rate_initial = kwargs.get('gng_neighbor_learning_rate_initial', 0.1)
        self.max_edge_age = kwargs.get('gng_max_edge_age', 50)
        self.n_iter_before_neuron_added = kwargs.get('gng_n_iter_before_neuron_added', 100)
        self.after_split_error_decay_rate = kwargs.get('gng_after_split_error_decay_rate', 0.5)
        self.error_decay_fast = kwargs.get('gng_error_decay_fast', 0.01)
        self.error_decay_slow = kwargs.get('gng_error_decay_slow', 0.001)
        self.utility_decay_rate = kwargs.get('gng_utility_decay_rate', 0.0005)
        self.utility_gain = kwargs.get('gng_utility_gain', 1.0)
        self.faiss_rebuild_interval = kwargs.get('gng_faiss_rebuild_interval', 100)
        self.pruning_grace_period = kwargs.get('gng_pruning_grace_period', 500)
        self.utility_floor = kwargs.get('gng_utility_floor', 0.1)
        self.utility_clip = kwargs.get('gng_utility_clip', 10.0)
        self.sf_dimension = kwargs.get('sf_dimension', 64)

        # State representation
        self.nodes = {} # { node_id: {'weight': np.array, 'error': float, 'utility': float} }
        self.edges = set()

        self._next_node_id = 0
        self._iterations = 0

        # FAISS index for fast nearest neighbor search
        self.faiss_index = None
        self.faiss_id_map = [] # Maps faiss index to gng node_id

        # Initialize with two random nodes
        self._add_node(np.random.randn(self.dimensions))
        self._add_node(np.random.randn(self.dimensions))

    def _build_faiss_index(self):
        """Builds or rebuilds the FAISS index for fast node lookups."""
        if faiss is None or len(self.nodes) < 2:
            self.faiss_index = None
            return

        # This was creating too much noise during training.
        # print(f"Building FAISS index for {len(self.nodes)} nodes...")
        self.faiss_id_map = list(self.nodes.keys())
        weights = np.array([self.nodes[nid]['weight'] for nid in self.faiss_id_map]).astype('float32')

        self.faiss_index = faiss.IndexFlatL2(self.dimensions)
        self.faiss_index.add(weights)

    def _add_node(self, weight_vector, error_fast=0.0, error_slow=0.0, utility=1.0):
        node_id = self._next_node_id

        # Normalize the weight vector to have a length of 1
        weight_vector = _safe_unit(weight_vector)

        self.nodes[node_id] = {
            'weight': weight_vector.astype('float32'),
            'error_fast': error_fast,
            'error_slow': error_slow,
            'utility': utility,
            'creation_iteration': self._iterations,
            'psi': np.zeros(self.sf_dimension)
        }
        self._next_node_id += 1
        self.faiss_index = None # Invalidate index
        return node_id

    def process_input(self, input_vector, reward=0):
        """
        Processes a single input vector according to the GNG algorithm.
        The reward signal modulates the utility gain for the winning node.
        """
        if len(self.nodes) < 2:
            return

        input_vector = _safe_unit(input_vector)

        # 1. Find Winners
        s1_id, s2_id = self._find_winners(input_vector)
        if s1_id is None: return # Not enough nodes to find winners

        # 2. Update Utility and Error Accumulation
        winner_dist_sq = np.sum((self.nodes[s1_id]['weight'] - input_vector) ** 2)
        self.nodes[s1_id]['error_fast'] += winner_dist_sq
        self.nodes[s1_id]['error_slow'] += winner_dist_sq
        reward_gain = np.tanh(reward)
        self.nodes[s1_id]['utility'] += self.utility_gain * (1 + reward_gain)

        # 3. Adaptation with Dynamic Learning Rates
        # Calculate decayed learning rates
        decay_factor = self.learning_rate_decay ** self._iterations
        current_winner_lr = self.winner_learning_rate_initial * decay_factor
        current_neighbor_lr = self.neighbor_learning_rate_initial * decay_factor

        winner_utility = self.nodes[s1_id]['utility']
        dynamic_winner_lr = current_winner_lr / (1e-5 + np.log1p(winner_utility))
        self.nodes[s1_id]['weight'] += dynamic_winner_lr * (input_vector - self.nodes[s1_id]['weight'])

        # Re-normalize the winner's weight vector
        self.nodes[s1_id]['weight'] = _safe_unit(self.nodes[s1_id]['weight'])

        neighbor_ids = self._get_neighbors(s1_id)
        for n_id in neighbor_ids:
            neighbor_utility = self.nodes[n_id]['utility']
            dynamic_neighbor_lr = current_neighbor_lr / (1e-5 + np.log1p(neighbor_utility))
            self.nodes[n_id]['weight'] += dynamic_neighbor_lr * (input_vector - self.nodes[n_id]['weight'])

            # Re-normalize the neighbor's weight vector
            self.nodes[n_id]['weight'] = _safe_unit(self.nodes[n_id]['weight'])

        self.faiss_index = None # Invalidate index due to weight changes

        # 4. Edge Management
        self._update_edges(s1_id, s2_id)

        # 5. Node Insertion (Growth)
        self._iterations += 1
        if self._iterations % self.n_iter_before_neuron_added == 0:
            self._insert_node()

        # 6. Global Damping (Error and Utility)
        for node_id in self.nodes:
            self.nodes[node_id]['error_fast'] *= (1 - self.error_decay_fast)
            self.nodes[node_id]['error_slow'] *= (1 - self.error_decay_slow)
            self.nodes[node_id]['utility'] *= (1 - self.utility_decay_rate)
            # Clip and floor utility
            self.nodes[node_id]['utility'] = np.clip(self.nodes[node_id]['utility'], self.utility_floor, self.utility_clip)


    def _find_winners(self, input_vector):
        """Finds the two nodes closest to the input vector using FAISS or numpy."""
        if len(self.nodes) < 2:
            return None, None

        # Rebuild index if it's invalidated or periodically
        if faiss and (self.faiss_index is None or (self._iterations % self.faiss_rebuild_interval == 0)):
            self._build_faiss_index()

        if self.faiss_index:
            input_vector_faiss = np.array([input_vector]).astype('float32')
            _, indices = self.faiss_index.search(input_vector_faiss, 2)
            s1_gng_id = self.faiss_id_map[indices[0][0]]
            s2_gng_id = self.faiss_id_map[indices[0][1]]
            return s1_gng_id, s2_gng_id
        else:
            # NumPy fallback
            node_ids = list(self.nodes.keys())
            weights = np.array([self.nodes[nid]['weight'] for nid in node_ids])
            distances_sq = np.sum((weights - input_vector) ** 2, axis=1)

            # Get the indices of the two smallest distances
            # Using argpartition is more efficient than argsort for finding k smallest items
            if len(distances_sq) > 2:
                two_smallest_indices = np.argpartition(distances_sq, 2)[:2]
            else:
                two_smallest_indices = np.argsort(distances_sq)[:2]

            s1_idx, s2_idx = two_smallest_indices
            return node_ids[s1_idx], node_ids[s2_idx]

    def find_k_nearest_neighbors(self, vector, k=5):
        """Finds the k nearest neighbors to a given vector."""
        if not self.nodes or k == 0:
            return [], []

        if self.faiss_index:
            query_vector = np.array([vector]).astype('float32')
            distances, indices = self.faiss_index.search(query_vector, k)
            neighbor_ids = [self.faiss_id_map[i] for i in indices[0]]
            return neighbor_ids, distances[0]
        else:
            # NumPy fallback
            node_ids = list(self.nodes.keys())
            weights = np.array([self.nodes[nid]['weight'] for nid in node_ids])
            distances_sq = np.sum((weights - vector) ** 2, axis=1)

            k = min(k, len(node_ids))

            # Use argpartition for efficiency
            k_smallest_indices = np.argpartition(distances_sq, k-1)[:k]

            # Sort only the k-smallest to get them in order
            sorted_indices = k_smallest_indices[np.argsort(distances_sq[k_smallest_indices])]

            neighbor_ids = [node_ids[i] for i in sorted_indices]
            return neighbor_ids, np.sqrt(distances_sq[sorted_indices])


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
        """
        edge = tuple(sorted((s1_id, s2_id)))
        found_edge = next((e for e in self.edges if e[0] == edge[0] and e[1] == edge[1]), None)
        if found_edge:
            self.edges.remove(found_edge)
            self.edges.add((found_edge[0], found_edge[1], 0))
        else:
            self.edges.add((edge[0], edge[1], 0))

        edges_to_update = {e for e in self.edges if s1_id in e[:2]}
        for e in edges_to_update:
            self.edges.remove(e)
            self.edges.add((e[0], e[1], e[2] + 1))

        old_edges = {e for e in self.edges if e[2] > self.max_edge_age}
        self.edges -= old_edges

        # Remove isolated nodes resulting from edge removal
        if old_edges:
            connected_nodes = {node for edge in self.edges for node in edge[:2]}
            all_node_ids = set(self.nodes.keys())
            isolated_node_ids = all_node_ids - connected_nodes
            if len(self.nodes) - len(isolated_node_ids) >= 2:
                for nid in isolated_node_ids:
                    if nid in self.nodes:
                        del self.nodes[nid]
                self.faiss_index = None # Invalidate index

    def _insert_node(self):
        """Inserts a new node into the network (step 5)."""
        if len(self.nodes) < 2: return

        # Use the slow-decaying error for finding where to insert
        q_id = max(self.nodes, key=lambda nid: self.nodes[nid]['error_slow'])
        q_neighbors = self._get_neighbors(q_id)
        if not q_neighbors: return
        f_id = max(q_neighbors, key=lambda nid: self.nodes[nid]['error_slow'])

        q_node, f_node = self.nodes[q_id], self.nodes[f_id]
        r_weight = 0.5 * (q_node['weight'] + f_node['weight'])
        r_utility = 0.5 * (q_node['utility'] + f_node['utility'])

        # Initialize new node's error based on the parent
        r_error_fast = q_node['error_fast'] * self.after_split_error_decay_rate
        r_error_slow = q_node['error_slow'] * self.after_split_error_decay_rate
        r_id = self._add_node(r_weight, error_fast=r_error_fast, error_slow=r_error_slow, utility=r_utility)
        logger.info(f"GNG: Inserted new node {r_id} between {q_id} and {f_id}")

        edge_to_remove = tuple(sorted((q_id, f_id)))
        original_edge = next((e for e in self.edges if e[0] == edge_to_remove[0] and e[1] == edge_to_remove[1]), None)
        if original_edge:
            self.edges.remove(original_edge)

        self.edges.add((tuple(sorted((q_id, r_id)))[0], tuple(sorted((q_id, r_id)))[1], 0))
        self.edges.add((tuple(sorted((f_id, r_id)))[0], tuple(sorted((f_id, r_id)))[1], 0))

        q_node['error_fast'] *= self.after_split_error_decay_rate
        q_node['error_slow'] *= self.after_split_error_decay_rate
        f_node['error_fast'] *= self.after_split_error_decay_rate
        f_node['error_slow'] *= self.after_split_error_decay_rate

        self.faiss_index = None # Invalidate index

    def get_state(self):
        """Returns the serializable state of the GNG."""
        # Convert numpy arrays in nodes to lists for JSON serialization
        serializable_nodes = {
            nid: {
                'weight': node['weight'].tolist(),
                'error_fast': node['error_fast'],
                'error_slow': node['error_slow'],
                'utility': node['utility']
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
                'error_fast': node.get('error_fast', node.get('error', 0.0)), # Backwards compatibility
                'error_slow': node.get('error_slow', node.get('error', 0.0)), # Backwards compatibility
                'utility': node['utility']
            }
            for nid, node in state_dict['nodes'].items()
        }
        gng.edges = {tuple(e) for e in state_dict['edges']}
        gng._next_node_id = state_dict['next_node_id']
        gng._iterations = state_dict['iterations']
        return gng

    def prune_low_utility_nodes(self, min_utility):
        """
        Removes nodes with utility below a given threshold.
        Also removes any edges connected to the pruned nodes.
        """
        # Ensure we don't prune the graph into a state with less than 2 nodes.
        if len(self.nodes) <= 2:
            return

        # Identify nodes to prune without modifying the dict during iteration.
        nodes_to_prune = []
        for nid, node_data in self.nodes.items():
            # Check for low utility
            if node_data['utility'] < min_utility:
                # Check if the node is outside its grace period
                node_age = self._iterations - node_data.get('creation_iteration', 0)
                if node_age > self.pruning_grace_period:
                    nodes_to_prune.append(nid)

        # Do nothing if it would leave the graph with less than 2 nodes.
        if len(self.nodes) - len(nodes_to_prune) < 2:
            return

        for node_id in nodes_to_prune:
            # Remove the node itself
            if node_id in self.nodes:
                del self.nodes[node_id]

            # Remove all edges connected to this node
            edges_to_remove = {e for e in self.edges if node_id in e[:2]}
            self.edges -= edges_to_remove

        # If any nodes were pruned, the FAISS index is now invalid.
        if nodes_to_prune:
            self.faiss_index = None

    def predict(self, current_node_id):
        """
        Predicts the next node based on the utility of the neighbors
        of the current node.
        """
        neighbors = self._get_neighbors(current_node_id)
        if not neighbors:
            # If no neighbors, the best prediction is the current node itself
            return self.nodes[current_node_id]

        # Find the neighbor with the highest utility
        best_neighbor_id = max(neighbors, key=lambda nid: self.nodes[nid]['utility'])
        return self.nodes[best_neighbor_id]
