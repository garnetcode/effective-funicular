# Implements the topological mapping layer (Cortical Layer).
# This is a single Growing Neural Gas (GNG) instance that learns the
# topology of the input data. This version is accelerated with FAISS
# for high-performance nearest-neighbor search.

import numpy as np
import itertools

try:
    import faiss
except ImportError:
    print("Warning: faiss library not found. GNG will run with a slow numpy fallback.")
    faiss = None

class GNG_Engine:
    def __init__(self, dimensions, **kwargs):
        self.dimensions = dimensions

        # Hyperparameters
        self.winner_learning_rate = kwargs.get('winner_learning_rate', 0.05)
        self.neighbor_learning_rate = kwargs.get('neighbor_learning_rate', 0.001)
        self.max_edge_age = kwargs.get('max_edge_age', 50)
        self.n_iter_before_neuron_added = kwargs.get('n_iter_before_neuron_added', 100)
        self.error_decay_rate = kwargs.get('error_decay_rate', 0.0005)
        self.after_split_error_decay_rate = kwargs.get('after_split_error_decay_rate', 0.5)
        self.utility_prune_threshold = kwargs.get('utility_prune_threshold', 0.05)
        self.n_iter_before_pruning = kwargs.get('n_iter_before_pruning', 400)
        self.merge_threshold = kwargs.get('merge_threshold', 0.1)
        self.n_iter_before_merging = kwargs.get('n_iter_before_merging', 500)

        self.nodes = {}
        self.edges = set()
        self._next_node_id = 0
        self._iterations = 0

        # FAISS Index for fast search
        self.faiss_index = self._initialize_faiss_index()

        # Initialize with two random nodes
        self._add_node(np.random.rand(self.dimensions))
        self._add_node(np.random.rand(self.dimensions))

    def _initialize_faiss_index(self):
        if not faiss: return None
        # Use L2 distance, which is equivalent to Euclidean distance
        base_index = faiss.IndexFlatL2(self.dimensions)
        # Wrap with IndexIDMap to allow adding/removing by custom ID
        return faiss.IndexIDMap(base_index)

    def _add_node(self, weight_vector, error=0.0):
        node_id = self._next_node_id
        self.nodes[node_id] = {'weight': weight_vector.astype('float32'), 'error': error}
        if self.faiss_index:
            self.faiss_index.add_with_ids(np.array([self.nodes[node_id]['weight']]), np.array([node_id]))
        self._next_node_id += 1
        return node_id

    def _remove_nodes(self, node_ids):
        if not node_ids: return
        for node_id in node_ids:
            if node_id in self.nodes:
                del self.nodes[node_id]
        if self.faiss_index:
            self.faiss_index.remove_ids(np.array(node_ids))

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
        if len(self.nodes) < 2: return list(self.nodes.keys())[0] if self.nodes else None, None

        query_vector = np.array([input_vector.astype('float32')])

        if self.faiss_index and self.faiss_index.ntotal > 1:
            distances, ids = self.faiss_index.search(query_vector, 2)
            return ids[0][0], ids[0][1]
        else: # Fallback to numpy if faiss is not available or index is too small
            node_ids = list(self.nodes.keys())
            weights = np.array([self.nodes[nid]['weight'] for nid in node_ids])
            distances = np.linalg.norm(weights - query_vector, axis=1)
            sorted_indices = np.argsort(distances)
            return node_ids[sorted_indices[0]], node_ids[sorted_indices[1]]

    def _adapt_weights(self, s1_id, input_vector):
        # Update winner
        winner_node = self.nodes[s1_id]
        winner_node['weight'] += self.winner_learning_rate * (input_vector - winner_node['weight'])

        # Update neighbors and their edges
        neighbor_ids = {e[1] if e[0] == s1_id else e[0] for e in self.edges if s1_id in e[:2]}
        nodes_to_update_in_index = {s1_id}
        for nid in neighbor_ids:
            self.nodes[nid]['weight'] += self.neighbor_learning_rate * (input_vector - self.nodes[nid]['weight'])
            nodes_to_update_in_index.add(nid)
            edge_tuple = tuple(sorted((s1_id, nid)))
            found_edge = next((e for e in self.edges if e[0] == edge_tuple[0] and e[1] == edge_tuple[1]), None)
            if found_edge:
                self.edges.remove(found_edge)
                self.edges.add((found_edge[0], found_edge[1], found_edge[2], found_edge[3] + 1))

        # Update vectors in the FAISS index
        if self.faiss_index:
            ids_to_update = list(nodes_to_update_in_index)
            self.faiss_index.remove_ids(np.array(ids_to_update))
            updated_weights = np.array([self.nodes[nid]['weight'] for nid in ids_to_update])
            self.faiss_index.add_with_ids(updated_weights, np.array(ids_to_update))

    def _remove_isolated_nodes(self):
        if len(self.nodes) <= 2: return
        connected_nodes = {node for edge in self.edges for node in edge[:2]}
        isolated_node_ids = [nid for nid in self.nodes if nid not in connected_nodes]
        self._remove_nodes(isolated_node_ids)

    def _merge_nodes(self):
        # ... (logic for finding pairs to merge)
        nodes_to_merge_pairs = []
        node_ids = list(self.nodes.keys())
        if len(node_ids) < 2: return

        for id1, id2 in itertools.combinations(node_ids, 2):
            dist = np.linalg.norm(self.nodes[id1]['weight'] - self.nodes[id2]['weight'])
            if dist < self.merge_threshold:
                nodes_to_merge_pairs.append((id1, id2))

        merged_already = set()
        for id1, id2 in nodes_to_merge_pairs:
            if id1 in merged_already or id2 in merged_already: continue

            node1, node2 = self.nodes[id1], self.nodes[id2]
            new_weight = (node1['weight'] + node2['weight']) / 2
            new_error = node1['error'] + node2['error']
            new_id = self._add_node(new_weight, new_error) # This also adds to faiss index

            # Re-route edges
            edges_to_remove = {e for e in self.edges if id1 in e[:2] or id2 in e[:2]}
            edges_to_add = set()
            for edge in edges_to_remove:
                other_node = edge[1] if edge[0] in [id1, id2] else edge[0]
                if other_node not in [id1, id2]:
                    edges_to_add.add((tuple(sorted((new_id, other_node)))[0], tuple(sorted((new_id, other_node)))[1], 0, edge[3]))

            self.edges -= edges_to_remove
            self.edges.update(edges_to_add)

            # Mark old nodes for deletion
            merged_already.add(id1)
            merged_already.add(id2)

        self._remove_nodes(list(merged_already))

    # Other methods like _update_winner_edges, _insert_node, _prune_by_utility remain largely the same...
    # [Code from previous version for these methods is assumed here to save space, but with _remove_nodes calls]
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

    def get_state(self):
        # NOTE: FAISS index is not serialized here; it's handled by the agent.
        return {
            'dimensions': self.dimensions, 'nodes': self.nodes, 'edges': self.edges,
            'next_node_id': self._next_node_id, 'iterations': self._iterations
        }

    @classmethod
    def from_state(cls, state_dict, faiss_index=None, **kwargs):
        gng = cls(state_dict['dimensions'], **kwargs)
        gng.nodes = state_dict['nodes']
        gng.edges = state_dict['edges']
        gng._next_node_id = state_dict['next_node_id']
        gng._iterations = state_dict['iterations']
        # If a pre-loaded faiss index is provided, use it.
        if faiss_index: gng.faiss_index = faiss_index
        else: gng._rebuild_faiss_index() # Otherwise, rebuild it from nodes
        return gng

    def _rebuild_faiss_index(self):
        self.faiss_index = self._initialize_faiss_index()
        if self.faiss_index:
            node_ids = list(self.nodes.keys())
            if not node_ids: return
            weights = np.array([self.nodes[nid]['weight'] for nid in node_ids]).astype('float32')
            self.faiss_index.add_with_ids(weights, np.array(node_ids))
