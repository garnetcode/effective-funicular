# Implements the hierarchical framework (STAG - Self-organizing Tree-like Adaptive Graph).
# This system organizes the GNGs into a multi-level taxonomy, inspired by GHSOM.
# It manages a tree of GNG instances, where high-error nodes in a parent GNG
# can expand into a new child GNG at the next level of the hierarchy.
# See Section 2.4 and 3.2 of the Project Chimera specification.

import numpy as np
from .gng_engine import GNG_Engine

class STAG_Framework:
    def __init__(self, dimensions, **kwargs):
        """
        Initializes the STAG Framework.

        Args:
            dimensions (int): The dimensionality of the input vectors.
            **kwargs: Hyperparameters for the GNGs and the STAG itself.
        """
        self.dimensions = dimensions
        self.gng_params = kwargs
        self._next_level_id = 0
        self.level_map = {} # Map level_id to the tree node containing the GNG

        # The hierarchy is a tree structure where each node holds a GNG instance.
        self.tree = self._create_tree_node()

    def _create_tree_node(self, parent_gng_node_id=None):
        """Helper to create a new node in the hierarchy tree."""
        level_id = self._next_level_id
        self._next_level_id += 1
        node = {
            'level_id': level_id,
            'gng': GNG_Engine(self.dimensions, **self.gng_params),
            'parent_node_id': parent_gng_node_id,
            'children': []  # List of child tree nodes
        }
        self.level_map[level_id] = node['gng']
        return node

    def find_terminal_node_and_path(self, input_vector):
        """
        Traverses the hierarchy to find the terminal GNG and winning node for an input.
        Also returns the activation path taken to reach the terminal node.

        Args:
            input_vector (np.array): The input vector to route through the hierarchy.

        Returns:
            tuple: A tuple containing (
                terminal_level_node (dict): The tree node containing the terminal GNG.
                winner_id (int): The ID of the winning node within that GNG.
                activation_path (list): A list of dicts, each with 'level_id' and 'winner_id'.
            )
        """
        current_level = self.tree
        activation_path = []
        while True:
            gng = current_level['gng']
            if len(gng.nodes) < 2:
                # This GNG is not yet populated enough to make a decision.
                return current_level, None, activation_path

            winner_id, _ = gng._find_winners(input_vector)
            activation_path.append({'level_id': current_level['level_id'], 'winner_id': winner_id})

            # Check if the winner node has a child GNG
            child_node = next((child for child in current_level['children'] if child['parent_node_id'] == winner_id), None)

            if child_node:
                # Traverse down the hierarchy
                current_level = child_node
            else:
                # This is the terminal node
                return current_level, winner_id, activation_path

    def expand_node(self, parent_level_node, parent_gng_node_id):
        """
        Expands a node in a parent GNG into a new child GNG.
        This method creates the new child GNG and attaches it to the tree.
        The caller is responsible for training the new GNG.

        Args:
            parent_level_node (dict): The tree node from the hierarchy containing the parent GNG.
            parent_gng_node_id (int): The ID of the node within the parent GNG to expand.

        Returns:
            GNG_Engine: The new, untrained GNG instance for the child level.
        """
        print(f"Expanding node {parent_gng_node_id} in parent GNG.")

        # Check if a child for this node already exists
        if any(child['parent_node_id'] == parent_gng_node_id for child in parent_level_node['children']):
            # This should not happen if the agent logic is correct
            print(f"Warning: Expansion attempted on node {parent_gng_node_id} which already has a child.")
            return None

        new_level_node = self._create_tree_node(parent_gng_node_id=parent_gng_node_id)
        parent_level_node['children'].append(new_level_node)

        return new_level_node['gng']

    def get_serializable_structure(self):
        """
        Returns a serializable representation of the entire STAG tree.
        This is used for saving the agent's state.
        """
        structure = {
            '_next_level_id': self._next_level_id,
            'tree': self._serialize_level(self.tree)
        }
        return structure

    def _serialize_level(self, level_node):
        return {
            'level_id': level_node['level_id'],
            'gng_state': level_node['gng'].get_state(),
            'parent_node_id': level_node['parent_node_id'],
            'children': [self._serialize_level(child) for child in level_node['children']]
        }

    def get_flattened_structure(self):
        """
        Returns a serializable representation of the STAG tree, flattened
        into a single graph for visualization.
        """
        final_nodes = {}
        final_edges = []
        node_id_counter = 0

        # A map from (level_id, gng_node_id) to a global sequential ID
        global_id_map = {}

        def traverse(level_node, parent_global_id=None):
            nonlocal node_id_counter
            gng = level_node['gng']
            level_id = level_node['level_id']

            # Add nodes for this level
            for node_id, node_data in gng.nodes.items():
                global_id = node_id_counter
                global_id_map[(level_id, node_id)] = global_id

                serializable_node = {
                    'level_id': level_id,
                    'gng_node_id': node_id,
                    'weight': node_data['weight'].tolist(),
                    'error': node_data.get('error', 0.0),
                    'utility': node_data.get('utility', 0.0)
                }
                final_nodes[str(global_id)] = serializable_node
                node_id_counter += 1

            # Add intra-level edges
            for u, v, age in gng.edges:
                if (level_id, u) in global_id_map and (level_id, v) in global_id_map:
                    global_u = global_id_map[(level_id, u)]
                    global_v = global_id_map[(level_id, v)]
                    final_edges.append({'source': str(global_u), 'target': str(global_v), 'type': 'intra-level', 'age': age})

            # Add inter-level edges (from parent to child)
            if parent_global_id is not None:
                # Find the 'entry point' node in the child GNG.
                # This is the node in the child GNG that is closest to the parent node's weight vector.
                parent_node_gng = self.level_map.get(parent_global_id['level_id'])
                if parent_node_gng:
                    parent_weight = parent_node_gng.nodes[parent_global_id['gng_node_id']]['weight']
                    child_winner_id, _ = gng._find_winners(parent_weight)

                    if (level_id, child_winner_id) in global_id_map:
                        child_global_id = global_id_map[(level_id, child_winner_id)]
                        parent_mapped_id = global_id_map[(parent_global_id['level_id'], parent_global_id['gng_node_id'])]
                        final_edges.append({'source': str(parent_mapped_id), 'target': str(child_global_id), 'type': 'inter-level'})

            # Recurse through children
            for child_node in level_node['children']:
                parent_info = {'level_id': level_id, 'gng_node_id': child_node['parent_node_id']}
                traverse(child_node, parent_info)

        traverse(self.tree)

        # Create adjacency list for efficient lookups
        adj_list = {node_id: [] for node_id in final_nodes.keys()}
        for edge in final_edges:
            adj_list[edge['source']].append(edge['target'])
            adj_list[edge['target']].append(edge['source'])

        return {
            'nodes': final_nodes,
            'edges': final_edges,
            'adj': adj_list,
            'dimensions': self.dimensions
        }

    @classmethod
    def from_serializable_structure(cls, structure, **kwargs):
        """
        Creates a STAG instance from a serialized structure.
        """
        # Determine dimensions, prioritizing the structure's value but falling back to kwargs.
        # This is key to supporting older save files where the agent's top-level
        # dimensions need to be passed down.
        dimensions = structure.get('dimensions', kwargs.get('dimensions'))

        # Create a clean set of kwargs for initialization, removing 'dimensions'
        # to prevent a TypeError from passing it both positionally and as a keyword.
        init_kwargs = kwargs.copy()
        init_kwargs.pop('dimensions', None)

        stag = cls(dimensions, **init_kwargs)
        stag._next_level_id = structure.get('_next_level_id', 0)

        # Pass the clean kwargs down to the level deserialization to prevent
        # the same TypeError in the GNG_Engine constructor.
        stag.tree = stag._deserialize_level(structure['tree'], init_kwargs)
        return stag

    def _deserialize_level(self, level_dict, kwargs):
        gng = GNG_Engine.from_state(level_dict['gng_state'], **kwargs)
        node = {
            'level_id': level_dict['level_id'],
            'gng': gng,
            'parent_node_id': level_dict['parent_node_id'],
            'children': [self._deserialize_level(child_dict, kwargs) for child_dict in level_dict['children']]
        }
        self.level_map[node['level_id']] = gng
        if node['level_id'] >= self._next_level_id:
            self._next_level_id = node['level_id'] + 1
        return node

    def prune_graph(self, min_utility):
        """
        Prunes all GNGs in the hierarchy based on a utility threshold.
        """
        for level_id, gng_instance in self.level_map.items():
            gng_instance.prune_low_utility_nodes(min_utility)

    def find_k_nearest_in_terminal_gng(self, vector, k=5):
        """Finds the k-nearest neighbors in the terminal GNG for a given vector."""
        if not self.level_map:
            return [], []

        terminal_gng = self.level_map[max(self.level_map.keys())]
        return terminal_gng.find_k_nearest_neighbors(vector, k)

    def generate_prediction(self, current_node_id):
        """
        Generates a top-down prediction from the STAG's terminal GNG.
        """
        if not self.level_map:
            return None

        terminal_gng = self.level_map[max(self.level_map.keys())]
        if current_node_id not in terminal_gng.nodes:
            return None # Current node not found

        predicted_node = terminal_gng.predict(current_node_id)
        # Return the latent representation (weight vector) of the predicted node
        return predicted_node['weight']
