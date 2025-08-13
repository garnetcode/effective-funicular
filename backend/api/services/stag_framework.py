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
        self.stag_expansion_threshold = kwargs.get('stag_expansion_threshold', 0.1)

        # The hierarchy is a tree structure where each node holds a GNG instance.
        self.tree = {
            'gng': GNG_Engine(dimensions, **kwargs),
            'parent_node_id': None,
            'children': [] # List of child tree nodes
        }

    def process_input(self, input_vector):
        """
        Processes an input vector, routing it down the hierarchy to the correct GNG.
        """
        current_level = self.tree

        while True:
            gng = current_level['gng']
            winner_id, _ = gng._find_winners(input_vector)

            if winner_id is None:
                # This GNG is empty, should not happen if initialized
                break

            # Check if the winner node has a child GNG
            child_node = next((child for child in current_level['children'] if child['parent_node_id'] == winner_id), None)

            if child_node:
                # Traverse down the hierarchy
                current_level = child_node
            else:
                # This is the terminal node, process the input here
                gng.process_input(input_vector)

                # Check for expansion condition
                if gng.nodes.get(winner_id) and gng.nodes[winner_id]['error'] > self.stag_expansion_threshold:
                    self._expand_node(current_level, winner_id)

                break # Stop traversing

    def _expand_node(self, parent_level_node, parent_gng_node_id):
        """
        Expands a node in a parent GNG into a new child GNG.
        """
        print(f"Expanding node {parent_gng_node_id} in GNG.")

        new_gng = GNG_Engine(self.dimensions, **self.gng_params)

        new_level_node = {
            'gng': new_gng,
            'parent_node_id': parent_gng_node_id,
            'children': []
        }

        parent_level_node['children'].append(new_level_node)

        # In a real implementation, we would now need to route the specific
        # data points that belong to the expanded node to the new GNG for training.
        # This simplified version just creates the new GNG. The `process_input`
        # logic will ensure future inputs are routed correctly.

    def get_serializable_structure(self):
        """
        Returns a serializable representation of the entire STAG tree.
        """
        def serialize_level(level_node):
            return {
                'gng_state': level_node['gng'].get_state(),
                'parent_node_id': level_node['parent_node_id'],
                'children': [serialize_level(child) for child in level_node['children']]
            }

        return serialize_level(self.tree)

    @classmethod
    def from_serializable_structure(cls, structure, faiss_index=None, **kwargs):
        """
        Creates a STAG instance from a serialized structure.

        Args:
            structure (dict): The serialized tree structure.
            faiss_index: A pre-loaded FAISS index for the root GNG.
            **kwargs: Hyperparameters.
        """
        # This is a simplified implementation for a single-level STAG (one GNG).
        # A full implementation would need to handle indexes for each GNG in the tree.
        def deserialize_level(level_dict, index=None):
            gng = GNG_Engine.from_state(level_dict['gng_state'], faiss_index=index, **kwargs)
            node = {
                'gng': gng,
                'parent_node_id': level_dict['parent_node_id'],
                # Recursively call without the index for children
                'children': [deserialize_level(child_dict) for child_dict in level_dict['children']]
            }
            return node

        stag = cls(structure['gng_state']['dimensions'], **kwargs)
        # Pass the loaded index only to the root level
        stag.tree = deserialize_level(structure, index=faiss_index)
        return stag
