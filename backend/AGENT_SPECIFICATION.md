# Chimera Agent Technical Specification

This document provides a detailed technical specification of the Chimera cognitive architecture. It is intended for developers who wish to understand the internal workings of the agent, its learning mechanisms, and how to interact with its API.

## 1. High-Level Architecture

The Chimera agent is a modular, biologically-inspired cognitive architecture. The flow of information for learning and decision-making follows three main stages:

1.  **Sensory Cortexes**: Raw input from the environment (e.g., vectors, text, images) is processed by a corresponding **Cortex** module. The cortex converts the raw data into a fixed-size numerical vector called an "embedding". This is the agent's internal representation of the sensory event.

2.  **Associative Memory (Hopfield Core)**: The embedding vector is then passed to a **Hopfield Network**. The Hopfield network is a form of associative memory. It takes the embedding and recalls the closest "stable pattern" or "attractor state". This process cleans up noise and results in a canonical, stable representation of the memory.

3.  **Hierarchical World Model (STAG Framework)**: The stable attractor from the Hopfield network is then used to train the **STAG (Self-organizing Tree-like Adaptive Graph)** framework. This is the agent's long-term memory and "world model". It is a hierarchical structure of **Growing Neural Gas (GNG)** networks that organizes memories and concepts, from general at the root to specific at the leaves.

## 2. Core Components

### 2.1. GNG Engine (`gng_engine.py`)

The Growing Neural Gas (GNG) is the fundamental building block of the agent's world model. Each GNG is a neural network that learns the topological structure of the data it is exposed to.

-   **Growth**: A GNG starts with two neurons (nodes) and adds new neurons in regions of high error, allowing it to increase its representational capacity where needed.
-   **Adaptation**: When an input vector is presented, the closest neuron (the "winner") and its direct neighbors are moved slightly closer to the input vector.
-   **Pruning**: Edges between neurons have an "age" that increments when they are not used. Old edges are removed, and any neuron that becomes disconnected from the graph is pruned. This ensures the network does not grow indefinitely and forgets unused information.
-   **Performance**: To ensure efficient scaling, the search for the winning neuron is accelerated using a `faiss` (Facebook AI Similarity Search) index. This is significantly faster than a brute-force search, especially when the GNG has many neurons.

#### Memory Protection via Dynamic Learning Rate

To prevent new memories from distorting old, important ones (a problem known as catastrophic forgetting), the GNG implements a utility-based dynamic learning rate.

-   **Utility**: Each neuron has a `utility` property. This value increases every time the neuron is a "winner" for an input and decays slowly over time.
-   **Dynamic Learning**: The learning rate for each neuron is now inversely proportional to its utility. This means:
    -   **New, low-utility neurons** are "plastic" and learn quickly.
    -   **Established, high-utility neurons** are "rigid" and resistant to change, thus protecting the important memories they represent.

### 2.2. STAG Framework (`stag_framework.py`)

The STAG framework organizes multiple GNGs into a hierarchy, similar to a taxonomic tree.

-   **Structure**: The STAG is a tree where each node contains a GNG instance. The root GNG represents the most general concepts, and its children represent more specific sub-concepts.
-   **Traversal**: When a new memory (a stable attractor from the Hopfield network) is presented, it is first given to the root GNG. The winning neuron is found. If that neuron has a child GNG, the memory is passed down to that child, and the process repeats. This continues until a terminal node (a neuron with no child GNG) is found.
-   **Expansion**: The terminal GNG is then trained with the new memory. If the error for the winning neuron in the terminal GNG exceeds a certain threshold, it signals that the current representation is not sufficient. The agent then **expands** this neuron, creating a new, empty child GNG beneath it. The patterns that were previously mapped to the parent neuron (identified by a robust `(level_id, node_id)` tuple) are then used to train this new, more specialized GNG.

### 2.3. Action Head (`action/modules.py`)

The Action Head is the component responsible for decision-making. It is a `torch.nn.Module` that implements a simple linear layer mapping the agent's internal state representation to a set of logits over the possible actions.

-   **PyTorch Integration**: By using PyTorch, the Action Head can be trained efficiently using standard deep learning techniques.
-   **Optimization**: It uses an `Adam` optimizer to update its weights based on the policy gradient loss calculated during the `train()` step. This is more robust and maintainable than a manual numpy-based gradient implementation.

## 3. Learning and Memory Processes

### 3.1. Online Learning

When new information is presented to the agent, it goes through the following steps:

1.  `perceive()`: A cortex turns the raw input into an embedding.
2.  `learn_associative()`: The embedding is stored in the Hopfield network. A unique `pattern_id` is created for this memory.
3.  `organize_memory()`: The agent recalls the stable attractor for the new pattern and uses it to train the STAG framework, finding the appropriate terminal node and updating the GNG. This may trigger an expansion if necessary.

### 3.2. Offline Memory Consolidation

The agent implements a form of memory consolidation to strengthen its knowledge during periods of inactivity.

-   **Process**: The `consolidate_memories()` method iterates through all the patterns the agent has learned. For each pattern, it replays it through the `organize_memory()` process.
-   **Effect**: This replay strengthens the connections in the GNGs and refines the overall structure of the STAG. It reinforces existing knowledge and further solidifies the utility of important neurons, making them more resistant to future change.

## 4. API Reference

### 4.1. Cortex Specifications

-   **Endpoint**: `GET /api/cortex_specifications/`
-   **Description**: Returns a list of available cortex modules and a specification of the input data they expect. This allows a client to know how to structure data for the agent.
-   **Example Response**:
    ```json
    [
      {
        "type": "DenseCortex",
        "description": "Processes a fixed-size vector input...",
        "input_spec": { "type": "vector", "dtype": "float", "shape": ["input_dim"] },
        "params": [{"name": "input_dim", "type": "integer", ...}]
      },
      ...
    ]
    ```

### 4.2. Activity Probing

-   **Endpoint**: `POST /api/agents/<agent_id>/probe_activity/`
-   **Description**: Allows you to "see" what the agent is "thinking". It takes a sensory input and returns the chain of winning neurons that were activated in the STAG hierarchy.
-   **Request Body**:
    ```json
    {
      "cortex_id": "text_input",
      "raw_input": "A sample sentence for the agent to process."
    }
    ```
-   **Response Body**: The response is a list of activated nodes. Each element corresponds to a level in the STAG hierarchy, from the root downwards. The `level_id` and `winner_id` can be used to identify the specific neuron in the graph structure returned by the `/structure/` endpoint.
    ```json
    {
      "activation_path": [
        { "level_id": 0, "winner_id": 4 },
        { "level_id": 2, "winner_id": 1 }
      ]
    }
    ```
