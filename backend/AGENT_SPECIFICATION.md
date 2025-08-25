# Chimera Agent Technical Specification

This document provides a detailed technical specification of the Chimera cognitive architecture. It is intended for developers who wish to understand the internal workings of the agent, its learning mechanisms, and how to interact with it.

## 1. High-Level Architecture

The Chimera agent is a modular, biologically-inspired cognitive architecture designed for **continual, multi-skill learning**. It is based on a **Recurrent Latent World Model** and a novel **Skill Manager** that leverages multiple hierarchical conceptual graphs to learn and master different tasks without catastrophic forgetting.

### 1.1. Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                                 Chimera Agent                                     |
|                                                                                   |
|  +---------------------------+     +-------------------------------------------+  |
|  |      Sensory Cortexes     | --> |               World Model                 |  |
|  | (Vision, Vector, etc.)    |     |    (Encoder, GRU, Decoder)                |  |
|  +---------------------------+     +----------------------+--------------------+  |
|      ^                             |                      |                       |
|      |                             | h (hidden_state)     |                       |
|      |                             v                      v                       |
|  +---+-------------------------+  +----------------------+--------------------+  |
|  |     Environment             |  |                 Skill Manager               |  |
|  | (e.g., "CartPole-v1")       |  |                                             |  |
|  +---------------------------+ |  | +------------------+   +------------------+ |  |
|      ^          |              |  | | Skill A: STAG    |   | Skill B: STAG    | |  |
|      | Action   | Observation  |  | | (for "CartPole") |   | (for "LunarLander")| |  |
|      |          v              |  | +------------------+   +------------------+ |  |
|  +---+-------------------------+  +-----------+--------------------------+----+  |
|  |         Action Head           |              ^ C_t (context vector)  |       |
|  |        (Policy)               |--------------------------------------+       |
|  +-------------------------------+                                              |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### 1.2. Information Flow

The flow of information for learning and decision-making follows these main stages:

1.  **Set Active Skill**: Before interacting with an environment, the training script informs the agent of the current task (e.g., `agent.set_active_skill("CartPole-v1")`).

2.  **Sensory Cortexes**: Raw input from the environment is processed by the appropriate **Cortex** module (e.g., `VisionCortex` for images, `DenseCortex` for vectors). The cortex converts the raw data into a fixed-size **embedding vector**.

3.  **World Model**: The embedding vector is passed to the **World Model**. This is the agent's core "brain" and consists of three parts:
    *   **Encoder**: Compresses the embedding into a smaller, more abstract **latent state (`z`)**.
    *   **Recurrent Core (GRU)**: The agent's temporal memory. It integrates the latent state `z`, the agent's last action, and its own previous memory state (`h_old`) to produce a new, context-rich memory state (`h_new`).
    *   **Decoder**: Takes the new memory state `h_new` and uses it to predict the future: the expected next observation and the expected reward.

4.  **Conceptualization (Skill Manager & STAG)**: The agent's memory state `h` is passed to the **Skill Manager**, which routes it to the appropriate **STAG (Self-organizing Tree-like Adaptive Graph)** instance for the active skill. This allows the agent to build a separate, isolated, hierarchical conceptual model for each task.

5.  **Action Selection**: The agent's memory state `h` and a **context vector (`C_t`)** derived from the active STAG are fed into an **Action Head**, which decides on the next action to take.

## 2. Core Components

### 2.1. Skill Manager (`skill_manager.py`)

This is the central component for enabling multi-skill learning. It replaces the previous single, monolithic STAG framework.
-   **Manages Multiple STAGs**: The `SkillManager` holds a dictionary of `STAG_Framework` instances, keyed by a `skill_id` (the environment ID).
-   **Dynamic Creation**: When the agent encounters a new skill ID, the manager creates a new, empty STAG for it automatically.
-   **Delegation**: All operations on the conceptual graph (finding paths, pruning, etc.) are delegated by the manager to the STAG instance corresponding to the agent's currently active skill. This ensures that learning in one environment does not interfere with the conceptual knowledge of another.

### 2.2. STAG Framework (`stag_framework.py`)

This class now represents a single, hierarchical conceptual graph for one specific skill. Its internal logic, including the GNG engine and the tree-like hierarchy, remains the same.

### 2.3. Cortexes (`cortex/`)

The cortex system has been made more robust.
-   **`CortexFactory`**: A new factory in `cortex/factory.py` automatically determines the correct cortex configuration (e.g., `DenseCortex` for vectors, `VisionCortex` for images) by inspecting an environment's observation space.
-   **`VisionCortex`**: A new `nn.Module` that uses a Convolutional Neural Network (CNN) to process image-based observations into the agent's standard `embedding_dim`.

### 2.4. World Model & Action Head

These components remain architecturally the same but have been given a larger "brain capacity" via increased default dimensions (`embedding_dim`, `hidden_dim`, `latent_dim`) to better handle the complexity of learning multiple tasks.

## 3. Learning Process

The learning process is now based on a **curriculum learning** paradigm.

1.  **Curriculum Training**: The primary training scripts (`train.py`, `run_colosseum_agents.py`) are designed to run a single agent through a sequence of environments (a "curriculum").
2.  **Set Active Skill**: Before starting training on a new environment, the script calls `agent.set_active_skill(env_id)`. This tells the `SkillManager` which conceptual graph to use.
3.  **Online Learning**: Within an environment, the agent learns continuously. It collects experiences in a `ReplayBuffer` and trains its World Model and Action Head after each step (or every N steps).
4.  **Two-Phase GNG Training**: To build stable conceptual graphs, the GNGs within each STAG now use a two-phase learning process. They start with a high learning rate to quickly establish a rough topology (ordering phase) and then switch to a lower learning rate for fine-tuning.
5.  **Graph Pruning**: To manage complexity, each STAG is periodically pruned. Nodes with utility below a certain threshold are removed, keeping the graphs efficient and focused on relevant concepts.

---

# User Manual & Procedures

This manual provides practical instructions for running and interacting with the ChimeraAgent.

## A. Configuration (`backend/config.yaml`)

All training and model parameters are controlled by `backend/config.yaml`.

-   **`agent_config`**:
    -   `embedding_dim`, `latent_dim`, `hidden_dim`: The dimensions of the agent's core neural networks. These have been increased to provide more capacity for multi-task learning.
    -   `hyperparams`: Contains learning parameters. New additions include:
        -   `world_model_pretrain_steps`, `stag_update_frequency`: Control the decoupling of World Model and STAG training.
        -   `gng_pruning_frequency`, `gng_min_utility_threshold`: Control the periodic pruning of the STAGs.
        -   `gng_ordering_phase_steps`, `gng_winner_learning_rate_initial`: Control the new two-phase learning for GNGs.
        -   `world_model_weight_decay`: Controls L2 regularization on the World Model.

## B. Training the Agent

The training scripts have been updated to support curriculum learning.

### 1. Local Training with Curriculum

The `train.py` script can train a single agent on a curriculum of local Gymnasium environments.

```bash
# Example: Train on CartPole first, then LunarLander
python backend/train.py --env-curriculum "CartPole-v1,LunarLander-v2" --steps-per-env 20000
```
-   `--env-curriculum`: A comma-separated list of environment IDs.
-   `--steps-per-env`: The number of steps to train on each environment before moving to the next.

### 2. Colosseum Training with Curriculum

The `run_colosseum_agents.py` script has been refactored to train a single agent on a curriculum of Colosseum environments.

```bash
python backend/run_colosseum_agents.py
```
-   The script now reads the `env_list` from within the file and trains a single agent sequentially on them.
-   The old `multi_session_manager.py` has been removed.

## C. Interactive Chat Mode

This remains the same. To chat with a trained agent:

```bash
python backend/interactive_mode.py
```
