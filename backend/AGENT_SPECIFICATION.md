# Chimera Agent Technical Specification

This document provides a detailed technical specification of the Chimera cognitive architecture. It is intended for developers who wish to understand the internal workings of the agent, its learning mechanisms, and how to interact with it.

## 1. High-Level Architecture

The Chimera agent is a modular, biologically-inspired cognitive architecture based on a **Recurrent Latent World Model**. This architecture allows the agent to build an internal, predictive model of its environment, enabling it to plan and learn from "imagined" futures.

The flow of information for learning and decision-making follows these main stages:

1.  **Sensory Cortexes**: Raw input from the environment (e.g., vectors, text from a user) is processed by a corresponding **Cortex** module. The cortex converts the raw data into a fixed-size numerical observation vector.

2.  **World Model**: The observation vector is passed to the **World Model**. This is the agent's core "brain" and consists of three parts:
    *   **Encoder**: Compresses the observation into a smaller, more abstract **latent state (`z`)**.
    *   **Recurrent Core (GRU)**: The agent's temporal memory. It integrates the latent state `z`, the agent's last action, and its own previous memory state (`h_old`) to produce a new, context-rich memory state (`h_new`).
    *   **Decoder**: Takes the new memory state `h_new` and uses it to predict the future: the expected next observation and the expected reward.

3.  **Action Selection**: The agent's internal memory state `h` is fed into an **Action Head**, which decides on the next action to take.

4.  **Hierarchical World Model (STAG Framework)**: The agent's memory states (`h`) are continuously organized by the **STAG (Self-organizing Tree-like Adaptive Graph)** framework. This provides a long-term, hierarchical conceptual model of the situations the agent has encountered.

## 2. Core Components

### 2.1. World Model (`world_model_core.py`)

This is the central predictive component of the agent. It is a single PyTorch `nn.Module` containing:
-   **Encoder**: A multi-layer perceptron (MLP) that maps observations to latent states.
-   **Recurrent Core**: A Gated Recurrent Unit (GRU) cell that updates the agent's hidden memory state.
-   **Decoder**: Two separate MLPs that predict the next observation and the next reward from the hidden state.

The entire World Model is trained to minimize the difference between its predictions and the actual outcomes from the environment.

### 2.2. GNG Engine & STAG Framework (`gng_engine.py`, `stag_framework.py`)

These components remain from the previous architecture, but their role has changed. Instead of organizing static memories, the STAG framework now organizes the **hidden states (`h`)** from the World Model's recurrent core. This means the agent builds a conceptual hierarchy of *dynamic situations* rather than just static observations.

### 2.3. Action Head (`action/modules.py`)

The Action Head is an `nn.Module` that maps the agent's current hidden state `h` to a probability distribution over possible actions. It is trained alongside the World Model to take actions that lead to states with high predicted rewards.

### 2.4. State History Manager (`state_history_manager.py`)

This service provides a "self-preservation" mechanism for the agent. It implements a Git-like versioning system for the agent's learnable parameters (the weights of the `WorldModel` and `ActionHead`).
-   **Base & Diff Snapshots**: To save memory, it periodically saves a full "base" snapshot of the weights. Between these, it only saves the *difference* (delta) from the previous version.
-   **History Log**: A `history.json` file tracks every version, allowing the agent's state to be reconstructed or reverted to any point in time.

## 3. Learning Process (Online Learning)

The agent learns in a continuous, online fashion after every interaction with the environment.
1.  **Perceive & Act**: The agent perceives the current state, updates its internal hidden state `h`, and selects an action.
2.  **Record Experience**: The experience tuple `(observation, action, reward, next_observation)` is stored in a `ReplayBuffer`.
3.  **Train**: After each step, the agent samples a batch of experiences from the `ReplayBuffer` and performs one gradient descent step to train its models:
    *   The **World Model** is trained to accurately predict `next_observation` and `reward`.
    *   The **Action Head** is trained to select actions that lead to states with high predicted rewards.

---

# User Manual & Procedures

This manual provides practical instructions for running and interacting with the ChimeraAgent.

## A. Configuration (`backend/config.yaml`)

All training and model parameters are controlled by `backend/config.yaml`.

-   **`env_name`**: The Gymnasium environment ID to use for the `train_agent` script (e.g., "CartPole-v1").
-   **`default_agent_id_prefix`**: A prefix for naming agent save files.
-   **`force_new_agent`**: If `true`, the agent will start with a fresh brain, ignoring any saved state.
-   **`episodes_per_env`**: The number of episodes to run for each environment in parallel when using the Colosseum trainer.

-   **`language_model`**:
    -   `enabled`: Set to `true` to enable the Gemma LLM for chatting.
    -   `model_id`: The model ID from the Hugging Face Hub (e.g., "google/gemma-3-270m").
    -   `local_model_path`: If you have the model files locally, provide the path to the directory here to prevent downloading. This path is prioritized over `model_id`.

-   **`agent_config`**:
    -   `latent_dim`, `hidden_dim`: The dimensions of the World Model's internal vectors.
    -   `hyperparams`: Contains learning parameters like `learning_rate`, `gamma`, `batch_size`, and `buffer_capacity`.

## B. Training the Agent

There are two ways to train the agent:

### 1. Local Training (Single Environment)

This is useful for simple debugging and testing.

```bash
python backend/manage.py train_agent
```
- This script uses the `env_name` specified in `config.yaml`.
- It saves the agent's state history in `backend/storage/<agent_id>_history/`.

### 2. Colosseum Training (Multi-Environment, Real-time)

This is the primary method for large-scale, parallel training. It requires the Colosseum server to be running.

```bash
python backend/run_colosseum_agents.py
```
- This script reads the `env_list` from the file itself (can be moved to config).
- It launches a separate agent for each environment and runs them all concurrently.

## C. Interactive Chat Mode

To chat with an agent that has a language model enabled:

1.  Ensure `language_model.enabled` is `true` in `config.yaml`.
2.  Ensure you have a trained agent state saved in the history directory.
3.  Run the interactive script:

```bash
python backend/interactive_mode.py
```
- The script will load the latest version of the agent specified in the config.
- You can then type messages in the console and receive responses from the agent.
- Type `quit` to exit.
