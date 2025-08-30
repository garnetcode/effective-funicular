# Implements the ChimeraAgent, the core orchestrator for the cognitive architecture.
# This class encapsulates the agent's "brain" (cognitive layers), its sensory
# cortexes, and its action-selection mechanism. It is responsible for managing
# the flow of information between the layers as specified in the Project Chimera doc.

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import random
import time
import threading
from collections import deque
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from .world_model_core import WorldModel
from .skill_manager import SkillManager
from .state_history_manager import StateHistoryManager
from .replay_buffer import Experience
from .per_sequence_buffer import PERSequenceBuffer
from . import cortex as cortex_modules
from .cortex.vision_cortex import VisionCortex # Import the new cortex
from .action.modules import ActionHead
from .action.latent_planner import LatentPlanner
from .action.graph_planner import GraphPlanner
from .world_model.rep_learning import ContrastiveLoss

class StagContextProcessor(nn.Module):
    """
    Processes the STAG's activation path to create a rich context vector C_t.
    This version uses a simple linear layer over the concatenated node weights.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.processor = nn.Linear(input_dim, output_dim)

    def forward(self, flat_path_tensor):
        # The input is now a pre-processed tensor of shape (N, input_dim).
        return self.processor(flat_path_tensor)


def _initialize_cortexes(configs, output_dim, device):
    """
    Initializes all cortex modules based on the provided configurations.
    It dynamically loads the class from the cortex package.
    """
    cortexes = {}
    if configs is None: return cortexes
    for cortex_id, config in configs.items():
        class_name = config['type']
        params = config.get('params', {})
        try:
            CortexClass = getattr(cortex_modules, class_name)
            cortex_instance = None
            # Pass parameters based on cortex type
            if class_name == "DenseCortex":
                cortex_instance = CortexClass(input_dim=params['input_dim'], output_dim=output_dim)
            elif class_name == "VisionCortex":
                cortex_instance = VisionCortex(input_shape=params['input_shape'], output_dim=output_dim)
            else: # For TextCortex etc.
                cortex_instance = CortexClass(output_dim=output_dim)

            if cortex_instance:
                if isinstance(cortex_instance, nn.Module):
                    cortex_instance.to(device)
                cortexes[cortex_id] = cortex_instance

        except (AttributeError, ImportError) as e:
            print(f"Warning: Could not initialize cortex '{cortex_id}' of type '{class_name}': {e}")
    return cortexes

def sanitize_state_dict(state):
    """
    Recursively iterates through a dictionary and converts numpy data types to native
    Python types to ensure compatibility with torch.save/load.
    """
    for key, value in state.items():
        if isinstance(value, dict):
            sanitize_state_dict(value)
        elif isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
            state[key] = int(value)
        elif isinstance(value, (np.float64, np.float16, np.float32)):
            state[key] = float(value)
        elif isinstance(value, np.ndarray):
            state[key] = value.tolist()
    return state

class ChimeraAgent:
    def __init__(self, agent_id, embedding_dim, max_action_dim, latent_dim=128, hidden_dim=512, cortex_configs=None, load_from_storage=True, hyperparams=None, history_config=None, replay_buffer=None, **kwargs):
        self.agent_id = agent_id
        self.embedding_dim = embedding_dim
        self.max_action_dim = max_action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Consolidate hyperparameters from explicit arg and kwargs
        self.hyperparams = hyperparams or {}
        self.hyperparams.update(kwargs)

        history_config = history_config or {}
        self.history_manager = StateHistoryManager(agent_id, **history_config)
        self.learning_rate = self.hyperparams.get('learning_rate', 0.0001)
        self.gamma = self.hyperparams.get('gamma', 0.99)
        self.stag_expansion_threshold = self.hyperparams.get('stag_expansion_threshold', 0.1)
        self.max_grad_norm = self.hyperparams.get('max_grad_norm', 1.0)
        # Max length of the STAG activation path for the context processor
        self.max_stag_path_length = self.hyperparams.get('max_stag_path_length', 10)
        self.stag_context_dim = self.hyperparams.get('stag_context_dim', 128)
        self.use_stag_in_ac_loss = self.hyperparams.get('use_stag_in_ac_loss', True)
        self.world_model_pretrain_steps = self.hyperparams.get('world_model_pretrain_steps', 5000)
        self.stag_update_frequency = self.hyperparams.get('stag_update_frequency', 10)
        self.gng_pruning_frequency = self.hyperparams.get('gng_pruning_frequency', 1000)
        self.gng_min_utility_threshold = self.hyperparams.get('gng_min_utility_threshold', 0.1)
        self.world_model_weight_decay = self.hyperparams.get('world_model_weight_decay', 1e-6)
        self.num_ensemble_models = self.hyperparams.get('num_ensemble_models', 5)
        self.uncertainty_penalty_weight = self.hyperparams.get('uncertainty_penalty_weight', 0.1)
        # Schedules
        self.imagination_horizon_start = self.hyperparams.get('imagination_horizon_start', 5)
        self.imagination_horizon_end = self.hyperparams.get('imagination_horizon_end', 15)
        self.imagination_horizon_schedule_steps = self.hyperparams.get('imagination_horizon_schedule_steps', 10000)
        self.entropy_coef_start = self.hyperparams.get('entropy_coef_start', 0.01)
        self.entropy_coef_end = self.hyperparams.get('entropy_coef_end', 0.001)
        self.entropy_coef_schedule_steps = self.hyperparams.get('entropy_coef_schedule_steps', 20000)
        self.use_planner = self.hyperparams.get('use_planner', False)
        self.high_level_replan_frequency = self.hyperparams.get('high_level_replan_frequency', 100)
        self.low_level_plan_frequency = self.hyperparams.get('low_level_plan_frequency', 10)
        self.goal_dim = self.hyperparams.get('goal_dim', 512)
        self.contrastive_loss_weight = self.hyperparams.get('contrastive_loss_weight', 0.1)
        self.sf_dimension = self.hyperparams.get('sf_dimension', 64)
        self.sf_learning_rate = self.hyperparams.get('sf_learning_rate', 0.01)
        self.her_replay_strategy = self.hyperparams.get('her_replay_strategy', 'future')
        self.her_replay_k = self.hyperparams.get('her_replay_k', 4)

        # --- Add Homeostatic Vitals ---
        self.max_energy = 100.0
        self.energy = self.max_energy
        self.integrity = 100.0
        self.max_integrity = 100.0
        self.metabolic_cost = 0.1

        # --- Initialize Agent State and Components ---
        self.steps_done = 0
        self.step_lock = threading.Lock()
        self.train_steps = 0
        self.last_stag_node_id = None
        self.subgoal_reward = 0
        self.subgoal_duration = 0
        # h_t and z_t for the RSSM, initialized for the hierarchy
        self.hidden_state, self.latent_state = None, None # Will be initialized after world models
        self.last_action = torch.tensor([0], device=self.device)
        self.high_level_plan = None
        self.current_subgoal = None
        self.current_goal = np.zeros(self.goal_dim)
        self.action_plan = None
        self.cortex_configs = cortex_configs or {}
        self.snn_history_length = self.hyperparams.get('snn_history_length', 10)
        self.state_history_for_snn = deque(maxlen=self.snn_history_length)

        # --- Initialize Architecture Components ---
        self.cortexes = _initialize_cortexes(self.cortex_configs, self.embedding_dim, self.device)

        # World Model Ensemble
        self.world_models = nn.ModuleList([
            WorldModel(self.embedding_dim, self.max_action_dim, latent_dim, hidden_dim, self.goal_dim, hyperparams=self.hyperparams)
            for _ in range(self.num_ensemble_models)
        ]).to(self.device)
        self.world_model_optimizers = [
            torch.optim.Adam(
                wm.parameters(),
                lr=self.hyperparams.get('world_model_lr', 1e-4),
                weight_decay=self.world_model_weight_decay
            ) for wm in self.world_models
        ]

        # Initialize hidden and latent states
        self.hidden_state, self.latent_state = self.world_models[0].get_initial_state(batch_size=1, device=self.device)

        # Normalization Hub
        self.h_norm = nn.LayerNorm(self.hidden_dim).to(self.device)
        self.z_norm = nn.LayerNorm(self.latent_dim).to(self.device)

        # Skill Manager (manages multiple STAGs)
        self.skill_manager = SkillManager(self.hidden_dim, **self.hyperparams)
        self.active_skill_id = None # The currently active skill/environment

        # STAG Context Processor
        # Input dimension is the max path length * the dimension of a node's weight vector (hidden_dim)
        self.stag_context_processor = StagContextProcessor(
            input_dim=self.max_stag_path_length * self.hidden_dim,
            output_dim=self.stag_context_dim
        ).to(self.device)

        # Actor-Critic Planner
        # Actor (Policy Head)
        # Input includes current state, STAG context, and SNN prediction
        self.action_head = ActionHead(
            input_dim=self.hidden_dim + self.stag_context_dim + self.hidden_dim,
            n_actions=self.max_action_dim,
            goal_dim=self.goal_dim,
            learning_rate=self.learning_rate
        ).to(self.device)
        # Critic (Value Head)
        # Input includes current state, latent state, goal, and SNN prediction
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim + self.goal_dim + self.hidden_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 1)
        ).to(self.device)

        self.planner = LatentPlanner(
            world_models=self.world_models,
            action_dim=self.max_action_dim,
            plan_horizon=self.hyperparams.get('cem_plan_horizon', 12),
            num_samples=self.hyperparams.get('cem_num_samples', 1000),
            top_k=self.hyperparams.get('cem_top_k', 100),
            iterations=self.hyperparams.get('cem_iterations', 10),
            uncertainty_penalty_weight=self.uncertainty_penalty_weight
        ).to(self.device)

        self.graph_planner = GraphPlanner()

        # Random projection matrix for state features phi(s)
        self.sf_projection_matrix = torch.randn(self.hidden_dim, self.sf_dimension, device=self.device)
        self.reward_weights = torch.zeros(self.sf_dimension, device=self.device)

        # MoCo-style queue for contrastive learning
        self.contrastive_queue_size = self.hyperparams.get('contrastive_queue_size', 4096)
        if self.contrastive_queue_size > 0:
            self.contrastive_queue = torch.randn(self.contrastive_queue_size, self.hidden_dim, device=self.device)
            self.contrastive_queue = F.normalize(self.contrastive_queue, dim=1)
            self.contrastive_queue_ptr = torch.zeros(1, dtype=torch.long, device=self.device)

        # KL Balancing
        self.use_kl_balancing = self.hyperparams.get('kl_balancing', True)
        if self.use_kl_balancing:
            self.kl_target = self.hyperparams.get('kl_target', 3.0)
            self.kl_coeff = self.hyperparams.get('kl_coeff_initial', 1.0)
            self.pid_kp = self.hyperparams.get('kl_coeff_pid_kp', 0.01)
            self.pid_ki = self.hyperparams.get('kl_coeff_pid_ki', 0.0001)
            self.pid_kd = self.hyperparams.get('kl_coeff_pid_kd', 0.001)
            self.kl_error_integral = 0
            self.kl_last_error = 0

        # Running stats for novelty z-scoring
        self.novelty_stats = {
            'error_fast': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'error_slow': {'mean': 0.0, 'std': 1.0, 'count': 0}
        }

        # Policy Distillation
        self.distill_params = self.hyperparams.get('policy_distillation', {})
        self.lambda_bc_start = self.distill_params.get('lambda_bc_start', 1.0)
        self.lambda_bc_end = self.distill_params.get('lambda_bc_end', 0.1)
        self.lambda_bc_schedule_steps = self.distill_params.get('lambda_bc_schedule_steps', 300000)

        # Hierarchical Control
        self.h_control_params = self.hyperparams.get('hierarchical_control', {})
        self.h_step_interval = self.h_control_params.get('h_step_interval', 50)
        self.h_step_counter = 0

        # This will be populated by load_state so the training script can access it
        self.loaded_snapshot_data = None

        if load_from_storage and self.history_manager._read_history():
            self.load_state()
        else:
            self.save_state(version_info={"message": "Initial state."})

        if replay_buffer:
            self.replay_buffer = replay_buffer
        else:
            buffer_capacity = self.hyperparams.get('buffer_capacity', 10000)
            sequence_length = self.hyperparams.get('sequence_length', 50)
            self.replay_buffer = PERSequenceBuffer(
                capacity=buffer_capacity,
                sequence_length=sequence_length,
                alpha=self.hyperparams.get('per_alpha', 0.6),
                beta_start=self.hyperparams.get('per_beta_start', 0.4),
                beta_frames=self.hyperparams.get('per_beta_frames', 100000),
                her_replay_strategy=self.her_replay_strategy,
                her_replay_k=self.her_replay_k
            )

    def save_state(self, version_info={}):
        """Saves the complete state of the agent for resumable training."""
        with self.step_lock:
            steps_done_safe = self.steps_done

        agent_state = {
            # --- Model States ---
            'world_models_state_dicts': [wm.state_dict() for wm in self.world_models],
            'action_head_state_dict': self.action_head.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'stag_context_processor_state_dict': self.stag_context_processor.state_dict(),

            # --- Optimizer States ---
            'world_model_optimizers_state_dicts': [opt.state_dict() for opt in self.world_model_optimizers],
            # Note: AC optimizer is re-created in train_policy_in_imagination, so no need to save.

            # --- Agent Internal State ---
            'steps_done': steps_done_safe,
            'train_steps': self.train_steps,
            'novelty_stats': self.novelty_stats,
            'skill_manager_state': self.skill_manager.get_serializable_structure(),

            # --- Learning Mechanism States ---
            'kl_coeff': self.kl_coeff,
            'kl_error_integral': self.kl_error_integral,
            'kl_last_error': self.kl_last_error,
            'contrastive_queue': self.contrastive_queue,
            'contrastive_queue_ptr': self.contrastive_queue_ptr,
        }

        # Combine with any additional info provided and sanitize for saving
        full_state_package = {**agent_state, **version_info}
        sanitized_state = sanitize_state_dict(full_state_package)
        self.history_manager.save_snapshot(sanitized_state, version_info)

    def load_state(self, version='latest'):
        """Loads the complete state of the agent for resumable training."""
        state_dict = self.history_manager.load_snapshot(version)
        if not state_dict:
            print("Warning: No state to load.")
            return

        # Store the loaded data so the training script can access it
        self.loaded_snapshot_data = state_dict

        # --- Load Model States ---
        if 'world_models_state_dicts' in state_dict:
            for i, sd in enumerate(state_dict['world_models_state_dicts']):
                if i < len(self.world_models): self.world_models[i].load_state_dict(sd)

        self.action_head.load_state_dict(state_dict['action_head_state_dict'])
        self.value_head.load_state_dict(state_dict['value_head_state_dict'])
        self.stag_context_processor.load_state_dict(state_dict.get('stag_context_processor_state_dict'))

        # --- Load Optimizer States ---
        if 'world_model_optimizers_state_dicts' in state_dict:
            for i, sd in enumerate(state_dict['world_model_optimizers_state_dicts']):
                if i < len(self.world_model_optimizers): self.world_model_optimizers[i].load_state_dict(sd)

        # --- Load Agent Internal State ---
        with self.step_lock:
            self.steps_done = state_dict.get('steps_done', 0)
        self.train_steps = state_dict.get('train_steps', 0)
        self.novelty_stats = state_dict.get('novelty_stats', self.novelty_stats)
        if 'skill_manager_state' in state_dict:
            # Pass the agent's hidden_dim to ensure correct initialization.
            skill_manager_kwargs = self.hyperparams.copy()
            skill_manager_kwargs['dimensions'] = self.hidden_dim
            self.skill_manager = SkillManager.from_serializable_structure(
                state_dict['skill_manager_state'], **skill_manager_kwargs
            )

        # --- Load Learning Mechanism States ---
        self.kl_coeff = state_dict.get('kl_coeff', self.kl_coeff)
        self.kl_error_integral = state_dict.get('kl_error_integral', 0)
        self.kl_last_error = state_dict.get('kl_last_error', 0)
        if 'contrastive_queue' in state_dict: self.contrastive_queue.copy_(state_dict['contrastive_queue'])
        if 'contrastive_queue_ptr' in state_dict: self.contrastive_queue_ptr.copy_(state_dict['contrastive_queue_ptr'])

        logger.info(f"Agent state loaded from version '{version}'. Resuming from step {self.steps_done}.")

    def set_active_skill(self, skill_id):
        """Sets the currently active skill/environment for the agent."""
        logger.info(f"Agent active skill set to: {skill_id}")
        self.active_skill_id = skill_id

    def set_goal(self, goal):
        """Sets the current goal for the agent."""
        self.current_goal = goal

    def perceive_and_update_state(self, cortex_id, raw_obs, damage_taken=0):
        """
        Processes observation, updates vitals, and updates the world model state.
        """
        # 1. Update vitals based on damage from the last step
        self._update_vitals(damage_taken=damage_taken)

        # 2. Process raw observation through the specified cortex
        obs_numpy = self.cortexes[cortex_id].process(raw_obs)
        obs_tensor = torch.from_numpy(obs_numpy).float().unsqueeze(0).to(self.device)

        # 3. Use the first world model in the ensemble to update the agent's internal state
        with torch.no_grad():
            h_next, z_next, kl_loss = self.world_models[0].rssm(
                obs_tensor, self.last_action, self.hidden_state, self.latent_state
            )

        # 4. Update the agent's internal state
        self.hidden_state = h_next
        self.latent_state = z_next

        # 5. Conditionally engage the STAG framework
        h_normalized = None
        activation_path = []
        novelty = 0
        winner_id = None
        # Only engage STAG if we are past the pre-training phase.
        if self.steps_done > self.world_model_pretrain_steps:
            # Use the hidden state for the STAG
            h_normalized = self.h_norm(self.hidden_state).detach().cpu().numpy().flatten()
            # The vector is now normalized by LayerNorm and will be normalized again
            # by _safe_unit in the GNG engine for double safety.

            # Find the activation path and get the potential novelty signal (error)
            terminal_node, winner_id, activation_path = self.skill_manager.find_terminal_node_and_path(self.active_skill_id, h_normalized)
            if terminal_node and winner_id is not None:
                winner_node = terminal_node['gng'].nodes[winner_id]
                error_fast = winner_node.get('error_fast', 0)
                error_slow = winner_node.get('error_slow', 0)

                # Update running stats for z-scoring
                self._update_running_stats('error_fast', error_fast)
                self._update_running_stats('error_slow', error_slow)

                # Calculate z-scores
                z_score_fast = (error_fast - self.novelty_stats['error_fast']['mean']) / self.novelty_stats['error_fast']['std']
                z_score_slow = (error_slow - self.novelty_stats['error_slow']['mean']) / self.novelty_stats['error_slow']['std']

                novelty = max(z_score_fast, z_score_slow)

                # Check for STAG node transition
                if self.last_stag_node_id is not None and self.last_stag_node_id != winner_id:
                    # Update option model
                    self.skill_manager.update_option_model(
                        self.active_skill_id,
                        self.last_stag_node_id,
                        winner_id,
                        self.subgoal_reward,
                        self.subgoal_duration
                    )
                    # Update successor features
                    with torch.no_grad():
                        state_features = torch.matmul(self.hidden_state, self.sf_projection_matrix).squeeze(0).cpu().numpy()
                    self.skill_manager.update_successor_features(
                        self.active_skill_id,
                        self.last_stag_node_id,
                        winner_id,
                        state_features
                    )
                    # Reset subgoal counters
                    self.subgoal_reward = 0
                    self.subgoal_duration = 0

                self.last_stag_node_id = winner_id

                # Check if subgoal has been reached
                if self.current_subgoal is not None and winner_id == self.current_subgoal:
                    print(f"Subgoal {self.current_subgoal} reached!")
                    self.current_subgoal = None
                    if self.high_level_plan and len(self.high_level_plan) > 0:
                        self.current_subgoal = self.high_level_plan.pop(0)


        # The GNG is now updated in a separate step after the reward is known.
        if winner_id is not None:
            self.state_history_for_snn.append(winner_id)

        return self.hidden_state, self.latent_state, h_normalized, activation_path, novelty

    def update_stag(self, h_normalized, r_env):
        """
        Updates the STAG/GNG with the given state and the environmental reward it produced.
        This is called in the main training loop after a reward is received.
        """
        # Gate the update based on pre-training and update frequency.
        if self.steps_done <= self.world_model_pretrain_steps or \
           self.steps_done % self.stag_update_frequency != 0:
            return

        if h_normalized is None: # Should not happen if logic is correct, but as a safeguard.
            return

        # We need to get the terminal node for the currently active skill
        terminal_node, _, _ = self.skill_manager.find_terminal_node_and_path(self.active_skill_id, h_normalized)
        if terminal_node:
            # The GNG's utility update is driven by the environmental reward
            terminal_node['gng'].process_input(h_normalized, reward=r_env)

    def select_action(self, actual_action_dim, activation_path, evaluation_mode=False):
        """
        Selects an action using the hierarchical planning and control system.
        In evaluation_mode, it disables exploration (e.g., epsilon-greedy).
        Also returns the probability distribution over actions.
        """
        start_time = time.time()
        if not evaluation_mode:
            with self.step_lock:
                self.steps_done += 1
        self.h_step_counter += 1
        decision_maker = "policy" # Default
        log_prob = torch.tensor(0.0) # Default
        stag_context_vector = torch.zeros(1, self.stag_context_dim, device=self.device) # Default
        action_probs = torch.zeros(self.max_action_dim, device=self.device) # Default probabilities
        h_normalized = torch.zeros(1, self.hidden_dim, device=self.device)
        snn_prediction = torch.zeros(1, self.hidden_dim, device=self.device)

        # --- Hierarchical Planning Logic ---
        # 1. High-Level Planner (GraphPlanner)
        if self.use_planner and (self.current_subgoal is None or self.h_step_counter >= self.h_step_interval):
            self.h_step_counter = 0
            stag_graph = self.skill_manager.get_flattened_structure(self.active_skill_id)
            if self.last_stag_node_id and len(stag_graph['nodes']) > 1:
                # --- Goal Curriculum Logic ---
                curriculum_params = self.h_control_params.get('goal_curriculum', {})
                if curriculum_params.get('enabled', False) and not evaluation_mode:
                    # Calculate current max distance
                    schedule_steps = curriculum_params.get('schedule_steps', 1)
                    progress = min(1.0, self.train_steps / schedule_steps)
                    start_dist = curriculum_params.get('initial_max_graph_dist', 2)
                    end_dist = curriculum_params.get('max_graph_dist_end', 10)
                    max_dist = int(start_dist + progress * (end_dist - start_dist))

                    # Find reachable goals within the curriculum distance
                    distances = self.graph_planner.bfs_distances(self.last_stag_node_id, stag_graph)
                    possible_goals = [nid for nid, dist in distances.items() if 1 <= dist <= max_dist]
                else:
                    # Fallback to original logic if curriculum is disabled or in eval mode
                    possible_goals = [nid for nid in stag_graph['nodes'] if nid != self.last_stag_node_id]

                if possible_goals:
                    goal_node_id = random.choice(possible_goals)
                    self.high_level_plan = self.graph_planner.plan(stag_graph, self.skill_manager.option_models.get(self.active_skill_id, {}), self.last_stag_node_id, goal_node_id)
                    if self.high_level_plan and len(self.high_level_plan) > 1:
                        self.current_subgoal = self.high_level_plan[1] # Target the next node in the path
                        logger.info(f"H-Planner: New subgoal set to STAG node {self.current_subgoal}")
                        self.action_plan = None # Invalidate low-level plan
                    else:
                        self.current_subgoal = None # Plan failed or is trivial

        # 2. Mid-Level Planner (LatentPlanner)
        if self.use_planner and self.current_subgoal is not None and self.action_plan is None:
            stag = self.skill_manager._get_or_create_stag(self.active_skill_id)
            # This assumes subgoals are in the terminal GNG
            terminal_gng = stag.level_map[max(stag.level_map.keys())]
            if self.current_subgoal in terminal_gng.nodes:
                subgoal_weight = torch.from_numpy(terminal_gng.nodes[self.current_subgoal]['weight']).float().to(self.device)
                with torch.no_grad():
                    logger.info(f"M-Planner: Planning trajectory to subgoal {self.current_subgoal}")
                    plan = self.planner.plan(self.hidden_state, self.latent_state, subgoal_weight=subgoal_weight)

                if plan is not None and len(plan) > 0:
                    self.action_plan = plan
                else:
                    # Fallback: Planner failed, find a nearby alternative subgoal
                    logger.warning(f"M-Planner failed for subgoal {self.current_subgoal}. Finding alternative.")
                    original_subgoal_weight = subgoal_weight.cpu().numpy()
                    neighbor_ids, _ = self.skill_manager.find_k_nearest_neighbors(self.active_skill_id, original_subgoal_weight, k=5)

                    # Try the next closest neighbor that isn't the one we just failed on
                    new_subgoal = next((nid for nid in neighbor_ids if nid != self.current_subgoal), None)

                    if new_subgoal:
                        logger.info(f"Fallback: Setting new subgoal to {new_subgoal}")
                        self.current_subgoal = new_subgoal
                    else:
                        logger.warning("Fallback failed, no alternative subgoals found. Clearing subgoal.")
                        self.current_subgoal = None
            else:
                # Fallback: Subgoal not found in GNG
                logger.warning(f"Subgoal {self.current_subgoal} not found in terminal GNG. Clearing subgoal.")
                self.current_subgoal = None

        # 3. Low-Level Execution (Plan Following or Policy)
        if self.action_plan is not None and len(self.action_plan) > 0:
            action_continuous = self.action_plan[0]
            action_tensor = torch.argmax(action_continuous, dim=-1)
            self.action_plan = self.action_plan[1:]
            decision_maker = "planner"
            with torch.no_grad():
                probs = F.softmax(action_continuous, dim=-1).squeeze()
                action_probs[:probs.shape[0]] = probs
        else:
            # Fallback to learned policy
            with torch.no_grad():
                # --- Get SNN Prediction (Temporarily Disabled) ---
                snn_prediction = torch.zeros(1, self.hidden_dim, device=self.device)

                # Use the hidden state for the policy
                h_normalized = self.h_norm(self.hidden_state)
                stag_context_vector = self._prepare_stag_context(activation_path)
                combined_input = torch.cat((h_normalized, stag_context_vector, snn_prediction), dim=1)

                goal_tensor = torch.from_numpy(self.current_goal).float().to(self.device)
                goal_tensor_expanded = goal_tensor.unsqueeze(0).expand(combined_input.size(0), -1)
                action_dist = self.action_head(combined_input, goal_tensor_expanded)

                mask = torch.full(action_dist.logits.shape, -float('inf'), device=self.device)
                mask[0, :actual_action_dim] = 0
                masked_logits = action_dist.logits + mask
                action_dist = Categorical(logits=masked_logits)

                # Store probabilities from the policy
                action_probs[:actual_action_dim] = action_dist.probs.squeeze()

                epsilon = self._get_epsilon()
                if not evaluation_mode and np.random.rand() < epsilon:
                    action_tensor = torch.tensor([np.random.randint(0, actual_action_dim)], device=self.device)
                    decision_maker = "random"
                    # For random action, show a uniform distribution
                    action_probs.fill_(0.0)
                    action_probs[:actual_action_dim] = 1.0 / actual_action_dim
                else:
                    action_tensor = action_dist.sample()
                    decision_maker = "policy"
                log_prob = action_dist.log_prob(action_tensor)

        action = action_tensor.item()
        self.last_action = action_tensor.reshape(1)
        action_time = time.time() - start_time
        epsilon = self._get_epsilon() if not evaluation_mode else 0.0 # Recalculate for logging

        return action, log_prob, stag_context_vector, decision_maker, epsilon, action_time, action_probs.cpu().numpy(), h_normalized, snn_prediction

    def _prepare_stag_context(self, activation_path):
        """
        Prepares the STAG context vector from an activation path.
        This involves fetching node weights, padding/truncating, and processing.
        """
        if not self.use_stag_in_ac_loss or not activation_path:
            return torch.zeros(1, self.stag_context_dim, device=self.device)

        stag = self.skill_manager._get_or_create_stag(self.active_skill_id)
        if not stag:
            return torch.zeros(1, self.stag_context_dim, device=self.device)

        path_tensors = []
        for step in activation_path:
            level = step.get('level')
            node_id = step.get('node_id')
            if level is not None and node_id is not None and \
               level in stag.level_map and node_id in stag.level_map[level]['gng'].nodes:
                weight = stag.level_map[level]['gng'].nodes[node_id]['weight']
                path_tensors.append(torch.from_numpy(weight).float().to(self.device))

        if not path_tensors:
            return torch.zeros(1, self.stag_context_dim, device=self.device)

        # Pad or truncate the list of tensors to the required path length
        padded_path_tensors = path_tensors[:self.max_stag_path_length]
        while len(padded_path_tensors) < self.max_stag_path_length:
            padded_path_tensors.append(torch.zeros(self.hidden_dim, device=self.device))

        # Concatenate into a single flat tensor and add a batch dimension
        flat_path_tensor = torch.cat(padded_path_tensors, dim=0).unsqueeze(0)

        # Process with the context processor
        return self.stag_context_processor(flat_path_tensor)

    def _get_epsilon(self):
        epsilon_start = self.hyperparams.get('epsilon_start', 0.9)
        epsilon_end = self.hyperparams.get('epsilon_end', 0.05)
        epsilon_decay_steps = self.hyperparams.get('epsilon_decay_steps', 20000)
        if self.steps_done < epsilon_decay_steps:
            return epsilon_start - (epsilon_start - epsilon_end) * (self.steps_done / epsilon_decay_steps)
        return epsilon_end

    def _update_vitals(self, damage_taken=0, energy_gain=0):
        """Updates agent's vitals."""
        # Update Integrity based on damage
        self.integrity = max(0, self.integrity - damage_taken)
        # Update Energy
        self.energy = min(self.max_energy, self.energy - self.metabolic_cost + energy_gain)

    def get_internal_reward(self, damage_taken, novelty_signal):
        """Calculates the total internal reward signal."""
        internal_reward = -self.metabolic_cost # Base metabolic cost

        # Penalty for low energy
        low_energy_threshold = self.hyperparams.get('low_energy_threshold', 0.2)
        if (self.energy / self.max_energy) < low_energy_threshold:
            internal_reward += self.hyperparams.get('low_energy_penalty', -10.0)

        # Penalty for taking damage
        damage_penalty_multiplier = self.hyperparams.get('damage_penalty_multiplier', -5.0)
        internal_reward += damage_penalty_multiplier * damage_taken

        # Reward for novelty (exploring new concepts in STAG)
        novelty_reward_weight = self.hyperparams.get('novelty_reward_weight', 0.1)
        internal_reward += novelty_reward_weight * novelty_signal

        return internal_reward

    def _update_running_stats(self, key, value, decay=0.99):
        """Welford's online algorithm for running mean and std."""
        stats = self.novelty_stats[key]
        stats['count'] += 1

        old_mean = stats['mean']
        new_mean = old_mean + (value - old_mean) / stats['count']

        # Using variance is more stable
        old_var = stats['std']**2
        new_var = old_var + (value - old_mean) * (value - new_mean)

        stats['mean'] = new_mean
        stats['std'] = np.sqrt(new_var / stats['count']) if stats['count'] > 1 else 1.0
        # Add a small epsilon to prevent division by zero
        stats['std'] = max(stats['std'], 1e-6)


    def record_experience(self, h_t, z_t, activation_path, obs, action, log_prob, reward, next_obs, done):
        """Pushes an experience to the replay buffer and updates subgoal counters."""
        h_t_np = h_t.detach().cpu().numpy()
        z_t_np = z_t.detach().cpu().numpy()

        experience = (h_t_np, z_t_np, activation_path, obs, action, log_prob.item(), reward, next_obs, done, self.current_goal)
        self.replay_buffer.push(*experience)

        self.subgoal_reward += reward
        self.subgoal_duration += 1

    def train_on_batch(self, batch, cortex_id="vector_input"):
        """
        Trains the World Model and Policy on a pre-sampled batch of experiences.
        """
        self.train_steps += 1
        total_start_time = time.time()
        timing_stats = {}

        # --- Part 1: Train the World Model ---
        wm_start_time = time.time()
        world_model_stats, _, _ = self.train_world_model_on_batch(batch, cortex_id)
        timing_stats['wm_train_time_ms'] = (time.time() - wm_start_time) * 1000

        # --- Part 2: Train the Policy ---
        policy_stats = {}
        policy_train_frequency = self.hyperparams.get('policy_train_frequency', 1)
        if self.train_steps % policy_train_frequency == 0:
            policy_start_time = time.time()
            policy_stats = self.train_policy_in_imagination(batch)
            timing_stats['policy_train_time_ms'] = (time.time() - policy_start_time) * 1000

        # --- Part 3: Prune the STAG ---
        if self.train_steps > 0 and self.train_steps % self.gng_pruning_frequency == 0:
            if self.active_skill_id:
                self.skill_manager.prune_graph(self.active_skill_id, self.gng_min_utility_threshold)

        # --- Part 4: Train the SNN Predictor ---
        snn_stats = {}
        snn_train_frequency = self.hyperparams.get('snn_train_frequency', 5)
        if self.train_steps > self.world_model_pretrain_steps and \
           self.train_steps % snn_train_frequency == 0:
            snn_start_time = time.time()
            snn_stats = self._train_snn_on_batch(batch)
            timing_stats['snn_train_time_ms'] = (time.time() - snn_start_time) * 1000

        # --- Combine stats for logging ---
        combined_stats = {**world_model_stats, **policy_stats, **snn_stats, **timing_stats}
        combined_stats['total_train_time_ms'] = (time.time() - total_start_time) * 1000
        return combined_stats

    def _train_snn_on_batch(self, batch):
        """
        Trains the NodePredictor on a batch of sequences of STAG node activations.
        """
        if not self.active_skill_id:
            return {}

        stag = self.skill_manager._get_or_create_stag(self.active_skill_id)
        terminal_gng = stag.level_map[max(stag.level_map.keys())]

        # This is a placeholder for getting node activation sequences from the replay buffer.
        # This part requires a more complex replay buffer that stores STAG activation paths.
        # For now, we'll generate dummy data to test the training loop.
        batch_size = batch['obs'].shape[0]
        seq_len = self.snn_history_length

        # Create dummy sequences of node IDs
        node_ids = list(terminal_gng.nodes.keys())
        if len(node_ids) < 2:
            return {"snn_predictor_loss": 0} # Not enough nodes to form a sequence

        input_node_ids = np.random.choice(node_ids, (batch_size, seq_len))
        target_node_ids = np.random.choice(node_ids, (batch_size,))

        # Convert node IDs to embeddings
        input_embeddings = np.array([[terminal_gng.nodes[nid]['weight'] for nid in seq] for seq in input_node_ids])

        input_tensor = torch.from_numpy(input_embeddings).float().to(self.device)
        target_tensor = torch.from_numpy(target_node_ids).long().to(self.device)

        stag.snn_predictor.train()
        loss = stag.snn_predictor.train_on_batch(input_tensor, target_tensor)

        return {"snn_predictor_loss": loss}

    def train(self, cortex_id="vector_input"):
        """
        The main training loop that samples from the agent's internal replay buffer.
        """
        batch_size = self.hyperparams.get('batch_size', 32)
        if len(self.replay_buffer) < batch_size:
            return {}

        batch, indices, weights = self.replay_buffer.sample(batch_size)

        if batch is None:
            return {} # Not enough data to sample a sequence

        # PER-specific logic
        if indices is not None:
            self.update_priorities(indices, weights) # This is a placeholder for actual priority calculation

        return self.train_on_batch(batch, cortex_id)

    def train_world_model_on_batch(self, batch, cortex_id="vector_input"):
        """
        Trains the World Model ensemble on a given batch of sequences.
        """
        weights = torch.ones(len(batch['obs']), device=self.device) / len(batch['obs'])
        total_wm_loss = 0

        for i, (world_model, wm_optimizer) in enumerate(zip(self.world_models, self.world_model_optimizers)):
            world_model_loss, _, last_hidden_states, last_reward_seq = self._calculate_world_model_loss(
                world_model, batch, cortex_id, weights
            )
            wm_optimizer.zero_grad()
            world_model_loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), self.max_grad_norm)
            wm_optimizer.step()
            total_wm_loss += world_model_loss.item()

        avg_wm_loss = total_wm_loss / self.num_ensemble_models
        stats = { "wm_loss_total": avg_wm_loss }
        return stats, None, None

    def update_priorities(self, indices, priorities):
        """Updates the priorities of experiences in the replay buffer."""
        self.replay_buffer.update_priorities(indices, priorities)

    def _calculate_world_model_loss(self, world_model, batch, cortex_id, weights):
        """Helper function to calculate the world model loss for a given batch."""
        obs_sequence = torch.from_numpy(batch['obs']).float().to(self.device)
        action_sequence = torch.from_numpy(batch['action']).float().to(self.device)
        reward_sequence = torch.from_numpy(batch['reward']).float().to(self.device)

        batch_size, seq_len, _ = obs_sequence.shape
        obs_processed = self.cortexes[cortex_id](obs_sequence.view(-1, obs_sequence.shape[-1])).view(batch_size, seq_len, -1)

        h_t, z_t = world_model.get_initial_state(batch_size, self.device)
        total_kl_loss = 0
        total_recon_loss = 0
        all_h = []

        for t in range(seq_len):
            obs_t = obs_processed[:, t]
            action_t = action_sequence[:, t]

            h_t, z_t, kl_loss = world_model.rssm(obs_t, action_t, h_t, z_t)

            recon_obs = world_model.obs_decoder(h_t, z_t)
            recon_loss = F.mse_loss(recon_obs, obs_t)

            total_kl_loss += kl_loss
            total_recon_loss += recon_loss
            all_h.append(h_t)

        total_kl_loss /= seq_len
        total_recon_loss /= seq_len

        free_nats = self.hyperparams.get('free_bits', 1.0)
        kl_loss = torch.max(torch.tensor(free_nats, device=self.device), total_kl_loss)

        final_loss = kl_loss + total_recon_loss

        h_states = torch.stack(all_h, dim=1)

        return final_loss, total_kl_loss, h_states, batch['reward']


    def train_policy_in_imagination(self, batch):
        """
        Trains the Actor (ActionHead) and Critic (ValueHead) in imagination.
        This version is corrected to work with a hierarchical state space.
        """
        if not batch:
            return {}

        horizon = self.planner.plan_horizon

        # --- Calculate scheduled parameters ---
        progress = min(1.0, self.train_steps / self.entropy_coef_schedule_steps)
        entropy_coef = self.entropy_coef_start - progress * (self.entropy_coef_start - self.entropy_coef_end)
        progress = min(1.0, self.train_steps / self.lambda_bc_schedule_steps)
        lambda_bc = self.lambda_bc_start - progress * (self.lambda_bc_start - self.lambda_bc_end)

        h_start = torch.from_numpy(batch['h'][:, 0]).float().to(self.device).squeeze(1)
        z_start = torch.from_numpy(batch['z'][:, 0]).float().to(self.device).squeeze(1)

        batch_size = h_start.size(0)
        goal_sequence = torch.from_numpy(batch['goal'][:, 0]).float().to(self.device)

        teacher_plan = None
        if self.use_planner:
            # --- Generate "teacher" plan from the latent planner ---
            with torch.no_grad():
                # Pass the top-level states to the planner
                teacher_plan = self.planner.plan(h_start, z_start, current_goal=goal_sequence.cpu().numpy())

            if teacher_plan is None:
                logger.warning("Planner failed to return a plan. Skipping policy update for this batch.")
                return {}
        else:
            # If planner is disabled, disable the behavioral cloning loss term
            lambda_bc = 0.0

        # --- Imagine Trajectories using the policy's actions ---
        h_t, z_t = h_start, z_start
        imagined_h = [h_t]
        imagined_z = [z_t]
        policy_actions = []
        policy_log_probs = []
        policy_entropies = []

        transition_model = self.world_models[0].rssm.transition_model

        logger.info(f"Imagination: Starting trajectory of length {horizon}")
        for t in range(horizon):
            norm_h = self.h_norm(h_t)
            stag_context_batch = torch.zeros(batch_size, self.stag_context_dim, device=self.device)
            snn_pred_batch = torch.zeros(batch_size, self.hidden_dim, device=self.device)
            action_input = torch.cat([norm_h, stag_context_batch, snn_pred_batch], dim=-1)
            action_dist = self.action_head(action_input, goal_sequence)

            action = action_dist.sample()
            policy_actions.append(action)
            policy_log_probs.append(action_dist.log_prob(action))
            policy_entropies.append(action_dist.entropy())

            logger.debug(f"Imagination Step {t+1}/{horizon}: action={action.float().mean().item():.2f}")

            with torch.no_grad():
                h_t, prior_mean, prior_std = transition_model(z_t, action, h_t)
                z_t = Normal(prior_mean, prior_std).rsample()

            imagined_h.append(h_t)
            imagined_z.append(z_t)

        imagined_h = torch.stack(imagined_h)
        imagined_z = torch.stack(imagined_z)
        policy_actions = torch.stack(policy_actions)
        policy_log_probs = torch.stack(policy_log_probs)
        policy_entropies = torch.stack(policy_entropies)

        # --- Predict Rewards and Values for the policy's imagined trajectory ---
        norm_imagined_h = self.h_norm(imagined_h)
        norm_imagined_z = self.z_norm(imagined_z)
        goal_expanded = goal_sequence.unsqueeze(0).expand(horizon + 1, -1, -1)

        snn_prediction_imagined = torch.zeros(horizon + 1, batch_size, self.hidden_dim, device=self.device)
        imagined_rewards = self.world_models[0].reward_model(torch.cat([norm_imagined_z, norm_imagined_h, goal_expanded], dim=-1)).squeeze(-1)
        imagined_values = self.value_head(torch.cat([norm_imagined_h, norm_imagined_z, goal_expanded, snn_prediction_imagined], dim=-1)).squeeze(-1)

        lambda_ = self.hyperparams.get('lambda', 0.95)
        returns = torch.zeros_like(imagined_values[-1])
        lambda_returns = []
        for t in reversed(range(horizon)):
            returns = imagined_rewards[t] + self.gamma * ((1 - lambda_) * imagined_values[t+1].detach() + lambda_ * returns)
            lambda_returns.append(returns)
        lambda_returns = torch.stack(list(reversed(lambda_returns)))

        advantage = (lambda_returns - imagined_values[:-1]).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        policy_loss = -(policy_log_probs * advantage).mean()
        critic_loss = F.mse_loss(imagined_values[:-1], lambda_returns.detach())

        imagined_states = norm_imagined_h[:-1]
        stag_context_bc = torch.zeros(horizon, batch_size, self.stag_context_dim, device=self.device)
        goal_bc = goal_sequence.unsqueeze(0).expand(horizon, -1, -1)
        imagined_states_flat = imagined_states.reshape(-1, self.hidden_dim)
        stag_context_bc_flat = stag_context_bc.reshape(-1, self.stag_context_dim)
        snn_pred_bc_flat = torch.zeros(horizon * batch_size, self.hidden_dim, device=self.device)
        goal_bc_flat = goal_bc.reshape(-1, self.goal_dim)
        combined_input_flat = torch.cat([imagined_states_flat, stag_context_bc_flat, snn_pred_bc_flat], dim=-1)
        policy_logits_flat = self.action_head(combined_input_flat, goal_bc_flat).logits
        policy_logits = policy_logits_flat.reshape(horizon, batch_size, -1)
        bc_loss = torch.tensor(0.0, device=self.device)
        if teacher_plan is not None:
            bc_loss = F.mse_loss(policy_logits, teacher_plan.detach())

        entropy_loss = -entropy_coef * policy_entropies.mean()

        total_loss = policy_loss + critic_loss + lambda_bc * bc_loss + entropy_loss

        ac_optimizer = torch.optim.Adam(
            list(self.action_head.parameters()) + list(self.value_head.parameters()),
            lr=self.hyperparams.get('actor_critic_lr', 0.0003)
        )
        ac_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.action_head.parameters()) + list(self.value_head.parameters()), self.max_grad_norm)
        ac_optimizer.step()

        return {
            "ac_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            "bc_loss": bc_loss.item(),
            "entropy": -entropy_loss.item() / entropy_coef if entropy_coef > 0 else 0,
            "lambda_bc": lambda_bc,
            "horizon": horizon,
            "entropy_coef": entropy_coef
        }

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the MoCo queue with a batch of keys."""
        batch_size = keys.shape[0]
        ptr = int(self.contrastive_queue_ptr)

        # Ensure the keys fit in the queue
        if ptr + batch_size > self.contrastive_queue_size:
            # If they don't, wrap around
            remaining = self.contrastive_queue_size - ptr
            self.contrastive_queue[ptr:] = keys[:remaining]
            self.contrastive_queue[:batch_size - remaining] = keys[remaining:]
            ptr = batch_size - remaining
        else:
            self.contrastive_queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.contrastive_queue_size

        self.contrastive_queue_ptr[0] = ptr


    def get_actor_state_dict(self):
        """Returns the state dict for the actor components."""
        return {
            'action_head_state_dict': self.action_head.state_dict(),
            'stag_context_processor_state_dict': self.stag_context_processor.state_dict(),
        }

    def state_dict(self):
        """Returns the complete state of the agent for serialization."""
        # This is similar to save_state but returns the dict instead of saving it.
        return {
            'world_models_state_dicts': [wm.state_dict() for wm in self.world_models],
            'action_head_state_dict': self.action_head.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'stag_context_processor_state_dict': self.stag_context_processor.state_dict(),
            'steps_done': self.steps_done,
            'train_steps': self.train_steps,
            'novelty_stats': self.novelty_stats,
            'skill_manager_state': self.skill_manager.get_serializable_structure(),
            'kl_coeff': self.kl_coeff,
            'kl_error_integral': self.kl_error_integral,
            'kl_last_error': self.kl_last_error,
            'contrastive_queue': self.contrastive_queue,
            'contrastive_queue_ptr': self.contrastive_queue_ptr,
        }

    def load_state_dict(self, state_dict):
        """Loads the agent's state from a state dictionary."""
        if 'world_models_state_dicts' in state_dict:
            for i, sd in enumerate(state_dict['world_models_state_dicts']):
                if i < len(self.world_models): self.world_models[i].load_state_dict(sd)

        if 'action_head_state_dict' in state_dict:
            self.action_head.load_state_dict(state_dict['action_head_state_dict'])
        if 'value_head_state_dict' in state_dict:
            self.value_head.load_state_dict(state_dict['value_head_state_dict'])
        if 'stag_context_processor_state_dict' in state_dict:
            self.stag_context_processor.load_state_dict(state_dict.get('stag_context_processor_state_dict'))

        self.steps_done = state_dict.get('steps_done', 0)
        self.train_steps = state_dict.get('train_steps', 0)
        self.novelty_stats = state_dict.get('novelty_stats', self.novelty_stats)

        if 'skill_manager_state' in state_dict:
            skill_manager_kwargs = self.hyperparams.copy()
            skill_manager_kwargs['dimensions'] = self.hidden_dim
            self.skill_manager = SkillManager.from_serializable_structure(
                state_dict['skill_manager_state'], **skill_manager_kwargs
            )

    def load_actor_state_dict(self, state_dict):
        """Loads the state dict for the actor components."""
        if 'action_head_state_dict' in state_dict:
            self.action_head.load_state_dict(state_dict['action_head_state_dict'])
        if 'stag_context_processor_state_dict' in state_dict:
            self.stag_context_processor.load_state_dict(state_dict['stag_context_processor_state_dict'])
        logger.info("Actor weights updated from learner.")

    def get_graph_structure(self, skill_id=None):
        """
        Gets the graph structure for the active skill, or a specified skill.
        If no skill_id is provided, it defaults to the currently active one.
        """
        skill_to_get = skill_id or self.active_skill_id
        if not skill_to_get:
            return {} # Return an empty graph if no skill is active/specified
        return self.skill_manager.get_flattened_structure(skill_to_get)