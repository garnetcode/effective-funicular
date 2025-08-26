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

logger = logging.getLogger(__name__)

class NumpyJSONEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

    def forward(self, activation_path_weights):
        # activation_path_weights is a list of weight vectors.
        # We concatenate them to form a single input vector.
        if not activation_path_weights:
            return torch.zeros(1, self.processor.out_features) # Return a zero vector if path is empty

        concatenated_weights = torch.cat(activation_path_weights, dim=0)
        # Ensure the tensor is flat before adding the batch dimension
        return self.processor(concatenated_weights.flatten().unsqueeze(0))


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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class ChimeraAgent:
    def __init__(self, agent_id, embedding_dim, max_action_dim, latent_dim=128, hidden_dim=512, cortex_configs=None, load_from_storage=True, hyperparams=None, history_config=None, **kwargs):
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
        self.train_steps = 0
        self.last_stag_node_id = None
        self.subgoal_reward = 0
        self.subgoal_duration = 0
        # h_t and z_t for the RSSM
        self.hidden_state = torch.zeros(1, self.hidden_dim, device=self.device)
        self.latent_state = torch.zeros(1, self.latent_dim, device=self.device)
        self.last_action = torch.tensor([0], device=self.device)
        self.high_level_plan = None
        self.current_subgoal = None
        self.current_goal = np.zeros(self.goal_dim)
        self.action_plan = None
        self.cortex_configs = cortex_configs or {}

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
        self.h_norm = nn.LayerNorm(self.hidden_dim).to(self.device)

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
        self.action_head = ActionHead(
            input_dim=self.hidden_dim + self.stag_context_dim,
            n_actions=self.max_action_dim,
            goal_dim=self.goal_dim,
            learning_rate=self.learning_rate
        ).to(self.device)
        # Critic (Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim + self.goal_dim, 400),
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

        if load_from_storage and self.history_manager._read_history():
            self.load_state()
        else:
            self.save_state(version_info={"message": "Initial state."})

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
        """Saves the agent's core models to a new version snapshot."""
        learnable_params = {
            'world_model_state_dicts': [wm.state_dict() for wm in self.world_models],
            'action_head_state_dict': self.action_head.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'stag_context_processor_state_dict': self.stag_context_processor.state_dict(),
            'skill_manager_state': self.skill_manager.get_serializable_structure()
        }
        self.history_manager.save_snapshot(learnable_params, version_info)

    def load_state(self, version='latest'):
        """Loads the agent's core models from a version snapshot."""
        learnable_params = self.history_manager.load_snapshot(version)
        if learnable_params:
            if 'world_model_state_dicts' in learnable_params:
                for i, state_dict in enumerate(learnable_params['world_model_state_dicts']):
                    if i < len(self.world_models):
                        self.world_models[i].load_state_dict(state_dict)
            elif 'world_model_state_dict' in learnable_params:
                # Handle old single-model checkpoints
                self.world_models[0].load_state_dict(learnable_params['world_model_state_dict'])

            self.action_head.load_state_dict(learnable_params['action_head_state_dict'])
            self.value_head.load_state_dict(learnable_params['value_head_state_dict'])
            self.stag_context_processor.load_state_dict(learnable_params.get('stag_context_processor_state_dict'))

            if 'skill_manager_state' in learnable_params:
                self.skill_manager = SkillManager.from_serializable_structure(
                    learnable_params['skill_manager_state'], **self.hyperparams
                )
        else:
            print("Warning: No state to load.")

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
            h_next, z_next, _ = self.world_models[0].rssm(obs_tensor, self.last_action, self.hidden_state, self.latent_state)

        # 4. Update the agent's internal state
        self.hidden_state = h_next
        self.latent_state = z_next

        # 5. Conditionally engage the STAG framework
        h_normalized = None
        activation_path = []
        novelty = 0
        # Only engage STAG if we are past the pre-training phase.
        if self.steps_done > self.world_model_pretrain_steps:
            h_normalized = self.h_norm(self.hidden_state).detach().cpu().numpy().flatten()
            # The vector is now normalized by LayerNorm and will be normalized again
            # by _safe_unit in the GNG engine for double safety.

            # Find the activation path and get the potential novelty signal (error)
            terminal_node, winner_id, activation_path = self.skill_manager.find_terminal_node_and_path(self.active_skill_id, h_normalized)
            if terminal_node and winner_id is not None:
                novelty = terminal_node['gng'].nodes[winner_id].get('error', 0)

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

    def select_action(self, actual_action_dim, activation_path):
        """
        Selects an action using the policy or a planner.
        """
        start_time = time.time()
        # 1. Construct the context vector C_t from the STAG's activation path
        path_weights = []
        if activation_path:
            active_stag = self.skill_manager._get_or_create_stag(self.active_skill_id)
            for step in activation_path:
                level_gng = active_stag.level_map[step['level_id']]
                node_weight = level_gng.nodes[step['winner_id']]['weight']
                path_weights.append(torch.from_numpy(node_weight).float().to(self.device))

        while len(path_weights) < self.max_stag_path_length:
            path_weights.append(torch.zeros(self.hidden_dim, device=self.device))

        with torch.no_grad():
            stag_context_vector = self.stag_context_processor(path_weights)
        if not self.use_stag_in_ac_loss:
            stag_context_vector = torch.zeros_like(stag_context_vector)

        combined_input = torch.cat((self.hidden_state, stag_context_vector), dim=1)

        # 3. Select action
        epsilon = self._get_epsilon()
        self.steps_done += 1

        # Decide whether to use the planner or the learned policy
        use_planner_this_step = self.use_planner and (self.action_plan is None or len(self.action_plan) == 0) and (self.steps_done % self.low_level_plan_frequency == 0)

        if use_planner_this_step:
            # Hierarchical planning logic
            if self.current_subgoal is None or self.train_steps % self.high_level_replan_frequency == 0:
                stag_graph = self.skill_manager.get_flattened_structure(self.active_skill_id)
                if self.last_stag_node_id and len(stag_graph['nodes']) > 1:
                    possible_goals = [nid for nid in stag_graph['nodes'] if nid != self.last_stag_node_id]
                    if possible_goals:
                        goal_node_id = random.choice(possible_goals)
                        self.high_level_plan = self.graph_planner.plan(stag_graph, self.skill_manager.option_models.get(self.active_skill_id, {}), self.last_stag_node_id, goal_node_id)

                if self.high_level_plan:
                    self.high_level_plan.pop(0)
                    if self.high_level_plan:
                        self.current_subgoal = self.high_level_plan.pop(0)

            subgoal_weight = None
            if self.current_subgoal:
                stag = self.skill_manager._get_or_create_stag(self.active_skill_id)
                terminal_gng = stag.level_map[max(stag.level_map.keys())]
                if self.current_subgoal in terminal_gng.nodes:
                    subgoal_weight = torch.from_numpy(terminal_gng.nodes[self.current_subgoal]['weight']).float().to(self.device)

            with torch.no_grad():
                # The planner returns the full sequence of actions
                self.action_plan = self.planner.plan(self.hidden_state, self.latent_state, subgoal_weight, self.current_goal)

            # Take the first action from the new plan
            action_continuous = self.action_plan[0]
            action_tensor = torch.argmax(action_continuous[:actual_action_dim], dim=-1)
            self.action_plan = self.action_plan[1:] # Consume the first action
            decision_maker = "planner"
            log_prob = torch.tensor(0.0)

        elif self.action_plan is not None and len(self.action_plan) > 0:
            # Execute the existing plan
            action_continuous = self.action_plan[0]
            action_tensor = torch.argmax(action_continuous[:actual_action_dim], dim=-1)
            self.action_plan = self.action_plan[1:] # Consume the action
            decision_maker = "plan_follower"
            log_prob = torch.tensor(0.0)

        else: # Use the policy
            with torch.no_grad():
                if self.current_goal is None:
                    logger.warning("self.current_goal was None in select_action. Defaulting to a zero vector.")
                    self.current_goal = np.zeros(self.goal_dim)
                goal_tensor = torch.from_numpy(self.current_goal).float().to(self.device)
                goal_tensor_expanded = goal_tensor.unsqueeze(0).expand(combined_input.size(0), -1)
                action_dist = self.action_head(combined_input, goal_tensor_expanded)
                mask = torch.full(action_dist.logits.shape, -float('inf'), device=self.device)
                mask[0, :actual_action_dim] = 0
                masked_logits = action_dist.logits + mask
                action_dist = Categorical(logits=masked_logits)

            if np.random.rand() < epsilon:
                action_tensor = torch.tensor([np.random.randint(0, actual_action_dim)], device=self.device)
                decision_maker = "random"
            else:
                action_tensor = action_dist.sample()
                decision_maker = "policy"

            log_prob = action_dist.log_prob(action_tensor)

        action = action_tensor.item()
        # Ensure last_action is always a 1D tensor of shape (1,) for consistency.
        self.last_action = action_tensor.reshape(1)
        print(f"Total select_action time: {time.time() - start_time:.4f}s, decision_maker: {decision_maker}")
        return action, log_prob, stag_context_vector, decision_maker, epsilon

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

    def record_experience(self, *args):
        """Pushes an experience to the replay buffer and updates subgoal counters."""
        experience = list(args)
        experience.append(self.current_goal)
        self.replay_buffer.push(*experience)
        # Assumes the reward is the 6th argument in *args
        self.subgoal_reward += args[6]
        self.subgoal_duration += 1

    def train(self, cortex_id="vector_input"):
        """
        The main training loop that orchestrates the "sleep" phase of the agent,
        which involves training the world model and then training the policy in imagination.
        """
        self.train_steps += 1
        # Part 1: Train the World Model on real, recently collected data.
        world_model_stats, indices, priorities = self.train_world_model(cortex_id)
        if indices is not None:
            self.update_priorities(indices, priorities)

        policy_stats = {}

        # Part 2: Train the Actor-Critic policy in imagined trajectories, but less frequently.
        policy_train_frequency = self.hyperparams.get('policy_train_frequency', 1)
        if self.train_steps % policy_train_frequency == 0:
            policy_stats = self.train_policy_in_imagination()

        # Part 3: Prune the STAG graph periodically for the active skill.
        if self.train_steps > 0 and self.train_steps % self.gng_pruning_frequency == 0:
            if self.active_skill_id:
                self.skill_manager.prune_graph(self.active_skill_id, self.gng_min_utility_threshold)

        # Combine stats for logging
        combined_stats = {**world_model_stats, **policy_stats}
        return combined_stats

    def update_priorities(self, indices, priorities):
        """Updates the priorities of experiences in the replay buffer."""
        self.replay_buffer.update_priorities(indices, priorities)

    def train_world_model(self, cortex_id="vector_input"):
        """
        Trains the World Model ensemble on batches of sequences, including a contrastive loss.
        """
        batch_size = self.hyperparams.get('batch_size', 32)
        if len(self.replay_buffer) < self.replay_buffer.sequence_length:
            return {}, None, None

        contrastive_loss_fn = ContrastiveLoss()

        # For PER, we sample a single batch and use it for all models
        batch, indices, weights = self.replay_buffer.sample(batch_size)
        weights = torch.from_numpy(weights).float().to(self.device)

        total_wm_loss = 0
        sequence_priorities = torch.zeros(batch_size, device=self.device)

        for i, (world_model, wm_optimizer) in enumerate(zip(self.world_models, self.world_model_optimizers)):
            # --- Prepare tensors ---
            obs_sequence = torch.from_numpy(batch['obs']).float().to(self.device)
            action_sequence = torch.from_numpy(batch['action']).float().to(self.device)
            reward_sequence = torch.from_numpy(batch['reward']).float().to(self.device)
            goal_sequence = torch.from_numpy(batch['goal']).float().to(self.device)

            # Process observations
            if obs_sequence.dim() == 5:
                batch_size, seq_len, h_dim, w_dim, c_dim = obs_sequence.shape
                obs_sequence_processed = self.cortexes[cortex_id](obs_sequence.view(-1, h_dim, w_dim, c_dim))
            else:
                batch_size, seq_len, obs_dim = obs_sequence.shape
                obs_sequence_processed = self.cortexes[cortex_id](obs_sequence.view(-1, obs_dim))
            obs_sequence_processed = obs_sequence_processed.view(batch_size, seq_len, -1)

            # --- Sequence-based Forward Pass and Loss Calculation ---
            h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
            z_t = torch.zeros(batch_size, self.latent_dim, device=self.device)

            total_recon_loss, total_reward_loss, total_kl_loss = 0, 0, 0
            hidden_states = []

            for t in range(self.replay_buffer.sequence_length):
                obs_t = obs_sequence_processed[:, t]
                action_t = action_sequence[:, t]
                reward_t = reward_sequence[:, t]

                h_t, z_t, kl_loss = world_model.rssm(obs_t, action_t, h_t, z_t)
                hidden_states.append(h_t)

                obs_recon = world_model.obs_decoder(z_t, h_t)
                reward_pred = world_model.reward_model(torch.cat([z_t, h_t, goal_sequence[:, t]], dim=-1))

                recon_loss = torch.mean((obs_recon - obs_t)**2, dim=list(range(1, obs_recon.dim())))
                reward_loss = torch.mean((reward_pred - reward_t.unsqueeze(-1))**2, dim=-1)

                free_bits = self.hyperparams.get('free_bits', 1.0)
                kl_loss = torch.clamp(kl_loss, min=free_bits)

                total_recon_loss += recon_loss
                total_reward_loss += reward_loss
                total_kl_loss += kl_loss

            hidden_states = torch.stack(hidden_states, dim=1) # (batch, seq_len, hidden_dim)

            # --- Contrastive Loss Calculation ---
            # Positive pairs are adjacent states in the sequence
            anchor = hidden_states[:, :-1].reshape(-1, self.hidden_dim)
            positive = hidden_states[:, 1:].reshape(-1, self.hidden_dim)

            # Negative pairs are all other states in the batch
            # This is a simplification. A better approach would be to use a memory bank of negatives.
            # For now, we'll just use other states in the same batch.
            negatives = positive.unsqueeze(0).expand(positive.size(0), -1, -1)
            contrastive_loss = contrastive_loss_fn(anchor, positive, negatives)

            # --- Total Loss ---
            world_model_loss = (
                (total_recon_loss + total_reward_loss + total_kl_loss) / self.replay_buffer.sequence_length
            )

            # The priority is based on the reconstruction error
            sequence_priorities += world_model_loss.detach()

            # Weight the loss by the importance sampling weights
            total_loss = (weights * world_model_loss).mean() + self.contrastive_loss_weight * contrastive_loss
            total_wm_loss += total_loss.item()

            # --- Backpropagation ---
            wm_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), self.max_grad_norm)
            wm_optimizer.step()

        # Average the priorities over the ensemble
        final_priorities = (sequence_priorities / self.num_ensemble_models).cpu().numpy()

        # Learn reward weights
        with torch.no_grad():
            # Use the last hidden state of the sequence to represent the state
            last_h = hidden_states[:, -1]
            state_features = torch.matmul(last_h, self.sf_projection_matrix)

            # Predict reward using current weights
            predicted_r = torch.einsum('bd,d->b', [state_features, self.reward_weights])

            # Get the actual reward for the last step of the sequence
            actual_r = reward_sequence[:, -1]

            # LMS update rule
            error = actual_r - predicted_r
            self.reward_weights += self.sf_learning_rate * torch.mean(error.unsqueeze(-1) * state_features, dim=0)

        return {"wm_loss": total_wm_loss / self.num_ensemble_models, "contrastive_loss": contrastive_loss.item()}, indices, final_priorities

    def train_policy_in_imagination(self):
        """
        Trains the Actor (ActionHead) and Critic (ValueHead) in imagination.
        This is the second part of the "Sleep" phase.
        """
        batch_size = self.hyperparams.get('batch_size', 32)
        if len(self.replay_buffer) < self.replay_buffer.sequence_length:
            return {}

        # --- Calculate scheduled parameters ---
        # Imagination Horizon
        progress = min(1.0, self.train_steps / self.imagination_horizon_schedule_steps)
        horizon = int(self.imagination_horizon_start + progress * (self.imagination_horizon_end - self.imagination_horizon_start))

        # Entropy Coefficient
        progress = min(1.0, self.train_steps / self.entropy_coef_schedule_steps)
        entropy_coef = self.entropy_coef_start - progress * (self.entropy_coef_start - self.entropy_coef_end)


        # --- Sample starting states from real data ---
        batch, _, _ = self.replay_buffer.sample(batch_size)
        h_start = torch.from_numpy(batch['h'][:, 0]).float().to(self.device)
        z_start = torch.from_numpy(batch['z'][:, 0]).float().to(self.device)
        goal_sequence = torch.from_numpy(batch['goal'][:, 0]).float().to(self.device)

        # --- Imagine Trajectories ---
        h_t, z_t = h_start, z_start
        imagined_h = [h_t]
        imagined_z = [z_t]
        imagined_actions = []
        imagined_stag_contexts = []

        # --- Imagine Trajectories with Dynamic STAG Context ---
        for _ in range(horizon):
            # 1. Generate STAG context dynamically for the current imagined state h_t
            # This part needs to have gradient tracking for the StagContextProcessor
            stag_contexts = []
            for i in range(h_t.size(0)): # Process each state in the batch
                h_numpy = h_t[i].detach().numpy().flatten()
                norm = np.linalg.norm(h_numpy)
                h_normalized = h_numpy / norm if norm > 0 else h_numpy

                _, _, activation_path = self.skill_manager.find_terminal_node_and_path(self.active_skill_id, h_normalized)

                path_weights = []
                if activation_path:
                    active_stag = self.skill_manager._get_or_create_stag(self.active_skill_id)
                    for step in activation_path:
                        level_gng = active_stag.level_map[step['level_id']]
                        node_weight = level_gng.nodes[step['winner_id']]['weight']
                        path_weights.append(torch.from_numpy(node_weight).float().to(self.device))

                while len(path_weights) < self.max_stag_path_length:
                    path_weights.append(torch.zeros(self.hidden_dim, device=self.device))

                stag_contexts.append(self.stag_context_processor(path_weights))

            stag_context_batch = torch.cat(stag_contexts, dim=0)

            # If STAG is disabled for training OR we are in the pre-training phase,
            # use a zero vector for the context.
            if not self.use_stag_in_ac_loss or self.steps_done <= self.world_model_pretrain_steps:
                stag_context_batch = torch.zeros_like(stag_context_batch)

            # 2. Actor selects action based on h_t and the dynamically generated C_t
            action_input = torch.cat([h_t, stag_context_batch], dim=-1)
            action_dist = self.action_head(action_input, goal_sequence)
            action = action_dist.sample()

            # 3. World model predicts next state (do this without tracking gradients)
            with torch.no_grad():
                h_t, prior_mean, prior_std = self.world_models[0].rssm.transition_model(z_t, action, h_t)
                z_t = Normal(prior_mean, prior_std).rsample()

            imagined_h.append(h_t)
            imagined_z.append(z_t)
            imagined_actions.append(action)
            imagined_stag_contexts.append(stag_context_batch)

        imagined_h = torch.stack(imagined_h) # Shape: [horizon+1, batch, hidden_dim]
        imagined_z = torch.stack(imagined_z) # Shape: [horizon+1, batch, latent_dim]
        imagined_actions = torch.stack(imagined_actions) # Shape: [horizon, batch]
        imagined_stag_contexts = torch.stack(imagined_stag_contexts) # Shape: [horizon, batch, context_dim]


        # --- Predict Rewards and Values for Imagined Trajectory ---
        # For simplicity, we assume the goal is constant for the whole trajectory
        goal_sequence = torch.from_numpy(batch['goal'][:, 0]).float().to(self.device)
        imagined_rewards = self.world_models[0].reward_model(torch.cat([imagined_z, imagined_h, goal_sequence.unsqueeze(0).expand(horizon + 1, -1, -1)], dim=-1)).squeeze(-1)

        # Expand goal for value prediction
        goal_sequence_expanded = goal_sequence.unsqueeze(0).expand(horizon + 1, -1, -1)
        imagined_values = self.value_head(torch.cat([imagined_h, imagined_z, goal_sequence_expanded], dim=-1)).squeeze(-1)

        # --- Calculate Value Targets (Lambda-Return) ---
        lambda_ = self.hyperparams.get('lambda', 0.95)
        returns = torch.zeros_like(imagined_values[-1])
        lambda_returns = []
        for t in reversed(range(horizon)):
            # V_target = r_t + gamma * ( (1-lambda) * V(s_{t+1}) + lambda * V_target_{t+1} )
            returns = imagined_rewards[t] + self.gamma * ((1 - lambda_) * imagined_values[t+1].detach() + lambda_ * returns)
            lambda_returns.append(returns)
        lambda_returns = torch.stack(list(reversed(lambda_returns)))

        # --- Actor-Critic Loss Calculation (ℒ_AC) ---
        # Actor Loss
        advantage = (lambda_returns - imagined_values[:-1]).detach()
        # Normalize advantages to stabilize training
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        action_input = torch.cat([imagined_h[:-1], imagined_stag_contexts], dim=-1)
        log_probs = self.action_head(action_input, goal_sequence.unsqueeze(0).expand(horizon, -1, -1)).log_prob(imagined_actions)
        policy_loss = -(log_probs * advantage).mean()

        # Critic Loss
        critic_loss = torch.nn.functional.mse_loss(imagined_values[:-1], lambda_returns.detach())

        # Entropy Bonus
        entropy = self.action_head(action_input, goal_sequence.unsqueeze(0).expand(horizon, -1, -1)).entropy().mean()

        # --- Total AC Loss and Backpropagation ---
        w_policy = self.hyperparams.get('w_policy', 1.0)
        w_critic = self.hyperparams.get('w_critic', 0.5)

        ac_loss = w_policy * policy_loss + w_critic * critic_loss - entropy_coef * entropy

        ac_optimizer = torch.optim.Adam(
            list(self.action_head.parameters()) + list(self.value_head.parameters()),
            lr=self.hyperparams.get('actor_critic_lr', 0.0003)
        )
        ac_optimizer.zero_grad()
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.action_head.parameters()) + list(self.value_head.parameters()), self.max_grad_norm)
        ac_optimizer.step()

        return {
            "ac_loss": ac_loss.item(),
            "policy_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "horizon": horizon,
            "entropy_coef": entropy_coef
        }

    def get_graph_structure(self, skill_id=None):
        """
        Gets the graph structure for the active skill, or a specified skill.
        If no skill_id is provided, it defaults to the currently active one.
        """
        skill_to_get = skill_id or self.active_skill_id
        if not skill_to_get:
            return {} # Return an empty graph if no skill is active/specified
        return self.skill_manager.get_flattened_structure(skill_to_get)