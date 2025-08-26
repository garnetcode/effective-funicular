import torch
import torch.nn as nn

class LatentPlanner(nn.Module):
    def __init__(self, world_models, action_dim, plan_horizon=12, num_samples=1000, top_k=100, iterations=10, uncertainty_penalty_weight=0.1):
        super().__init__()
        self.world_models = world_models
        self.action_dim = action_dim
        self.plan_horizon = plan_horizon
        self.num_samples = num_samples
        self.top_k = top_k
        self.iterations = iterations
        self.uncertainty_penalty_weight = uncertainty_penalty_weight

    def plan(self, h_t, z_t, goal=None):
        """
        Plans the best action sequence using the Cross-Entropy Method (CEM) with an ensemble of world models.
        """
        device = h_t.device
        batch_size = h_t.size(0)
        num_models = len(self.world_models)

        action_mean = torch.zeros(self.plan_horizon, batch_size, self.action_dim, device=device)
        action_std = torch.ones(self.plan_horizon, batch_size, self.action_dim, device=device)

        for _ in range(self.iterations):
            actions = torch.normal(
                mean=action_mean.unsqueeze(2).expand(-1, -1, self.num_samples, -1),
                std=action_std.unsqueeze(2).expand(-1, -1, self.num_samples, -1)
            )

            all_model_returns = torch.zeros(num_models, batch_size, self.num_samples, device=device)

            for i, world_model in enumerate(self.world_models):
                h_sim = h_t.unsqueeze(1).expand(-1, self.num_samples, -1)
                z_sim = z_t.unsqueeze(1).expand(-1, self.num_samples, -1)

                with torch.no_grad():
                    for t in range(self.plan_horizon):
                        action_t_samples = actions[t]

                        h_sim_flat = h_sim.view(-1, h_sim.size(-1))
                        z_sim_flat = z_sim.view(-1, z_sim.size(-1))

                        # Convert continuous actions to discrete for the transition model
                        action_t_discrete = torch.argmax(action_t_samples, dim=-1)
                        action_t_flat = action_t_discrete.view(-1, 1)

                        h_sim_flat, prior_mean, prior_std = world_model.rssm.transition_model(z_sim_flat, action_t_flat, h_sim_flat)
                        z_sim_flat = torch.distributions.Normal(prior_mean, prior_std).rsample()

                        if goal is not None:
                            # AGENT_FIX: The goal needs to be expanded to match the simulation batch size
                            num_sims = z_sim_flat.shape[0]
                            goal_expanded = goal.unsqueeze(0).expand(num_sims, -1)
                            # AGENT_FIX: Call the reward model with the correct concatenated input
                            reward_input = torch.cat([z_sim_flat, h_sim_flat, goal_expanded], dim=1)
                            reward_pred = world_model.reward_model(reward_input).squeeze(-1)
                        else:
                            # Fallback if no goal is provided (though the agent should always have one)
                            # This part is tricky as the model requires a goal. We can send a zero goal.
                            goal_dim = world_model.reward_model[0].in_features - z_sim_flat.shape[1] - h_sim_flat.shape[1]
                            zero_goal = torch.zeros(z_sim_flat.shape[0], goal_dim, device=device)
                            reward_input = torch.cat([z_sim_flat, h_sim_flat, zero_goal], dim=1)
                            reward_pred = world_model.reward_model(reward_input).squeeze(-1)

                        h_sim = h_sim_flat.view(batch_size, self.num_samples, -1)
                        z_sim = z_sim_flat.view(batch_size, self.num_samples, -1)
                        all_model_returns[i] += reward_pred.view(batch_size, self.num_samples)

            # Calculate mean reward and uncertainty penalty
            mean_returns = all_model_returns.mean(dim=0)
            reward_variance = all_model_returns.var(dim=0)

            # Objective is to maximize mean reward and minimize variance
            objective = mean_returns - self.uncertainty_penalty_weight * reward_variance

            _, top_k_indices = torch.topk(objective, self.top_k, dim=1)

            top_actions = []
            for t in range(self.plan_horizon):
                top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, self.action_dim)
                action_t_samples = actions[t]
                top_action_t = torch.gather(action_t_samples, 1, top_k_indices_expanded)
                top_actions.append(top_action_t)
            top_actions = torch.stack(top_actions)

            action_mean = top_actions.mean(dim=2)
            action_std = top_actions.std(dim=2) + 1e-6

        return action_mean[0]
