import torch
import torch.nn as nn

class LatentPlanner(nn.Module):
    def __init__(self, world_model, action_dim, plan_horizon=12, num_samples=1000, top_k=100, iterations=10):
        super().__init__()
        self.world_model = world_model
        self.action_dim = action_dim
        self.plan_horizon = plan_horizon
        self.num_samples = num_samples
        self.top_k = top_k
        self.iterations = iterations

    def plan(self, h_t, z_t):
        """
        Plans the best action sequence using the Cross-Entropy Method (CEM).
        """
        device = h_t.device
        batch_size = h_t.size(0)

        # Initialize the belief over action sequences
        action_mean = torch.zeros(self.plan_horizon, batch_size, self.action_dim, device=device)
        action_std = torch.ones(self.plan_horizon, batch_size, self.action_dim, device=device)

        for _ in range(self.iterations):
            # 1. Sample action sequences from the current belief
            # Shape: [horizon, batch_size, num_samples, action_dim]
            actions = torch.normal(
                mean=action_mean.unsqueeze(2).expand(-1, -1, self.num_samples, -1),
                std=action_std.unsqueeze(2).expand(-1, -1, self.num_samples, -1)
            )
            # For discrete action spaces, we would sample from a categorical distribution
            # and convert to one-hot. Here we assume continuous actions for simplicity of CEM.
            # We can clamp the actions to a valid range if necessary.
            actions = torch.clamp(actions, -1, 1) # Assuming action space is [-1, 1]

            # 2. Rollout the sequences in the latent space
            # Initialize hidden and latent states for all samples
            h_sim = h_t.unsqueeze(1).expand(-1, self.num_samples, -1)
            z_sim = z_t.unsqueeze(1).expand(-1, self.num_samples, -1)

            returns = torch.zeros(batch_size, self.num_samples, device=device)

            with torch.no_grad():
                for t in range(self.plan_horizon):
                    action_t_samples = actions[t] # Shape: [batch_size, num_samples, action_dim]

                    # Reshape for world model batch processing
                    h_sim_flat = h_sim.view(-1, h_sim.size(-1))
                    z_sim_flat = z_sim.view(-1, z_sim.size(-1))
                    action_t_flat = action_t_samples.view(-1, action_t_samples.size(-1))

                    h_sim_flat, prior_mean, prior_std = self.world_model.rssm.transition_model(z_sim_flat, action_t_flat, h_sim_flat)
                    z_sim_flat = torch.distributions.Normal(prior_mean, prior_std).rsample()

                    # Predict rewards
                    reward_pred = self.world_model.reward_model(z_sim_flat, h_sim_flat)

                    # Reshape back
                    h_sim = h_sim_flat.view(batch_size, self.num_samples, -1)
                    z_sim = z_sim_flat.view(batch_size, self.num_samples, -1)
                    returns += reward_pred.view(batch_size, self.num_samples)

            # 3. Select the top-K action sequences based on their returns
            _, top_k_indices = torch.topk(returns, self.top_k, dim=1)

            # Extract the action sequences corresponding to the top-K returns
            # top_k_indices shape: [batch_size, top_k]
            # actions shape: [horizon, batch_size, num_samples, action_dim]
            # We need to gather the top actions for each step in the horizon
            top_actions = []
            for t in range(self.plan_horizon):
                # indices for gather need to match the dimensions of the input tensor
                top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, self.action_dim)
                action_t_samples = actions[t] # [batch_size, num_samples, action_dim]
                top_action_t = torch.gather(action_t_samples, 1, top_k_indices_expanded)
                top_actions.append(top_action_t)
            top_actions = torch.stack(top_actions) # [horizon, batch_size, top_k, action_dim]

            # 4. Refit the belief (mean and std) to the top-K action sequences
            action_mean = top_actions.mean(dim=2)
            action_std = top_actions.std(dim=2) + 1e-6 # Add epsilon for numerical stability

        # Return the first action of the mean sequence
        return action_mean[0]
