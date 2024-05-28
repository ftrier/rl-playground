from gymnasium.spaces import Box
from src.rl.networks.conv_net import ConvNet
from src.rl.agent.buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np


class ConvNetAgent:
    def __init__(self, env, buffer_size, lr):
        self.env = env
        self.device = torch.device('mps')
        self.policy_net = ConvNet(
            self.env.observation_space.shape[0], self.env.action_space.n, 0.001).to(self.device)
        self.target_net = ConvNet(
            self.env.observation_space.shape[0], self.env.action_space.n, 0.001).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    @torch.no_grad()
    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.env.action_space.n)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.policy_net(state)
        return q_values.argmax().item()

    def memorize(self, item):
        self.memory.append(item)

    def train(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states = zip(*batch)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool)
        non_final_next_states = torch.tensor(np.array(
            [s for s in next_states if s is not None]), dtype=torch.float32, device=self.device)
        states = torch.tensor(
            np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)

        state_action_values = self.policy_net(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_state_values = torch.zeros(batch_size, device=self.device)
            a = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = a.max(dim=1).values

        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device)
        expected_state_action_values = (next_state_values * gamma) + rewards
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_n_actions(self):
        # Check if Box or Discrete, then return the action space size (n_actions)
        if isinstance(self.env.action_space, Box):
            return self.env.action_space.shape[0]
        else:
            return self.env.action_space.n

    def get_max_actions(self):
        # Check if Box or Discrete, then return the max_action value for the actor network
        if isinstance(self.env.action_space, Box):
            return self.env.action_space.high
        else:
            raise NotImplementedError
