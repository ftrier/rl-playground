import numpy as np
import src.rl.networks.conv_net as conv_net
import torch
import torch.nn.functional as F
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict


class ConvNetAgent:
    def __init__(self, env, params):
        self.env = env
        self.device = torch.device('mps')
        self.policy_net = conv_net.registry[params.network](
            self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.target_net = conv_net.registry[params.network](
            self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=params.lr)
        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(
            params.buffer_size), collate_fn=lambda x: x)

    @torch.no_grad()
    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.env.action_space.n)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.policy_net(state)
        return q_values.argmax().item()

    def memorize(self, state, action, reward, next_state):
        data = TensorDict(
            {"state": torch.tensor(state[None, :], dtype=torch.float32),
             "action": [action],
             "reward": torch.tensor([reward], dtype=torch.float32),
             "next_state": torch.tensor(next_state[None, :], dtype=torch.float32)},
            batch_size=[1])
        self.memory.extend(data)

    def train(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size).to(self.device)
        state_action_values = self.policy_net(batch['state']).gather(
            1, batch['action'].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_state_values = self.target_net(
                batch['next_state']).max(dim=1).values
        expected_state_action_values = (
            next_state_values * gamma) + batch['reward']
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
