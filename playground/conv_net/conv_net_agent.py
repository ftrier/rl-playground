import numpy as np
import torch
import torch.nn.functional as F
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from playground.utils import Device
from playground.conv_net.conv_net import ConvNet


class ConvNetAgent:
    def __init__(self, env, network, buffer_size, lr, weights=None):
        self.env = env
        self.policy_net = ConvNet(
            self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_net = ConvNet(
            self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if weights:
            self.policy_net.load_state_dict(weights)
            self.target_net.load_state_dict(weights)
            print("ConvNetAgent: Loaded weights")

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = TensorDictReplayBuffer(
            storage=LazyTensorStorage(buffer_size), collate_fn=lambda x: x)

    @torch.no_grad()
    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.env.action_space.n)
        return self.target_net.predict(state)

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

        batch = self.memory.sample(batch_size).to(Device)
        state_action_values = self.policy_net(batch['state']).gather(
            1, batch['action'].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_state_values = self.target_net(
                batch['next_state']).max(dim=1).values
        expected_state_action_values = next_state_values * \
            gamma + batch['reward']

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self, adaption_rate):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                adaption_rate + target_net_state_dict[key]*(1-adaption_rate)
        self.target_net.load_state_dict(target_net_state_dict)
