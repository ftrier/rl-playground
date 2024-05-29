import numpy as np
from tqdm import tqdm
import mlflow
import argparse
from src.rl.agent.conv_net_agent import ConvNetAgent
from src.rl.utils.car_racing_env import CarRacingEnv
from src.rl.utils.experiment import Experiment


def train(experiment: Experiment):
    params = experiment.config

    env = CarRacingEnv()
    agent = ConvNetAgent(env, params.buffer_size, params.lr)
    epsilon = params.epsilon
    for episode in tqdm(range(params.n_episodes)):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            if not done and not truncated:
                agent.memorize(state, action, reward, next_state)
            agent.train(params.batch_size, params.gamma)
            state = next_state
            episode_reward += reward
            if done or truncated:
                break

        mlflow.log_metric("reward", episode_reward, episode)
        mlflow.log_metric("epsilon", epsilon, episode)
        experiment.save_checkpoint(
            agent.policy_net, f"checkpoint_{episode}.pth")
        agent.update_target_net()
        epsilon = max(params.min_epsilon,
                      epsilon * params.epsilon_decay)
        if episode % 10 == 0:
            print(f"Ep: {episode}, Reward: {episode_reward}, Epsilon: {epsilon}")


def params():
    parser = argparse.ArgumentParser(description='CarRacing')
    parser.add_argument('-c', '--config', type=str,
                        default="configs/train.yaml", help='config file')
    return parser.parse_args()


if __name__ == "__main__":
    cmd_params = params()
    with Experiment("CarRacing") as exp:
        exp.load_config(cmd_params.config)
        exp.save_configs()
        train(exp)
