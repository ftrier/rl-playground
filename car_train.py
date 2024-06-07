import numpy as np
from tqdm import tqdm
import mlflow
import argparse
from src.rl.utils.experiment import Experiment
from src.rl.utils.car_racing_env import CarRacingEnv
from src.rl.conv_net.conv_net_agent import ConvNetAgent


def train(exp: Experiment):
    params = exp.params

    env = CarRacingEnv()
    agent = ConvNetAgent(env, params.network,
                         params.buffer_size, params.lr, weights=exp.checkpoint)

    epsilon = params.epsilon
    for episode in tqdm(exp.range()):
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
            agent.update_target_net(params.adaption_rate)

        exp.save(agent.policy_net, episode,
                 reward=episode_reward, epsilon=epsilon)
        epsilon = max(params.min_epsilon, epsilon * params.epsilon_decay)


def params():
    parser = argparse.ArgumentParser(description='CarRacing')
    parser.add_argument('-p', '--params', type=str,
                        default="configs/train.yaml", help='params file')
    parser.add_argument('-u', '--use_mlflow', action='store_true', default=False,
                        help='whether to use mlflow or not')
    return parser.parse_args()


if __name__ == "__main__":
    p = params()

    with Experiment(param_file=p.params, use_mlflow=p.use_mlflow) as exp:
        train(exp)
