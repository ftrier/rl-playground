import argparse
import torch
import imageio
from tqdm import tqdm
from playground.conv_net.conv_net import ConvNet
from playground.utils.experiment import Experiment
from playground.utils.car_racing_env import CarRacingEnv
from playground.conv_net.conv_net_agent import ConvNetAgent


def main():
    parser = argparse.ArgumentParser(description='CarRacing')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)

    train_parser.add_argument('-p', '--params', type=str,
                              default="params/car-train.yaml", help='params file')
    train_parser.add_argument('-u', '--use_mlflow', action='store_true', default=False,
                              help='whether to use mlflow or not')

    val_parser = subparsers.add_parser("val")
    val_parser.set_defaults(func=val)
    val_parser.add_argument('-w', '--weights', type=str, help='weights')
    val_parser.add_argument('-o', '--out', type=str,
                            help='output image', default='carracing.gif')
    args = parser.parse_args()
    args.func(args)


def train(args):
    with Experiment(param_file=args.params, use_mlflow=args.use_mlflow) as exp:
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


def val(args):
    weights = torch.load(args.weights)

    env = CarRacingEnv(render_mode="rgb_array")
    net = ConvNet(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(weights)

    env.reset()
    state = env.reset()
    done, truncated = False, False
    frames = []
    while not done and not truncated:
        frame = env.render()
        frames.append(frame)
        action = net.predict(state)
        next_state, _, done, truncated, _ = env.step(action)
        state = next_state
    imageio.mimsave(args.out, frames, duration=20)


if __name__ == "__main__":
    main()
