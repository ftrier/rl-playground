import argparse
import torch
import imageio
import sys
from src.rl.conv_net.conv_net import ConvNet
from src.rl.utils.car_racing_env import CarRacingEnv


def get_args(args):
    parser = argparse.ArgumentParser(description='CarRacing')
    parser.add_argument('-w', '--weights', type=str, help='weights')
    parser.add_argument('-o', '--out', type=str,
                        help='output image', default='carracing.gif')
    return parser.parse_args(args)


def val(args):
    p = get_args(args)
    weights = torch.load(p.weights)

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
        print(action)
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
    imageio.mimsave(p.out, frames, duration=20)


if __name__ == "__main__":
    val(sys.argv[1:])
