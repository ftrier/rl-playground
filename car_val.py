import argparse
import torch
import imageio
from src.rl.conv_net.conv_net import ConvNet
from src.rl.utils.car_racing_env import CarRacingEnv


def draw(weights):
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
    imageio.mimsave('carracing.gif', frames, duration=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CarRacing')
    parser.add_argument('-w', '--weights', type=str, help='weights')
    p = parser.parse_args()
    weights = torch.load(p.weights)
    draw(weights)
