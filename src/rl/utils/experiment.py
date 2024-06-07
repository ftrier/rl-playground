import os
import time
import yaml
from dotmap import DotMap
from pathlib import Path
import mlflow
import torch
from dotenv import dotenv_values


def Load_params(param_file: str) -> DotMap:
    if not os.path.exists(param_file):
        return DotMap()
    return DotMap(yaml.safe_load(Path(param_file).read_text()))

def save_params(param_file: str, values: DotMap):
    with open(param_file, "w") as f:
        yaml.dump(values.toDict(), f)

class Experiment:
    def __init__(self, param_file: str, use_mlflow: bool = True) -> None:
        self.p = {}
        self.params = Load_params(param_file)

        if use_mlflow:
            config = dotenv_values(".env")
            mlflow.set_tracking_uri(uri=config['URL_MLFLOW'])
        print("Experiment: is using mlflow at:", mlflow.get_tracking_uri())
        mlflow.set_experiment(self.params.experiment_name)

        if not self.params.dir:
            x = self.params.experiment_name + \
                time.strftime("_%m-%d_%H-%M-%S%f")[:-2]
            self.p['base'] = os.path.join("experiments", x)
        else:
            self.p['base'] = self.params.dir
        print("Experiment directory:", self.p['base'])

        self.p |= {
            "params": os.path.join(self.p['base'], "params.yaml"),
            "running": os.path.join(self.p['base'], "running.yaml"),
            "checkpoints": os.path.join(self.p['base'], "checkpoints"),
            "best_weight": os.path.join(self.p['base'], "checkpoints", "best.pth"),
            "episode_weight": lambda e: os.path.join(self.p['base'], "checkpoints", str(e) + ".pth"),
        }
        os.makedirs(self.p['checkpoints'], exist_ok=True)

        self.running = Load_params(self.p["running"])

        if not self.running.last_ep:
            self.running.last_ep = -1

        self.checkpoint = None
        weights = self.p['episode_weight'](self.running.last_ep)
        if os.path.exists(weights):
            self.checkpoint = torch.load(weights)

        self.running.best_episode = None
        self.running.best_reward = -float('inf')

    def range(self):
        return range(int(self.running.last_ep + 1), self.params.n_episodes)

    def __enter__(self):
        mlflow.start_run(
            run_id=self.params.run_id if self.params.run_id else None)

        active_run = mlflow.active_run()
        if active_run:
            self.params.run_id = active_run.info.run_id

        save_params(self.p['params'], self.params)
        mlflow.log_params(self.params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()

    def save(self, model, episode, **kwargs):
        torch.save(model.state_dict(), self.p["episode_weight"](episode))

        if kwargs.get('reward', 0) > self.running.best_reward:
            self.running.best_reward = kwargs['reward']
            self.running.best_episode = episode
            torch.save(model.state_dict(), self.p['best_weight'])

        kwargs['best_reward'] = self.running.best_reward
        kwargs['best_episode'] = self.running.best_episode

        for k, v in kwargs.items():
            mlflow.log_metric(k, v, episode)

        self.running.last_ep = episode
        save_params(self.p['running'], self.running)
