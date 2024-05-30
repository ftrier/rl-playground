import os
import time
import yaml
from dotmap import DotMap
from pathlib import Path
import mlflow
import torch


class Experiment:
    def __init__(self, param_file: str) -> None:
        self.running_params = DotMap()
        self.params = self._load_params(param_file)
        mlflow.set_experiment(self.params.experiment_name)

        if not self.params.dir:
            t = time.strftime("_%m-%d_%H-%M-%S%f")[:-2]
            timed_name = self.params.experiment_name + t
            self.params.dir = os.path.join("experiments", timed_name)
            os.makedirs(os.path.join(self.params.dir,
                        "checkpoints"), exist_ok=True)

        if os.path.exists(os.path.join(self.params.dir, "running_params.yaml")):
            self.running_params = self._load_params(
                os.path.join(self.params.dir, "running_params.yaml"))

        if not self.running_params.last_episode:
            self.running_params.last_episode = -1
        self.running_params.last_episode += 1

    def range(self):
        return range(int(self.running_params.last_episode), self.params.n_episodes)

    def __enter__(self):
        mlflow.start_run(
            run_id=self.params.run_id if self.params.run_id else None)
        self.params.run_id = mlflow.active_run().info.run_id
        self._save_params('params.yaml')
        mlflow.log_params(self.params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()

    def _load_params(self, param_file: str):
        return DotMap(yaml.safe_load(Path(param_file).read_text()))

    def _save_params(self, param_file):
        with open(os.path.join(self.params.dir, param_file), "w") as f:
            yaml.dump(self.params.toDict(), f)

    def _save_running_params(self, param_file):
        with open(os.path.join(self.params.dir, param_file), "w") as f:
            yaml.dump(self.running_params.toDict(), f)

    def save_checkpoint(self, model, episode, **kwargs):
        checkpoint_dir = os.path.join(self.params.dir, "checkpoints")
        self.running_params.last_episode = episode
        self._save_running_params('running_params.yaml')
        torch.save(model.state_dict(), os.path.join(
            checkpoint_dir, str(episode) + ".pth"))
        for k, v in kwargs.items():
            mlflow.log_metric(k, v, episode)
