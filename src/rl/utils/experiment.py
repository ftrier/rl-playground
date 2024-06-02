import os
import time
import yaml
from dotmap import DotMap
from pathlib import Path
import mlflow
import torch
from dotenv import dotenv_values


class Experiment:
    def __init__(self, param_file: str, use_mlflow: bool = True) -> None:
        self.running_params = DotMap()
        self.params = self._load_params(param_file)
        if use_mlflow:
            config = dotenv_values(".env")
            mlflow.set_tracking_uri(uri=config['URL_MLFLOW'])
        print("Experiment: is using mlflow at:", mlflow.get_tracking_uri())

        mlflow.set_experiment(self.params.experiment_name)

        if not self.params.dir:
            t = time.strftime("_%m-%d_%H-%M-%S%f")[:-2]
            timed_name = self.params.experiment_name + t
            self.params.dir = os.path.join("experiments", timed_name)
            os.makedirs(os.path.join(self.params.dir,
                        "checkpoints"), exist_ok=True)
            print("Experiment: created directory at", self.params.dir)

        # params cannot be overwritten in mlflow, therefore we are using running params as well
        if os.path.exists(os.path.join(self.params.dir, "running_params.yaml")):
            self.running_params = self._load_params(
                os.path.join(self.params.dir, "running_params.yaml"))

        if not self.running_params.last_episode:
            self.running_params.last_episode = -1
        ckp = os.path.join(self.params.dir, 'checkpoints', str(
            self.running_params.last_episode) + ".pth")
        self.ckp = torch.load(ckp) if os.path.exists(ckp) else None
        self.running_params.last_episode += 1

    def range(self):
        return range(int(self.running_params.last_episode), self.params.n_episodes)

    def __enter__(self):
        mlflow.start_run(
            run_id=self.params.run_id if self.params.run_id else None)

        active_run = mlflow.active_run()
        if active_run:
            self.params.run_id = active_run.info.run_id

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
