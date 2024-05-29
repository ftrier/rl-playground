import os
import time
import yaml
from dotmap import DotMap
from pathlib import Path
import mlflow
import torch

class Experiment:
    def __init__(self, experiment_name="experiment") -> None:
        mlflow.set_experiment(experiment_name)
        self.timestamp = time.strftime("_%m-%d_%H-%M-%S%f")[:-2]
        self.experiments_dir = "experiments"

        if not os.path.exists(self.experiments_dir):
            os.makedirs(self.experiments_dir)

        self.experiment_name = experiment_name + self.timestamp
        self.experiment_dir = os.path.join(
            self.experiments_dir, self.experiment_name)

        for folder in ["plots", "checkpoints", "configs", "rendering", "episodes"]:
            os.makedirs(os.path.join(self.experiment_dir, folder))
        print(f"Created: {self.experiment_dir}")

    def __enter__(self):
        mlflow.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()

    def load_config(self, config_file):
        raw_config = yaml.safe_load(Path(config_file).read_text())
        self.config = DotMap(raw_config)
        mlflow.log_params(self.config)

    def save_configs(self, config_file="train.yaml"):
        experiment_configs = os.path.join(self.experiment_dir, "configs")
        with open(os.path.join(experiment_configs, config_file), "w") as f:
            yaml.dump(self.config.toDict(), f)

    def save_checkpoint(self, model, checkpoint_name):
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, checkpoint_name))