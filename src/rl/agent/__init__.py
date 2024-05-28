from dotenv import dotenv_values
import mlflow

config = dotenv_values(".env")
mlflow.set_tracking_uri(uri=config['URL_MLFLOW'])
print("MLflow is now using", mlflow.get_tracking_uri())