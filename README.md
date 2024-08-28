# Reinforcement Learning Playground
Welcome to my reinforcement learning playground.

To use the repository with mlflow, you will need a `.env`file which includes:
```
URL_MLFLOW=https://mlflow.your-server.com
```

Currently, there is only one environment which you can execute:
```
python car.py -h
python car.py train -p params/car-train.yaml -u
python car.py val -w .../best.pth -o result.gif
```