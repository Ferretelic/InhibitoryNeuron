import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

histories = []
models = [
  "NormalRelu",
  "DistributionKReluGradientMuRandom",
  "DistributionKReluGradientMuRandomIncreased",
  "DistributionKReluGradientMuVarianceRandom",
  "DistributionKReluGradientMuVarianceRandomAbsolute"
]

for history_type in ["accuracy", "loss"]:
  plt.figure()
  for index, directory in enumerate(models):
    path = os.path.join("../", directory, "model", "model_history.pkl")
    if os.path.exists(path) == True:
      with open(path, "rb") as f:
        history = pickle.load(f)
        history[history_type] = history[history_type][:50]
        plt.plot(np.arange(len(history[history_type])), history[history_type], label="Model{}".format(index + 1))

  plt.xlabel("epochs")
  plt.ylabel(history_type)
  plt.legend()
  plt.savefig("./{}.png".format(history_type))
