import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

histories = []

models = [
  "NormalRelu",
  "NormalReluReduced",
  "DistributionKReluGradientMuRandom","DistributionKReluGradientMuRandomReduced",
  "DistributionKReluGradientMuRandomIncreased",
  "DistributionKReluGradientMuVarianceRandom",
  "DistributionKReluGradientMuVarianceRandomReduced",
  "DistributionKReluGradientMuVarianceRandomAbsolute",
  "DistributionKReluGradientMuVarianceRandomAbsoluteReduced"
]

for history_type in ["accuracy", "loss"]:
  plt.figure()
  for index, directory in enumerate(models):
    path = os.path.join("../", directory, "model", "model_history.pkl")
    if os.path.exists(path) == True:
      with open(path, "rb") as f:
        history = pickle.load(f)
        plt.plot(np.arange(len(history[history_type])), history[history_type], label="Model{}".format(index + 1))

  plt.xlabel("epochs")
  plt.ylabel(history_type)
  plt.legend()
  plt.savefig("./{}.png".format(history_type))
