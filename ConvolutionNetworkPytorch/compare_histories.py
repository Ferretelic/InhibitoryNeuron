import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def show_compare_histories(path_list):
  accuracies = []
  losses = []

  for path in path_list:
    history_path = os.path.join(path, "model", "model_history.pkl")
    with open(history_path, "rb") as f:
      history = pickle.load(f)
      accuracies.append(history["accuracy"])
      losses.append(history["loss"])

  plt.figure()
  for index, accuracy in enumerate(accuracies):
    plt.plot(np.arange(len(accuracy)), accuracy)
  plt.legend(path_list)
  plt.savefig("./accuracy.png")

  plt.figure()
  for loss in losses:
    line = plt.plot(np.arange(len(loss)), loss, label=path_list[index])
  plt.legend(path_list)
  plt.savefig("./loss.png")



path_list = ["DistributionKReluGradientMuRandom", "NormalRelu"]
show_compare_histories(path_list)
