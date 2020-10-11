import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def show_compare_histories(path_list, name_list):
  accuracies = []
  losses = []

  for path, name in zip(path_list, name_list):
    history_path = os.path.join(path, name)
    with open(history_path, "rb") as f:
      history = pickle.load(f)
      accuracies.append(history["accuracy"])
      losses.append(history["loss"])

  plt.figure()
  for index, accuracy in enumerate(accuracies):
    plt.plot(np.arange(len(accuracy)), accuracy)
  plt.legend(name_list)
  plt.savefig("../image/accuracy.png")

  plt.figure()
  for loss in losses:
    line = plt.plot(np.arange(len(loss)), loss)
  plt.legend(name_list)
  plt.savefig("../image/loss.png")

name_list = []
path_list = []

for i in np.linspace(0.1, 1, 10):
  i = np.round(i, decimals=1)
  name = "model_history_{}.pkl".format(i)
  path = "../model"

  name_list.append(name)
  path_list.append(path)

show_compare_histories(path_list, name_list)