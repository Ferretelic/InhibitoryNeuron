import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

histories = []

for history_type in ["accuracy", "loss"]:
  plt.figure()
  for directory in os.listdir("../"):
    path = os.path.join("../", directory, "model", "model_history.pkl")
    if os.path.exists(path) == True:
      with open(path, "rb") as f:
        history = pickle.load(f)
        plt.plot(np.arange(len(history[history_type])), history[history_type], label=directory)

  plt.xlabel("epochs")
  plt.ylabel(history_type)
  plt.legend()
  plt.savefig("./{}.png".format(history_type))
