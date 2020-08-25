import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_history(path):
  with open(path, "rb") as f:
    history = pickle.load(f)

  accuracy = history["accuracy"]
  loss = history["loss"]

  plt.figure()
  plt.plot(np.arange(len(accuracy)), accuracy)
  plt.savefig(os.path.join(path, "accuracy.png"))

  plt.figure()
  plt.plot(np.arange(len(loss)), loss)
  plt.savefig(os.path.join(path, "loss.png"))
