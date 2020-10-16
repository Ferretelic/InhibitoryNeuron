import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def get_distribution_3d_random(input_shape):
  channels, input_height, input_width = input_shape
  y = np.linspace(-1, 1, input_height)
  x = np.linspace(-1, 1, input_width)

  mu_x, mu_y, variance_x, variance_y =  np.random.randn(4)
  variance_x = np.abs(variance_x)
  variance_y = np.abs(variance_y)
  x, y = np.meshgrid(x, y)

  position = np.empty(x.shape + (2,))

  position[:, :, 0] = x
  position[:, :, 1] = y

  distributions = []
  for i in range(channels):
    rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
    distribution = rv.pdf(position)
    max, min = np.max(distribution), np.min(distribution)
    distribution = distribution * (1.0 / (max - min))
    distributions.append(distribution)

  distributions = np.array(distributions)
  return distributions

def plot_distribution(distribution, name="distribution.png"):
  y = np.linspace(-1, 1, distribution.shape[1])
  x = np.linspace(-1, 1, distribution.shape[0])
  x, y = np.meshgrid(x, y)

  figure = plt.figure()
  ax = figure.gca(projection="3d")
  ax.plot_surface(x, y, distribution, cmap="viridis", linewidth=0)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  plt.savefig("../image/{}".format(name))

def plot_distribution_2d(distribution, name="distribution.png"):
  plt.matshow(distribution)
  plt.savefig("../image/{}".format(name))
