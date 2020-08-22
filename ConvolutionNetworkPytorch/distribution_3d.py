import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def get_distribution_3d(shape, variance, mu, channels):
  x = np.linspace(-1, 1, shape[0])
  y = np.linspace(-1, 1, shape[1])

  mu_x = mu[0]
  mu_y = mu[1]
  variance_x = variance
  variance_y = variance

  x, y = np.meshgrid(x, y)

  position = np.empty(x.shape + (2,))

  position[:, :, 0] = x
  position[:, :, 1] = y

  distributions = []
  for i in range(channels):
    rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
    distribution = rv.pdf(position)
    distributions.append(distribution)

  distributions = np.array(distributions)

  return distributions

def plot_distribution(distribution):
  figure = plt.figure()
  ax = figure.gca(projection="3d")
  ax.plot_surface(x, y, distribution, cmap="viridis", linewidth=0)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  plt.savefig("./data/distribution.png")
