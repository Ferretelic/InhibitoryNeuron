import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def get_distribution_3d(input_shape, variance, mu):
  channels, input_height, input_width = input_shape
  y = np.linspace(-1, 1, input_height)
  x = np.linspace(-1, 1, input_width)

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
    distribution = rv.pdf(position) * 80 - 5.0
    distributions.append(distribution)

  distributions = np.array(distributions)
  return distributions

def plot_distribution(distribution):
  y = np.linspace(-1, 1, distribution.shape[1])
  x = np.linspace(-1, 1, distribution.shape[0])
  x, y = np.meshgrid(x, y)

  figure = plt.figure()
  ax = figure.gca(projection="3d")
  ax.plot_surface(x, y, distribution, cmap="viridis", linewidth=0)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  plt.savefig("../image/distribution.png")

# distribution = get_distribution_3d((1, 100, 100), 2, (0, 0))
# plot_distribution(distribution[0])