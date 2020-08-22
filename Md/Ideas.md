# Inhibitory Neuron Ideas
## Activations
* ### Sigmoid
#### Mathematics
``` math
f(x) = \frac{1}{1 + e^{-x}} \\
```

#### Code
``` python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

#### Mathematics
``` math
f'(x) =  f(x) (1 - f(x))\\
```

#### Code
``` python
def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

![Sigmoid Function](./Images/sigmoid.png)

* ### Reverse Sigmoid
#### Mathematics
``` math
f(x) = -\frac{1}{1 + e^{-x}} \\
```

#### Code
``` python
def sigmoid(x):
    return - (1 / (1 + np.exp(-x)))
```

#### Mathematics
``` math
f'(x) =  f(x) (1 - f(x))\\
```

#### Code
``` python
def dsigmoid(x):
    return sigmoid(x) * (sigmoid(x) - 1)
```

![Sigmoid Function](./Images/reverse_sigmoid.png)

## Distribution
* ### Normal Distribution 2d
#### Mathematics
``` math
f(x) = \frac{1}{\sqrt{2{\pi}}\sigma}
        e^{-\frac{1}{2}(\frac{x - \mu}{\sigma})}
```

#### Code
``` python
def normal_distribution_2d(mu, variance):
  mu = mu
  variance = variance
  sigma = math.sqrt(variance)
  x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
  plt.plot(x, stats.norm.pdf(x, mu, sigma))
  plt.show()
```

![normal distribution 2d](./Images/normal_distribution_2d.png)

* ### Normal Distribution 3d
#### Code
``` python
def normal_distribution_3d(mu, variance):
  mu_x = mu
  variance_x = variance

  mu_y = mu
  variance_y = variance

  x = np.linspace(-10,10,500)
  y = np.linspace(-10,10,500)

  X, Y = np.meshgrid(x,y)
  pos = np.empty(X.shape + (2,))
  pos[:, :, 0] = X
  pos[:, :, 1] = Y

  rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

  fig = plt.figure()
  ax = fig.gca(projection="3d")
  ax.plot_surface(X, Y, rv.pdf(pos), cmap="viridis",linewidth=0)
  ax.set_xlabel('X axis')
  ax.set_ylabel('Y axis')
  ax.set_zlabel('Z axis')
  plt.show()
```

#### Image
![normal distribution 2d](./Images/normal_distribution_3d.png)

##  Layer
* ### Convolution Normal Distribution 3d Activations
This layer has two types of activations, sigmoid and reverse sigmoid.
To reproduction inhibitory newuron, I set reverse sigmoid as activation based on normal distibution 3d in convolution layers.

#### 1: Get probably of inhibitory neuron.
#### 2: Set activations each neuron.

* ### Convolution Normal Distibution 3d Inclusion Activations
This layer has two types of activations, sigmoid and reverse sigmoid.
`Convolution Normal Distribution 3d Activations` has the limit of expression inhibitory neuron, so we create new layer that contain sigmoid and reverse sigmoid in one layer.