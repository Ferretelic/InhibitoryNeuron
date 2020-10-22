import torch
from distribution_3d import plot_distribution, plot_distribution_2d
from network import DistributionConvolutionModelKReluGradientMuRandom
import matplotlib.pyplot as plt

model = DistributionConvolutionModelKReluGradientMuRandom(image_size=(100, 100))
model.load_state_dict(torch.load("../model/model_final.pth")["model_state_dict"])

layers = [
  [model.conv1_1, model.conv1_2],
  [model.conv2_1, model.conv2_2],
  [model.conv3_1, model.conv3_2]
]
shapes = [[4, 8], [8, 8], [8, 16]]

for i, layer in enumerate(layers):
  for j, conv in enumerate(layer):
    print("conv{}_{}.png".format(i + 1, j + 1))
    distributions = conv.distribution.detach().numpy()
    print(distributions.shape)

    figure, graphs = plt.subplots(*shapes[i])
    print(distributions.shape)
    print(graphs.shape)
    print(shapes[i][0] * shapes[i][1])

    for graph, distribution in zip(graphs.reshape(shapes[i][0] * shapes[i][1]), distributions):
      graph.matshow(distribution)
      graph.set_xticklabels([])
      graph.set_yticklabels([])
      graph.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    figure.savefig("../image/distribution_trained_conv{}_{}.png".format(i + 1, j + 1))

plot_distribution(distributions[0], name="distribution_trained.png")
plot_distribution_2d(distributions[0], name="distribution_trained_2d.png")