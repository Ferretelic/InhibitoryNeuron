import torch
from distribution_3d import plot_distribution, plot_distribution_2d
from network import DistributionConvolutionModelKReluGradientMuRandom
import matplotlib.pyplot as plt

model = DistributionConvolutionModelKReluGradientMuRandom()
model.load_state_dict(torch.load("../model/model_final.pth")["model_state_dict"])
distributions = model.conv3_1.distribution.detach().numpy()

figure, graphs = plt.subplots(8, 16)
for graph, distribution in zip(graphs.reshape(128), distributions):
  graph.matshow(distribution)
  graph.set_xticklabels([])
  graph.set_yticklabels([])
  graph.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
figure.savefig("../image/distribution_trained_conv3_1.png")

plot_distribution(distributions[0], name="distribution_trained.png")
plot_distribution_2d(distributions[0], name="distribution_trained_2d.png")