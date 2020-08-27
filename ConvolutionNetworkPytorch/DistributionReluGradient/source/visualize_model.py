import torch
from network import DistributionConvolutionModelReluGradient
import matplotlib.pyplot as plt

model = DistributionConvolutionModelReluGradient()
model.load_state_dict(torch.load("../model/model_final.pth"), strict=False)
print(model.conv1_1.weights.size())
distribution = model.conv1_1.weights.detach().numpy().T[0].reshape(3, 3, 3)[0]

plt.imshow(distribution)
plt.savefig("./doistribution.png")