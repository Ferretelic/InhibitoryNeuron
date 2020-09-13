import torch
from network import DistributionConvolutionModelKReluGradient
import matplotlib.pyplot as plt

model = DistributionConvolutionModelKReluGradient()
model.load_state_dict(torch.load("../model/model_final.pth"), strict=False)
distribution = model.conv1_1.distribution.detach().numpy()[0]

plt.imshow(distribution)
plt.savefig("./distribution.png")