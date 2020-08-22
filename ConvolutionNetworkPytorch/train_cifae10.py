import torch
import torch.nn as nn
import torch.optim as optim
import os
from cifar_data import get_cifar10
from train_util import train_model
from convolution_lnetwork import ConvolutionModel

model_name = "conv_normal"
model_directory = os.path.join("./models", model_name)

if os.path.isdir(model_directory) == False:
  os.mkdir(model_directory)

epochs = 150
device = torch.device("cuda")
model = ConvolutionModel()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
trainloader, testloader = get_cifar10(64)

model = train_model(model, criterion, optimizer, epochs, trainloader, testloader, device, model_directory)

model.load_state_dict(torch.load(os.path.join(model_path, "model_final.pth")))