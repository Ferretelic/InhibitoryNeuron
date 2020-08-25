import torch
import torch.nn as nn
import torch.optim as optim
import os
from cifar_data import get_cifar10
from train_util import train_model
from convolution_network import NormalConvolutionModelRelu, NormalConvolutionModelSigmoid, DistributionConvolutionModelReluNoGrad, DistributionConvolutionModelReluGrad
import pickle
import shutil

model_name = "conv_distribution_relu_grad"
model_directory = os.path.join("./models", model_name)
log_directory = os.path.join(model_directory, "logs")

if os.path.isdir(model_directory) == False:
  os.mkdir(model_directory)

shutil.rmtree(log_directory, ignore_errors=True)
if os.path.isdir(log_directory) == False:
  os.mkdir(log_directory)

epochs = 150
device_name = "cuda"
learning_rate = 0.001
batch_size = 128
model = DistributionConvolutionModelReluGrad()
background = False

device = torch.device(device_name)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
trainloader, testloader = get_cifar10(batch_size)

model, history = train_model(model, criterion, optimizer, epochs, trainloader, testloader, device, model_directory, background)

with open(os.path.join(model_directory, "model_history.pkl"), "wb") as f:
  pickle.dump(history, f)

torch.save({"model_state_dict": model.state_dict()}, os.path.join(model_directory, "model_final.pth"))
