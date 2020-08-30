import torch
import torch.nn as nn
import torch.optim as optim
import os
from cifar_data import get_cifar10
from train_util import train_model
from network import DistributionConvolutionModelKReluRandomGradient
import pickle
from show_history import plot_history

if os.path.isdir("../model") == False:
  os.mkdir("../model")

variances = np.linspace(0.1, 1, 10)
epochs = 300
device_name = "cuda"
learning_rate = 0.001
batch_size =

for variance in variances:
  model = DistributionConvolutionModelKReluRandomGradient(variance)

  device = torch.device(device_name)
  model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  trainloader, testloader = get_cifar10(batch_size)

  model, history = train_model(model, criterion, optimizer, epochs, trainloader, testloader, device)

  with open("../model/model_history_{}.pkl".format(variance), "wb") as f:
    pickle.dump(history, f)

  plot_history(history)

  torch.save({"model_state_dict": model.state_dict()}, "../model/model_final_{}.pth".format(variance))
