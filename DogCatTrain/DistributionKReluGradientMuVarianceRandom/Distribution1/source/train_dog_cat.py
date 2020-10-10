import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle

from train_util import train_model
from dog_cat import load_datasets, prepare_datasets
from network import DistributionConvolutionModelKReluGradientMuRandom
from show_history import plot_history

if os.path.isdir("../model") == False:
  os.mkdir("../model")

prepared = True
epochs = 1
device_name = "cuda"
learning_rate = 0.001
batch_size = 32
image_size = (100, 100)
model = DistributionConvolutionModelKReluGradientMuRandom(image_size=image_size)

device = torch.device(device_name)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if prepared == False:
  prepare_datasets(image_size)
x_train, y_train, x_test, y_test = load_datasets()

print("Start Training")
model, history = train_model(model, criterion, optimizer, epochs, x_train, y_train, x_test, y_test, device, batch_size)

with open("../model/model_history.pkl", "wb") as f:
  pickle.dump(history, f)

plot_history(history)

torch.save({"model_state_dict": model.state_dict()}, "../model/model_final.pth")
