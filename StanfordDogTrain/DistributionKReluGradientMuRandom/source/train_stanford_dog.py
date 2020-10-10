import torch
import torch.nn as nn
import torch.optim as optim
import os
from stanford_dog import load_images
from train_util import train_model
from network import DistributionConvolutionModelKReluGradientMuRandom
import pickle
from show_history import plot_history

if os.path.isdir("../model") == False:
  os.mkdir("../model")

epochs = 500
device_name = "cpu"
learning_rate = 0.001
batch_size = 128
model = DistributionConvolutionModelKReluGradientMuRandom()

device = torch.device(device_name)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
x_train, y_train, x_test, y_test = load_images()

model, history = train_model(model, criterion, optimizer, epochs, x_train, y_train, x_test, y_test, device, batch_size)

with open("../model/model_history.pkl", "wb") as f:
  pickle.dump(history, f)

plot_history(history)

torch.save({"model_state_dict": model.state_dict()}, "../model/model_final.pth")
