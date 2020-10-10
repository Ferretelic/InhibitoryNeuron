import torch
import torch.nn.functional as F
import pyprind
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import shutil

def train_model(model, criterion, optimizer, epochs, x_train, y_train, x_test, y_test, device, batch_size):
  shutil.rmtree("../logs", ignore_errors=True)
  os.mkdir("../logs")
  writer = SummaryWriter("../logs")

  history = {"loss": [], "accuracy": []}

  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    bar = pyprind.ProgBar(y_train.shape[0], track_time=True, title="Training Model")

    for i in range(y_train.shape[0] // batch_size):
      # get the inputs; data is a list of [inputs, labels]
      inputs = torch.tensor(x_train[i * batch_size : (i + 1) * batch_size], device=device, dtype=torch.float32)
      labels = torch.tensor(y_train[i * batch_size : (i + 1) * batch_size], device=device, dtype=torch.long)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      bar.update()


    with torch.no_grad():
      inputs = torch.tensor(x_test, device=device, dtype=torch.float32)
      labels = torch.tensor(y_test, device=device, dtype=torch.float32)

      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total = labels.size(0)
      correct = (predicted == labels).sum().item()

      history["loss"].append(running_loss / len(trainloader))
      history["accuracy"].append(correct / total)

      writer.add_scalar("Train Loss", running_loss / len(trainloader), epoch)
      writer.add_scalar("Test Accuracy", (100 * correct / total), epoch)

      print("Epoch: {} Loss: {:.3f}".format(epoch + 1, running_loss / len(trainloader)))
      print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

  print("Finished Training")
  return model, history