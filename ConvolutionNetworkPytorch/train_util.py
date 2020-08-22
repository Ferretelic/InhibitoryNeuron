import torch
import torch.nn.functional as F
import pyprind
import os

def train_model(model, criterion, optimizer, epochs, trainloader, testloader, device, model_directory):
  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    bar = pyprind.ProgBar(len(trainloader), track_time=True, title="Training Model")

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        bar.update()

    correct = 0
    total = 0

    with torch.no_grad():
      for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

      model_path = os.path.join(model_directory, "model_{}.pth".format(epoch + 1))
      torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "loss": running_loss / len(trainloader)}, model_path)

      print("Epoch: {} Loss: {:.3f}".format(epoch + 1, running_loss / len(trainloader)))
      print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))

  print("Finished Training")
  return model