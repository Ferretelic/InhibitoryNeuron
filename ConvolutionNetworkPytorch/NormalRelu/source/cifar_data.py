import torch
import torchvision
import torchvision.transforms as transforms
import PIL

def get_cifar10(batch_size):
  transform = transforms.Compose([
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  trainset = torchvision.datasets.CIFAR10(root="../../CIFAR10", train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=12)

  testset = torchvision.datasets.CIFAR10(root="../../CIFAR10", train=False,
                                        download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=12)

  return trainloader, testloader