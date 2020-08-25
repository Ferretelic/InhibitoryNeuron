import torch
import numpy as np
import matplotlib.pyplot as plt

def ksigmoid(x, k):
  return k * torch.div(1, torch.add(1, torch.exp(-x)))
