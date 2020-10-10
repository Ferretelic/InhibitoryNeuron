import torch

def krelu(x, k):
  return torch.max(torch.tensor([0], device="cpu", dtype=torch.float32), torch.mul(k, x))