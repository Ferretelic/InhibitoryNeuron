import torch

def krelu(x, k):
  return torch.max(torch.tensor([0], device="cuda", dtype=torch.float32), torch.mul(k, x))