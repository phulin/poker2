import torch

counts = torch.tensor([0, 1, 0], device="mps")
data = torch.arange(2, device="mps")
data.repeat_interleave(counts[1:3], dim=0)  # segfault
