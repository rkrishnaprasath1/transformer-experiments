import torch

def create_causal_mask(size):
    """Generates a square mask to block future positions [cite: 79]"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask  # Returns True for allowed, False for masked