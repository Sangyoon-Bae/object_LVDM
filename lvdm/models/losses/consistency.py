import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
import numpy as np


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, slot_list, length, temperature):
        consistency_loss_whole = []
        for i in range(length):
            for j in range(length):
                if i<j:
                    numerator = torch.exp(self.sim(slot_list[i], slot_list[j])/temperature)
                    complent_list = list(range(length))[:j]+list(range(length))[j+1:]
                    denominator = sum([torch.exp(self.sim(slot_list[i], slot_list[k])/temperature) for k in complent_list])
                    device = numerator.get_device()
                    penalty = torch.abs(torch.tensor([i])-torch.tensor([j])).to(device)
                    consistency_loss_whole.append(-torch.log((numerator/denominator))*penalty)
        
        total_consistency_loss = sum(consistency_loss_whole)/((i*j)/2)
        log = {"total_consistency_loss": total_consistency_loss}
        
        return total_consistency_loss, log