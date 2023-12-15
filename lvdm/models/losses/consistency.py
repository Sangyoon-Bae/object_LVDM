import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.consistency_loss = torch.tensor([0.0])

    def forward(self, slot_list, length, temperature):
        for i in range(length):
            for j in range(length):
                if i<j:
                    numerator = torch.exp(self.sim(slot_list[i], slot_list[j])/temperature)
                    complent_list = list(range(length))[:j]+list(range(length))[j+1:]
                    denominator = sum([torch.exp(sim(slot_list[i], slot_list[k])/temperature) for k in complent_list])
                    penalty = np.abs(i-j)
                    self.consistency_loss += -torch.log((numerator/denominator))*penalty
                    
        log = {"total_consistency_loss": self.consistency_loss/((i*j)/2)}
        
        return self.consistency_loss, log