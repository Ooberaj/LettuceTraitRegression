import torch
import torch.nn as nn

class NMSELoss(nn.Module):
    '''
    Source: https://github.com/plant-ai-biophysics-lab/DeformableCNN-PlantTraits
    '''
    def __init__(self):
        super(NMSELoss, self).__init__()
    def forward(self, pred, target):
        if target.size() != pred.size():
              raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), pred.size()))
        
        # Sum along batch dimension when getting normalized means, then sum normalized means along the label dimension
        num=torch.sum((target-pred)**2, 0)
        den=torch.sum(target**2, 0)

        return torch.sum(num/den)