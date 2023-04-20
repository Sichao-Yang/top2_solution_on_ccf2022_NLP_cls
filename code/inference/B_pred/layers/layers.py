
from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F



class MaskedGlobalMaxPool1D(nn.Module):
    
    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPool1D, self).__init__(**kwargs)
        self.supports_masking = True       
        
    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)
    
    def forward(self,inputs,mask = None):
        if mask is not None:
            mask = mask.float()
            inputs = inputs - torch.unsqueeze((1.0-mask)*1e6,dim = -1)
        return torch.max(inputs,dim = -2).values
    
class MaskedGlobalAveragePooling1D(nn.Module):

    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1],)
    
    def forward(self,inputs, mask=None):
        if mask is not None:
            mask = mask.float()
            mask = torch.unsqueeze(mask,dim = -1)
            inputs = inputs*mask
            return torch.sum(inputs,dim = 1)/torch.sum(mask,dim = 1)
        else:
            return torch.mean(inputs,dim = 1)
        
        
class Spatial_Dropout(nn.Module):
    def __init__(self,drop_prob):

        super(Spatial_Dropout,self).__init__()
        self.drop_prob = drop_prob

    def forward(self,inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self,input):
        return input.new().resize_(input.size(0),*repeat(1, input.dim() - 2),input.size(2))