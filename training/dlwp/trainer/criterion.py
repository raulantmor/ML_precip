import torch as th
import torch
from torch.nn import functional as F
import numpy as np
import xarray as xr 
from typing import Any, Dict, Optional, Sequence, Union

"""
Custom loss classes allow for more sophisticated training optimization.

Each custom loss should inherit all methods of th.nn._Loss base class or subclasses thereof. 
Additionally, custom loss classes should define a setup function which receives the trainer object. 
The setup function should be used to move tensors to appropriate gpus and finalize configuration
of the loss calculation using information about the model (trainer.model) and trainer. Custom
losses should also redefine the forward function to contain a flag indicating whether or not to 
average output channels. This is used in the varible wise logging of validation loss by the trainer. 

"""
class BaseMSE( th.nn.MSELoss ):
    """
    Base MSE class offers impementaion for basic MSE loss compatable with dlwp custom loss training
    """
    def __init__(
        self,
    ):
        super().__init__()
        self.device = None
    def setup(self, trainer):
        """
        Nothing to implement here  
        """
        pass
    def forward(self, prediction, target, average_channels=True ):
        d = ((target-prediction)**2).mean(dim=(0, 1, 2, 4, 5))
        if average_channels: 
            return th.mean(d)
        else: 
            return d
    
class ConditionalWeightLoss( th.nn.MSELoss ):
    """
    Conditional loss for precipitation diagnostic model
    """
    def __init__(
        self,
        weight=(0.15,1.0), #EI:(0.15, 1.0), EE:(0.10, 1.0)
        b=None,
        w=1

        ):
        super().__init__()
        self.weight_zero = weight[0]
        self.weight_nonzero = weight[1]
        self.device = None
        self.b = b
        self.w = w
    def setup(self, trainer):
        self.b = th.tensor(self.b,device=trainer.device)
        self.w = th.tensor(self.w,device=trainer.device)
    def forward(self, prediction, target, average_channels=True ):
        weights_for_zero = th.ones_like(target) * self.weight_zero
        weights_for_nonzero = (th.ones_like(target) * self.weight_nonzero) * th.exp(self.b*target)
        weights = th.where(target > -0.99848, weights_for_nonzero, weights_for_zero)
        loss = (th.mean(weights * (prediction - target) ** 2))*self.w
        return loss
    