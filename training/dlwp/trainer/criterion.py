import torch as th
import torch
from torch.nn import functional as F
import numpy as np
import xarray as xr 
from typing import Any, Dict, Optional, Sequence, Union

#from pytorch_msssim import SSIM

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

class LogLoss( th.nn.MSELoss ):
    """
    Loss for precipitation diagnostic model
    """
    def __init__(
        self,
        b=None,
        w=1,

        ):
        super().__init__()
        self.b = b
        self.device = None
        self.w = w
    def setup(self, trainer):
        self.b = th.tensor(self.b,device=trainer.device)
        self.w = th.tensor(self.w,device=trainer.device)
    def forward(self, prediction, target, average_channels=True ):
        
        #loss = self.w*th.sum(th.exp(self.b*prediction)*(target-prediction)**2)
        #loss = ((th.exp(self.b*prediction)*(target-prediction)**2).mean(dim=(0,1,2,4,5)))*self.w
        #loss = (th.mean(th.exp(self.b*prediction)*(target-prediction)**2))*self.w
        loss = (th.mean(th.exp(self.b*target)*(target-prediction)**2))*self.w
        return loss
    
class ConditionalWeightLoss( th.nn.MSELoss ):
    """
    Conditional loss for precipitation diagnostic model
    """
    def __init__(
        self,
        weight=(0.10,1.0), #EI:(0.15, 1.0), EE:(0.10, 1.0)
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
        weights = th.where(target > -2.18223, weights_for_nonzero, weights_for_zero)
        loss = (th.mean(weights * (prediction - target) ** 2))*self.w
        return loss
    
class ConditionalWeightLossSSIM(th.nn.Module):
    """
    Conditional loss for precipitation diagnostic model incorporating SSIM.
    """
    def __init__(
        self,
        weight=(0.01, 1.0),
        b=None,
        w=1,
        w2=1,
        ssim_weight=0.2,  # Weight for SSIM loss component
        mse_weight=0.8,  # Weight for MSE loss component
        ssim_params=None
    ):
        super().__init__()
        self.weight_zero = weight[0]
        self.weight_nonzero = weight[1]
        self.b = b
        self.w = w
        self.w2 = w2
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight

        if ssim_params is None:
            self.ssim = SSIM()  
        else:
            self.ssim = SSIM(**ssim_params)

    def setup(self, trainer):
        self.b = th.tensor(self.b, device=trainer.device)
        self.w = th.tensor(self.w, device=trainer.device)
        self.w2 = th.tensor(self.w2, device=trainer.device)

    def forward(self, prediction, target, average_channels=True):
        # Compute weighted MSE
        #print(f"Prediction shape: {prediction.shape}")
        
        weights_for_zero = th.ones_like(target) * self.weight_zero
        weights_for_nonzero = (th.ones_like(target) * self.weight_nonzero) * th.exp(self.b * target)
        weights = th.where(target > 0, weights_for_nonzero, weights_for_zero)
        mse_loss = (th.mean(weights * (prediction - target) ** 2)) * self.w
        #print(f"MSE: {mse_loss}")

        # Compute DSSIM loss (1 - SSIM)
        #print(f"pred shape:{prediction.shape}")
        ssim_loss = (1 - self.ssim(prediction, target)) * self.w2
        #print(f"DSSIM: {ssim_loss}")

        # Combine MSE and DSSIM
        loss = (self.mse_weight * mse_loss) + (self.ssim_weight * ssim_loss)
        return loss
    
class LogLossSqrt( th.nn.MSELoss ):
    """
    Loss for precipitation diagnostic model
    """
    def __init__(
        self,
        r=None,
        w=1,

        ):
        super().__init__()
        self.r = r
        self.device = None
        self.w = w
    def setup(self, trainer):
        self.r = th.tensor(self.r,device=trainer.device)
        self.w = th.tensor(self.w,device=trainer.device)
    def forward(self, prediction, target, average_channels=True ):
        
        #loss = self.w*th.sum(th.exp(self.b*prediction)*(target-prediction)**2)
        #loss = ((th.exp(self.b*prediction)*(target-prediction)**2).mean(dim=(0,1,2,4,5)))*self.w
        loss = (th.mean(th.sqrt(th.abs(prediction)/self.r)*(target-prediction)**2))#*self.w 
        return loss

class WeightedMSE( th.nn.MSELoss ):

    """
    Loss object that allows for user defined weighting of variables when calculating MSE
    """

    def __init__(
        self,
        weights: Sequence = [],
    ):
    
        """
        params: weights: list of floats that determine weighting of variable loss, assumed to be
            in order consistent with order of model output channels
        """
        super().__init__()
        self.loss_weights = th.tensor(weights)
        self.device = None

    def setup(self, trainer):
        """
        pushes weights to cuda device 
        """
        
        try:
            assert len(trainer.output_variables) == len(self.loss_weights)
        except AssertionError:
            raise ValueError('Length of outputs and loss_weights is not the same!')

        self.loss_weights = self.loss_weights.to(device=trainer.device)

    def forward(self, prediction, target, average_channels=True ):

        d = ((target-prediction)**2).mean(dim=(0, 1, 2, 4, 5))*self.loss_weights
        if average_channels: 
            return th.mean(d)
        else: 
            return d

class OceanMSE( th.nn.MSELoss ):
    """
    Ocean MSE class offers impementaion for MSE loss weighted by a land-sea-mask field. 
    """
    def __init__(
        self,
        lsm_file: str,
        open_dict: dict = {
            'engine':'zarr'},
        selection_dict: dict = {
            'channel_c':'lsm'},
    ):
        """
        """
        super().__init__()
        self.device = None
        self.lsm_file = lsm_file
        self.lsm_ds = None
        self.open_dict = open_dict
        self.selection_dict = selection_dict
        self.lsm_tensor = None 
        self.lsm_sum_calculated = False
        self.lsm_sum = None
        self.lsm_var_sum = None

    def setup(self, trainer):
        """
        reshape lsm and put on device 
        """
        self.lsm_ds = xr.open_dataset(self.lsm_file,**self.open_dict).constants.sel(self.selection_dict)
        # 1-lsm gives the percentage of pixel that has ocean 
        self.lsm_tensor = 1 - th.tensor(np.expand_dims(self.lsm_ds.values,(0,2,3))).to(trainer.device)
        #self.lsm_sum = self.lsm_tensor.sum()
        ## weighting sum that retains varible dimension for variable wise loss calculation
        #self.lsm_var_sum = self.lsm_tensor.sum(dim=(0, 1, 2, 4, 5))
        
    def forward(self, prediction, target, average_channels=True ):
         
        if not self.lsm_sum_calculated:
            self.lsm_sum = th.broadcast_to(self.lsm_tensor,target.shape).sum()
            self.lsm_var_sum = th.broadcast_to(self.lsm_tensor,target.shape).sum(dim=(0,1,2,4,5))
            self.lsm_sum_calculated = True
        # average weighted 
        ocean_err = (((target-prediction)**2)*self.lsm_tensor)
        ocean_mean_err = ocean_err.sum(dim=(0, 1, 2, 4, 5))
        if average_channels: 
            return th.sum(ocean_mean_err)/self.lsm_sum
        else: 
            return ocean_mean_err/self.lsm_var_sum
        
class SSIM(torch.nn.Module):
    """
    This class provides a differential structural similarity (SSIM) as loss for training an artificial neural network. The
    advantage of SSIM over the conventional mean squared error is a relation to images where SSIM incorporates the local
    neighborhood when determining the quality of an individual pixel. Results are less blurry, as demonstrated here
    https://ece.uwaterloo.ca/~z70wang/research/ssim/

    Code is origininally taken from https://github.com/Po-Hsun-Su/pytorch-ssim
    Modifications include comments and an optional training phase with the mean squared error (MSE) preceding the SSIM
    loss, to bring the weights on track. Otherwise, SSIM often gets stuck early in a local minimum.
    """

    def __init__(
            self,
            window_size: int = 11,
            time_series_forecasting: bool = False,
            padding_mode: str = "constant",
            mse: bool = False,
            mse_epochs: int = 0):
        """
        Constructor method.

        :param window_size: (Optional) The patch size over which the SSIM is computed
        :param time_series_forecasting: (Optional) Boolean indicating whether time series forecasting is the task
        :param padding_mode: Padding mode used for padding input images, e.g. 'zeros', 'replicate', 'reflection'
        :param mse: Uses MSE parallel
        :param mse_epochs: (Optional) Number of MSE epochs preceding the SSIM epochs during training
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.time_series_forecasting = time_series_forecasting
        self.padding_mode = padding_mode
        self.mse = torch.nn.MSELoss() if mse else None
        self.mse_epochs = mse_epochs
        self.c1, self.c2 = 0.01 ** 2, 0.03 ** 2

        self.register_buffer("window", self._create_window(window_size), persistent=False)

    def forward(
            self,
            img1: torch.Tensor,
            img2: torch.Tensor,
            mask: torch.Tensor = None,
            epoch: int = 0) -> torch.Tensor:
        """
        Forward pass of the SSIM loss

        :param img1: Predicted image of shape [B, T, C, F, H, W]
        :param img2: Ground truth image of shape [B, T, C, F, H, W]
        :param mask: (Optional) Mask for excluding pixels
        :param epoch: (Optional) The current epoch
        :return: The structural similarity loss
        """
        img1 = img1.transpose(1, 2)
        img2 = img2.transpose(1, 2)
        #if self.time_series_forecasting:
            # Join Batch and time dimension
        img1 = torch.flatten(img1, start_dim=0, end_dim=2)
        img2 = torch.flatten(img2, start_dim=0, end_dim=2)

        window = self.window.expand(img1.shape[1], -1, -1, -1)

        if window.dtype != img1.dtype:
            window = window.to(dtype=img1.dtype)

        return self._ssim(img1, img2, window, mask, epoch)

    @staticmethod
    def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
        """
        Computes a Gaussian over the size of the window to weigh distant pixels less.

        :param window_size: The size of the patches
        :param sigma: The width of the Gaussian curve
        :return: A tensor representing the weights for each pixel in the window or patch
        """
        x = torch.arange(0, window_size) - window_size // 2
        gauss = torch.exp(-((x.div(2 * sigma)) ** 2))
        return gauss / gauss.sum()

    def _create_window(
            self,
            window_size: int,
            sigma: float = 1.5) -> torch.Tensor:
        """
        Creates the weights of the window or patches.

        :param window_size: The size of the patches
        :param sigma: The width of the Gaussian curve
        """
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(-1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        return _2D_window

    def _ssim(
            self,
            img1: torch.Tensor,
            img2: torch.Tensor,
            window: torch.Tensor,
            mask: torch.Tensor = None,
            epoch: int = 0) -> torch.Tensor:
        """
        Computes the SSIM loss between two image tensors

        :param _img1: The predicted image tensor
        :param _img2: The target image tensor
        :param window: The weights for each pixel in the window over which the SSIM is computed
        :param mask: Mask for excluding pixels
        :param epoch: The current epoch
        :return: The SSIM between img1 and img2
        """
        if epoch < self.mse_epochs:
            # If specified, the MSE is used for the first self.mse_epochs epochs
            return F.mse_loss(img1, img2)

        channels = window.shape[0]
        window_size = window.shape[2]

        window = window.to(device=img1.device)

        _img1 = F.pad(
            img1,
            pad=[(window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2,
                 (window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2],
            mode=self.padding_mode
        )

        _img2 = F.pad(
            img2,
            pad=[(window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2,
                 (window_size - 1) // 2, (window_size - 1) // 2 + (window_size - 1) % 2],
            mode=self.padding_mode
        )

        mu1 = F.conv2d(_img1, window, padding=0, groups=channels)
        mu2 = F.conv2d(_img2, window, padding=0, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(_img1 * _img1, window, padding=0, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(_img2 * _img2, window, padding=0, groups=channels) - mu2_sq
        sigma12_sq = F.conv2d(_img1 * _img2, window, padding=0, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12_sq + self.c2)) /\
                   ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))

        if mask is not None:
            ssim_map = ssim_map[..., mask]
            img1 = img1[..., mask]
            img2 = img2[..., mask]

        ssim = ssim_map.mean().abs()

        if self.mse:
            ssim = ssim + self.mse(img1, img2)
    
        return ssim
