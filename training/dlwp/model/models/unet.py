import logging
from typing import Any, Dict, Optional, Sequence, Union

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import torch as th
import pandas as pd

from training.dlwp.model.modules.healpix import HEALPixPadding, HEALPixLayer
from training.dlwp.model.modules.encoder import UNetEncoder
from training.dlwp.model.modules.decoder import UNetDecoder
from training.dlwp.model.modules.blocks import FoldFaces, UnfoldFaces
from training.dlwp.model.modules.utils import Interpolate

logger = logging.getLogger(__name__)

class HEALPixUNet(th.nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            input_channels: int,
            output_channels: int,
            n_constants: int,
            decoder_input_channels: int,
            input_time_dim: int,
            output_time_dim: int,
            presteps: int = 0,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False,
    ):
        """
        Deep Learning Weather Prediction (DLWP) UNet on the HEALPix mesh.

        :param encoder: dictionary of instantiable parameters for the U-net encoder
        :param decoder: dictionary of instantiable parameters for the U-net decoder
        :param input_channels: number of input channels expected in the input array schema. Note this should be the
            number of input variables in the data, NOT including data reshaping for the encoder part.
        :param output_channels: number of output channels expected in the output array schema, or output variables
        :param n_constants: number of optional constants expected in the input arrays. If this is zero, no constants
            should be provided as inputs to `forward`.
        :param decoder_input_channels: number of optional prescribed variables expected in the decoder input array
            for both inputs and outputs. If this is zero, no decoder inputs should be provided as inputs to `forward`.
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param enable_nhwc: Model with [N, H, W, C] instead of [N, C, H, W] oder
        :param enable_healpixpad: Enable CUDA HEALPixPadding if installed
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_constants = n_constants
        self.decoder_input_channels = decoder_input_channels
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.channel_dim = 2  # Now 2 with [B, F, C*T, H, W]. Was 1 in old data format with [B, T*C, F, H, W]
        self.enable_nhwc = enable_nhwc
        self.enable_healpixpad = enable_healpixpad

        #self.output_dim = self.output_channels*self.input_time_dim

        # Number of passes through the model, or a diagnostic model with only one output time
        self.is_diagnostic = self.output_time_dim == 1 and self.input_time_dim > 1
        if not self.is_diagnostic and (self.output_time_dim % self.input_time_dim != 0):
            raise ValueError(f"'output_time_dim' must be a multiple of 'input_time_dim' (got "
                             f"{self.output_time_dim} and {self.input_time_dim})")

        # Build the model layers
        self.fold = FoldFaces()
        self.unfold = UnfoldFaces(num_faces=12)
        self.encoder = instantiate(config=encoder,
                                   input_channels=self._compute_input_channels(),
                                   enable_nhwc=self.enable_nhwc,
                                   enable_healpixpad=self.enable_healpixpad)
        self.encoder_depth = len(self.encoder.n_channels)
        self.decoder = instantiate(config=decoder,
                                   output_channels=self._compute_output_channels(),
                                   enable_nhwc = self.enable_nhwc,
                                   enable_healpixpad = self.enable_healpixpad)

    @property
    def integration_steps(self):
        return max(self.output_time_dim // self.input_time_dim, 1)

    def _compute_input_channels(self) -> int:
        return self.input_time_dim * (self.input_channels + self.decoder_input_channels) + self.n_constants

    def _compute_output_channels(self) -> int:
        return (1 if self.is_diagnostic else self.input_time_dim) * self.output_channels

    def _reshape_inputs(self, inputs: Sequence, step: int = 0) -> th.Tensor:
        """
        Returns a single tensor to pass into the model encoder/decoder. Squashes the time/channel dimension and
        concatenates in constants and decoder inputs.
        :param inputs: list of expected input tensors (inputs, decoder_inputs, constants)
        :param step: step number in the sequence of integration_steps
        :return: reshaped Tensor in expected shape for model encoder
        """
        if not (self.n_constants > 0 or self.decoder_input_channels > 0):
            return inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1)
        if self.n_constants == 0:
            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(self.channel_dim, self.channel_dim+1)  # DI
            ]
            res = th.cat(result, dim=self.channel_dim)

            # fold faces into batch dim
            res = self.fold(res)
            
            return res
        if self.decoder_input_channels == 0:
            result = [
                inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
                inputs[1].expand(*tuple([inputs[0].shape[0]] + len(inputs[1].shape)*[-1]))  # constants
                #th.tile(self.constants, (inputs[0].shape[0], 1, 1, 1, 1)) # constants
            ]
            res = th.cat(result, dim=self.channel_dim)

            # fold faces into batch dim
            res = self.fold(res)
            
            return res
        
        result = [
            inputs[0].flatten(start_dim=self.channel_dim, end_dim=self.channel_dim+1),  # inputs
            inputs[1][:, :, slice(step*self.input_time_dim, (step+1)*self.input_time_dim), ...].flatten(self.channel_dim, self.channel_dim+1),  # DI
            inputs[2].expand(*tuple([inputs[0].shape[0]] + len(inputs[2].shape) * [-1]))  # constants
            #th.tile(self.constants, (inputs[0].shape[0], 1, 1, 1, 1)) # constants
        ]
        res = th.cat(result, dim=self.channel_dim)

        # fold faces into batch dim
        res = self.fold(res)
        
        return res

    def _reshape_outputs(self, outputs: th.Tensor) -> th.Tensor:

        # unfold:
        outputs = self.unfold(outputs)
        
        # extract shape and reshape
        shape = tuple(outputs.shape)
        res = th.reshape(outputs, shape=(shape[0], shape[1], 1 if self.is_diagnostic else self.input_time_dim, -1, *shape[3:]))
        
        return res

    def forward(self, inputs: Sequence, output_only_last=False) -> th.Tensor:

        outputs = []
        for step in range(self.integration_steps):
            if step == 0:
                inputs_0 = inputs[0]
                input_tensor = self._reshape_inputs(inputs, step)
            else:
                inputs_0 = outputs[-1]
                input_tensor = self._reshape_inputs([outputs[-1]] + list(inputs[1:]), step)

            encodings = self.encoder(input_tensor)
            decodings = self.decoder(encodings)

            reshaped = self._reshape_outputs(decodings)  # Absolute prediction
            #reshaped = self._reshape_outputs(input_tensor[:, :self.input_channels*self.input_time_dim] + decodings)  # Residual prediction
            outputs.append(reshaped)
            
        if output_only_last:
            res = outputs[-1]
        else:
            res = th.cat(outputs, dim=self.channel_dim)

        return res
