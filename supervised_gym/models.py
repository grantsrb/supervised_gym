import numpy as np
import torch
import torch.nn as nn
from supervised_gym.utils.torch_modules import Flatten, Reshape, GaussianNoise
from supervised_gym.utils.utils import update_shape
# update_shape(shape, kernel=3, padding=0, stride=1, op="conv"):

class Model(torch.nn.Module):
    """
    This is the base class for all models within this project. It
    ensures the appropriate members are added to the model.
    """
    def __init__(self,
        inpt_shape,
        actn_size,
        h_size=128,
        bnorm=False,
        conv_noise=0,
        dense_noise=0,
        *args, **kwargs
    ):
        """
        Args: 
            inpt_shape: tuple or listlike (..., C, H, W)
                the shape of the input
            actn_size: int
                the number of potential actions
            h_size: int
                the size of the hidden dimension for the dense layers
            bnorm: bool
                if true, the model uses batch normalization
        """
        super().__init__()
        self.inpt_shape = inpt_shape
        self.actn_size = actn_size
        self.h_size = h_size
        self.bnorm = bnorm
        self.conv_noise = conv_noise
        self.dense_noise = dense_noise

    def reset(self):
        """
        Only necessary to override if building a recurrent network.
        This function should reset any recurrent state in a model.
        """
        pass

    def step(self, x):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            pred: torch Float Tensor (B, K)
        """
        pass

    def forward(self, x):
        """
        Performs multiple steps in time rather than a single step.

        Args:
            x: torch FloatTensor (B, S, C, H, W)
        Returns:
            pred: torch Float Tensor (B, S, K)
        """
        pass

class RandomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(inpt_shape=None, **kwargs)

    def forward(self, x, dones=None):
        """
        Args:
            x: torch Float Tensor (B, S, C, H, W)
            dones: torch LongTensor (B, S)
        """
        if len(x.shape) == 4:
            rand = torch.randint(
                low=0,
                high=self.actn_size,
                size=(len(x),)
            )
            actn = torch.zeros(len(x), self.actn_size).float()
            actn[torch.arange(len(x)).long(), rand] = 1
            if x.is_cuda: actn.cuda()
            return actn
        else:
            actn = torch.zeros(*x.shape[:2], self.actn_size).float()
            rand = torch.randint(
                low=0,
                high=self.actn_size,
                size=(int(np.prod(x.shape[:2])),)
            )
            actn = actn.reshape(int(np.prod(x.shape[:2])), -1)
            actn[torch.arange(len(actn)).long(), rand] = 1
            if x.is_cuda: actn.cuda()
            return actn

class SimpleCNN(Model):
    """
    A simple convolutional network with no recurrence.
        conv2d
        bnorm
        relu
        conv2d
        bnorm
        relu
        linear
        bnorm
        relu
        linear
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depths = [self.inpt_shape[-3], 32, 48]
        self.kernels = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]
        modules = []
        shape = [*self.inpt_shape[-3:]]
        self.shapes = [shape]
        for i in range(len(self.depths)-1):
            # CONV
            modules.append(
                nn.Conv2d(
                    self.depths[i],
                    self.depths[i+1],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.paddings[i]
                )
            )
            # RELU
            modules.append(GaussianNoise(self.conv_noise))
            modules.append(nn.ReLU())
            # Batch Norm
            if self.bnorm:
                modules.append(nn.BatchNorm2d(self.depths[i+1]))
            # Track Activation Shape Change
            shape = update_shape(
                shape, 
                depth=self.depths[i+1],
                kernel=self.kernels[i],
                stride=self.strides[i],
                padding=self.paddings[i]
            )
            self.shapes.append(shape)
        self.features = nn.Sequential(*modules)

        # Make Output MLP
        self.flat_size = int(np.prod(shape))
        modules = [
            Flatten(),
            nn.Linear(self.flat_size, self.h_size),
            GaussianNoise(self.dense_noise),
            nn.ReLU()
        ]
        if self.bnorm:
            modules.append(nn.BatchNorm1d(self.h_size))
        self.dense = nn.Sequential(
            *modules,
            nn.Linear(self.h_size, self.actn_size)
        )
        # Full Model All Together
        self.full_model = nn.Sequential(
            self.features,
            self.dense
        )

    def step(self, x, *args, **kwargs):
        """
        Performs a single step rather than a complete sequence of steps

        Args:
            x: torch FloatTensor (B, C, H, W)
        Returns:
            pred: torch Float Tensor (B, K)
        """
        return self.full_model(x)

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: torch FloatTensor (B, S, C, H, W)
        Returns:
            actns: torch FloatTensor (B, S, N)
                N is equivalent to self.actn_size
        """
        b,s = x.shape[:2]
        fx = self.full_model(x.reshape(-1, *x.shape[2:]))
        return fx.reshape(b,s,-1)

