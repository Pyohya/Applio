from typing import Optional
import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn import functional as F

from rvc.lib.algorithm.modules import WN
from rvc.lib.algorithm.commons import get_padding, init_weights

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    """Residual block with three dilated convolutional layers.

    Args:
        channels (int): Number of channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple, optional): Dilation rates of the convolutional layers. Defaults to (1, 3, 5).

    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
            x_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, time_steps).
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, time_steps).

        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        """Remove weight normalization from the module."""
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    """Residual block with two dilated convolutional layers.

    Args:
        channels (int): Number of channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple, optional): Dilation rates of the convolutional layers. Defaults to (1, 3).

    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
            x_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, time_steps).
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, time_steps).

        """
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        """Remove weight normalization from the module."""
        for l in self.convs:
            remove_weight_norm(l)


class Log(nn.Module):
    """Logarithm module for flow-based models.

    This module computes the logarithm of the input and its log determinant.
    During reverse, it computes the exponential of the input.
    """

    def forward(self, x, x_mask, reverse=False, **kwargs):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            x_mask (torch.Tensor): Mask tensor.
            reverse (bool, optional): Whether to reverse the operation. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: Output tensor.
                - torch.Tensor: Log determinant tensor.
        """
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    """Flip module for flow-based models.

    This module flips the input along the time dimension.
    """

    def forward(self, x, *args, reverse=False, **kwargs):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            reverse (bool, optional): Whether to reverse the operation. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: Output tensor.
                - torch.Tensor: Log determinant tensor.
        """
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    """Elementwise affine transformation module for flow-based models.

    This module performs an elementwise affine transformation on the input.

    Args:
        channels (int): Number of channels.

    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            x_mask (torch.Tensor): Mask tensor.
            reverse (bool, optional): Whether to reverse the operation. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: Output tensor.
                - torch.Tensor: Log determinant tensor.
        """
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class ResidualCouplingBlock(nn.Module):
    """Residual Coupling Block for normalizing flow.

    Args:
        channels (int): Number of channels in the input.
        hidden_channels (int): Number of hidden channels in the coupling layer.
        kernel_size (int): Kernel size of the convolutional layers.
        dilation_rate (int): Dilation rate of the convolutional layers.
        n_layers (int): Number of layers in the coupling layer.
        n_flows (int, optional): Number of coupling layers in the block. Defaults to 4.
        gin_channels (int, optional): Number of channels for the global conditioning input. Defaults to 0.

    Inputs:
        x (torch.Tensor): Input tensor with shape (batch_size, channels, length).
        x_mask (torch.Tensor): Mask tensor with shape (batch_size, 1, length).
        g (torch.Tensor, optional): Global conditioning input with shape (batch_size, gin_channels, 1). Defaults to None.
        reverse (bool, optional): Whether to reverse the flow. Defaults to False.

    Outputs:
        x (torch.Tensor): Output tensor with shape (batch_size, channels, length).
    """

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows[::-1]:
                x = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        """Removes weight normalization from the coupling layers."""
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

    def __prepare_scriptable__(self):
        """Prepares the module for scripting."""
        for i in range(self.n_flows):
            for hook in self.flows[i * 2]._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.flows[i * 2])

        return self


class ResidualCouplingLayer(nn.Module):
    """Residual coupling layer for flow-based models.

    Args:
        channels (int): Number of channels.
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation_rate (int): Dilation rate of the convolution.
        n_layers (int): Number of convolutional layers.
        p_dropout (float, optional): Dropout probability. Defaults to 0.
        gin_channels (int, optional): Number of conditioning channels. Defaults to 0.
        mean_only (bool, optional): Whether to use mean-only coupling. Defaults to False.

    """

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
            x_mask (torch.Tensor): Mask tensor of shape (batch_size, 1, time_steps).
            g (torch.Tensor, optional): Conditioning tensor of shape (batch_size, gin_channels, time_steps).
                Defaults to None.
            reverse (bool, optional): Whether to reverse the operation. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: Output tensor of shape (batch_size, channels, time_steps).
                - torch.Tensor: Log determinant tensor.
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self):
        """Remove weight normalization from the module."""
        self.enc.remove_weight_norm()
