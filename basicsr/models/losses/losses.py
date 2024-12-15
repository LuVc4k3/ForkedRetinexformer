import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class LuminanceL1Loss(nn.Module):
    """
    Luminance L1 Loss for Low-Light Image Enhancement.

    Computes loss based on luminance (grayscale) and combines it with the RGB L1 loss
    for improved low-light image correction.
    """

    def __init__(self, luminance_weight=0.5, loss_weight=1.0, reduction="mean"):
        super(LuminanceL1Loss, self).__init__()
        self.luminance_weight = luminance_weight
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted tensor of shape (N, C, H, W).
            target (Tensor): Ground truth tensor of shape (N, C, H, W).

        Returns:
            Tensor: Combined L1 loss for luminance and RGB.
        """
        # Convert to luminance (grayscale)
        luminance_pred = (
            0.2989 * pred[:, 0, :, :]
            + 0.5870 * pred[:, 1, :, :]
            + 0.1140 * pred[:, 2, :, :]
        )
        luminance_target = (
            0.2989 * target[:, 0, :, :]
            + 0.5870 * target[:, 1, :, :]
            + 0.1140 * target[:, 2, :, :]
        )

        # Compute L1 loss for luminance
        luminance_loss = F.l1_loss(
            luminance_pred, luminance_target, reduction=self.reduction
        )

        # Compute standard RGB L1 loss
        rgb_loss = F.l1_loss(pred, target, reduction=self.reduction)

        # Combine losses with weights
        combined_loss = self.loss_weight * (
            self.luminance_weight * luminance_loss
            + (1 - self.luminance_weight) * rgb_loss
        )

        return combined_loss


class GradientGraphLaplacianRegularizer(nn.Module):
    """
    Gradient Graph Laplacian Regularizer (GGLR) for smoothness and edge preservation.

    Enforces smooth illumination maps while maintaining gradients at sharp transitions.
    """

    def __init__(self, loss_weight=0.1):
        """
        Args:
            loss_weight (float): Weight for the GGLR loss term. Default is 0.1.
        """
        super(GradientGraphLaplacianRegularizer, self).__init__()
        self.loss_weight = loss_weight

    @staticmethod
    def compute_gradients(tensor):
        """
        Compute gradients in the x and y directions.

        Args:
            tensor (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tuple[Tensor, Tensor]: Gradients in the x and y directions.
        """
        grad_x = torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:])
        grad_y = torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :])
        return grad_x, grad_y

    def forward(self, illumination_map):
        """
        Compute the GGLR loss.

        Args:
            illumination_map (Tensor): Illumination map of shape (N, C, H, W).

        Returns:
            Tensor: GGLR loss value.
        """
        # Compute gradients
        grad_x, grad_y = self.compute_gradients(illumination_map)

        # Smoothness penalty
        smoothness_loss = torch.mean(grad_x**2) + torch.mean(grad_y**2)

        # Total GGLR loss with weight
        total_loss = self.loss_weight * smoothness_loss
        return total_loss


# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Module):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction


#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss


##########testing##########

# https://github.com/TaoWangzj/LLFormer/blob/main/utils/losses.py

from typing import Tuple


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()

def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d

def get_gaussian_kernel2d(ksize: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d

# class SSIMLoss(nn.Module):
#     def __init__(self, window_size: int = 11, reduction: str = 'mean', max_val: float = 1.0) -> None:
#         super(SSIMLoss, self).__init__()
#         self.window_size: int = window_size
#         self.max_val: float = max_val
#         self.reduction: str = reduction

#         self.window: torch.Tensor = get_gaussian_kernel2d(
#             (window_size, window_size), (1.5, 1.5))
#         self.padding: int = self.compute_zero_padding(window_size)

#         self.C1: float = (0.01 * self.max_val) ** 2
#         self.C2: float = (0.03 * self.max_val) ** 2

#     @staticmethod
#     def compute_zero_padding(kernel_size: int) -> int:
#         """Computes zero padding."""
#         return (kernel_size - 1) // 2

#     def filter2D(
#             self,
#             input: torch.Tensor,
#             kernel: torch.Tensor,
#             channel: int) -> torch.Tensor:
#         return F.conv2d(input, kernel, padding=self.padding, groups=channel)

#     def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
#         # prepare kernel
#         b, c, h, w = img1.shape
#         tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
#         kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

#         # compute local mean per channel
#         mu1: torch.Tensor = self.filter2D(img1, kernel, c)
#         mu2: torch.Tensor = self.filter2D(img2, kernel, c)

#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2

#         # compute local sigma per channel
#         sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
#         sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
#         sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2

#         ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
#             ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

#         loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.

#         if self.reduction == 'mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         elif self.reduction == 'none':
#             pass
#         return loss
    
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss



