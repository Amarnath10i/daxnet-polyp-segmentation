# -*- coding: utf-8 -*-
"""BAMPolyp reference implementation, copied verbatim from
repo_original/bampolyp.py (github.com/farhad324/BAMPolyp @ bebb67b).

Only edits: imports gathered here, `pretrained=True` -> the equivalent
IMAGENET1K_V1 weights enum (the kwarg is removed in current torchvision).
Do NOT "fix" anything in this file: it is the baseline being reproduced.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Axial Mamba Module

class AxialMambaBlock(nn.Module):
    """
    Axial Mamba block: applies Mamba mixing along height and width axes separately.
    """
    def __init__(self, in_channels, out_channels, args):
        super(AxialMambaBlock, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Projections for height-axis Mamba
        self.height_down = nn.Conv2d(in_channels, args.model_input_dims, kernel_size=1, bias=False)
        self.height_mamba = MambaBlock(args)
        self.height_up = nn.Conv2d(args.model_input_dims, out_channels, kernel_size=1, bias=False)
        self.height_norm = nn.BatchNorm2d(out_channels)

        # Projections for width-axis Mamba
        self.width_down = nn.Conv2d(in_channels, args.model_input_dims, kernel_size=1, bias=False)
        self.width_mamba = MambaBlock(args)
        self.width_up = nn.Conv2d(args.model_input_dims, out_channels, kernel_size=1, bias=False)
        self.width_norm = nn.BatchNorm2d(out_channels)

        self.skip_proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        b, c, h, w = x.shape

        h_proj = self.height_down(x)
        h_seq = rearrange(h_proj, 'b d h w -> (b w) h d')
        h_seq = self.height_mamba(h_seq)
        h_mixed = rearrange(h_seq, '(b w) h d -> b d h w', b=b, w=w)
        h_out = self.height_up(h_mixed)
        h_out = self.height_norm(h_out)

        w_proj = self.width_down(x)
        w_seq = rearrange(w_proj, 'b d h w -> (b h) w d')
        w_seq = self.width_mamba(w_seq)
        w_mixed = rearrange(w_seq, '(b h) w d -> b d h w', b=b, h=h)
        w_out = self.width_up(w_mixed)
        w_out = self.width_norm(w_out)

        skip = self.skip_proj(x)
        out = h_out + w_out + skip
        return out

# Basic Double Convolution Block

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def selective_scan(u, delta, A, B, C, D):
    A = torch.clamp(A, min=-5.0, max=5.0)
    dA = torch.einsum('bld,dn->bldn', delta, A)
    dB_u = torch.einsum('bld,bld,bln->bldn', delta, u, B)
    dA_cumsum = torch.cat([dA[:, 1:], torch.zeros_like(dA[:, :1])], dim=1)
    dA_cumsum = torch.flip(dA_cumsum, dims=[1])
    dA_cumsum = torch.cumsum(dA_cumsum, dim=1)
    dA_cumsum = torch.clamp(dA_cumsum, max=15.0)
    dA_cumsum = torch.exp(dA_cumsum)
    dA_cumsum = torch.flip(dA_cumsum, dims=[1])
    x = dB_u * dA_cumsum
    x = torch.cumsum(x, dim=1) / (dA_cumsum + 1e-6)
    y = torch.einsum('bldn,bln->bld', x, C)
    return y + u * D


class MambaBlock(nn.Module):
    def __init__(self, args):
        super(MambaBlock, self).__init__()
        self.args = args
        self.in_proj = nn.Linear(args.model_input_dims, args.model_internal_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            args.model_internal_dim,
            args.model_internal_dim,
            kernel_size=args.conv_kernel_size,
            padding=args.conv_kernel_size - 1,
            groups=args.model_internal_dim
        )
        self.x_proj = nn.Linear(args.model_internal_dim, args.delta_t_rank + args.model_states * 2, bias=False)
        self.delta_proj = nn.Linear(args.delta_t_rank, args.model_internal_dim)

        A_vals = torch.arange(1, args.model_states + 1).float() / args.model_states * 3
        self.A_log = nn.Parameter(torch.log(repeat(A_vals, 'n -> d n', d=args.model_internal_dim)))
        self.D = nn.Parameter(torch.ones(args.model_internal_dim))
        self.out_proj = nn.Linear(args.model_internal_dim, args.model_input_dims, bias=args.dense_use_bias)

    def forward(self, x):
        if self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        b, l, d = x.shape
        x_and_res = self.in_proj(x)
        x1, res = x_and_res.chunk(2, dim=-1)

        x1 = rearrange(x1, 'b l d -> b d l')
        x1 = self.conv1d(x1)[..., :l]
        x1 = rearrange(x1, 'b d l -> b l d')
        x1 = F.silu(x1)

        A = -torch.exp(torch.clamp(self.A_log, min=-5, max=5))
        D = self.D
        x_dbl = self.x_proj(x1)
        delta, B, C = torch.split(
            x_dbl,
            [self.args.delta_t_rank, self.args.model_states, self.args.model_states],
            dim=-1
        )
        delta = F.softplus(self.delta_proj(delta))

        y = selective_scan(x1, delta, A, B, C, D)
        y = y * F.silu(res)
        return self.out_proj(y)


class ResidualBlock(nn.Module):
    def __init__(self, args):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm(args.model_input_dims)
        self.mixer = MambaBlock(args)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.norm2 = nn.LayerNorm(args.model_input_dims)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = self.dropout(x)
        x = residual + x
        return self.norm2(x)

class ModelArgs:
    def __init__(self):
        self.model_input_dims = 96
        self.model_states = 96
        self.projection_expand_factor = 2
        self.conv_kernel_size = 4
        self.dense_use_bias = False
        self.dropout_rate = 0.2
        self.num_layers = 6
        self.num_classes = 2
        self.model_internal_dim = self.projection_expand_factor * self.model_input_dims
        self.delta_t_rank = math.ceil(self.model_input_dims / 16)

class IoULoss(nn.Module):

    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):

        inputs_soft = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (2, 3)
        intersection = (inputs_soft * targets_one_hot).sum(dim=dims)
        union = (inputs_soft + targets_one_hot - inputs_soft * targets_one_hot).sum(dim=dims) + self.eps

        iou = (intersection + self.eps) / union
        iou_loss = 1.0 - iou
        return iou_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):

        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        intersection = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
        cardinality = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):

        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        focal_term = ((1 - pt) ** self.gamma) * logpt
        loss = -focal_term
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, inputs, targets):

        inputs_soft = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (2, 3)
        TP = (inputs_soft * targets_one_hot).sum(dim=dims)
        FP = (inputs_soft * (1 - targets_one_hot)).sum(dim=dims)
        FN = ((1 - inputs_soft) * targets_one_hot).sum(dim=dims)

        tversky_index = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        return (1.0 - tversky_index).mean()


class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):

        inputs_soft = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (2, 3)
        intersection = (inputs_soft * targets_one_hot).sum(dim=dims)
        union = inputs_soft.sum(dim=dims) + targets_one_hot.sum(dim=dims) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return (1.0 - iou).mean()


class BoundaryLoss(nn.Module):

    def __init__(self, eps: float = 1e-6):
        super(BoundaryLoss, self).__init__()
        self.eps = eps

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:

        m = mask.unsqueeze(1)
        inv = 1.0 - m
        eroded_inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - eroded_inv
        b = m - eroded
        return b.squeeze(1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inputs.shape
        probs = F.softmax(inputs, dim=1)
        gt = F.one_hot(targets, num_classes=C).permute(0,3,1,2).float()

        loss = 0.0
        for c in range(C):
            p_c = probs[:, c, :, :]
            g_c = gt[:, c, :, :]
            b_pred = self._get_boundary(p_c)
            b_gt   = self._get_boundary(g_c)
            inter = (b_pred * b_gt).sum(dim=(1,2))
            union = b_pred.sum(dim=(1,2)) + b_gt.sum(dim=(1,2))
            dice_b = (2.0 * inter + self.eps) / (union + self.eps)
            loss += (1.0 - dice_b).mean()
        return loss / C


class ComboLoss(nn.Module):
    """
    Weighted sum of BCE, Dice, Focal, Tversky, IoU, and Boundary losses.
    Loss weights are dataset-specific and detailed in the README and paper.
    """
    def __init__(self,
                 bce_weight=0.15,
                 dice_weight=0.25,
                 focal_weight=0.2,
                 tversky_weight=0.15,
                 iou_weight=0.15,
                 boundary_weight=0.1,
                 focal_gamma=2.0,
                 tversky_alpha=0.5,
                 tversky_beta=0.5):
        super(ComboLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight

        self.ce_loss       = nn.CrossEntropyLoss()
        self.dice_loss     = DiceLoss()
        self.focal_loss    = FocalLoss(gamma=focal_gamma, reduction='mean')
        self.tversky_loss  = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.iou_loss      = IoULoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, inputs, targets):
        loss_bce      = self.ce_loss(inputs, targets)
        loss_dice     = self.dice_loss(inputs, targets)
        loss_focal    = self.focal_loss(inputs, targets)
        loss_tversky  = self.tversky_loss(inputs, targets)
        loss_iou      = self.iou_loss(inputs, targets)
        loss_boundary = self.boundary_loss(inputs, targets)

        total = (
            self.bce_weight      * loss_bce
            + self.dice_weight   * loss_dice
            + self.focal_weight  * loss_focal
            + self.tversky_weight* loss_tversky
            + self.iou_weight    * loss_iou
            + self.boundary_weight * loss_boundary
        )
        return total


class DeepSupervisionLoss(nn.Module):

    def __init__(self, main_weight=0.6, deep2_weight=0.2, deep3_weight=0.2):
        super(DeepSupervisionLoss, self).__init__()
        self.main_weight = main_weight
        self.deep2_weight = deep2_weight
        self.deep3_weight = deep3_weight
        self.criterion = ComboLoss(
            bce_weight=0.2,
            dice_weight=0.3,
            focal_weight=0.2,
            tversky_weight=0.2,
            iou_weight=0.3,
            focal_gamma=2.0,
            tversky_alpha=0.5,
            tversky_beta=0.5
        )

    def forward(self, outputs, target):

        main_out, deep2, deep3, *_ = outputs
        loss_main = self.criterion(main_out, target)
        loss_deep2 = self.criterion(deep2, target)
        loss_deep3 = self.criterion(deep3, target)
        total_loss = (
            self.main_weight * loss_main +
            self.deep2_weight * loss_deep2 +
            self.deep3_weight * loss_deep3
        )
        return total_loss


class MambaUNetWithAxialMambaEfficientNetB0(nn.Module):
    def __init__(self, args, in_channels=3):
        super().__init__()
        self.args = args

        #EfficientNetB0 pretrained
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # == pretrained=True

        if in_channels != 3:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
                backbone.features[0][1],
                backbone.features[0][2],
            )
        else:
            self.stem = backbone.features[0]

        # Encoder stages
        self.stage1 = backbone.features[1]
        self.stage2 = backbone.features[2]
        self.stage3 = backbone.features[3]
        self.stage4 = backbone.features[4]
        self.stage5 = nn.Sequential(
            backbone.features[5],
            backbone.features[6],
            backbone.features[7],
        )

        # Axial Mamba bottleneck
        self.axial_mamba = AxialMambaBlock(320, 320, args)


        self.bridge_down = nn.Conv2d(320, args.model_input_dims, kernel_size=1)
        self.mamba_blocks = nn.Sequential(*[ResidualBlock(args) for _ in range(args.num_layers)])
        self.bridge_up = nn.Conv2d(args.model_input_dims, 320, kernel_size=1)

        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(320, 160, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(160 + 80, 160)

        self.upconv3 = nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(80 + 40, 80)

        self.upconv2 = nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(40 + 24, 40)

        self.upconv1 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(24 + in_channels, 24)

        # Final conv and deep supervision
        self.final_conv = nn.Conv2d(24, args.num_classes, kernel_size=1)
        self.deep_sup4 = nn.Conv2d(160, args.num_classes, kernel_size=1)
        self.deep_sup3 = nn.Conv2d(80, args.num_classes, kernel_size=1)
        self.deep_sup2 = nn.Conv2d(40, args.num_classes, kernel_size=1)

    def forward(self, x, return_deep=False):
        x0 = self.stem(x)
        e1 = self.stage1(x0)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        e5 = self.stage5(e4)

        axial_out = self.axial_mamba(e5)
        bottleneck_feat = axial_out

        mamba_in = self.bridge_down(bottleneck_feat)
        b, c, h, w = mamba_in.shape
        m_in = mamba_in.permute(0, 2, 3, 1).reshape(b, h * w, c)
        m_out = self.mamba_blocks(m_in)
        m_out = m_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        mamba_out = self.bridge_up(m_out)

        bottleneck_out = self.bottleneck_refine(mamba_out + bottleneck_feat)

        d4 = self.upconv4(bottleneck_out)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))
        ds4 = self.deep_sup4(d4)

        d3 = self.upconv3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        ds3 = self.deep_sup3(d3)

        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        ds2 = self.deep_sup2(d2)

        d1 = self.upconv1(d2)
        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d1, x], dim=1))

        out = self.final_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        if return_deep:
            ds2_up = F.interpolate(ds2, size=x.shape[2:], mode='bilinear', align_corners=False)
            ds3_up = F.interpolate(ds3, size=x.shape[2:], mode='bilinear', align_corners=False)
            ds4_up = F.interpolate(ds4, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, ds2_up, ds3_up, ds4_up

        return out



def build_bampolyp(**_):
    return MambaUNetWithAxialMambaEfficientNetB0(ModelArgs())
