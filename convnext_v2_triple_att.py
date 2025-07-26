import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential
from mmpretrain.registry import MODELS
from ..utils import GRN, build_norm_layer
from .convnext import ConvNeXt, ConvNeXtBlock

class TripletAttention(BaseModule):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        #print(f"TripletAttention input shape: {x.shape}, expected channels: {self.in_channels}")
        if c != self.in_channels:
            #print(f"Warning: Adjusting convolution from {self.in_channels} to {c} channels.")
            self.conv = nn.Conv2d(c, c, kernel_size=1, bias=False).to(x.device)
            self.bn = nn.BatchNorm2d(c).to(x.device)

        # Rama 1: Atención canal-espacial (H x W)
        x_perm1 = x.transpose(1, 2)  # [B, H, C, W]
        x_proj1 = self.conv(x)  # Proyectar en la forma original [B, C, H, W]
        act1 = torch.mean(x_proj1, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        act1 = self.sigmoid(act1)
        out1 = x * act1

        # Rama 2: Atención canal-espacial (W x H)
        x_perm2 = x.transpose(2, 3)  # [B, C, W, H]
        x_proj2 = self.conv(x)  # Proyectar en la forma original [B, C, H, W]
        act2 = torch.mean(x_proj2, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        act2 = self.sigmoid(act2)
        out2 = x * act2

        # Rama 3: Atención canal
        x_perm3 = x  # [B, C, H, W]
        x_proj3 = self.conv(x_perm3)  # [B, C, H, W] después de conv
        act3 = torch.mean(x_proj3, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        act3 = self.sigmoid(act3)
        out3 = x * act3

        # Combinación residual
        out = out1 + out2 + out3 + x
        return out

class ConvNeXtBlockTriplet(ConvNeXtBlock):
    def __init__(self, triplet_enabled=True, **kwargs):
        super().__init__(**kwargs)
        self.triplet_enabled = triplet_enabled
        if triplet_enabled:
            self.triplet_attention = TripletAttention(self.depthwise_conv.out_channels)

    def forward(self, x):
        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)
            if self.triplet_enabled and hasattr(self, 'triplet_attention'):
                #print(f"Channels before Triplet Attention: {x.shape[1]}")
                x = self.triplet_attention(x)  # Aplicar Triplet Attention después de depthwise_conv
            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

@MODELS.register_module()
class ConvNeXtV2Triplet(ConvNeXt):
    def __init__(self, triplet_stages=[2], **kwargs):
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        norm_cfg = kwargs.get('norm_cfg', dict(type='LN2d', eps=1e-6))
        act_cfg = kwargs.get('act_cfg', dict(type='GELU'))
        linear_pw_conv = kwargs.get('linear_pw_conv', True)
        layer_scale_init_value = kwargs.get('layer_scale_init_value', 1e-6)
        use_grn = kwargs.get('use_grn', False)
        with_cp = kwargs.get('with_cp', False)

        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate
        self.triplet_stages = triplet_stages

        block_idx = 0
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]
            stage = Sequential(*[
                ConvNeXtBlockTriplet(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp,
                    triplet_enabled=(i in triplet_stages)
                ) for j in range(depth)
            ])
            block_idx += depth
            self.stages[i] = stage