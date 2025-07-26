# mmpretrain/models/backbones/convnext_v2_eca.py
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential
from mmpretrain.registry import MODELS
from ..utils import GRN, build_norm_layer
from .convnext import ConvNeXt, ConvNeXtBlock

class ChannelAttention(BaseModule):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(BaseModule):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(BaseModule):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
    
class ConvNeXtBlockCBAM(ConvNeXtBlock):
    def __init__(self, cbam_reduction=16, cbam_kernel_size=7, cbam_enabled=True, **kwargs):
        super().__init__(**kwargs)
        self.cbam_enabled = cbam_enabled
        if cbam_enabled:
            self.cbam = CBAM(self.depthwise_conv.out_channels, reduction=cbam_reduction, kernel_size=cbam_kernel_size)

    def forward(self, x):
        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)
            if self.cbam_enabled and hasattr(self, 'cbam'):
                x = self.cbam(x)  # Aplicar CBAM después de depthwise_conv
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
class ConvNeXtV2CBAM(ConvNeXt):
    def __init__(self, cbam_stages=[1, 2], cbam_reduction=16, cbam_kernel_size=7, **kwargs):
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        norm_cfg = kwargs.get('norm_cfg', dict(type='LN2d', eps=1e-6))
        act_cfg = kwargs.get('act_cfg', dict(type='GELU'))
        linear_pw_conv = kwargs.get('linear_pw_conv', True)
        layer_scale_init_value = kwargs.get('layer_scale_init_value', 1e-6)
        use_grn = kwargs.get('use_grn', False)
        with_cp = kwargs.get('with_cp', False)

        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate
        self.cbam_stages = cbam_stages

        block_idx = 0
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]
            stage = Sequential(*[
                ConvNeXtBlockCBAM(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp,
                    cbam_enabled=(i in cbam_stages),
                    cbam_reduction=cbam_reduction,
                    cbam_kernel_size=cbam_kernel_size
                ) for j in range(depth)
            ])
            block_idx += depth
            self.stages[i] = stage