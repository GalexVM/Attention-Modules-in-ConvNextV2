import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential
from mmpretrain.registry import MODELS
from ..utils import GRN, build_norm_layer
from .convnext import ConvNeXt, ConvNeXtBlock

class SEBlock(BaseModule):
    """Squeeze-and-Excitation Block.
    
    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for the squeeze operation. Defaults to 16.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class ConvNeXtBlockSE(ConvNeXtBlock):
    """ConvNeXt Block with Squeeze-and-Excitation.
    
    This block adds an SE module after the depthwise convolution when enabled.
    """
    def __init__(self, se_enabled=True, se_reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.se_enabled = se_enabled
        if se_enabled:
            self.se = SEBlock(self.depthwise_conv.out_channels, reduction=se_reduction)

    def forward(self, x):
        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)
            if self.se_enabled and hasattr(self, 'se'):
                x = self.se(x)  # Add SE block after depthwise convolution

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
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
class ConvNeXtV2SE(ConvNeXt):
    """ConvNeXt V2 with Squeeze-and-Excitation.
    
    Args:
        se_stages (list): List of stages where SE is applied. Defaults to [0, 1, 2, 3].
        se_reduction (int): Reduction ratio for the SE block. Defaults to 16.
        **kwargs: Arguments for the base ConvNeXt class.
    """
    def __init__(self, se_stages=[0, 1, 2, 3], se_reduction=16, **kwargs):
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        norm_cfg = kwargs.get('norm_cfg', dict(type='LN2d', eps=1e-6))
        act_cfg = kwargs.get('act_cfg', dict(type='GELU'))
        linear_pw_conv = kwargs.get('linear_pw_conv', True)
        layer_scale_init_value = kwargs.get('layer_scale_init_value', 1e-6)
        use_grn = kwargs.get('use_grn', False)
        with_cp = kwargs.get('with_cp', False)

        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate
        self.se_stages = se_stages

        block_idx = 0
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]
            stage = Sequential(*[
                ConvNeXtBlockSE(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp,
                    se_enabled=(i in se_stages),
                    se_reduction=se_reduction
                ) for j in range(depth)
            ])
            block_idx += depth
            self.stages[i] = stage