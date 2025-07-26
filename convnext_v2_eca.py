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

class ECALayer(BaseModule):
    """Efficient Channel Attention (ECA) Layer.
    
    Args:
        channel (int): Number of input channels.
        gamma (float): Hyperparameter for adaptive kernel size calculation. Defaults to 2.
    """
    def __init__(self, channel, gamma=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Calcular k_size adaptativamente
        k_size = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) / gamma)))
        k_size = max(3, k_size if k_size % 2 else k_size + 1)  # Asegurar que sea impar y al menos 3
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)  # [b, c, 1]
        y = self.conv(y.transpose(-1, 1)).transpose(-1, 1)  # [b, c, 1]
        y = self.sigmoid(y)
        return x * y.view(b, c, 1, 1)

class ConvNeXtBlockECA(ConvNeXtBlock):
    """ConvNeXt Block with Efficient Channel Attention (ECA).
    
    This block adds an ECA module after the depthwise convolution.
    
    Args:
        eca_gamma (float): Hyperparameter for ECA kernel size calculation. Defaults to 2.
        **kwargs: Arguments for the base ConvNeXtBlock class.
    """
    def __init__(self, eca_gamma=2, **kwargs):
        super().__init__(**kwargs)
        if eca_gamma is not None: #Porque no se aplica ECA siempre
            self.eca = ECALayer(self.depthwise_conv.out_channels, gamma=eca_gamma)
        else:
            self.eca = None

    def forward(self, x):
        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)
            if self.eca is not None: #Porque no se aplica ECA siempre
                x = self.eca(x)  # Add ECA block after depthwise convolution

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
class ConvNeXtV2ECA(ConvNeXt):
    """ConvNeXt V2 with Efficient Channel Attention (ECA).
    
    Args:
        eca_gamma (float): Hyperparameter for ECA kernel size calculation. Defaults to 2.
        **kwargs: Arguments for the base ConvNeXt class.
    """
    def __init__(self, eca_gamma=2, eca_stages=[1,2], **kwargs): #eca_stages: intentar que sol actúe en algunas capas
        # Extraer los argumentos necesarios de kwargs
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        norm_cfg = kwargs.get('norm_cfg', dict(type='LN2d', eps=1e-6))
        act_cfg = kwargs.get('act_cfg', dict(type='GELU'))
        linear_pw_conv = kwargs.get('linear_pw_conv', True)
        layer_scale_init_value = kwargs.get('layer_scale_init_value', 1e-6)
        use_grn = kwargs.get('use_grn', False)
        with_cp = kwargs.get('with_cp', False)

        # Llamar al inicializador de la clase base con todos los argumentos
        super().__init__(**kwargs)
        
        # Asegurarnos de que drop_path_rate esté definido
        self.drop_path_rate = drop_path_rate

        # Índices de etapas donde aplicar ECA (0-based)
        self.eca_stages = eca_stages
        
        # Calcular dpr
        block_idx = 0
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        ]
        
        # Reemplazar ConvNeXtBlock con ConvNeXtBlockECA en todas las etapas
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]
            stage = Sequential(*[
                ConvNeXtBlockECA(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp,
                    eca_gamma=eca_gamma if i in eca_stages else None #elegir capas
                ) for j in range(depth)
            ])
            block_idx += depth
            self.stages[i] = stage