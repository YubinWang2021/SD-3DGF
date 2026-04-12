import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from torch.utils.checkpoint import checkpoint



class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, relu=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion_rate=6, se=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_rate = expansion_rate
        self.se = se

        expansion_channels = in_channels * expansion_rate
        se_channels = max(1, int(in_channels * 0.25))

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            padding = 0

        if expansion_rate != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expansion_channels),
                nn.ReLU()
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=expansion_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=expansion_channels, bias=False),
            nn.BatchNorm2d(expansion_channels),
            nn.ReLU()
        )

        if se:
            self.se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(expansion_channels, se_channels, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(se_channels, expansion_channels, 1, bias=False),
                nn.Sigmoid()
            )

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(expansion_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):
        x = inputs
        if self.expansion_rate != 1:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.se:
            x = self.se_block(x) * x
        x = self.pointwise_conv(x)
        if self.in_channels == self.out_channels and self.stride == 1:
            x = x + inputs
        return x

class EffUNet(nn.Module):
    def __init__(self, in_channels=3, classes=256):
        super().__init__()

        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.down_block_2 = nn.Sequential(
            MBConvBlock(32, 16, 3, 1, expansion_rate=1),
            MBConvBlock(16, 24, 3, 2),
            MBConvBlock(24, 24, 3, 1),
        )

        self.down_block_3 = nn.Sequential(
            MBConvBlock(24, 40, 5, 2),
            MBConvBlock(40, 40, 5, 1),
        )

        self.down_block_4 = nn.Sequential(
            MBConvBlock(40, 80, 3, 2),
            MBConvBlock(80, 80, 3, 1),
            MBConvBlock(80, 80, 3, 1),
            MBConvBlock(80, 112, 5, 1),
        )

        self.down_block_5 = nn.Sequential(
            MBConvBlock(112, 112, 5, 1),
            MBConvBlock(112, 112, 5, 1),
            MBConvBlock(112, 160, 5, 2),
            MBConvBlock(160, 160, 5, 1),
            MBConvBlock(160, 160, 5, 1),
            MBConvBlock(160, 160, 5, 1),
            MBConvBlock(160, 240, 3, 1),
        )

        self.up_block_4 = DecoderBlock(352, 192)
        self.up_block_3 = DecoderBlock(232, 96)
        self.up_block_2 = DecoderBlock(120, 48)
        self.up_block_1a = DecoderBlock(80, 32)

        self.outc = nn.Conv2d(32, classes, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down_block_2(x1)
        x3 = self.down_block_3(x2)
        x4 = self.down_block_4(x3)
        x5 = self.down_block_5(x4)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x4], dim=1)
        x5 = self.up_block_4(x5)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.up_block_3(x5)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x2], dim=1)
        x5 = self.up_block_2(x5)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.up_block_1a(x5)

        output = self.outc(x5)
        return output




class DCEPredictor(nn.Module):
    """
    predictor of DCE representation and DensePose 15 coarse segmentation (14 parts / background)
    """
    def __init__(self, dim_in, dce_chan):
        super(DCEPredictor, self).__init__()
        self.dce_chan = dce_chan
        self.decode = BasicBlock(
            dim_in,
            dce_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            relu=False
        )

    def interp2d(self, size):
        """
        Args:
            tensor_nchw: shape (N, C, H, W)
        Return:
            tensor of shape (N, C, Hout, Wout) by applying the scale factor to H and W
        """
        return nn.functional.interpolate(
            size, scale_factor=self.scale_factor, mode='bilinear', align_corners=False
        )

    def forward(self, head_outputs):
        x = F.interpolate(head_outputs, scale_factor=2, mode='nearest')
        x = self.decode(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        #p_out, dce_out = torch.split(x, [self.p_chan, self.dce_chan], dim=1)
        dce_out = F.normalize(x, 1)
        return dce_out


class DCEModel(nn.Module):
    def __init__(self, dce_chan, backbone='effunet'):
        super(DCEModel, self).__init__()
        self.backbone = EffUNet()
  
        self.DCEPredictor = DCEPredictor(dim_in=256, dce_chan=dce_chan)
        print(sum(p.numel() for p in self.DCEPredictor.parameters()))

    
    def forward(self, img, dp_masks_gt=None, dp_x=None, dp_y=None, dp_I=None, dp_U=None, dp_V=None,
                mode='loss'):
        #print(img.shape)
        if mode == 'loss':
            x = checkpoint(self.backbone, img, use_reentrant=False)
        else:
            x = self.backbone(img)
        #dp_masks_pred, 

        dce_pred = self.DCEPredictor(x)
        if mode == 'feat':
            return dce_pred #dp_masks_pred, dce_pred
        elif mode == 'loss':
            return dce_pred
            # return self.loss(
            #     #dp_masks_pred, 
            #     dp_masks_gt,
            #     dce_pred, dp_x, dp_y, dp_I, dp_U, dp_V, img
            # )
        elif mode == 'eval':
            return dce_pred
            # self.loss(
            #     #dp_masks_pred, 
            #     dp_masks_gt,
            #     dce_pred, dp_x, dp_y, dp_I, dp_U, dp_V, img, evaluate=True
            # )

@register_model
def dce_effunet64(pretrained=True, **kwargs):
    model = DCEModel(dce_chan=64, backbone='effunet')
    return model