import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import segmentation_models_pytorch as smp




# ============================================================
#                     ENCODER (ResNet50)
# ============================================================
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        resnet = models.resnet50(pretrained=pretrained)

        # Keep only encoder layers
        self.encoder0 = nn.Sequential(
            resnet.conv1, 
            resnet.bn1, 
            resnet.relu
        )                     # -> 64, 112×112

        self.pool = resnet.maxpool               # -> 64, 56×56
        self.encoder1 = resnet.layer1            # -> 256, 56×56
        self.encoder2 = resnet.layer2            # -> 512, 28×28
        self.encoder3 = resnet.layer3            # -> 1024,14×14
        self.encoder4 = resnet.layer4            # -> 2048, 7×7 (Bottleneck)

    def forward(self, x):
        x0 = self.encoder0(x)                    # 112×112
        x1 = self.encoder1(self.pool(x0))        # 56×56
        x2 = self.encoder2(x1)                   # 28×28
        x3 = self.encoder3(x2)                   # 14×14
        x4 = self.encoder4(x3)                   # 7×7 (Bottleneck)

        # return ALL SKIP CONNECTIONS
        return x0, x1, x2, x3, x4



# ============================================================
#                     DECODER (UNet Decoder)
# ============================================================
class ConvBlock(nn.Module):
    """Basic UNet convolution block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)



class UpBlock(nn.Module):
    """Upsampling block with skip connection."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)  # Upsample

        # If mismatch size due to rounding
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)  # SKIP CONNECTION
        return self.conv(x)



class UNetDecoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()

        # Channels of encoder outputs:
        # x4: 2048, x3: 1024, x2: 512, x1: 256, x0: 64
        self.up4 = UpBlock(2048, 1024, 1024)
        self.up3 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up1 = UpBlock(256, 64, 128)

        self.final = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x0, x1, x2, x3, x4):
        d4 = self.up4(x4, x3)  # 14×14
        d3 = self.up3(d4, x2)  # 28×28
        d2 = self.up2(d3, x1)  # 56×56
        d1 = self.up1(d2, x0)  # 112×112

        # Final upsample to original size (224×224)
        d1 = F.interpolate(d1, size=x0.shape[-2:], mode="bilinear", align_corners=False)

        return self.final(d1)



# ============================================================
#                   FULL UNET (Encoder + Decoder)
# ============================================================
class ResNet50UNet(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=True)
        self.decoder = UNetDecoder(out_channels)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)     # Encoder path
        out = self.decoder(x0, x1, x2, x3, x4)   # Decoder path
        return out


# ------------------------------------------------------------
#  SMP-based UNet (ResNet50 encoder) for image->image output
#  Uses segmentation_models_pytorch to build a UNet where the
#  final activation is linear (suitable for regression/reconstruction)
# ------------------------------------------------------------
class SMPResNet50UNet(nn.Module):
    """Wrapper around `segmentation_models_pytorch.Unet` configured
    for image-to-image regression (no final activation).

    Args:
        in_channels (int): number of input image channels (default 3).
        out_channels (int): number of output channels (default 3).
        encoder_weights (str|None): weights for encoder, e.g. 'imagenet' or None.
    """
    def __init__(self, in_channels=3, out_channels=3, encoder_weights='imagenet'):
        super().__init__()
        # segmentation_models_pytorch uses `classes` for number of output channels
        self.model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,  # linear output for reconstruction/regression
        )

    def forward(self, x):
        return self.model(x)


class SMPDenseNet121UNet(nn.Module):
    """UNet using a DenseNet-121 encoder via segmentation_models_pytorch.

    Args:
        in_channels (int): number of input image channels (default 3).
        out_channels (int): number of output channels (default 3).
        encoder_weights (str|None): weights for encoder, e.g. 'imagenet' or None.
    """
    def __init__(self, in_channels=3, out_channels=3, encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='densenet121',
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)




# class SMPDenseNet121UNet(nn.Module):
#     """UNet using a DenseNet-121 encoder via segmentation_models_pytorch.

#     Args:
#         in_channels (int): number of input image channels (default 3).
#         out_channels (int): number of output channels (default 3).
#         encoder_weights (str|None): weights for encoder, e.g. 'imagenet' or None.
#     """
#     def __init__(self, in_channels=3, out_channels=3, encoder_weights='imagenet'):
#         super().__init__()
#         self.model = smp.Unet(
#             encoder_name='densenet121',
#             encoder_weights=encoder_weights,
#             in_channels=in_channels,
#             classes=out_channels,
#             activation=None,
#         )

#     def forward(self, x):
#         return self.model(x)



class GenePredictorFromDenseNetUNet(nn.Module):
    def __init__(self, encoder, req_grad, num_layers, n_genes=460):
        super().__init__()
        self.encoder = encoder
        
        # DenseNet121 last channel size = 1024 (for UNet it may be 512)
        for param in self.encoder.parameters():
            param.requires_grad = req_grad

        if req_grad == False:
            children = list(self.encoder.children())
            for layer in children[-num_layers:]:
                param.requires_grad = True

        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # simple MLP
        self.fc = nn.Linear(1024, n_genes)

    def forward(self, x):
        feats = self.encoder(x)[-1]     # deepest feature
        pooled = self.gap(feats).squeeze(-1).squeeze(-1)  # (B, C)
        pred = self.fc(pooled)
        return pred