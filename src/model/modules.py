"""Module providing classes to define the modules of the network."""
from typing import Tuple
import torch
from torch import nn

import torchvision.models  as models
import torch.nn.functional as F

class DCNN(nn.Module):
    """A Deep Convolutional Neural Network (DCNN) module based on MobileNetV3
    architecture.

    This class implements a DCNN module with four sequential blocks, where the
    first block is the input convolutional layer, and the next three blocks
    are sequences of MobileNetV3 bottlenecks.

    Attributes
    ----------
        input_convolution : Sequential
            A sequential block consisting of a 2D convolutional layer,
            followed by batch normalization and Hardswish activation.
        bottlenecks_sequential_1 : Sequential 
            A sequential block consisting of a MobileNetV3 bottleneck
            layer.
        bottlenecks_sequential_2 : Sequential
            A sequential block consisting of two MobileNetV3 bottleneck
            layers.
        bottlenecks_sequential_3 : Sequential
            A sequential block consisting of three MobileNetV3 bottleneck
            layers.
        bottlenecks_sequential_4 : Sequential
            A sequential block consisting of six MobileNetV3 bottleneck
            layers.

    Methods
    -------
    forward(x: FloatTensor) -> (
    FloatTensor, FloatTensor, FloatTensor, FloatTensor)
        Forward pass of the DCNN module. Takes a tensor as input and 
        returns four tensors: `f1`, `f2`, `f3`, and `out`, which
        represent the intermediate feature maps after each of the
        three first bottleneck blocks and the final output of the module,
        respectively.
    """
    def __init__(self) -> None:
        """Initialize the DCNN module."""
        super().__init__()

        backbone_model   = models.mobilenet_v3_large(pretrained=True)
        base_layers      = nn.ModuleList(list(backbone_model.features))
        
        # Input sequential block.
        self.input_convolution = nn.Sequential( base_layers[0], )
        # First MobileNetV3 bottlenecks sequential block.
        self.bottlenecks_sequential_1 =  nn.Sequential(base_layers[1],)

        # Second MobileNetV3 bottlenecks sequential block.
        self.bottlenecks_sequential_2 =  nn.Sequential(*base_layers[2:4])

        # Third MobileNetV3 bottlenecks sequential block.
        self.bottlenecks_sequential_3 =  nn.Sequential(*base_layers[4:7])
        
        # Last MobileNetV3 bottlenecks sequential block.
        self.bottlenecks_sequential_4 =  nn.Sequential(*base_layers[7:] )

    def forward(self, x: torch.FloatTensor) -> Tuple[
        torch.FloatTensor, torch.FloatTensor,
        torch.FloatTensor, torch.FloatTensor]:
        """Forward pass of the DCNN block.

        Parameters
        ----------
        x : FloatTensor
            The input tensor.

        Returns
        -------
        FloatTensor
            The first shallow intermediate feature `f1`.
        FloatTensor
            The second shallow intermediate feature `f2`.
        FloatTensor
            The third shallow intermediate feature `f3`.
        FloatTensor
            The output tensor.
        """
        # Apply the initial convolution.
        out = self.input_convolution(x)
        # Apply the series of bottleneck blocks and get the intermediate
        # feature results.
        f1 = self.bottlenecks_sequential_1(out)
        f2 = self.bottlenecks_sequential_2(f1)
        f3 = self.bottlenecks_sequential_3(f2)
        out = self.bottlenecks_sequential_4(f3)
        
        return f1, f2, f3, out

class ASPP(nn.Module):
    """An Atrous Spatial Pyramid Pooling (ASPP) module based on DeepLabV3+
    architecture.

    This class applies a series of atrous convolutions on an input tensor
    followed by global average pooling to extract multi-scale contextual
    information.

    Attributes
    ----------
    standard_convolution : Sequential
        A sequential block consisting of a convolution layer, batch
        normalization layer, and ReLU activation function.
    atrous_convolution_1 : Sequential
        A sequential block performing an atrous convolution with a dilation
        rate of 6.
    atrous_convolution_2 : Sequential
        A sequential block performing an atrous convolution with a dilation
        rate of 12.
    atrous_convolution_3 : Sequential
        A sequential block performing an atrous convolution with a dilation
        rate of 18.
    global_average_pooling : Sequential 
        A sequential block performing global average pooling.
    final_convolution : Sequential
        A final sequential block consisting of a convolution layer followed
        by ReLU activation and dropout.

    Methods
    -------
    forward(x: FloatTensor) -> FloatTensor
        Computes the forward pass of the ASPP module. Takes an input
        tensor and applies the series of atrous convolutions and global
        average pooling, concatenates the resulting tensors, and applies
        a final convolution.
    """
    def __init__(self) -> None:
        """Initialize the ASPP module."""
        super().__init__()
        # Set the number of input and output channels.
        in_channels  = 960
        out_channels = 256

        # Set the standard convolution sequential block.
        self.standard_convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
        )

        # Set the atrous convolution sequential blocks.
        self.atrous_convolution_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      bias=False, dilation=6, padding=6),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
        )
        self.atrous_convolution_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      bias=False, dilation=12, padding=12),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
        )
        self.atrous_convolution_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      bias=False, dilation=18, padding=18),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
        )

        # Set the global average pooling sequential block.
        self.global_average_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
        )

        # Set the output convolution sequential block.
        self.final_convolution = nn.Sequential(
            nn.Conv2d(out_channels*5, out_channels, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass of the ASPP block.

        Parameters
        ----------
        x : FloatTensor
            The input tensor.

        Returns
        -------
        FloatTensor
            The output tensor.
        """
        # Apply the series of atrous convolutions on the input.
        out1 = self.standard_convolution(x)
        out2 = self.atrous_convolution_1(x)
        out3 = self.atrous_convolution_2(x)
        out4 = self.atrous_convolution_3(x)

        # Apply Global Average Pooling on the input and replicate
        # spatially the result.
        out5 = self.global_average_pooling(x)
        out5 = F.interpolate(out5, size=x.shape[-2:], mode='bilinear',
                             align_corners=False)

        # Concatenate the tensors of the atrous convolutions and the pooling
        # operation and apply a final convolution.
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        f4 = self.final_convolution(out)

        return f4

class Encoder(nn.Module):
    """The encoder module of the network.
    It is composed of a Deep Convolutional Neural Network (DCNN) module and
    an Atrous Spatial Pyramid Pooling (ASPP) module.

    Attributes
    ----------
    dcnn : DCNN
        The DCNN module of the encoder.
    aspp : ASPP
        The ASPP module of the encoder.
        
    Methods
    -------
    forward(x: FloatTensor) -> (
    FloatTensor, FloatTensor, FloatTensor, FloatTensor)
        Forward pass of the encoder module. Takes a tensor as input and 
        returns four tensors: `f1`, `f2`, `f3`, and `f4`, which
        represent the intermediate feature maps after each of the
        three first bottleneck blocks and the final output of the encoder,
        respectively.
    """
    def __init__(self) -> None:
        """Initialize the encoder module."""
        super().__init__()
        # Set the DCNN module of the encoder.
        self.dcnn = DCNN()
        # Set the ASPP module of the encoder.
        self.aspp = ASPP()

    def forward(self, x: torch.FloatTensor) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, 
        torch.FloatTensor, torch.FloatTensor]:
        """Forward pass of the encoder block.

        Parameters
        ----------
        x : FloatTensor
            The input tensor.

        Returns
        -------
        FloatTensor
            The first shallow intermediate feature `f1`.
        FloatTensor
            The second shallow intermediate feature `f2`.
        FloatTensor
            The third shallow intermediate feature `f3`.
        FloatTensor
            The output of the encoder's ASPP module: `f4`.
        """
        # Get the first 3 shallow intermediate features and the
        # DCNN module output.
        f1, f2, f3, out = self.dcnn(x)
        # Get the ASPP module output.
        f4 = self.aspp(out)
        return f1, f2, f3, f4

class Decoder(nn.Module):
    """The decoder module of the network.
    It is composed of a Deep Convolutional Neural Network (DCNN) module and
    an Atrous Spatial Pyramid Pooling (ASPP) module.

    Attributes
    ----------
    convolution_f1 : Sequential
        The sequential block of convolutional layers for the shallow
        intermediate feature `f1`.
    convolution_f2 : Sequential
        The sequential block of convolutional layers for the shallow
        intermediate feature `f2`.
    convolution_f3 : Sequential
        The sequential block of convolutional layers for the shallow
        intermediate feature `f3`.
    upsample_f1 : Upsample
        The upsampling layer for the shallow intermediate feature `f1`.
    upsample_f3 : Upsample
        The upsampling layer for the shallow intermediate feature `f3`.
    upsample_f4 : Upsample
        The upsampling layer for the shallow intermediate feature `f4`.
    final_convolution : Sequential
        The final sequential block of convolutional layers for the
        channel-wise concatenated intermediate features.
    final_upsample : Upsample
        The final upsampling layer for the concatenated intermediate
        features.
        
    Methods
    -------
    forward(f1: FloatTensor, f2: FloatTensor, f3: FloatTensor,
    f4: FloatTensor) ->  FloatTensor
        Forward pass of the decoder module.
    """
    def __init__(self, target_size: Tuple[int, int]) -> None:
        """Initialize the decoder module.

        Parameters
        ----------
        target_size : (int, int)
            The size of the output tensor.
        """
        super().__init__()
        # Set the output channels of each convolution applied to the
        # shallow intermediate features.
        out_channels = 256
        # Set the size of the shallow feature `f2`, which is the target
        # size of the upsampling of the other shallow features
        f2_size = (128, 128)

        # Set the convolution sequential blocks to assign the desired equal
        # output channels to each shallow feature.
        self.convolution_f1 = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
        )
        self.convolution_f2 = nn.Sequential(
            nn.Conv2d(24, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
        )
        self.convolution_f3 = nn.Sequential(
            nn.Conv2d(40, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
        )

        # Set the upsampling operation layers to change the shape of
        # each shallow feature to the size of `f2`.
        self.upsample_f1 = nn.Upsample(f2_size, mode='bilinear',
                                       align_corners=False)
        self.upsample_f3 = nn.Upsample(f2_size, mode='bilinear',
                                       align_corners=False)
        self.upsample_f4 = nn.Upsample(f2_size, mode='bilinear',
                                       align_corners=False)

        # Set the final convolution sequential block.
        self.final_convolution = nn.Sequential(
            nn.Conv2d(out_channels * 4, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 2, kernel_size=1, stride=1),
        )

        # Set the final upsample block to change the image to the
        # target size.
        self.final_upsample = nn.Upsample(target_size, mode='bilinear',
                                          align_corners=False)

    def forward(
        self, f1: torch.FloatTensor, f2: torch.FloatTensor,
        f3: torch.FloatTensor, f4: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass of the decoder block.

        f1 : FloatTensor
            The first shallow intermediate feature.
        f2 : FloatTensor
            The second shallow intermediate feature.
        f3 : FloatTensor
            The third shallow intermediate feature.
        f4 : FloatTensor
            The output of the encoder.

        Returns
        -------
        FloatTensor
            The output tensor.
        """
        # Apply the convolution operations to assign the desired equal
        # output channels to each input feature.
        out_f1 = self.convolution_f1(f1)
        out_f2 = self.convolution_f2(f2)
        out_f3 = self.convolution_f3(f3)

        # Upsample each feature to the size of `f2`.
        out_f1 = self.upsample_f1(out_f1)
        out_f3 = self.upsample_f3(out_f3)
        out_f4 = self.upsample_f4(f4)

        # Concatenate the results channel wise.
        out = torch.cat([out_f1, out_f2, out_f3, out_f4], dim=1)

        # Apply the final convolution and upsample.
        out = self.final_convolution(out)
        out = self.final_upsample(out)

        return out
