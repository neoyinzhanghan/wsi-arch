import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

class DeformConvNet(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_dim,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=0,
        output_size=(10, 10)
    ):
        super(DeformConvNet, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define the weight and bias for the deformable convolutional layer
        self.weight = nn.Parameter(torch.Tensor(out_channels, feature_dim, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        # nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

        # Offset and mask for deformable convolution
        self.offset_mask = nn.Conv2d(
            feature_dim,
            3 * kernel_size * kernel_size,  # 2 * kernel_size^2 for offset and 1 * kernel_size^2 for mask
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ) ### IN GENERAL HOW IS THE OFFSET COMPUTED? DEFINITELY NOT A CONVOLUTIONAL LAYER???

        # Define a pooling layer that outputs a fixed size
        self.pool = nn.AdaptiveAvgPool2d(output_size)

        # Calculate the number of features going into the linear layer from the output size
        self.num_features_before_fc = out_channels * output_size[0] * output_size[1]

        # Define a linear layer that dynamically adapts to the output of the pooling layer
        self.fc1 = nn.Linear(self.num_features_before_fc, num_classes)

    def forward(self, x):
        # Generate offset and mask
        offset_mask = self.offset_mask(x)
        offset = offset_mask[:, :2*self.kernel_size**2, :, :]
        mask = torch.sigmoid(offset_mask[:, 2*self.kernel_size**2:, :, :])

        x = F.relu(deform_conv2d(x, self.weight, self.bias, offset, mask, stride=self.stride, padding=self.padding))

        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        x = self.fc1(x)

        return x
