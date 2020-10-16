from layer import DistributionConvolution2DGradientKReluMuRandom
import torch.nn as nn
import torch

class DistributionConvolutionModelKReluGradientMuRandom(nn.Module):
  def __init__(self):
    super(DistributionConvolutionModelKReluGradientMuRandom, self).__init__()

    # Conv1
    self.conv1_1 = DistributionConvolution2DGradientKReluMuRandom(filters=32, kernel_size=(3, 3), input_size=(3, 32, 32), padding=1)
    self.batch_normalization1_1 = nn.BatchNorm2d(self.conv1_1.output_size[0])
    conv1_output = self.get_pool_output(self.conv1_1)

    # Conv2
    self.conv2_1 = DistributionConvolution2DGradientKReluMuRandom(filters=64, kernel_size=(3, 3), input_size=conv1_output, padding=1)
    self.batch_normalization2_1 = nn.BatchNorm2d(self.conv2_1.output_size[0])
    conv2_output = self.get_pool_output(self.conv2_1)

    ## Conv3
    self.conv3_1 = DistributionConvolution2DGradientKReluMuRandom(filters=128, kernel_size=(3, 3), input_size=conv2_output, padding=1)
    self.batch_normalization3_1 = nn.BatchNorm2d(self.conv3_1.output_size[0])
    conv3_output = self.get_pool_output(self.conv3_1)

    # Linear
    self.linear_input = conv3_output[0] * conv3_output[1] * conv3_output[2]

    self.linear = nn.Linear(self.linear_input, 10)
    self.pool = nn.MaxPool2d(2, 2)
    self.drop_out = nn.Dropout(p=0.2)

  def forward(self, x):
    x = self.batch_normalization1_1(self.conv1_1.forward(x))
    x = self.drop_out(self.pool(x))

    x = self.batch_normalization2_1(self.conv2_1.forward(x))
    x = self.drop_out(self.pool(x))

    x = self.batch_normalization3_1(self.conv3_1.forward(x))
    x = self.drop_out(self.pool(x))

    x = x.reshape(-1, self.linear_input)
    x = self.linear(x)
    return x

  def get_pool_output(self, layer):
    return (layer.output_size[0], layer.output_size[1] // 2, layer.output_size[2] // 2)