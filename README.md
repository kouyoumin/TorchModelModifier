# TorchModelModifier
Modify model input, output, and dimension.
In addition to model architecture, weight is also recalculated to approximate original output.
---
## modify_model_in
Finds first conv layer in the model and change in_channels as you need. Useful to convert a RGB model (in_channels=3) to grayscale model (in_channels=1).
### Highlights:
- Reduces computation when you have grayscale input.
- Output is (almost) equivalent to original model (when channel copy applied to grayscale data).
### Limitations:
- Does not work with Inception3 because its transform_input always transform images into 3-channel.

---
## modify_model_out
Finds classifier in the model and change num_output as you need. Can alse extract specific subset of original classes.

---
## modify_model_dim
Change all 2d layers to specified dimension.

### Supported layers:
- nn.Conv1d
- nn.Conv2d
- nn.Conv3d
- nn.LazyConv1d
- nn.LazyConv2d
- nn.LazyConv3d
- nn.ConvTranspose1d
- nn.ConvTranspose2d
- nn.ConvTranspose3d
- nn.LazyConvTranspose1d
- nn.LazyConvTranspose2d
- nn.LazyConvTranspose3d
- nn.MaxPool1d
- nn.MaxPool2d
- nn.MaxPool3d
- nn.AvgPool1d
- nn.AvgPool2d
- nn.AvgPool3d
- nn.AdaptiveMaxPool1d
- nn.AdaptiveMaxPool2d
- nn.AdaptiveMaxPool3d
- nn.AdaptiveAvgPool1d
- nn.AdaptiveAvgPool2d
- nn.AdaptiveAvgPool3d
- nn.BatchNorm1d
- nn.BatchNorm2d
- nn.BatchNorm3d
- nn.LazyBatchNorm1d
- nn.LazyBatchNorm2d
- nn.LazyBatchNorm3d
- nn.InstanceNorm1d
- nn.InstanceNorm2d
- nn.InstanceNorm3d
- nn.LazyInstanceNorm1d
- nn.LazyInstanceNorm2d
- nn.LazyInstanceNorm3d
