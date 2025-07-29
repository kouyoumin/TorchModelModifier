import torch.nn as nn

convnd = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]
lazyconvnd = [None, nn.LazyConv1d, nn.LazyConv2d, nn.LazyConv3d]

convtransposend = [None, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
lazyconvtransposend = [None, nn.LazyConvTranspose1d, nn.LazyConvTranspose2d, nn.LazyConvTranspose3d]

maxpoolnd = [None, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]
avgpoolnd = [None, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]

adaptivemaxpoolnd = [None, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d]
adaptiveavgpoolnd = [None, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]

batchnormnd = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
lazybatchnormnd = [None, nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d]

instancenormnd = [None, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
lazyinstancenormnd = [None, nn.LazyInstanceNorm1d, nn.LazyInstanceNorm2d, nn.LazyInstanceNorm3d]

dropoutnd = [None, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d]

linear_interpolation_modes = [None, 'linear', 'bilinear', 'trilinear']

class AdaptiveAvgPool3d(nn.Module):
    """
    Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    This module only supports output_size=1 or output_size=(1,1,1).

    Args:
        output_size (int or tuple): The target output size of the form D x H x W. Must be 1 or (1,1,1).

    Raises:
        NotImplementedError: If output_size is not 1 or (1,1,1).

    Methods:
        forward(x):
            Applies the adaptive average pooling operation to the input tensor x.

            Args:
                x (Tensor): The input tensor of shape (N, C, D, H, W).

            Returns:
                Tensor: The output tensor of shape (N, C, 1, 1, 1) after applying adaptive average pooling.
    """
    def __init__(self, output_size):
        super(AdaptiveAvgPool3d, self).__init__()
        if output_size != 1 and output_size != (1,1,1):
            raise NotImplementedError('This AdaptiveAvgPool3d only supports output_size=1')

    def forward(self, x):
        return x.mean((-3,-2,-1), keepdim=True)


class AdaptiveMaxPool3d(nn.Module):
    """
    A custom implementation of 3D adaptive max pooling layer that only supports output size of 1.

    Args:
        output_size (int or tuple): The target output size. Must be 1 or (1, 1, 1).

    Raises:
        NotImplementedError: If the output_size is not 1 or (1, 1, 1).

    Methods:
        forward(x):
            Applies the adaptive max pooling operation to the input tensor `x`.

            Args:
                x (torch.Tensor): The input tensor of shape (N, C, D, H, W).

            Returns:
                torch.Tensor: The output tensor after applying adaptive max pooling, 
                              with shape (N, C, 1, 1, 1).
    """
    def __init__(self, output_size):
        super(AdaptiveMaxPool3d, self).__init__()
        if output_size != 1 and output_size != (1,1,1):
            raise NotImplementedError('This AdaptiveMaxPool3d only supports output_size=1')

    def forward(self, x):
        return x.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0].max(-3, keepdim=True)[0]


def _modify_tuple_dim(original, target_size):
    if not isinstance(original, int):
        if len(original) >= target_size:
            return original[:target_size]
        else:
            return original[0]
    else:
        return original


def _modify_weight_dim(original, target_dim):
    new_sd = {}
    for key in original:
        if key.endswith('weight'):
            original_dim = original[key].ndim - 2
            num_diff_dim = target_dim - original_dim
            weight = original[key]
            if num_diff_dim > 0:
                for i in range(num_diff_dim):
                    repeats = [1] * weight.ndim + [weight.shape[-1]]
                    weight = weight.unsqueeze(-1).repeat(*repeats)
                    
                weight_correction_factor = weight.shape[-1] ** original_dim / weight.shape[-1] ** target_dim
                print(weight_correction_factor)
                weight = weight * weight_correction_factor
            elif num_diff_dim < 0:
                for i in range(abs(num_diff_dim)):
                    weight = weight.sum(dim=weight.ndim-1, keepdim=True).squeeze(-1)
            new_sd[key] = weight
            print('Changed', key, 'from', original[key].shape, 'to', new_sd[key].shape)
        else:
            new_sd[key] = original[key]
    return new_sd


def modify_model_dim(model, new_dim, coreml_compatibility=False):
    """
    Modify the dimensions of the layers in a given PyTorch model to a new dimension.
    Args:
        model (torch.nn.Module): The PyTorch model to modify.
        new_dim (int): The new dimension to modify the model layers to. Must be between 1 and 3.
        coreml_compatibility (bool, optional): If True, ensures compatibility with CoreML for certain layers. Default is False.
    Raises:
        AssertionError: If new_dim is not between 1 and 3.
    The function iterates through the layers of the model and modifies the dimensions of convolutional, pooling, 
    batch normalization, and instance normalization layers to the specified new dimension. It handles both 
    standard and lazy versions of these layers. For adaptive pooling layers, it ensures CoreML compatibility 
    if specified.
    """
    assert(1 <= new_dim <= 3)
    for n, m in model.named_modules():
        hiers = n.split('.')
        levels = [model]
        for hier in hiers[:-1]:
            levels.append(getattr(levels[-1], hier))
        #print('Processing layer (%s): %s' % (n, m))
        if isinstance(m, nn.modules.conv._ConvTransposeNd):
            kernel_size = _modify_tuple_dim(m.kernel_size, new_dim)
            stride = _modify_tuple_dim(m.stride, new_dim)
            padding = _modify_tuple_dim(m.padding, new_dim)
            dilation = _modify_tuple_dim(m.dilation, new_dim)
            
            # Assign new
            if isinstance(m, nn.modules.conv._LazyConvXdMixin):
                conv_module = lazyconvtransposend[new_dim](m.out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=m.groups, bias=False if m.bias==None else True, padding_mode=m.padding_mode)
                conv_module.load_state_dict(_modify_weight_dim(m.state_dict(), new_dim))
                print(f'Found convtransposend layer ({n}): {m} -> {conv_module}')
                setattr(levels[-1], hiers[-1], conv_module)
            else:
                conv_module = convtransposend[new_dim](m.in_channels, m.out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=m.groups, bias=False if m.bias==None else True, padding_mode=m.padding_mode)
                conv_module.load_state_dict(_modify_weight_dim(m.state_dict(), new_dim))
                print(f'Found convtransposend layer ({n}): {m} -> {conv_module}')
                setattr(levels[-1], hiers[-1], conv_module)
        elif isinstance(m, nn.modules.conv._ConvNd):
            kernel_size = _modify_tuple_dim(m.kernel_size, new_dim)
            stride = _modify_tuple_dim(m.stride, new_dim)
            padding = _modify_tuple_dim(m.padding, new_dim)
            dilation = _modify_tuple_dim(m.dilation, new_dim)

            # Assign new
            if isinstance(m, nn.modules.conv._LazyConvXdMixin):
                conv_module = lazyconvnd[new_dim](m.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=m.groups, bias=False if m.bias==None else True, padding_mode=m.padding_mode)
                conv_module.load_state_dict(_modify_weight_dim(m.state_dict(), new_dim))
                print(f'Found conv layer ({n}): {m} -> {conv_module}')
                setattr(levels[-1], hiers[-1], conv_module)
            else:
                conv_module = convnd[new_dim](m.in_channels, m.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=m.groups, bias=False if m.bias==None else True, padding_mode=m.padding_mode)
                conv_module.load_state_dict(_modify_weight_dim(m.state_dict(), new_dim))
                print(f'Found conv layer ({n}): {m} -> {conv_module}')
                setattr(levels[-1], hiers[-1], conv_module)
        elif isinstance(m, nn.modules.pooling._MaxPoolNd):
            kernel_size = _modify_tuple_dim(m.kernel_size, new_dim)
            stride = _modify_tuple_dim(m.stride, new_dim)
            padding = _modify_tuple_dim(m.padding, new_dim)
            dilation = _modify_tuple_dim(m.dilation, new_dim)

            # Assign new 
            pool_module = maxpoolnd[new_dim](kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)
            print(f'Found maxpool layer ({n}): {m} -> {pool_module}')
            setattr(levels[-1], hiers[-1], pool_module)
        elif isinstance(m, nn.modules.pooling._AvgPoolNd):
            kernel_size = _modify_tuple_dim(m.kernel_size, new_dim)
            stride = _modify_tuple_dim(m.stride, new_dim)
            padding = _modify_tuple_dim(m.padding, new_dim)

            # Assign new 
            pool_module = avgpoolnd[new_dim](kernel_size, stride=stride, padding=padding, ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad)
            print(f'Found avgpool layer ({n}): {m} -> {pool_module}')
            setattr(levels[-1], hiers[-1], pool_module)
        elif isinstance(m, nn.modules.pooling._AdaptiveMaxPoolNd):
            output_size = _modify_tuple_dim(m.output_size, new_dim)

            # Assign new 
            if coreml_compatibility and new_dim == 3 and (m.output_size == 1 or m.output_size == (1,1)):
                pool_module = AdaptiveMaxPool3d(output_size, return_indices=m.return_indices)
            else:
                pool_module = adaptivemaxpoolnd[new_dim](output_size, return_indices=m.return_indices)
            print(f'Found adaptivemaxpool layer ({n}): {m} -> {pool_module}')
            setattr(levels[-1], hiers[-1], pool_module)
        elif isinstance(m, nn.modules.pooling._AdaptiveAvgPoolNd):
            output_size = _modify_tuple_dim(m.output_size, new_dim)

            # Assign new
            if coreml_compatibility and new_dim == 3 and (m.output_size == 1 or m.output_size == (1,1)):
                pool_module = AdaptiveAvgPool3d(output_size)
            else:
                pool_module = adaptiveavgpoolnd[new_dim](output_size)
            print(f'Found adaptiveavgpool layer ({n}): {m} -> {pool_module}')
            setattr(levels[-1], hiers[-1], pool_module)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            # Assign new 
            if isinstance(m, nn.modules.batchnorm._LazyNormBase):
                norm_module = lazybatchnormnd[new_dim](eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
                print(f'Found batchnorm layer ({n}): {m} -> {norm_module}')
                setattr(levels[-1], hiers[-1], norm_module)
            else:
                norm_module = batchnormnd[new_dim](m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
                print(f'Found batchnorm layer ({n}): {m} -> {norm_module}')
                setattr(levels[-1], hiers[-1], norm_module)
        elif isinstance(m, nn.modules.instancenorm._InstanceNorm):
            # Assign new 
            if isinstance(m, nn.modules.batchnorm._LazyNormBase):
                norm_module = lazyinstancenormnd[new_dim](eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
                print(f'Found instancenorm layer ({n}): {m} -> {norm_module}')
                setattr(levels[-1], hiers[-1], norm_module)
            else:
                norm_module = instancenormnd[new_dim](m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
                print(f'Found instancenorm layer ({n}): {m} -> {norm_module}')
                setattr(levels[-1], hiers[-1], norm_module)
        elif isinstance(m, nn.modules.upsampling.Upsample) and 1 <= new_dim <= 3:
            if m.mode.endswith('linear'):
                upsample_module = nn.Upsample(scale_factor=m.scale_factor, mode=linear_interpolation_modes[new_dim], align_corners=m.align_corners)
                print(f'Found upsample layer ({n}): {m} -> {upsample_module}')
                setattr(levels[-1], hiers[-1], upsample_module)
        elif isinstance(m, nn.modules.dropout._DropoutNd):# and not isinstance(m, nn.modules.dropout.Dropout):
            if isinstance(m, nn.modules.dropout.Dropout):
                print(f'Found dropout layer ({n}): {m} - no change')
                continue
            
            # Assign new 
            dropout_module = dropoutnd[new_dim](p=m.p, inplace=m.inplace)
            print(f'Found dropout layer ({n}): {m} -> {dropout_module}')
            setattr(levels[-1], hiers[-1], dropout_module)


def test(model_str, new_dim):
    import torch
    import torchvision
    model = torchvision.models.__dict__[model_str](pretrained=True)
    modify_model_dim(model, new_dim)
    print(model)

    dummy_data_size = [1,3]
    for i in range(new_dim):
        dummy_data_size.append(128)
    dummy_data = torch.randn(dummy_data_size)
    print('Testing model with dummy data:', dummy_data.shape)
    with torch.no_grad():
        out = model(dummy_data)
    print('Result tensor shape:', out.shape)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: %s [model] [new_dim]' % (sys.argv[0]))
    test(sys.argv[1], int(sys.argv[2]))