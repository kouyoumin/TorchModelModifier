import torch.nn as nn


def modify_last_dropout_rate(model, new_drop_rate=-1):
    for n, m in model.named_modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            print('Found dropout layer (%s): %s' % (n, m))
            drop_layer = m
   
    if new_drop_rate >= 0:
        print(f'Set new drop rate to {new_drop_rate}')
        drop_layer.p = new_drop_rate
        
    return drop_layer.p


def test():
    import torchvision
    
    old_droprate = 0.1
    model = torchvision.models.efficientnet_v2_s(dropout=old_droprate)
    retrieved_droprate = modify_last_dropout_rate(model, -1)
    assert(old_droprate == retrieved_droprate)
    new_droprate = 0.2
    retrieved_droprate = modify_last_dropout_rate(model, new_droprate)
    assert(new_droprate == retrieved_droprate)


if __name__ == '__main__':
    import sys
    test()