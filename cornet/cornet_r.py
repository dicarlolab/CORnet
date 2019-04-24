from collections import OrderedDict
import torch
from torch import nn


HASH = '5930a990'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_R(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp=None, state=None, batch_size=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        x = self.norm1(x)
        x = self.nonlin1(x)

        state = self.output(x)
        output = state
        return output, state


class CORnet_R(nn.Module):

    def __init__(self, times=5):
        super().__init__()
        self.times = times

        self.V1 = CORblock_R(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = CORblock_R(64, 128, stride=2, out_shape=28)
        self.V4 = CORblock_R(128, 256, stride=2, out_shape=14)
        self.IT = CORblock_R(256, 512, stride=2, out_shape=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000))
        ]))

    def forward(self, inp):
        outputs = {'inp': inp}
        states = {}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']

        for block in blocks[1:]:
            if block == 'V1':  # at t=0 input to V1 is the image
                inp = outputs['inp']
            else:  # at t=0 there is no input yet to V2 and up
                inp = None
            new_output, new_state = getattr(self, block)(inp, batch_size=outputs['inp'].shape[0])
            outputs[block] = new_output
            states[block] = new_state

        for t in range(1, self.times):
            for block in blocks[1:]:
                prev_block = blocks[blocks.index(block) - 1]
                prev_output = outputs[prev_block]
                prev_state = states[block]
                new_output, new_state = getattr(self, block)(prev_output, prev_state)
                outputs[block] = new_output
                states[block] = new_state

        out = self.decoder(outputs['IT'])
        return out
