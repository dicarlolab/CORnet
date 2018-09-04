from collections import OrderedDict
import torch
from torch import nn


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    def forward(self, x):
        return x


class CORblock_R(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, w=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = w

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU()

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU()

        self.output = Identity()

    def forward(self, inp=None, state=None, batch_size=None):
        if inp is None:
            inp = torch.zeros([batch_size, self.out_channels, self.w, self.w])
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:
            skip = inp
        else:
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

        self.V1 = CORblock_R(3, 64, kernel_size=7, stride=4, w=56)
        self.V2 = CORblock_R(64, 128, stride=2, w=28)
        self.V4 = CORblock_R(128, 256, stride=2, w=14)
        self.IT = CORblock_R(256, 512, stride=2, w=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000))
        ]))

    def forward(self, inp):
        outputs = {'inp': inp}
        states = {}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']

        for bno, block in enumerate(blocks[1:]):
            if block == 'V1':
                inp = outputs['inp']
            else:
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
