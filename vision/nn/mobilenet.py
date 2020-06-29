# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024, alpha=1.0):
        super(MobileNetV1, self).__init__()
        self.alpha = alpha

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(int(3 * self.alpha), int(32 * self.alpha), 2),
            conv_dw(int(32 * self.alpha), int(64 * self.alpha), 1),
            conv_dw(int(64 * self.alpha), int(128 * self.alpha), 2),
            conv_dw(int(128 * self.alpha), int(128 * self.alpha), 1),
            conv_dw(int(128 * self.alpha), int(256 * self.alpha), 2),
            conv_dw(int(256 * self.alpha), int(256 * self.alpha), 1),
            conv_dw(int(256 * self.alpha), int(512 * self.alpha), 2),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(512 * self.alpha), 1),
            conv_dw(int(512 * self.alpha), int(1024 * self.alpha), 2),
            conv_dw(int(1024 * self.alpha), int(1024 * self.alpha), 1),
        )
        self.fc = nn.Linear(int(1024 * self.alpha), num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, int(1024 * self.alpha))
        x = self.fc(x)
        return x
