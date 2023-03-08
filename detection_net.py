import torch.nn as nn
import torch
import torchvision

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Detection network
        self.embed_color = torchvision.models.densenet121(pretrained=True)
        self.embed_color = torch.nn.Sequential(*(list(self.embed_color.children())[:-1]))

        self.embed_state = torch.nn.Sequential(*([nn.BatchNorm2d(1024),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(1024, 512, (2,1), stride=1, bias=False),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 256, 1, stride=1, bias=False),
                                                  nn.BatchNorm2d(256),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(256, 128, 1, stride=1, bias=False),
                                                  nn.BatchNorm2d(128),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(128, 64, 1, stride=1, bias=False),
                                                  nn.BatchNorm2d(64),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 32, 1, stride=1, bias=False),
                                                  nn.BatchNorm2d(32),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(32, 16, 1, stride=1, bias=False),
                                                  nn.BatchNorm2d(16),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(16, 8, 1, stride=1, bias=False),
                                                  nn.BatchNorm2d(8),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(8, 1, 1, stride=1, bias=False),
                                                  ])
                                               )

    def forward(self, img):
        img = self.embed_color(img)
        x = self.embed_state(img).view(-1, 1)
        x = torch.sigmoid(x)

        return x

