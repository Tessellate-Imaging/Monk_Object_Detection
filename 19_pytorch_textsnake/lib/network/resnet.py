import torchvision.models.resnet as resnet
import torch.nn as nn

class ResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = resnet.resnet50(pretrained=True)

        self.stage1 = nn.Sequential(
            self.net.conv1,
            self.net.bn1,
            self.net.relu,
            self.net.maxpool
        )
        self.stage2 = self.net.layer1
        self.stage3 = self.net.layer2
        self.stage4 = self.net.layer3
        self.stage5 = self.net.layer4

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        return C1, C2, C3, C4, C5


if __name__ == '__main__':
    import torch
    input = torch.randn((4, 3, 512, 512))
    net = ResNet50()
    C1, C2, C3, C4, C5 = net(input)
    print(C1.size())
    print(C2.size())
    print(C3.size())
    print(C4.size())
    print(C5.size())
