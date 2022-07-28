import paddle.nn as nn
from paddle.vision.models import resnet18


class ProjectionNet(nn.Layer):
    def __init__(self, pretrained=True, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=2):
        super(ProjectionNet, self).__init__()
        self.resnet18 = resnet18(pretrained=pretrained)

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1D(num_neurons))
            sequential_layers.append(nn.ReLU())
            last_layer = num_neurons

        head = nn.Sequential(
            *sequential_layers
        )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        embeds = self.resnet18(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits

    def freeze_resnet(self):
        for param in self.resnet18.parameters():
            param.requires_grad = False

        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
