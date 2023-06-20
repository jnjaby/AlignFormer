import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from collections import OrderedDict
from basicsr.utils.registry import ARCH_REGISTRY


class ContrastExtractorLayer(nn.Module):

    def __init__(self):
        super().__init__()

        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
            'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv3_1_idx = vgg16_layers.index('conv3_1')
        features = getattr(vgg,
                           'vgg16')(pretrained=True).features[:conv3_1_idx + 1]
        modified_net = OrderedDict()
        for k, v in zip(vgg16_layers, features):
            modified_net[k] = v

        self.model = nn.Sequential(modified_net)
        # the mean is for image with range [0, 1]
        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # the std is for image with range [0, 1]
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, batch):
        batch = (batch - self.mean) / self.std
        output = self.model(batch)
        return output


@ARCH_REGISTRY.register()
class ContrastExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_extractor1 = ContrastExtractorLayer()
        self.feature_extractor2 = ContrastExtractorLayer()

    def forward(self, image1, image2):
        dense_features1 = self.feature_extractor1(image1)
        dense_features2 = self.feature_extractor2(image2)

        return {
            'dense_features1': dense_features1,
            'dense_features2': dense_features2
        }


@ARCH_REGISTRY.register()
class SingleExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = ContrastExtractorLayer()

    def forward(self, image1, image2):
        dense_features1 = self.feature_extractor(image1)
        dense_features2 = self.feature_extractor(image2)

        return {
            'dense_features1': dense_features1,
            'dense_features2': dense_features2
        }


if __name__ == '__main__':
    height = 256
    width = 256
    model = ContrastExtractor()
    print(model)

    src = torch.randn((2, 3, height, width))
    ref = torch.randn((2, 3, height, width))
    model.eval()
    with torch.no_grad():
        out = model(src, ref)
    model.train()

    import pdb
    pdb.set_trace()
    print(out.shape)
