import torch
import torch.nn as nn
import torchvision.models as models
from .Res2Net.res2net import res2net101
import sys

import logging

import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        # model = getattr(models, self.visual_extractor)(
        #     pretrained=self.pretrained)
        model = res2net101(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.avgpool = model.avgpool
        self.Linear = nn.Linear(model.fc.in_features, 121)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feature = self.avgpool(patch_feats)
        flatten_feature = avg_feature.view(avg_feature.size(0), -1)
        # flatten_feature = torch.flatten(avg_feature, 1)
        avg_feats = self.avg_fnt(patch_feats).squeeze(
        ).reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(
            batch_size, feat_size, -1).permute(0, 2, 1)
        images_feature = self.Linear(flatten_feature)
        # images_feature = self.softmax(images_feature)
        return patch_feats, avg_feats, images_feature
