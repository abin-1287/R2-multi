import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.model = getattr(models, self.visual_extractor)(
            pretrained=self.pretrained)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.Linear = nn.Linear(self.model.fc.in_features, 121)
        self.projection2 = nn.Linear(2048, 512)
        self.projection1 = nn.Linear(1024, 512)

    def forward(self, images):
        projections = []
        x = self.model.conv1(images)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)

        x = self.model.layer2(x)
        batch_size, feat_size, _, _ = x.shape
        patch_feats0 = x.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        projections.append(patch_feats0)

        x = self.model.layer3(x)
        batch_size, feat_size, _, _ = x.shape
        patch_feats1 = x.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        project1 = self.projection1(patch_feats1)
        projections.append(project1)

        x = self.model.layer4(x)
        batch_size, feat_size, _, _ = x.shape
        patch_feats2 = x.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        project2 = self.projection2(patch_feats2)
        projections.append(project2)

        avg_feature = self.model.avgpool(x)
        flatten_feature = torch.flatten(avg_feature, 1)
        patch_feats = torch.cat(projections, dim=1)

        avg_feats = self.avg_fnt(x).squeeze().reshape(-1, x.size(1))
        images_feature = self.Linear(flatten_feature)
        # images_feature = self.softmax(images_feature)
        return patch_feats, avg_feats, images_feature
