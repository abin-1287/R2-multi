import torch
import torch.nn as nn
import torchvision.models as models


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):

        B,  _, _, C = x.shape
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.model = getattr(models, self.visual_extractor)(
            pretrained=self.pretrained)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.Linear = nn.Linear(self.model.fc.in_features, 121)
        self.patchmerging3 = PatchMerging(1024)
        self.patchmerging2 = PatchMerging(512)
        self.patchmerging1 = PatchMerging(256)

    def forward(self, images):
        projections = []
        x = self.model.conv1(images)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        patch1 = x.permute(0, 2, 3, 1)
        patch1 = self.patchmerging1(patch1)  # [b,56,56,256]->[b,28,28,512]
        patch1 = self.patchmerging2(patch1)  # [b,28,28,512]->[b,14,14,1024]
        project1 = self.patchmerging3(patch1)  # [b,14,14,1024]->[b,7,7,2048]
        b, h, w, c = project1.shape
        project1 = project1.view(b, h*w, c)
        projections.append(project1)

        x = self.model.layer2(x)
        patch2 = x.permute(0, 2, 3, 1)
        patch2 = self.patchmerging2(patch2)  # [b,28,28,512]->[b,14,14,1024]
        project2 = self.patchmerging3(patch2)  # [b,14,14,1024]->[b,7,7,2048]
        b, h, w, c = project2.shape
        project2 = project2.view(b, h*w, c)
        projections.append(project2)

        x = self.model.layer3(x)
        patch3 = x.permute(0, 2, 3, 1)
        project3 = self.patchmerging3(patch3)  # [b,14,14,1024]->[b,7,7,2048]
        b, h, w, c = project3.shape
        project3 = project3.view(b, h*w, c)
        projections.append(project3)

        x = self.model.layer4(x)
        patch4 = x.permute(0, 2, 3, 1)
        b, h, w, c = patch4.shape
        project4 = patch4.view(b, h*w, c)
        projections.append(project4)

        avg_feature = self.model.avgpool(x)
        flatten_feature = torch.flatten(avg_feature, 1)
        patch_feats = torch.cat(projections, dim=1)

        avg_feats = self.avg_fnt(x).squeeze().reshape(-1, x.size(1))
        images_feature = self.Linear(flatten_feature)
        # images_feature = self.softmax(images_feature)
        return patch_feats, avg_feats, images_feature
