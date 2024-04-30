import torch
import torch.nn as nn
import numpy as np
import open_clip
import torch.nn.functional as F

from coca_modules.visual_extractor import VisualExtractor
from coca_modules.encoder_decoder import EncoderDecoder


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        cocamodel, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-B-32",
            pretrained=".cache/huggingface/hub/models--laion--CoCa-ViT-B-32-laion2B-s13B-b90k/open_clip_pytorch_model.bin"
        )
        self.cocamodel = cocamodel
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        # att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        # att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        # att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
             output, image_embed, text_embed = self.encoder_decoder(images, targets, mode='forward')
             return text_embed, image_embed, output
        elif mode == 'sample':
            output, _ = self.encoder_decoder(images, mode='sample')
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        # att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output, image_embed, text_embed = self.encoder_decoder(images, targets, mode='forward')
            # text_latents = F.normalize(text_embed, dim=-1)  # (b 512)
            # image_latents = F.normalize(image_embed, dim=-1)  # (b 512)
            return text_embed, image_embed, output
        elif mode == 'sample':
            output, _ = self.encoder_decoder(images, mode='sample')
            return output
        else:
            raise ValueError

