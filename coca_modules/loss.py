import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ce = F.cross_entropy


class LanguageModelCriterion(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=0,
        world_size=1,
    ):
        super(LanguageModelCriterion, self).__init__()
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.contrastive_loss_weight = 1.0
        self.caption_loss_weight = 1.0
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=0)

        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    # [b, 512] * [512, b] -> [b, b] 余弦相似度
    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = image_features @ text_features.T * logit_scale
        logits_per_text = text_features @ image_features.T * logit_scale

        return logits_per_image, logits_per_text

    def forward(
        self, output, reports_ids, reports_masks, text_latents=None, image_latents=None
    ):
        device = output.device
        # print(f"temperature: {self.temperature}")
        # if text_latents != None and image_latents !=None:
        #     sim = torch.einsum('i d, j d -> i j', text_latents, image_latents)
        #     sim = sim * self.temperature.exp().to(tensor_device)
        #     contrastive_labels = torch.arange(reports_ids.shape[0], device=tensor_device)#!!!
        #     contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5

        # logits = rearrange(output, 'b n c -> b c n')
        # caption_loss = ce(logits, reports_ids, ignore_index=0)
        # caption_loss = caption_loss * self.caption_loss_weight
        contrastive_loss = 0
        if text_latents != None and image_latents != None:
            logits_per_image, logits_per_text = self.get_logits(
                image_latents, text_latents, self.temperature.exp()
            )

            labels = self.get_ground_truth(device, logits_per_image.shape[0])

            contrastive_loss = (
                F.cross_entropy(logits_per_image, labels)
                + F.cross_entropy(logits_per_text, labels)
            ) / 2

        contrastive_loss = self.contrastive_loss_weight * contrastive_loss

        reports_ids = reports_ids[:, : output.size(1)]
        reports_masks = reports_masks[:, : output.size(1)]
        output = (
            -output.gather(2, reports_ids.long().unsqueeze(2)).squeeze(2)
            * reports_masks
        )
        caption_loss = torch.sum(output) / torch.sum(reports_masks)
        # label = reports_ids[:,-output.shape[1]:]
        # caption_loss = self.caption_loss(output.permute(0, 2, 1), reports_ids)
        caption_loss = caption_loss * self.caption_loss_weight
        return contrastive_loss, caption_loss


def compute_loss(
    loss_fn, output, reports_ids, reports_masks, text_latents=None, image_latents=None
):
    criterion = loss_fn
    if text_latents != None and image_latents != None:
        contrastive_loss, caption_loss = criterion(
            output,
            reports_ids[:, 1:],
            reports_masks[:, 1:],
            text_latents,
            image_latents,
        )
    else:
        contrastive_loss, caption_loss = criterion(
            output, reports_ids[:, 1:], reports_masks[:, 1:]
        )
    loss = contrastive_loss + caption_loss
    return contrastive_loss, caption_loss, loss
