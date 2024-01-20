from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel
import copy
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from .segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer, LayerNorm2d, MaskDecoderMultiScale
from utils.matcher import match_pred
from utils.loss import L1Loss
from typing import Any, Dict, List, Tuple

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def overlap_loss(inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    batch_seg_token_count: int):
    if num_masks == 0:
        return inputs.sum() * 0
    batch_seg_token_count = batch_seg_token_count.cumsum(-1)  
    batch_seg_token_count = torch.cat(
            [torch.zeros(1).long().cuda(), batch_seg_token_count], dim=0
        )
    loss = 0

    for i in range(len(batch_seg_token_count) -1):
        start_i = batch_seg_token_count[i]
        end_i = batch_seg_token_count[i+1]
        assert end_i <= len(targets), (targets.shape, batch_seg_token_count)
        question_inputs = inputs[start_i:end_i]
        question_targets = targets[start_i:end_i]
        if len(question_targets) == 0:
            continue
        n, h, w = question_inputs.shape
        all_targets = torch.zeros_like(question_targets[0]).bool()
        for target in question_targets:
            all_targets = (all_targets | target.bool())
        bg_area = all_targets < 0
        bg_area = bg_area[None].repeat(n, 1, 1)

        overlap_area = (question_inputs > 0).sum(dim=0)
        overlap_area = overlap_area >= 2

        overlap_area = overlap_area[None].repeat(n, 1, 1)
        weight = torch.ones_like(question_inputs)
        weight[bg_area] = 0

        q_loss = F.binary_cross_entropy_with_logits(question_inputs, question_targets, weight=weight, reduction="none")
        q_loss = q_loss.flatten(1, 2).mean(1).sum() 
        loss = loss + q_loss
    loss = loss / (num_masks + 1e-8)
    return loss


class PixelLMMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):  
        super(PixelLMMetaModel, self).__init__(config)
        self.logger = kwargs.get("logger", None)
        self.local_rank = kwargs.get("local_rank", 1)
        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_pixellm_modules(self.config)

    def initialize_pixellm_modules(self, config):
        # SAM
        if self.config.vision_tower_for_mask:
            
            prompt_embed_dim = 256 
            image_size = config.resize_vision_tower_size
            mask_decoder_transformer_depth = 2
            if self.local_rank == 0 and self.logger is not None:
                self.logger.info('--------build_sam_decoder--------')
                self.logger.info('--------sam decoder image size {}--------'.format(image_size))
            vit_patch_size = 14
            image_embedding_size = image_size // vit_patch_size
            self.prompt_encoder=PromptEncoder(
                        embed_dim=prompt_embed_dim,
                        image_embedding_size=(image_embedding_size, image_embedding_size),
                        input_image_size=(image_size, image_size),
                        mask_in_chans=16,
                    )
             
            # if config.image_feature_scale_num > 1:
            #     decoder_type = MaskDecoderMultiScale
            #     self.mask_decoder=nn.ModuleList(decoder_type(
            #         num_multimask_outputs=3,
            #         transformer=TwoWayTransformer(
            #             depth=mask_decoder_transformer_depth,
            #             embedding_dim=prompt_embed_dim,
            #             mlp_dim=2048,
            #             num_heads=8,
            #         ),
            #         transformer_dim=prompt_embed_dim,
            #         iou_head_depth=3,
            #         iou_head_hidden_dim=256,
            #         image_feature_scale_num=config.image_feature_scale_num
            #     ) for _ in range(config.image_feature_scale_num))

            # else:
            self.mask_decoder=MaskDecoderMultiScale(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=mask_decoder_transformer_depth,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                image_feature_scale_num=config.image_feature_scale_num
            ) 
            
            
            embed_dim = self.config.hidden_size 
            out_chans = prompt_embed_dim
            self.image_feature_neck = nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
                nn.Conv2d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_chans),
            )

        else:
            self.visual_model = build_sam_vit_h(self.vision_pretrained)
            for param in self.visual_model.parameters():
                param.requires_grad = False
            if config.train_mask_decoder:
                self.visual_model.mask_decoder.train()
                for param in self.visual_model.mask_decoder.parameters():
                    param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class PixelLMModel(PixelLMMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(PixelLMModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class PixelLMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # import pdb;pdb.set_trace()
        # 
        kwargs.update({
            "image_feature_scale_num": 2, 
            "pad_train_clip_images": True,
            "resize_vision_tower": True,
            "resize_vision_tower_size": 448,
            "vision_tower_for_mask": True,
            "separate_mm_projector": True,
        })
        self.logger = kwargs.get("logger", None)
        config.resize_vision_tower = kwargs.get("resize_vision_tower", False)
        config.resize_vision_tower_size = kwargs.get("resize_vision_tower_size", 224)
        config.pad_train_clip_images = kwargs.get("pad_train_clip_images", False)
        config.vision_tower_for_mask = kwargs.get("vision_tower_for_mask", False)
        config.separate_mm_projector = kwargs.get("separate_mm_projector", False)
        config.mm_projector_hidden_dim = 2
        config.mm_projector_out_dim = 1
        self.image_feature_scale_num = kwargs.get("image_feature_scale_num", 1)
        config.image_feature_scale_num = kwargs.get("image_feature_scale_num", 1)
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
        
        self.vision_tower_for_mask = kwargs.get("vision_tower_for_mask", False)
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.seg_token_num = kwargs.get("seg_token_num", 1)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.local_rank = kwargs.get("local_rank", 1)
        self.pad_train_clip_images = kwargs.get("pad_train_clip_images", False)
        self.masks_process_with_clip = kwargs.get("masks_process_with_clip", False)
        
        logger = kwargs.get("logger", None)
        if isinstance(self.seg_token_idx, list):
            if self.local_rank == 0 and logger is not None:
                print('--------initialize multiseg scalar--------')
            seg_token_num = len(self.seg_token_idx)
            scalar = 1 / seg_token_num
            self.multiseg_scalar = [torch.nn.Parameter(torch.ones([]) * scalar) for _ in range(seg_token_num)]
        if self.image_feature_scale_num > 1:
            scalar = 1 / self.image_feature_scale_num
            self.multiscale_scalar = [torch.nn.Parameter(torch.ones([]) * scalar) for _ in range(self.image_feature_scale_num)]
        super().__init__(config)
        self.model = PixelLMModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        self.iter = 0
        self.iter1 = 0
         
        if config.resize_vision_tower_size != 224:
            if self.local_rank == 0 and self.logger is not None:
                self.logger.info('--------mm_projector requires grad--------')
            for n, p in self.model.named_parameters():
                if any([x in n for x in ["mm_projector"]]):
                    p.requires_grad = True
                    
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        clip_resize_list = None,
        **kwargs,
    ):
         
        # clip_resize_list = kwargs.get('clip_resize_list', None)
        multi_reason_list = kwargs.get('multi_reason_list', None)
        if not self.vision_tower_for_mask:
            image_embeddings = self.get_visual_embs(images)
        batch_size = images.shape[0]
        assert batch_size == len(offset) - 1
        

        if isinstance(self.seg_token_idx, list):
            seg_token_num = self.seg_token_num
            seg_token_mask = torch.zeros_like(input_ids[:, 1:]).bool()
            for seg_token_idx in self.seg_token_idx:
                seg_token_mask = seg_token_mask | (input_ids[:, 1:] == seg_token_idx)  
        else:
            seg_token_num = self.seg_token_num
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx

        # if self.use_depth_token:
        

        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            extend_clip_resize_list = [clip_resize_list[0]] * length
            output_hidden_states = []
            output_image_features = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                    clip_resize_list=extend_clip_resize_list
                )
                output_image_feature_i = torch.stack(output_i.image_features, dim=0)
                output_hidden_states.append(output_i.hidden_states)
                output_image_features.append(output_image_feature_i)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output_image_features = torch.cat(output_image_features, dim=1)
            output = None

        else:
            images_clip_list = []
            extend_clip_resize_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                extend_clip_resize_list.extend([clip_resize_list[i]] * (end_i - start_i))
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                clip_resize_list=extend_clip_resize_list
            )
            output_hidden_states = output.hidden_states
            output_image_features = output.image_features

        hidden_states = []

         
        
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )
        
        seg_token_offset = seg_token_offset[offset]
        feat_scale_num = self.image_feature_scale_num
        pred_embeddings_ = []
        batch_seg_token_counts = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            batch_pred_embeddings = pred_embeddings[start_i:end_i]
            batch_seg_token_counts.append(seg_token_counts[offset[i]:offset[i+1]] // seg_token_num)
            assert len(batch_pred_embeddings) % seg_token_num == 0
            batch_pred_embeddings = batch_pred_embeddings.view(len(batch_pred_embeddings) // (seg_token_num*feat_scale_num), feat_scale_num, seg_token_num, batch_pred_embeddings.shape[-1]) #N, scale_num, seg_num, dim
            if seg_token_num > 1:
                fused_batch_pred_embeddings = batch_pred_embeddings[:, :, 0] * 0 #N, scale_num, dim
                for i in range(seg_token_num):
                    fused_batch_pred_embeddings = fused_batch_pred_embeddings + self.multiseg_scalar[i] * batch_pred_embeddings[:, :, i]
                batch_pred_embeddings = fused_batch_pred_embeddings
            else:
                batch_pred_embeddings = batch_pred_embeddings[:, :, 0]
            
            pred_embeddings_.append(batch_pred_embeddings)
       
        pred_embeddings = pred_embeddings_  #number of image[seg token num in each image, ]
        multi_scale_num = len(output_image_features)


        if not inference:
            output_image_features = torch.stack(output_image_features, dim=0)  #[Lev, Q*1024, dim]
        img_embeddings = output_image_features.flatten(1, 2)  #[Lev, Q*1024, dim]
        img_token_mask = torch.ones(output_image_features.shape[1], output_image_features.shape[2]).to(seg_token_mask)
        img_token_counts = img_token_mask.int().sum(-1) 
        patch_count = int(img_token_counts[0])
        
        patch_size = int(patch_count**0.5)
        img_token_offset = img_token_counts.cumsum(-1) #[256, 512, ...]
        img_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), img_token_offset], dim=0 #[0, 256, 512, ...]
        )
        img_token_offset = img_token_offset[offset]   #[0, 768, ...]
        img_embeddings_ = []
        single_img_embeddings = [] #Lev, dim, 32, 32
        for i in range(len(img_token_offset) - 1):
            start_i, end_i = img_token_offset[i], img_token_offset[i + 1]
            question_num = pred_embeddings_[i].shape[0] 
            img_num = img_embeddings[:, start_i:end_i].shape[1] // patch_count
            single_img_embeddings.append(img_embeddings[:, start_i:end_i].view(multi_scale_num, img_num, patch_count, img_embeddings.shape[-1]).permute(0, 1, 3, 2).view(multi_scale_num, img_num, img_embeddings.shape[-1], patch_size, patch_size)[:, 0])
            if question_num == 0:
                batch_img_embeddings = torch.zeros(multi_scale_num, 0, 4096, patch_size, patch_size).to(img_embeddings)
            else:
                batch_img_embeddings = img_embeddings[:, start_i:end_i].view(multi_scale_num, img_num, patch_count, img_embeddings.shape[-1])
                batch_img_embeddings = batch_img_embeddings.permute(0, 1, 3, 2).view(multi_scale_num, img_num, img_embeddings.shape[-1], patch_size, patch_size)
            img_embeddings_.append(batch_img_embeddings)

        img_embeddings = img_embeddings_

        
        multimask_output = False
        pred_masks = []
        mask_scores = []
        pred_depths = []
         
        for i in range(len(pred_embeddings)):
            if self.vision_tower_for_mask:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=None,masks=None,text_embeds=pred_embeddings[i],) #sparse_embeddings:N, Lev, 256
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                _img_embeddings = self.model.image_feature_neck(single_img_embeddings[i]) #[Lev, 4096, 32, 32]
                out_size = 128
                low_res_masks = torch.zeros([sparse_embeddings.shape[0], 1, out_size, out_size]).to(_img_embeddings)
                if self.image_feature_scale_num > 1:
                    for l in range(self.image_feature_scale_num):
                        l_low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=_img_embeddings[l].unsqueeze(0), image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings[:, l].unsqueeze(1), dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, previous_masks=l_low_res_masks if l>0 else None, level_num=l)
                        low_res_masks = low_res_masks + self.multiscale_scalar[l] * F.interpolate(l_low_res_masks.float(), (out_size, out_size),mode="bilinear",align_corners=False,).to(l_low_res_masks)
                else:
                    low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=_img_embeddings[0].unsqueeze(0), image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings[:, 0].unsqueeze(1), dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, )
            
                pred_mask = self.postprocess_masks(
                    low_res_masks,
                    input_size=clip_resize_list[i],
                    original_size=label_list[i].shape,
                )

            

            else:

                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i],
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_depths.append([])
            

            
            pred_masks.append(pred_mask[:, 0])
            mask_score = (pred_mask[:, 0].sigmoid().flatten(1) * (pred_mask[:, 0] > 0).flatten(1)).sum(1) / ((pred_mask[:, 0] > 0).flatten(1).sum(1) + 1e-6)
            mask_scores.append(mask_score)


        model_output = output
        gt_masks = masks_list


        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "batch_seg_token_counts": batch_seg_token_counts,
                "mask_scores": mask_scores,
            }

        output = model_output.logits
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = ce_loss
        mask_bce_loss = pred_masks[0].sum() * 0
        mask_dice_loss = pred_masks[0].sum() * 0
        mask_overlap_loss = pred_masks[0].sum() * 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            batch_seg_token_count = batch_seg_token_counts[batch_idx]
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
             
      
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_overlap_loss += (
                    overlap_loss(pred_mask, gt_mask, gt_mask.shape[0], batch_seg_token_count)
                    * gt_mask.shape[0]
                    )
            num_masks += gt_mask.shape[0]
             
            
        
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_overlap_loss = self.bce_loss_weight * mask_overlap_loss / (num_masks + 1e-8)


        mask_loss = mask_bce_loss + mask_dice_loss + mask_overlap_loss
        
       
        loss = loss + mask_loss


        
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        clip_resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
        # instance_out=False
    ):
         
        all_pred_embeddings = []
        all_output_ids = []
        batch_seg_token_counts = []
        with torch.no_grad():
            _, _, output_image_features = self.encode_images(images_clip, clip_resize_list, return_project=True)
            multi_scale_num = self.image_feature_scale_num
            output_image_features = torch.stack(output_image_features, dim=0)
            for idx, input_id in enumerate(input_ids):
                if 0 in input_id:
                    unk_start = torch.where(input_id==0)[0].min()
                    _input_id = input_id[:unk_start]
                else:
                    _input_id = input_id
                outputs = self.generate(
                    images=images_clip,
                    input_ids=_input_id[None],
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    clip_resize_list=clip_resize_list
                )
                output_hidden_states = outputs.hidden_states[-1]
                output_ids = outputs.sequences
                all_output_ids.append(output_ids)
       
                if isinstance(self.seg_token_idx, list):
                    seg_token_num = self.seg_token_num
                    seg_token_mask = torch.zeros_like(output_ids[:, 1:]).bool()
                      
                    for seg_token_idx in self.seg_token_idx:
                        seg_token_mask = seg_token_mask | (output_ids[:, 1:] == seg_token_idx)  
                
                else:
                    seg_token_num = self.seg_token_num
                    seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
                # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
                seg_token_mask = torch.cat(
                    [
                        torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                        seg_token_mask,
                    ],
                    dim=1,
                )

                hidden_states = []
            
                assert len(self.model.text_hidden_fcs) == 1
                hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
                feat_scale_num = self.image_feature_scale_num
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                pred_embeddings = last_hidden_state[seg_token_mask]

                if len(pred_embeddings) % (seg_token_num*feat_scale_num) != 0:
                    seg_token_mask = (seg_token_mask*0).bool()
                    pred_embeddings = last_hidden_state[seg_token_mask]

                seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
                seg_token_offset = seg_token_counts.cumsum(-1)
                seg_token_offset = torch.cat(
                    [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
                )
                seg_token_offset = seg_token_offset[[0, len(seg_token_offset)-1]]
                pred_embeddings_ = []
                
                
                for i in range(len(seg_token_offset) - 1):
                    start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                    batch_pred_embeddings = pred_embeddings[start_i:end_i]
                    # if seg_token_num*feat_scale_num > 1:
                    assert len(batch_pred_embeddings) % (seg_token_num*feat_scale_num) == 0
                    batch_pred_embeddings = batch_pred_embeddings.view(len(batch_pred_embeddings) // (seg_token_num*feat_scale_num), feat_scale_num, seg_token_num, batch_pred_embeddings.shape[-1]) #N, scale_num, seg_num, dim
                    if seg_token_num > 1:
                        fused_batch_pred_embeddings = batch_pred_embeddings[:, :, 0] * 0 #N, scale_num, dim
                        for i in range(seg_token_num):
                            fused_batch_pred_embeddings = fused_batch_pred_embeddings + self.multiseg_scalar[i] * batch_pred_embeddings[:, :, i]
                        batch_pred_embeddings = fused_batch_pred_embeddings
                    else:
                        batch_pred_embeddings = batch_pred_embeddings[:, :, 0]
                    pred_embeddings_.append(batch_pred_embeddings)
                batch_seg_token_counts.append(len(batch_pred_embeddings))
                pred_embeddings = pred_embeddings_  #number of image[seg token num in each image, ]
                all_pred_embeddings.extend(pred_embeddings)
            
            batch_seg_token_counts = [torch.tensor(batch_seg_token_counts).to(seg_token_counts)]
            pred_embeddings = [torch.cat(all_pred_embeddings)]
            
            multimask_output = False
            pred_masks = []
            mask_scores = []
            if not self.vision_tower_for_mask:
                image_embeddings = self.get_visual_embs(images)
            else:
                
                img_embeddings = output_image_features.flatten(1, 2)  #[number of question in a batch, dim]
                img_embeddings = [img_embeddings.view(multi_scale_num, 1024, img_embeddings.shape[-1]).permute(0, 2, 1).view(multi_scale_num, img_embeddings.shape[-1], 32, 32)]
             
            for i in range(len(pred_embeddings)):
                if self.vision_tower_for_mask:
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=None,masks=None,text_embeds=pred_embeddings[i],)
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    _img_embeddings = self.model.image_feature_neck(img_embeddings[i]) 
                    out_size = 128
                    low_res_masks = torch.zeros([sparse_embeddings.shape[0], 1, out_size, out_size]).to(_img_embeddings)
                    if self.image_feature_scale_num > 1:
                        for l in range(self.image_feature_scale_num):
                            l_low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=_img_embeddings[l].unsqueeze(0), image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings[:, l].unsqueeze(1), dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, previous_masks=l_low_res_masks if l>0 else None, level_num=l)
                            low_res_masks = low_res_masks + self.multiscale_scalar[l] * F.interpolate(l_low_res_masks.float(), (out_size, out_size),mode="bilinear",align_corners=False,).to(l_low_res_masks)
                    else:
                        low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=_img_embeddings[0].unsqueeze(0), image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings[:, 0].unsqueeze(1), dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, )


                    pred_mask = self.postprocess_masks(
                        low_res_masks,
                        input_size=clip_resize_list[i],
                        original_size=original_size_list[i],
                    )
                else:
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i],
                    )

                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=original_size_list[i],
                    )
                
                pred_masks.append(pred_mask[:, 0])
                mask_score = (pred_mask[:, 0].sigmoid().flatten(1) * (pred_mask[:, 0] > 0).flatten(1)).sum(1) / ((pred_mask[:, 0] > 0).flatten(1).sum(1) + 1e-6)
                mask_scores.append(mask_score)
             
        
        return all_output_ids, pred_masks, batch_seg_token_counts, mask_scores


    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
        ) -> torch.Tensor:
            """
            Remove padding and upscale masks to the original image size.

            Arguments:
            masks (torch.Tensor): Batched masks from the mask_decoder,
                in BxCxHxW format.
            input_size (tuple(int, int)): The size of the image input to the
                model, in (H, W) format. Used to remove padding.
            original_size (tuple(int, int)): The original size of the image
                before resizing for input to the model, in (H, W) format.

            Returns:
            (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
                is given by original_size.
            """
           
            target_size = max(input_size)
            dtype = masks.dtype
            if self.vision_tower_for_mask:
                masks = F.interpolate(
                    masks.float(),
                    (target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
            
            if not self.masks_process_with_clip:
                assert input_size[0] <= target_size
                assert input_size[1] <= target_size
                masks = masks[..., : input_size[0], : input_size[1]]
                masks = F.interpolate(
                    masks, original_size, mode="bilinear", align_corners=False
                )
            
            masks = masks.to(dtype)
            # 
            return masks    

    