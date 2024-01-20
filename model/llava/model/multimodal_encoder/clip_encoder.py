import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
from .custom_clip import _CLIPVisionModel
import torch.nn.functional as F
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.pad_vit = getattr(args, "pad_train_clip_images", False)
        self.resize_vision_tower = getattr(args, "resize_vision_tower", False)
        self.resize_vision_tower_size = getattr(args, "resize_vision_tower_size", 224)
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        # import pdb;pdb.set_trace()
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        if self.pad_vit:
            self.vision_tower = _CLIPVisionModel.from_pretrained(
                self.vision_tower_name, low_cpu_mem_usage=True
            )
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name, low_cpu_mem_usage=True
            )
        vision_tower = self.vision_tower
        resize_vision_tower_size = self.resize_vision_tower_size
        if self.resize_vision_tower:
            origin_p_num = int(vision_tower.vision_model.embeddings.num_patches ** 0.5)
            vision_tower_embed_dim = vision_tower.vision_model.embeddings.embed_dim
            vision_tower.vision_model.embeddings.image_size = resize_vision_tower_size
            vision_tower.vision_model.embeddings.num_patches = (resize_vision_tower_size // vision_tower.vision_model.embeddings.patch_size) **2
            vision_tower.vision_model.embeddings.num_positions = vision_tower.vision_model.embeddings.num_patches + 1
            vision_tower.vision_model.embeddings.register_buffer("position_ids", torch.arange(vision_tower.vision_model.embeddings.num_positions).expand((1, -1)))
            new_p_num = int(vision_tower.vision_model.embeddings.num_patches ** 0.5)

            origin_position_embedding_weight = vision_tower.vision_model.embeddings.position_embedding.weight
            origin_position_embedding_weight_cls = origin_position_embedding_weight[-1:]
            origin_position_embedding_weight = origin_position_embedding_weight[:-1].permute(1, 0).view(1, vision_tower_embed_dim, origin_p_num, origin_p_num)
            new_position_embedding_weight = F.interpolate(origin_position_embedding_weight, (new_p_num, new_p_num), mode='bilinear', align_corners=False)[0]
            new_position_embedding_weight = new_position_embedding_weight.flatten(-2).permute(1, 0)
            new_position_embedding_weight = torch.cat((new_position_embedding_weight, origin_position_embedding_weight_cls), dim=0)
            vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(vision_tower.vision_model.embeddings.num_positions, vision_tower_embed_dim)
            vision_tower.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(new_position_embedding_weight).to(origin_position_embedding_weight)
            vision_tower.vision_model.embeddings.position_ids = vision_tower.vision_model.embeddings.position_ids.to(origin_position_embedding_weight.device)

        self.vision_tower = vision_tower
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features, [image_forward_outs.hidden_states[-11][:, 1:]]

    @torch.no_grad()
    def forward(self, images, attention_mask=None):
        pre_image_features = []
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True, attention_mask=attention_mask
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            if isinstance(self.vision_tower, _CLIPVisionModel):
                image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True, attention_mask=attention_mask
                )
            else:
                image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True
                )
            image_features, pre_image_features = self.feature_select(image_forward_outs)
            image_features = image_features.to(images.dtype)
            pre_image_features = [f.to(images.dtype) for f in pre_image_features]
        torch.cuda.empty_cache()
        return image_features, pre_image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
