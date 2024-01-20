import glob
import os
from queue import Empty
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
import transformers

# from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
#                                    IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide, ResizeShortestSide

# from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset

from .vqa_dataset import VQADataset
from .multi_reason_seg_dataset import MultiReasonSegDataset

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

from model.llava import conversation as conversation_lib
# import conversation as conversation_lib
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    clip_resize_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    multi_reasons = []

    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        clip_resize,
        questions,
        sampled_classes,
        multi_reason,
        inference,

    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        clip_resize_list.append(clip_resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        multi_reasons.append(multi_reason)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.conv_templates['chatml'].copy() if conv_type == "chatml" else conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1" or "chatml":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    # print(conv)
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if conv.sep2 not in conversation:
            break
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if conv_type == "chatml":
                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(rou+sep, tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(rou+sep).input_ids) - 2

                if i == 0:
                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                # cur_len += round_len
                    
            else:
                parts = rou.split(sep)
                # if len(parts) != 2:
                #     break
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
        if conv_type == "chatml":
            cur_len = total_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )
        # print(tokenizer.model_max_length, cur_len, total_len)
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "clip_resize_list": clip_resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "multi_reason_list": multi_reasons,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        seg_token_num=1,
        num_classes_per_question=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,

    ):
        self.pad_train_clip_images = pad_train_clip_images
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()
        self.seg_token_num = seg_token_num
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")
        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list,
               
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list,
    
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
          
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list,
                    )
                )
            elif dataset == "multi_reason_seg":
                self.all_datasets.append(
                    MultiReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        num_classes_per_question,
                        seg_token_num,
                        pad_train_clip_images,
                        masks_process_with_clip,
                        preprocessor_config,
                        use_expand_question_list
                    )
                )
            
           
                

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        # print(type(data))
        output = data[0]

        if isinstance(data, MultiReasonSegDataset) :
            return *output, inference
        else:
            return *output, False, inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
        seg_token_num=1,
        pad_val_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',

    ):
       
        self.seg_token_num=seg_token_num
        self.base_image_dir = base_image_dir
        self.pad_val_clip_images = pad_val_clip_images
        self.masks_process_with_clip = masks_process_with_clip
        self.multiseg_inference = False
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            if 'multi' in ds:
                self.multiseg_inference = True
                ds = ds.split('multi')[-1]
            refer_api = REFER(self.base_image_dir+'/refer_seg/', ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "refer_seg/images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "refer_seg/images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size) 
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor, decoder_image_size) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = decoder_image_size - h
        padw = decoder_image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        _seg = "[SEG]" if self.seg_token_num == 1 else ' '.join(["[SEG{}]".format(i) for i in range(self.seg_token_num)])
        multi_sample_num = [6, 5, 4]
        multi_sample_index = 0

        while i < len(sampled_sents):
            conv.messages = []
            if self.multiseg_inference:
                sample_num = multi_sample_num[multi_sample_index]
                texts = [sampled_sents[k].strip() for k in range(i, i+sample_num)] if len(sampled_sents) - i >= sample_num else [sampled_sents[k].strip() for k in range(i, len(sampled_sents))]
                text = ', '.join(texts[:-1]) + ' and {}'.format(texts[-1]) if len(texts) > 1 else texts[0]
            else:
                text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "{}.".format(_seg))
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                if self.multiseg_inference:
                    answer = [_seg] * len(texts)
                    answer = ', '.join(answer[:-1]) + ' and ' + answer[-1] + '.' if len(answer) > 1 else answer[0]
                    conv.append_message(conv.roles[1], answer)
                else:
                    conv.append_message(conv.roles[1], "{}.".format(_seg))
            conversations.append(conv.get_prompt())
            if self.multiseg_inference:
                i += sample_num
                multi_sample_index = (multi_sample_index + 1) % len(multi_sample_num)
            else:
                i += 1

        # preprocess image for clip
        if self.pad_val_clip_images:
            image_clip = self.transform_clip.apply_image(image)
            clip_resize = image_clip.shape[:2]
            # print("self.clip_image_processor.size['shortest_edge']:", self.clip_image_processor.size['shortest_edge'])
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
            # clip_image_process(self.clip_image_processor, images)
        else:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        if self.masks_process_with_clip:
            mask_shape =  image_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = transform_mask(masks, mask_shape)

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            clip_resize,
            None,
            None,
            False,
            inference,
        )


def transform_mask(masks, size):
    height, width = masks.shape[-2:]
    short, long = (width, height) if width <= height else (height, width)
    requested_new_short = size
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)
    new_shape = (new_long, new_short) if width <= height else (new_short, new_long)
    masks = F.interpolate(masks[None].float(), size=new_shape, mode="nearest")[0].bool()

    orig_height, orig_width = new_shape
    crop_height, crop_width = size, size
    crop_height, crop_width = int(crop_height), int(crop_width)
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width
    assert top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width
    # if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
    masks = masks[..., top:bottom, left:right]

    return masks


def center_crop_image(image, size):
    orig_height, orig_width = image.shape[:2]
    crop_height, crop_width = size, size
    crop_height, crop_width = int(crop_height), int(crop_width)
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = (orig_width - crop_width) // 2
    right = left + crop_width
    assert top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width
    # if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
    image = image[top:bottom, left:right]

    return image


   