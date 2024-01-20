
        
import glob
import json
import os
import random
from pycocotools import mask
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, PretrainedConfig
import transformers
import copy
from model.segment_anything.utils.transforms import ResizeLongestSide

from model.llava import conversation as conversation_lib

from .utils import (
    ANSWER_LIST,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    EXPLANATORY_QUESTION_LIST,
    LONG_QUESTION_LIST,
    SHORT_QUESTION_LIST,
)

from transformers.image_utils import make_list_of_images, to_numpy_array, infer_channel_dimension_format
from transformers.image_transforms import convert_to_rgb, to_channel_dimension_format
from transformers.image_processing_utils import get_size_dict
class MultiReasonSegValDataset(torch.utils.data.Dataset):
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
        crop_sam_image=False
        
    ):
        self.pad_val_clip_images= pad_val_clip_images
        self.masks_process_with_clip = masks_process_with_clip
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST 
        
        reason_seg_data, split = val_dataset.split("|")
        assert split == 'val'
        print(base_image_dir)

        json_file_name = "./dataset/muse_val.json"
        with open(json_file_name, 'r') as f:
            reason_file = json.load(f)
        images = []
        anns = []
        questions = []
        answers = []
        
       
        self.reason_seg_data = reason_file


        print("number of reason_seg samples: ", len(images))

       
    def __len__(self):
        return len(self.reason_seg_data)

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
        # import pdb;pdb.set_trace()
        image_info = self.reason_seg_data[idx]
        if 'file_name' in image_info:
            image_root = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2014")
            image_path = os.path.join(image_root, image_info['file_name'])
        else:
            if 'train2017' in image_info['coco_url']:
                image_root = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2017")
                image_path = os.path.join(image_root, image_info['coco_url'].split('/')[-1])
            else:
                image_root = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/val2017")
                image_path = os.path.join(image_root, image_info['coco_url'].split('/')[-1])

        # print(image_path)
        segs = image_info['ann_list']
        question = image_info['questions']
        gt_answer = image_info['answers']
        gt_target_count = []
        gt_category_name = []

        name_list = [ann['rephrased_name'] if 'rephrased_name' in ann else ann['category_name'] for ann in segs]
        _name_list = []
        name_count = {}
        for name in name_list:
            if name not in name_count:
                name_count[name] = 1
            else:
                name_count[name] += 1
        max_name_count = copy.deepcopy(name_count)
        name_loc = []
        phrase_loc = []
        for name, ann in zip(name_list, segs):
            x, y, w, h = ann['bbox']
            x0 = x
            x1 = x + w
            y0 = y
            y1 = y + h
            bbox_str = str([x0, y0, x1, y1])
            # bbox_str = str([x, y, w, h])
            if max_name_count[name] == 1:
                _name_list.append(name)
                name_loc.append('{} at {}'.format(name, bbox_str))
            else:
                name_loc.append('{} {} at {}'.format(name, str(max_name_count[name] - name_count[name] + 1), bbox_str))
                _name_list.append('{} {}'.format(name, str(max_name_count[name] - name_count[name] + 1)))
                name_count[name] -= 1
        name_loc = ', '.join(name_loc)
        name_str = ', '.join(_name_list)
        prompt_ins = "These objects in the image and their respective bounding box coordinates are as follows: {}. The image height is {}, width is {}.".format(name_loc, image_info['height'], image_info['width'])
        # prompt_ins_phrase = "These objects in the image and their respective bounding box coordinates are as follows: {}. The image height is {}, width is {}.".format(name_loc, image_info['height'], image_info['width'])

  
        img = cv2.imread(image_path)
        images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_size = images.shape[:2]
        # preprocess images for clip
        if self.pad_val_clip_images:
            images_clip = self.transform_clip.apply_image(images)
            clip_resize = images_clip.shape[:2]
            images_clip = self.preprocess(torch.from_numpy(images_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
            # clip_image_process(self.clip_image_processor, images)
        else:
            images_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = images_clip.shape[:2]


        
        images = self.transform.apply_image(images)  # preprocess images for sam
        resize = images.shape[:2]
        masks = []
        if len(segs) == 0:
            return self[0]
        for answer_list in gt_answer:
            gt_target_count.append(len(answer_list))
            gt_category_name.append(['(' + ann['rephrased_name'] + ' ' + str([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]]) + ')' for ann in answer_list])
            for answer in answer_list:
                rle = mask.frPyObjects(answer["segmentation"], image_info["height"], image_info["width"])
                m = mask.decode(rle)
                if len(m.shape) > 2:
                    # assert m.shape[-1] == 1, m.shape
                    m = np.sum(m, axis=2)  # so
                m = m.astype(np.uint8)
                masks.append(m)
        

        sampled_sents = question
        sampled_answers = gt_answer
        sampled_masks = masks
       

        image_name = image_path.split("/")[-1]
        # if self.explanatory != -1 and image_name in self.img_to_explanation:
        #     if random.random() < self.explanatory:
        #         choice = 2
        #     else:
        #         choice = random.randint(0, 1)

        questions = []
        answers = []
        for text, answer in zip(sampled_sents, sampled_answers):
            # if is_sentence:
            question_template = random.choice(self.long_question_list)
            questions.append(question_template.format(sent=text))
     

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], "")
                conversations.append(conv.get_prompt())
                i += 1



        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous(), self.img_size)

        image_name = image_path.split("/")[-1]
      
        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        if self.masks_process_with_clip:
            mask_shape =  images_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = transform_mask(masks, mask_shape)
 
        return (
            image_path,
            images,
            images_clip,
            conversations,
            masks,
            label,
            resize,
            clip_resize,
            (questions, gt_target_count, gt_category_name, prompt_ins),
            sampled_sents,
            False,
            True
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


def clip_image_process(clip_image_processor, images):
    # import pdb;pdb.set_trace()
    images = make_list_of_images(images)
    # crop_size = clip_image_processor.crop_size
    images = [convert_to_rgb(image) for image in images]
    images = [to_numpy_array(image) for image in images]
    input_data_format = infer_channel_dimension_format(images[0])
    resize_transform = ResizeLongestSide(clip_image_processor.size['shortest_edge'])
    images = [resize_transform.apply_image(image) for image in images]
    # images = [
    #         to_channel_dimension_format(image, clip_image_processor.data_format) for image in images
    #     ]

    images = [
                clip_image_processor.rescale(image=image, scale=clip_image_processor.rescale_factor)
                for image in images]
    images = [
                    clip_image_processor.normalize(image=image, mean=clip_image_processor.image_mean, std=clip_image_processor.image_std)
                    for image in images
                ]


    images = [
                F.pad(torch.tensor(image).permute(2, 0, 1), (0, 224-image.shape[1], 0, 224-image.shape[0])) for image in images
            ]
    #images[0]: 3, h, w
    return images[0]

if __name__ == "__main__":
    version = '/opt/tiger/PointVIS/LISA/ckpt/Llava-7B-V1-1'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    ret_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids
    dataset = MultiReasonSegValDataset('/opt/tiger/PointVIS/LISA/dataset', tokenizer, 'openai/clip-vit-large-patch14', val_dataset='MultiReasonseg|val')
    for i in range(len(dataset)):
        # import pdb;pdb.set_trace()
        data = dataset[i]
