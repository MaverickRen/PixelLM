import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide, ResizeShortestSide

from .grefer import G_REFER
from .refer import REFER
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST, SINGLE_ANSWER_LIST, MULTI_ANSWER_LIST, EXPAND_QUESTION_LIST
# from grefer import G_REFER
# from refer import REFER
# from utils import ANSWER_LIST, SHORT_QUESTION_LIST, SINGLE_ANSWER_LIST, MULTI_ANSWER_LIST, EXPAND_QUESTION_LIST

class ReferSegDataset(torch.utils.data.Dataset):
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
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        num_classes_per_question=1,
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,

    ):
        self.pad_train_clip_images = pad_train_clip_images
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size) 

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.single_answer_list = SINGLE_ANSWER_LIST
        self.multi_answer_list = MULTI_ANSWER_LIST   
        self.seg_token_num = seg_token_num
        self.num_classes_per_question = num_classes_per_question

        self.masks_process_with_clip = masks_process_with_clip
        self.pad_train_clip_images = pad_train_clip_images
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])
        if use_expand_question_list:
            self.short_question_list.extend(EXPAND_QUESTION_LIST)
            
        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

    def __len__(self):
        return self.samples_per_epoch

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
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        
        max_num_classes_per_sample = self.num_classes_per_question * self.num_classes_per_sample
        if len(sents) >= max_num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=max_num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        sampled_ann_ids, sampled_classes = allocate_class(sampled_ann_ids, sampled_classes, max_question_num=self.num_classes_per_sample, max_class_per_question=self.num_classes_per_question)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess image for clip
        if self.pad_train_clip_images:
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

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        questions = []
        answers = []
        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token)
        
        # for text in sampled_classes:
        #     text = text.strip()
        #     assert len(text.split("||")) == 1
        #     question_template = random.choice(self.short_question_list)
        #     questions.append(question_template.format(class_name=text.lower()))
        #     answers.append(random.choice(self.answer_list))
        for text_per_question in sampled_classes:
            target = ''
            _seg = []
            for i, text in enumerate(text_per_question):
                assert len(text.split("||")) == 1
                if i == len(text_per_question) - 1:
                    _seg.append('[SEG]') if self.seg_token_num == 1 else _seg.append(seg_token)
                    target = target + (' and '  + text) if i != 0 else target + text
                elif i == 0:
                    target += text
                    _seg.append('[SEG]') if self.seg_token_num == 1 else _seg.append(seg_token)
                    continue
                else:
                    _seg.append('[SEG]') if self.seg_token_num == 1 else _seg.append(seg_token)
                    target += (', '  + text)
            # _seg = ', '.join(_seg)
            if len(_seg) > 1:
                part1 = ', '.join(_seg[:-1])
                part2 = ' and ' + _seg[-1]
                _seg = part1 + part2 
            else:
                _seg = _seg[0]
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=target.lower()))
            separate_answer = random.randint(0, 1)
            
            if len(text_per_question) == 1:
                choice_list = self.answer_list
                answer_temp = random.choice(choice_list) if self.seg_token_num == 1 else random.choice(choice_list).replace('[SEG]', seg_token)
                answer_temp = answer_temp.format(class_name=target.lower()) if "{class_name}" in answer_temp else answer_temp
                answers.append(answer_temp)
            elif separate_answer:
                target_answer = []
                answer_temp = random.choice(self.single_answer_list) if self.seg_token_num == 1 else random.choice(self.single_answer_list).replace('[SEG]', seg_token)
                for i, sampled_cls in enumerate(text_per_question):
                    _answer_temp = answer_temp.format(class_name=sampled_cls) if "{class_name}" in answer_temp else answer_temp
                    target_answer.append(_answer_temp[:-1])
                if len(target_answer) > 1:
                    part1 = ', '.join(target_answer[:-1])
                    part2 = ' and ' + target_answer[-1]
                    target_answer = part1 + part2 + '.'
                else:
                    target_answer = target_answer[0] + '.'
                answers.append(target_answer)
            else:
                answer_temp = random.choice(self.multi_answer_list)
                _answer_temp = answer_temp.format(class_name=target.lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                answers.append(_answer_temp) 


        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            # print(conv.get_prompt())
            i += 1

        
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)

        flag = False
        masks = []
        for ann_id_per_question in sampled_ann_ids:
            for ann_id in ann_id_per_question:
                if isinstance(ann_id, list):
                    flag = True
                    if -1 in ann_id:
                        assert len(ann_id) == 1
                        m = np.zeros((image_info["height"], image_info["width"])).astype(
                            np.uint8
                        )
                    else:
                        m_final = np.zeros(
                            (image_info["height"], image_info["width"])
                        ).astype(np.uint8)
                        for ann_id_i in ann_id:
                            ann = annotations[ann_id_i]

                            if len(ann["segmentation"]) == 0:
                                m = np.zeros(
                                    (image_info["height"], image_info["width"])
                                ).astype(np.uint8)
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
                            m_final = m_final | m
                        m = m_final
                    masks.append(m)
                    continue

                ann = annotations[ann_id]

                if len(ann["segmentation"]) == 0:
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                    masks.append(m)
                    continue

                if type(ann["segmentation"][0]) == list:  # polygon
                    rle = mask.frPyObjects(
                        ann["segmentation"], image_info["height"], image_info["width"]
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

        masks = np.stack(masks, axis=0)

        # if ds == 'grefcoco' and flag:
        #     import shutil
        #     image_name = image_path.split("/")[-1]
        #     save_dir = os.path.join("/group/30042/xlai/LISA_refactor_final/debug", image_name.split(".")[0])
        #     os.makedirs(save_dir, exist_ok=True)
        #     shutil.copy(image_path, save_dir)
        #     for i in range(masks.shape[0]):
        #         cv2.imwrite(os.path.join(save_dir, "{}_{}_{}.jpg".format(image_name, i, sampled_classes[i])), masks[i].astype(np.int32) * 100)

        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
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
            label,
            resize,
            clip_resize,
            questions,
            sampled_classes,
        )



def allocate_class(sampled_ann_ids, sampled_ann_classes, max_question_num=3, max_class_per_question=3):
    if len(sampled_ann_ids) < max_question_num:
        max_question_num = len(sampled_ann_ids)
    sample_num = len(sampled_ann_classes)
    question_id = np.arange(max_question_num)
    class_counts = np.arange(max_question_num) * 0
    new_sampled_ann_ids = [[] for _ in range(max_question_num)] 
    new_sampled_ann_classes = [[] for _ in range(max_question_num)] 
    sample_ids = np.arange(sample_num)
    np.random.shuffle(sample_ids)
    for i in range(sample_num):
        if 0 in class_counts:
            choose_id = np.random.choice(np.where(class_counts == 0)[0], size=1)[0]
        else:
            choose_id = np.random.choice(np.where(class_counts < max_class_per_question)[0], size=1)[0]
        
        class_counts[choose_id] += 1
        sample_id = sample_ids[i]
        new_sampled_ann_ids[choose_id].append(sampled_ann_ids[sample_id])
        new_sampled_ann_classes[choose_id].append(sampled_ann_classes[sample_id])

    return new_sampled_ann_ids, new_sampled_ann_classes



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