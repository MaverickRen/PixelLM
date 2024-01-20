import glob
import json
import os
import random
from unicodedata import category
from pycocotools import mask
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, PretrainedConfig
import transformers
# from dataset import clip_image_process
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
# from .conversation import get_default_conv_template

# from .utils import (
#     MR_SINGLE_ANSWER_LIST,
#     MR_MULTI_ANSWER_LIST,
#     ANSWER_LIST,
#     DEFAULT_IM_END_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IMAGE_PATCH_TOKEN,
#     DEFAULT_IMAGE_TOKEN,
#     EXPLANATORY_QUESTION_LIST,
#     LONG_QUESTION_LIST,
#     SHORT_QUESTION_LIST,
#     EXPAND_LONG_QUESTION_LIST,
#     INSTANCE_QUESTION_LIST
# )

from .utils import (
    MR_SINGLE_ANSWER_LIST,
    MR_MULTI_ANSWER_LIST,
    ANSWER_LIST,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    EXPLANATORY_QUESTION_LIST,
    LONG_QUESTION_LIST,
    SHORT_QUESTION_LIST,
    EXPAND_LONG_QUESTION_LIST,
)
from transformers.image_utils import make_list_of_images, to_numpy_array, infer_channel_dimension_format
from transformers.image_transforms import convert_to_rgb, to_channel_dimension_format
from transformers.image_processing_utils import get_size_dict
class MultiReasonSegDataset(torch.utils.data.Dataset):
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
        reason_seg_data="MultiReasonSeg|train",
        explanatory=0.1,
        num_classes_per_question=1,
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.single_answer_list = MR_SINGLE_ANSWER_LIST
        self.multi_answer_list = MR_MULTI_ANSWER_LIST   
        self.seg_token_num = seg_token_num
        self.num_classes_per_question = num_classes_per_question
        
        self.masks_process_with_clip = masks_process_with_clip
        self.pad_train_clip_images = pad_train_clip_images
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge']) 
        
        if use_expand_question_list:
            self.long_question_list.extend(EXPAND_LONG_QUESTION_LIST)
        
        print("___________self.single_answer_list:", self.single_answer_list)
        print("___________self.multi_answer_list:", self.multi_answer_list)
                
        reason_seg_data, split = reason_seg_data.split("|")
        json_file_name = './dataset/lvistrain_gpt4v_phrase27k_conversation50k.json'
        lvis_name_path = './dataset/livs_category_name_with_sys.json'
        with open(json_file_name, 'r') as f:
            reason_file = json.load(f)
        with open(lvis_name_path, 'r') as f:
            lvis_name_dict = json.load(f)
        images = []
        anns = []
        questions = []
        answers = []
        
        # for image_ann in reason_file:
        #     image_path = os.path.join(image_root, image_ann['file_name'])
        #     seg_list = image_ann['ann_list']
        #     images.append(image_path)
        #     anns.append(seg_list)
        #     questions.append(image_ann['question'])
        #     answers.append(image_ann['answer'])
        self.reason_seg_data = reason_file
        self.lvis_name_dict = lvis_name_dict


        print("number of reason_seg samples: ", len(images))

        
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
        # import pdb;pdb.set_trace()
        # images_ann, anns, questions, answers = self.reason_seg_data
        idx = random.randint(0, len(self.reason_seg_data) - 1)
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

        anns = image_info['ann_list']
        question = image_info['questions'] if 'questions' in image_info else None
        gt_answer = image_info['answers'] if 'answers' in image_info else None
        if question is not None:
            text_answers = image_info['text_answers'] if 'text_answers' in image_info else [None] * len(gt_answer)
        else:
            text_answers = None

        img = cv2.imread(image_path)
        images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_size = images.shape[:2]
        # preprocess images for clip
        if self.pad_train_clip_images:
            image_clip = self.transform_clip.apply_image(images)
            clip_resize = image_clip.shape[:2]
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_image_processor.size['shortest_edge'])
            # clip_image_process(self.clip_image_processor, images)
        else:
            image_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]

        
        images = self.transform.apply_image(images)  # preprocess images for sam
        resize = images.shape[:2]
        masks = []
        if len(anns) == 0:
            return self[0]

        category_ids = [ann['category_id'] for ann in anns]
        category_ids = list(set(category_ids))
        sampled_num = min(self.num_classes_per_sample, len(category_ids))
        sampled_category_ids = np.random.choice(category_ids, size=sampled_num, replace=False)

        sampled_sents = question
        sampled_answers = gt_answer
        sampled_masks = masks
        sample_text_answers = text_answers

        image_name = image_path.split("/")[-1]
        questions = []
        answers = []
        use_assign_list = []
        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token)

        if question is not None:
            for text, answer_list, text_answer in zip(sampled_sents, sampled_answers, sample_text_answers):
                # if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))

                
                for answer in answer_list:
                    rle = mask.frPyObjects(answer["segmentation"], image_info["height"], image_info["width"])
                    m = mask.decode(rle)
                    if len(m.shape) > 2:
                        # assert m.shape[-1] == 1, m.shape
                        m = np.sum(m, axis=2)  # so
                    m = m.astype(np.uint8)
                    masks.append(m)

        
                use_assign = False
                if text_answer is not None:
                    _text_answer = text_answer.format(seg='[SEG]') if self.seg_token_num == 1 else text_answer.format(seg=seg_token)
                    answers.append(_text_answer)
                    use_assign_list.append(False)
                else:
                    target_list = [a['rephrased_name'] if (random.random() > 0.1 and 'rephrased_name' in a) else a['category_name'] for a in answer_list ]
                    target_answer = []
                    separate_answer = random.randint(0, 1)
                    _seg = ['[SEG]'] * len(target_list)
                    if len(target_list) > 1:
                        part1 = ', '.join(_seg[:-1])
                        part2 = ' and ' + _seg[-1]
                        _seg = part1 + part2 
                    else:
                        _seg = _seg[0]
                    
                    if separate_answer:
                        choice_list = self.single_answer_list
                        answer_temp = random.choice(choice_list) if self.seg_token_num == 1 else random.choice(choice_list).replace('[SEG]', seg_token)
                        use_assign = False if "{class_name}" in answer_temp else True
                        for i, sampled_cls in enumerate(target_list):
                            _answer_temp = answer_temp.format(class_name=sampled_cls) if "{class_name}" in answer_temp else answer_temp
                            target_answer.append(_answer_temp[:-1])
                        if len(target_answer) > 1:
                            part1 = ', '.join(target_answer[:-1])
                            part2 = ' and ' + target_answer[-1]
                            target_answer = part1 + part2 + '.'
                        else:
                            target_answer = target_answer[0] + '.'
                    else:
                        answer_temp = random.choice(self.multi_answer_list)
                        _answer_temp = answer_temp.format(class_name=', '.join(target_list).lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                        use_assign = False if "{class_name}" in answer_temp else True
                        _answer_temp = _answer_temp if self.seg_token_num == 1 else _answer_temp.replace('[SEG]', seg_token)
                        target_answer = _answer_temp

                    answers.append(target_answer)
                    use_assign_list.append(use_assign)
            
        else:
            for sampled_category_id in sampled_category_ids:
                question_template = random.choice(self.instance_question_list)
                category_names = self.lvis_name_dict[str(sampled_category_id)]
                category_name = random.choice(category_names)
                questions.append(question_template.format(class_name=category_name))
                answer_list = [ann for ann in anns if ann['category_id'] == sampled_category_id]
                for answer in answer_list:
                    rle = mask.frPyObjects(answer["segmentation"], image_info["height"], image_info["width"])
                    m = mask.decode(rle)
                    if len(m.shape) > 2:
                        # assert m.shape[-1] == 1, m.shape
                        m = np.sum(m, axis=2)  # so
                    m = m.astype(np.uint8)
                    masks.append(m)

                target_list = [a['rephrased_name'] if random.random() > 0.1 else a['category_name'] for a in answer_list ]
                target_answer = []
                separate_answer = random.randint(0, 1)
                _seg = ['[SEG]'] * len(target_list)
                if len(target_list) > 1:
                    part1 = ', '.join(_seg[:-1])
                    part2 = ' and ' + _seg[-1]
                    _seg = part1 + part2 
                else:
                    _seg = _seg[0]

                separate_answer = random.randint(0, 1)
                # if len(answer_list) == 1 or separate_answer:
                choice_list = self.single_answer_list
                answer_temp = random.choice(choice_list) if self.seg_token_num == 1 else random.choice(choice_list).replace('[SEG]', seg_token)
                use_assign = False if "{class_name}" in answer_temp else True
                for i, sampled_cls in enumerate(target_list):
                    _answer_temp = answer_temp.format(class_name=sampled_cls) if "{class_name}" in answer_temp else answer_temp
                    target_answer.append(_answer_temp[:-1])
                if len(target_answer) > 1:
                    part1 = ', '.join(target_answer[:-1])
                    part2 = ' and ' + target_answer[-1]
                    target_answer = part1 + part2 + '.'
                else:
                    target_answer = target_answer[0] + '.'
                # else:
                #     answer_temp = random.choice(self.multi_answer_list)
                #     _answer_temp = answer_temp.format(class_name=', '.join(target_list).lower(), seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(seg=_seg)
                #     use_assign = False 
                #     _answer_temp = _answer_temp if self.seg_token_num == 1 else _answer_temp.replace('[SEG]', seg_token)
                #     target_answer = _answer_temp

                answers.append(target_answer)
                use_assign_list.append(use_assign)
    
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1


        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous(), self.img_size)

        image_name = image_path.split("/")[-1]
        # if (
        #     self.explanatory != -1
        #     and image_name in self.img_to_explanation
        #     and choice == 2
        # ):
        #     masks = torch.rand(0, *ori_size)
        #     label = torch.ones(ori_size) * self.ignore_label
        # else:
        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        if self.masks_process_with_clip:
            mask_shape =  image_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = transform_mask(masks, mask_shape)
        # print(question)
        # visualize(images, masks, resize, image_name)
        # print(conversations)
        return (
            image_path,
            images,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            clip_resize,
            questions,
            sampled_sents,
            use_assign_list
        )

def visualize(image, masks, resize, image_name):
    import cv2
    import numpy as np
    import copy
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711])
    h, w = masks.shape[-2:]
    # import pdb;pdb.set_trace()
    image = image[:, :resize[0], :resize[1]]
    image = F.interpolate(
                    image[None].float(),
                    (h, w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
    image = image.cpu().permute(1, 2, 0)
    image = (image * std) + mean
    image = (image * 255).int().numpy()
    color = [np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([0, 0, 255]), np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255]), np.array([100, 100, 0]), np.array([100, 0, 100]), np.array([0, 100, 100]), np.array([30, 30, 100])]
    masks = masks.cpu().numpy()
    # img_show = image
    # img_show = (img_show * std) + mean
    # img_show = (img_show * 255).int().numpy()
    # _img_show = copy.deepcopy(img_show)
    # masks_show = np.zeros([masks.shape[1], masks.shape[2], 3])
    for i, mask in enumerate(masks):
        # import pdb;pdb.set_trace()
        fg = mask > 0
        image[fg] = image[fg] *0.5 + color[i]*0.5
        
    cv2.imwrite('/mnt/bn/mmdataset/zhongwei/visualize/multi_reason_train/{}.jpg'.format(image_name), image[:, :, ::-1])
    
    

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
    dataset = MultiReasonSegDataset('/opt/tiger/PointVIS/LISA/dataset', tokenizer, 'openai/clip-vit-large-patch14')
    for i in range(len(dataset)):
        import pdb;pdb.set_trace()
        data = dataset[i]

