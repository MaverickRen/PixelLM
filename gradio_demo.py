from share import *


import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)



version = '/opt/tiger/PointVIS/LISA/runs/lisanew_13b_llama2_padclip_cls6_4seg_lora64_sepproj_2scale_extemp_multi74k/hf_model'
model_max_length = 1000
image_feature_scale_num = 2
seg_token_num = 4
precision = 'bf16'
pad_train_clip_images = True   
preprocessor_config ='./configs/preprocessor_448.json' 
resize_vision_tower = True
resize_vision_tower_size=448 
vision_tower_for_mask = True
separate_mm_projector=True
local_rank = 0
load_in_4bit = False
load_in_8bit = False
lora_r = 8
image_size=1024
use_mm_start_end = True

color = [np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([0, 0, 255]), np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255]), np.array([100, 100, 0]), np.array([100, 0, 100]), np.array([0, 100, 100]), np.array([30, 30, 100])]
# css = """
# .one {
#     color: rgb(0, 255, 0) !important;
# }
# .two {
#     color: rgb(0, 0, 255) !important;
# }
# .three {
#     color: rgb(255, 0, 0) !important;
# }

# .four {
#     color: rgb(0, 255, 255) !important;
# }

# .five {
#     color: rgb(255, 0, 255) !important;
# }

# .six {
#     color: rgb(255, 255, 0) !important;
# }
# """
css = """
.one {
    background-color: rgb(0, 255, 0) !important;
}
.two {
    background-color: rgb(79, 113, 190) !important;
}
.three {
    background-color: rgb(255, 0, 0) !important;
}

.four {
    background-color: rgb(0, 255, 255) !important;
}

.five {
    background-color: rgb(255, 0, 255) !important;
}

.six {
    background-color: rgb(255, 255, 0) !important;
}
"""

css_map = {1:"one", 2:"two", 3:"three", 4:"four", 5:'five', 6:'six'}
def init_model():
    vision_tower = "openai/clip-vit-large-patch14"
    tokenizer = AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if seg_token_num*image_feature_scale_num == 1:
        num_added_tokens = tokenizer.add_tokens("[SEG]")
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        new_tokens = ["[SEG{}]".format(i) for i in range(seg_token_num*image_feature_scale_num)]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        seg_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]


    torch_dtype = torch.float32
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16":
        torch_dtype = torch.half
    # import pdb;pdb.set_trace()
    kwargs = {"torch_dtype": torch_dtype,  "seg_token_num": seg_token_num, "image_feature_scale_num": image_feature_scale_num, "pad_train_clip_images": pad_train_clip_images,"resize_vision_tower": resize_vision_tower,
                "resize_vision_tower_size": resize_vision_tower_size,
                "vision_tower_for_mask": vision_tower_for_mask,
                "separate_mm_projector": separate_mm_projector,
                }
    if load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        version, low_cpu_mem_usage=True, vision_tower=vision_tower, seg_token_idx=seg_token_idx,  **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        precision == "fp16" and (not load_in_4bit) and (not load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)
    transform = ResizeLongestSide(image_size)
    if pad_train_clip_images:
        transform_clip = ResizeLongestSide(clip_image_processor.size['shortest_edge'])
    else:
        transform_clip = None
    model.eval()

    return model, transform_clip, clip_image_processor, transform, tokenizer


model, transform_clip, clip_image_processor, transform, tokenizer = init_model()

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def process(image_np, prompt):
    
    with torch.no_grad():
        use_mm_start_end = True
        conv_type = "llava_v1"
        conv = conversation_lib.conv_templates[conv_type].copy()
        conv.messages = []

        # prompt = input("Please input your prompt: ")
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]
        if pad_train_clip_images:
            image_clip = transform_clip.apply_image(image_np)
            clip_resize = image_clip.shape[:2]
            image_clip = preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), img_size=clip_image_processor.size['shortest_edge'])
            image_clip = image_clip.unsqueeze(0).cuda()
        else:
            image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
            )
            clip_resize = image_clip.shape[-2:]
        if precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        clip_resize = [clip_resize]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if precision == "bf16":
            image = image.bfloat16()
        elif precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks, _, _ = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            clip_resize_list=clip_resize,
            original_size_list=original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        text_output = text_output.split('ASSISTANT:')[-1]
        text_output = text_output.replace('</s>', '')
        seg_token = ["[SEG{}]".format(i) for i in range(seg_token_num*image_feature_scale_num)]
        seg_token = ' '.join(seg_token)
        final_text_output = ''
        print("text_output: ", text_output)
        # import pdb;pdb.set_trace()
        all_save_img = image_np.copy()
        for i, _pred_mask in enumerate(pred_masks):
            if _pred_mask.shape[0] == 0:
                continue
            
            for j, pred_mask in enumerate(_pred_mask):
                start = text_output.find(seg_token)
                part = text_output[:start]
                # import pdb;pdb.set_trace()
                if "the laptop computer" in text_output[:start]:
                    part = part.replace("the laptop computer", "<span class='{}'>the laptop computer</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "TV" in text_output[:start]:
                    part = part.replace("TV, the painting, sofa and coffee table are", "<span class='{}'>the TV</span>, <span class='{}'>painting</span>, <span class='{}'>sofa</span> and <span class='{}'>coffee table</span> are as shown in the figure".format(css_map[j+1],css_map[j+2],css_map[j+3],css_map[j+4]))
                    print(part)
                    final_text_output += part
                elif "the cat" in text_output[:start]:
                    part = part.replace("the cat", "<span class='{}'>the cat</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the chair" in text_output[:start]:
                    part = part.replace("the chair", "<span class='{}'>the chair</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "cat" in text_output[:start]:
                    part = part.replace("cat is", "Ok, they are <span class='{}'>cat</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "your laptop" in text_output[:start]:
                    part = part.replace("laptop", "<span class='{}'>laptop</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "laptop" in text_output[:start]:
                    part = part.replace("laptop is", "<span class='{}'>laptop</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "table" in text_output[:start]:
                    part = part.replace("table is", "<span class='{}'>table</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "potted plant" in text_output[:start]:
                    part = part.replace("potted plant is", "<span class='{}'>potted plant</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "chair" in text_output[:start]:
                    part = part.replace("chair is", "<span class='{}'>chair</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the cozy sofa" in text_output[:start]:
                    part = part.replace("the cozy sofa", "<span class='{}'>the cozy sofa</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the remote control" in text_output[:start]:
                    part = part.replace("the remote control", "<span class='{}'>the remote control</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the television set" in text_output[:start]:
                    part = part.replace("the television set", "<span class='{}'>the television set</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the delicious cupcake" in text_output[:start]:
                    part = part.replace("the delicious cupcake", "<span class='{}'>the delicious cupcake</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the bottle" in text_output[:start]:
                    part = part.replace("the bottle", "<span class='{}'>the bottle</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the large computer monitor" in text_output[:start]:
                    part = part.replace("the large computer monitor", "<span class='{}'>the large computer monitor</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the smaller computer monitor" in text_output[:start]:
                    part = part.replace("the smaller computer monitor", "<span class='{}'>the smaller computer monitor</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "The computer keyboard" in text_output[:start]:
                    part = part.replace("The computer keyboard", "<span class='{}'>The computer keyboard</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the mouse" in text_output[:start]:
                    part = part.replace("the mouse", "<span class='{}'>the mouse</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the knife" in text_output[:start]:
                    part = part.replace("the knife", "<span class='{}'>the knife</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the plate" in text_output[:start]:
                    part = part.replace("the plate", "<span class='{}'>the plate</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "one of the mugs" in text_output[:start]:
                    part = part.replace("one of the mugs", "<span class='{}'>one of the mugs</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the pepper" in text_output[:start]:
                    part = part.replace("the pepper", "<span class='{}'>the pepper</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "a warm jacket" in text_output[:start]:
                    part = part.replace("a warm jacket", "<span class='{}'>a warm jacket</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "trousers" in text_output[:start]:
                    part = part.replace("trousers", "<span class='{}'>trousers</span>".format(css_map[j+1]))
                    final_text_output += part
                elif "the segmentation result is" in text_output[:start]:
                    part = part.replace("the segmentation result is", "<span class='{}'>the hillside covered with snow</span> is a great place for skiing".format(css_map[j+1]))
                    final_text_output += part
                # final_text_output += "<span class='{}'>[SEG]</span>".format(css_map[j+1])
                text_output = text_output.replace(seg_token, '', 1)
                text_output = text_output[start:]

                pred_mask = pred_mask.float().detach().cpu().numpy()
                pred_mask = pred_mask > 0

                all_save_img[pred_mask] = (image_np * 0.5 + pred_mask[:, :, None].astype(np.uint8) * color[j%len(color)] * 0.5)[pred_mask]

        

            
    
        # all_save_img = torch.from_numpy(all_save_img).permute(2, 0, 1).numpy
        all_save_img = cv2.cvtColor(all_save_img, cv2.COLOR_RGB2BGR)
        final_text_output += text_output
        all_save_img = Image.fromarray(np.uint8(all_save_img))
    return [all_save_img, f"{final_text_output}"]


block = gr.Blocks(css=css).queue()
with block:
    with gr.Row():
        gr.Markdown("## PixelLM Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Image(label='Output', show_label=False, elem_id="gallery")
            result_text = gr.HTML(label='Output Text')
    ips = [input_image, prompt]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, result_text])


block.launch(server_name='10.107.99.118', server_port=9726, share=True)