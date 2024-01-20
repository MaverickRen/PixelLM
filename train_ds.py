import argparse
import os
import shutil
import sys
import time
from functools import partial
import logging
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
import copy
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from model.PixelLM import PixelLMForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)
from utils.matcher import match_pred
from utils.multi_reason_seg_val_dataset import MultiReasonSegValDataset

import requests
import json
import base64

# from azure.identity import DefaultAzureCredential
def parse_args(args):
    parser = argparse.ArgumentParser(description="PixelLM Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="pixellm", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--seg_token_num", default=1, type=int)
    parser.add_argument("--num_classes_per_question", default=1, type=int)
    parser.add_argument("--pad_train_clip_images", action="store_true", default=False)
    parser.add_argument("--masks_process_with_clip", default=False, action="store_true")
    parser.add_argument("--preprocessor_config", default='', type=str)
    parser.add_argument("--resize_vision_tower", action="store_true", default=False)
    parser.add_argument("--resize_vision_tower_size", default=224, type=int)
    parser.add_argument("--vision_tower_for_mask", action="store_true", default=False)
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--use_expand_question_list", action="store_true", default=False)
    parser.add_argument("--separate_mm_projector", action="store_true", default=False)
    parser.add_argument("--image_feature_scale_num", default=1, type=int)

    
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)

        log_filename = os.path.join(args.log_dir, 'meta.log')
        i = 1
        while os.path.exists(log_filename):
            log_filename = os.path.join(args.log_dir, 'meta_{}.log'.format(str(i)))
            i += 1
        logger = logging.getLogger('pixellm_logger')
        logger.setLevel(logging.INFO)


        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(args)
    
    else:
        writer = None
        logger = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if args.seg_token_num*args.image_feature_scale_num == 1:
        num_added_tokens = tokenizer.add_tokens("[SEG]")
        args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    else:
        new_tokens = ["[SEG{}]".format(i) for i in range(args.seg_token_num*args.image_feature_scale_num)]
        num_added_tokens = tokenizer.add_tokens(new_tokens)
        args.seg_token_idx = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in new_tokens]


    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_token_num": args.seg_token_num,
        "logger": logger,
        "tokenizer": tokenizer,
        "local_rank": args.local_rank,
        "pad_train_clip_images": args.pad_train_clip_images,
        "resize_vision_tower": args.resize_vision_tower,
        "resize_vision_tower_size": args.resize_vision_tower_size,
        "vision_tower_for_mask": args.vision_tower_for_mask,
        "separate_mm_projector": args.separate_mm_projector,
        "masks_process_with_clip": args.masks_process_with_clip,
        "image_feature_scale_num": args.image_feature_scale_num,

    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    
    model = PixelLMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_pixellm_modules(model.get_model().config)
    
    
    for p in vision_tower.parameters():
        p.requires_grad = False
    if args.resize_vision_tower_size == 224:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "mask_decoder",
                                "image_feature_neck",
                                "prompt_encoder",
                                # "out_mm_projector"
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    if args.weight:
        # with open(args.weight, "rb") as f:
        #     state_dict = torch.load(f)
        state_dict = torch.load(args.weight, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    trainable_list = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
    if args.resize_vision_tower_size != 224:
        trainable_list.append('mm_projector')
        # trainable_list.append('out_mm_projector')
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in trainable_list
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
        seg_token_num=args.seg_token_num*args.image_feature_scale_num,
        num_classes_per_question=args.num_classes_per_question,
        pad_train_clip_images=args.pad_train_clip_images,
        masks_process_with_clip=args.masks_process_with_clip,
        preprocessor_config=args.preprocessor_config,
        use_expand_question_list=args.use_expand_question_list,

    )
    print("____seg_token_num in data:________: ", args.seg_token_num*args.image_feature_scale_num)
    multi_val = False
    if args.no_eval == False:
        token_num = args.seg_token_num*args.image_feature_scale_num 
        if len(args.val_dataset.split('||')) == 1:
            if args.val_dataset.split('|')[0] == 'MultiReasonSeg':
                ValDataset_type = MultiReasonSegValDataset
            else:   
                ValDataset_type = ValDataset

            val_dataset_names = [args.val_dataset]
            val_dataset = ValDataset_type(
                args.dataset_dir,
                tokenizer,
                args.vision_tower,
                args.val_dataset,
                args.image_size,
                seg_token_num=token_num,
                pad_val_clip_images=args.pad_train_clip_images,
                masks_process_with_clip=args.masks_process_with_clip,
                preprocessor_config=args.preprocessor_config,
            )
            print(
                f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
            )
        else:
            multi_val = True
            val_dataset_names = args.val_dataset.split('||')
            val_dataset = []
            for val_dataset_name in val_dataset_names:
                if val_dataset_name.split('|')[0] == 'MultiReasonSeg':
                    ValDataset_type = MultiReasonSegValDataset
                else:   
                    ValDataset_type = ValDataset
                val_dataset.append(
                    ValDataset_type(
                        args.dataset_dir,
                        tokenizer,
                        args.vision_tower,
                        val_dataset_name,
                        args.image_size,
                        seg_token_num=token_num,
                        pad_val_clip_images=args.pad_train_clip_images,
                        masks_process_with_clip=args.masks_process_with_clip,
                        preprocessor_config=args.preprocessor_config,
                    )                   
                )

            # print(
            #     f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
            # )

    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        if multi_val:
            val_sampler = [torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False, drop_last=False
            ) for dataset in val_dataset]
            val_loader = [torch.utils.data.DataLoader(
                dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    local_rank=args.local_rank,
                ),
            ) for dataset, sampler in zip(val_dataset, val_sampler)]
        else:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    local_rank=args.local_rank,
                ),
            )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        if args.val_dataset.split('|')[0] == 'MultiReasonSeg':
            ar_validate(val_loader, model_engine, 0, writer, args, logger, val_dataset_names, tokenizer, args.seg_token_num, args.image_feature_scale_num)
        else:
            giou, ciou = validate(val_loader, model_engine, 0, writer, args, logger, val_dataset_names)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args, logger, val_dataset_names)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "best_ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)

        save_dir = os.path.join(args.log_dir, "ckpt_model")
        if args.local_rank == 0:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))

            # print("____loss____:", loss, "_____losses.val___", losses.val)
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()


            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()


        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter



def ar_validate(val_loader, model_engine, epoch, writer, args, logger, val_dataset_names, tokenizer, seg_token_num=1, image_feature_scale_num=1):

    pred_file = []
    acc_iou_list = []
    log_dir = args.log_dir
    out_file = os.path.join(log_dir, 'out_file_{}.json'.format(args.local_rank))
    acc_iou_out_file = os.path.join(log_dir, 'acc_list_{}.json'.format(args.local_rank))
    model_engine.eval()
    if not isinstance(val_loader, list):
        val_loader = [val_loader]
    assert len(val_dataset_names) == len(val_loader)
    k = 0
    for loader, dataset_name in zip(val_loader, val_dataset_names):
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        for input_dict in tqdm.tqdm(loader):
            image_pred = {}
            image_pred['answers'] = []
            image_pred['question_gt_category_name'] = []
            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            image_paths = input_dict['image_paths']
            images = input_dict['images']
            images_clip = input_dict['images_clip']
            # conversations = input_dict['conversations']
            resize_list = input_dict['resize_list']
            clip_resize_list = input_dict['clip_resize_list']
            label_list = input_dict['label_list']
            input_ids = input_dict['input_ids']
            gt_masks = input_dict['masks_list']
            questions_list = input_dict['questions_list']
            original_size_list = [label.shape for label in label_list]

          
            if k == 0:
                model_engine(**input_dict)
            
            output_ids, pred_masks, batch_seg_token_counts, mask_scores = model_engine.base_model.evaluate(images_clip, images, input_ids, resize_list, clip_resize_list, original_size_list, max_new_tokens=512, tokenizer=tokenizer)
            text_outputs = []
            for output_id in output_ids:
                _output_id = copy.deepcopy(output_id[0])
                _output_id[_output_id==-200] = 31999
                text_output = tokenizer.decode(_output_id, skip_special_tokens=False)
                text_output = (
                    text_output.replace(DEFAULT_IMAGE_PATCH_TOKEN, "")
                    .replace("\n", "")
                    .replace("  ", "")
                )
                text_outputs.append(text_output)
     
            image_path = input_dict['image_paths'][0]
            print("idx:", k, "image_path:", input_dict['image_paths'][0], "text_output: ", text_outputs)
            k += 1

            batch_seg_token_count = batch_seg_token_counts[0]
            batch_seg_token_count = batch_seg_token_count.cumsum(-1)  
            batch_seg_token_count = torch.cat(
                    [torch.zeros(1).long().cuda(), batch_seg_token_count], dim=0
                )
            pred_mask = pred_masks[0]
            gt_mask = gt_masks[0]
            mask_score = mask_scores[0]
            max_num = max(len(pred_masks[0]), len(gt_masks[0]))
            assigned_gt_masks = []
            assigned_pred_masks = []

            questions_list = input_dict['questions_list']
            gt_target_count = questions_list[0][1]
            gt_category_name = questions_list[0][2]
            prompt_ins = questions_list[0][3]
            gt_target_count = torch.tensor(gt_target_count).to(batch_seg_token_count).cumsum(-1)  
            gt_target_count = torch.cat(
                    [torch.zeros(1).long().cuda(), gt_target_count], dim=0
                )

            assign_length = []
            assign_indice = []
            assign_acc = []
            total_pred_count = []
            pred_count = []
            assert len(batch_seg_token_count) == len(gt_target_count)
            for j in range(len(batch_seg_token_count) -1):
                start_i = batch_seg_token_count[j]
                end_i = batch_seg_token_count[j+1]
                q_start_i = gt_target_count[j]
                q_end_i = gt_target_count[j+1]
                question_inputs = pred_mask[start_i:end_i]
                question_mask_scores = mask_score[start_i:end_i]
                question_targets = gt_mask[q_start_i:q_end_i]

                indice = match_pred(question_inputs.detach(), question_targets.detach())
                assigned_pred_mask = pred_mask[start_i:end_i][indice[0]]
                assigned_pred_mask = (assigned_pred_mask > 0).int()
                assigned_gt_mask = gt_mask[q_start_i:q_end_i][indice[1]]
                unassugned_indice = []
                unassugned_indice_pred = []
                for i in range(len(gt_mask[q_start_i:q_end_i])):
                    if i not in indice[1]:
                        unassugned_indice.append(i)
                for i in range(len(pred_mask[start_i:end_i])):
                    if i not in indice[0]:
                        unassugned_indice_pred.append(i)

                unassugned_indice = np.array(unassugned_indice)
                unassugned_indice_pred = np.array(unassugned_indice_pred)
                unassigned_gt_mask = gt_mask[q_start_i:q_end_i][unassugned_indice]
                unassigned_pred = pred_mask[start_i:end_i][unassugned_indice_pred]

                empty_gt = torch.zeros_like(unassigned_pred)
                empty_pred = torch.zeros_like(unassigned_gt_mask)

                assigned_gt_mask = torch.cat((assigned_gt_mask, unassigned_gt_mask))
                assigned_pred_mask = torch.cat((assigned_pred_mask, empty_pred))

                assigned_gt_mask = torch.cat((assigned_gt_mask, empty_gt))
                assigned_pred_mask = torch.cat((assigned_pred_mask, unassigned_pred))

                assigned_gt_masks.append(assigned_gt_mask)
                assigned_pred_masks.append(assigned_pred_mask)

                question_gt_category_name = gt_category_name[j]
                text_output = text_outputs[j]
                sorted_id = sorted(range(len(indice[0])), key=lambda k: indice[0][k], reverse=False)
                sorted_gt_indice = indice[1][sorted_id]
                sorted_pred_indice = indice[0][sorted_id]
                # sorted_category_name = question_gt_category_name[]
                seg_token = ' '.join(['[SEG{}]'.format(str(s)) for s in range(seg_token_num*image_feature_scale_num)]) if seg_token_num*image_feature_scale_num > 1 else '[SEG]'
                _text_output = text_output
                in_count = 0
                question_gt_category_name_list = []
                for count in range(text_output.count(seg_token)):
                    if count in sorted_pred_indice:
                        _text_output = _text_output.replace(seg_token, question_gt_category_name[sorted_gt_indice[in_count]], 1)  
                        question_gt_category_name_list.append(question_gt_category_name[sorted_gt_indice[in_count]][1:-1])
                        in_count += 1
                    else:
                        question_gt_category_name_list.append('None []')
                        _text_output = _text_output.replace(seg_token, '(None [])', 1) 
  
                image_pred['image_path'] = input_dict['image_paths'][0]
                image_pred['questions'] = questions_list[0][0]
                answer = _text_output.split('ASSISTANT:')[-1]
                answer = answer.replace('<unk>', '')
                image_pred['answers'].append(answer)
                image_pred['question_gt_category_name'].append(question_gt_category_name_list)
                assign_length.extend([True]*len(indice[0]))
                assign_length.extend([False]*(len(assigned_gt_mask)-len(indice[0])))
                assign_indice.append(indice[0].tolist())
                total_pred_count.append(len(assigned_gt_mask))
                pred_count.append(len(pred_mask[start_i:end_i]))

            assigned_gt_masks = torch.cat(assigned_gt_masks)
            output_list = torch.cat(assigned_pred_masks)
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i, is_assign in zip(assigned_gt_masks, output_list, assign_length):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
                # if is_assign:
                assign_acc.append((intersection_i.tolist(), union_i.tolist()))
            image_pred['assign_length'] = assign_length
            image_pred['assign_indice'] = assign_indice
            image_pred['assign_acc'] = assign_acc
            image_pred['total_pred_count'] = total_pred_count
            image_pred['pred_count'] = pred_count
            image_pred['prompt_ins'] = prompt_ins
            pred_file.append(image_pred)
            
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / max_num
            intersection_meter.update(intersection), union_meter.update(
                union
            ), acc_iou_meter.update(acc_iou, n=max_num)
            print(acc_iou)

            _acc_iou = acc_iou.tolist()
            _acc_iou.append(max_num)
            _acc_iou.append(input_dict['image_paths'][0])
            acc_iou_list.append(_acc_iou)
        

        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()

        with open(acc_iou_out_file, 'w') as f:
            json.dump(acc_iou_list, f)
        with open(out_file, 'w') as f:
            json.dump(pred_file, f)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]

        if args.local_rank == 0:
            writer.add_scalar("val/giou", giou, epoch)
            writer.add_scalar("val/ciou", ciou, epoch)
            print("{}, epoch: {}, giou: {:.4f}, ciou: {:.4f}".format(dataset_name, epoch, giou, ciou))
            logger.info("{}, epoch: {}, giou: {:.4f}, ciou: {:.4f}".format(dataset_name, epoch, giou, ciou))



def validate(val_loader, model_engine, epoch, writer, args, logger, val_dataset_names):
    

    model_engine.eval()
    if not isinstance(val_loader, list):
        val_loader = [val_loader]
    for loader, dataset_name in zip(val_loader, val_dataset_names):
        if 'NYU' in dataset_name:
            continue
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        for input_dict in tqdm.tqdm(loader):
            torch.cuda.empty_cache()

            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            with torch.no_grad():
                output_dict = model_engine(**input_dict)

            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection), union_meter.update(
                union
            ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]

        if args.local_rank == 0:
            writer.add_scalar("val/giou", giou, epoch)
            writer.add_scalar("val/giou", ciou, epoch)
            logger.info("{}, epoch: {}, giou: {:.4f}, ciou: {:.4f}".format(dataset_name, epoch, giou, ciou))
            print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

            

    return giou, ciou




if __name__ == "__main__":
    main(sys.argv[1:])
