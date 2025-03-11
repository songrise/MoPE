# -*- coding : utf-8 -*-
# @FileName  : main_classification.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Sep 08, 2023
# @Github    : https://github.com/songrise
# @Description: main script for multimodal classification


# general libs
import os,  argparse

from typing import Any, Dict
import warnings


warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("medium")
from utils import *
from models import swin, bert, classifier

from pytorch_lightning import LightningModule, Trainer, seed_everything
import pytorch_lightning as pl

from utils.datasets import create_loaders



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
    A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Full Pipeline Training")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="food101",
        choices=["food", "imdb", "snli"],
        help="which dataset to use.",
    )



    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Set the maximum batch size for training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of workers for pytorch's dataloader.",
    )


    # Encoder

    # General
    parser.add_argument("--name", default="", type=str, help="model name")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="If true, only validate segmentation.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="If true, only validate segmentation.",
    )

    parser.add_argument(
        "--max_epoch",
        type=int,
        nargs="+",
        default=10,
        help="max num of epoches for training",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed to provide (near-)reproducibility.",
    )
    parser.add_argument(
        "--exp_name",
        default="model",
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: model)",
    )
    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )



    # Optimisers
    parser.add_argument(
        "--lr_vis", type=float, default=4e-4, help="Learning rate for visual encoder."
    )
    parser.add_argument(
        "--lr_text", type=float, default=5e-4, help="Learning rate for text encoder."
    )

    parser.add_argument(
        "--wd_vis", type=float, default=1e-3, help="Weight decay for visual encoder."
    )
    parser.add_argument(
        "--wd_text", type=float, default=1e-3, help="Weight decay for text encoder."
    )

    # parser.add_argument("--lamda", type=float, default=LAMDA, help="Lamda for L1 norm.")
    parser.add_argument(
        "--warmup_epochs", type=float, default=1, help="warmup epochs for lr scheduler"
    )
    # parser.add_argument('-t', '--bn-threshold', type=float, default=BN_threshold,
    #                     help='Threshold for slimming BNs.')
    parser.add_argument(
        "--backbone",
        default="swinb_224",
        type=str,
        choices=["swinb_224", "swinb_384", "vitb"],
    )

    # ---------------model setting----------------
    parser.add_argument(
        "--fuse_method",
        default="late_concat",
        type=str,
        choices=[
            "late_concat",
            "instruct_v2t",
            "instruct_t2v",
            "instruct_moe_t2v",
            "instruct_moe_v2t",
            "instruct_mm_moe_t2v",
            "mope",
            "sequentialfuse",
            "img_only",
            "text_only",
            "p_sequential",
        ],
        help="how to fuse to modality",
    )
    parser.add_argument(
        "--train_instructor",
        action="store_true",
        default=False,
        help="whether the instructor should be trained at the same time.",
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=False,
        help="Whether to freeze (vision) encoder.",
    )
    parser.add_argument(   
        "--route_per_layer",
        action="store_true",
        default=True,
        help="whether to learn a routing weight for each layers",
    )
    parser.add_argument(
        "--dense_routing",
        action="store_true",
        default=True,
        help="whether to densely route expert (all experts are used)). temporarily deprecated arg",
    )
    # ----------------prompt learning---------------
    parser.add_argument(
        "--use_vpt", action="store_true", default=False, help="Whether to use VPT."
    )
    parser.add_argument(
        "--use_pbert",
        action="store_true",
        default=False,
        help="Whether to use Prompted Bert.",
    )

    parser.add_argument(
        "--vis_prompt_type",
        default="vpt",
        choices=["vpt"],
        type=str,
        help="how to apply prompt tuning?",
    )

    parser.add_argument(
        "--d_cross", type=int, default=8, help="dimension of cross-feature embd"
    )
    parser.add_argument(
        "--d_inter", type=int, default=2, help="dimension of cross-feature embd"
    )


    parser.add_argument(
        "--moe_n_experts", type=int, default=4, help="number of experts for moe"
    )
   
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=6,
        help="number of learnable visual prompts",
    )
    parser.add_argument(
        "--t_prompt_length",
        type=int,
        default=4,
        help="number of learnable text prompts",
    )

    parser.add_argument(
        "--use_static_prompt",
        action="store_true",
        default=True,
        help="whether to use additional static visual prompt, temporarily deprecated arg (always true)",
    )
    parser.add_argument(
        "--use_instruct",
        action="store_true",
        default=True,
        help="whether to use instructor, temporarily deprecated arg (always true)",
    )
    parser.add_argument(
        "--moe_top_k", type=int, default=1, help="number of top k experts to use, has to be used together with dense_routing, temporarily deprecated arg"
    )
    parser.add_argument(
        "--prompt_init",
        type=str,
        default="uniform",
        choices=["uniform", "normal", "othorgonal"],
        help="how prompt and experts are inited",
    )
    # --------------loss---------------
    
    parser.add_argument(
        "--w_imp", type=float, default=0.01, help="weight for importance loss"
    )
    parser.add_argument(
        "--smooth_label",
        action="store_true",
        default=False,
        help="whether to use lable smoothing",
    )
    parser.add_argument(
        "--w_othor", type=float, default=0.0, help="weight for othor loss"
    )
    parser.add_argument(
        "--w_contrast", type=float, default=0.0, help="weight for contrastive loss"
    )

    # ---------------debug------------------
    parser.add_argument("--exp_note", default="", type=str, help="experiment note")
    # ----------experiment setting----------------
    parser.add_argument(
        "--n_shot", default=0, type=int, help="number of shots for low shot learning"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="whether to finetune the model",
    )
    return parser.parse_args()


class Model(LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.validation_step_outputs = []
        # if args is a dictionary, convert to Namespace
        if self.args is not None and type(self.args) is dict:
            self.args = argparse.Namespace(**self.args)

        self.save_hyperparameters(self.args)
        if self.args.dataset == "food":
            self.num_classes = 101
        elif self.args.dataset == "imdb":
            self.num_classes = 23
        else:  # snli
            self.num_classes = 3


        if self.args.dataset == "food" or self.args.dataset == "snli":
            self.final_act = lambda x: F.log_softmax(x, dim=1)
            self.loss = nn.NLLLoss(ignore_index=self.args.ignore_label)
        elif self.args.dataset == "imdb":
            self.final_act = lambda x: F.sigmoid(x)
            self.loss = nn.BCEWithLogitsLoss()
        self.vision_classifier = swin.get_swin_classifier(
            num_classes=self.num_classes,
            backbone=self.args.backbone,
            use_vpt=self.args.use_vpt,
            moe_n_experts=self.args.moe_n_experts,
            prompt_length=self.args.prompt_length,
            use_static_prompt=self.args.use_static_prompt,
            prompt_init=self.args.prompt_init,
            use_instruct=self.args.use_instruct,
            d_cross=self.args.d_cross,
            d_inter=self.args.d_inter,
        )

        self.text_classifier = bert.BertClassifier(
            self.num_classes,
            use_prompt=self.args.use_pbert,
            prompt_length=self.args.t_prompt_length,
        )

        self.fuse_method = self.args.fuse_method
        self.classifier = classifier.VisionTextClassifiers(
            self.vision_classifier,
            self.text_classifier,
            self.num_classes,
            fusion_method=self.fuse_method,
            train_instructor=self.args.train_instructor,
            moe_n_experts=self.args.moe_n_experts,
            dense_routing=self.args.dense_routing,
            moe_top_k=self.args.moe_top_k,
            route_per_layer=self.args.route_per_layer,
        )
        if not self.args.train_instructor:
            self.text_classifier.requires_grad_(False)
        if self.args.finetune:
            self.vision_classifier.requires_grad_(True)
            self.text_classifier.requires_grad_(True)
        if self.fuse_method == "promptfuse":
            self.text_classifier.requires_grad_(True)
            self.text_classifier.freeze_backbone()
            self.vision_classifier.requires_grad_(False)
        if self.args.fuse_method == "sequentialfuse":
            self.vision_classifier.requires_grad_(True)
            self.text_classifier.requires_grad_(True)
        if self.args.fuse_method == "img_only":
            # self.vision_classifier.requires_grad_(True)
            self.text_classifier.requires_grad_(False)
        if self.args.fuse_method == "text_only":
            self.vision_classifier.requires_grad_(False)
            # self.text_classifier.requires_grad_(True)
        if self.args.fuse_method == "p_sequential":
            self.text_classifier.requires_grad_(True)
            self.text_classifier.freeze_backbone()
        if self.fuse_method == "instruct_v2t":
            self.text_classifier.freeze_backbone()
        # self.text_classifier.freeze_backbone()
        # self.vision_classifier.freeze_backbone()
        # TODO Sep 12: deprecated when prompt tuning, always freeze encoder
        if self.args.use_vpt and not self.args.freeze_encoder:
            print("[Warning] Using VPT without freezing encoder")

        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                num_params = np.prod(param.size())
                num_params_k = num_params / 1000
                print("{}, num_params: {}K".format(name, num_params_k))

    def training_step(self, batch, batch_idx):
        text_input, img_input, gt_label = batch
        # Compute outputs
        outputs, extra_out = self.classifier(img_input, text_input)

        # for amp compatibility, BCEWithLogitsLoss is used so no sigmoid act for multilabel during training
        if self.args.dataset == "food" or self.args.dataset == "snli":
            outputs = self.final_act(outputs)
        loss_val = self.loss(outputs, gt_label.squeeze())
        if extra_out is not None:
            if "importance_loss" in extra_out:
                imp_loss = extra_out["importance_loss"]
                self.log("imp_loss", imp_loss)
                loss_val += imp_loss * self.args.w_imp
            if "othor_loss" in extra_out:
                othor_loss = extra_out["othor_loss"]
                self.log("othor_loss", othor_loss)
                loss_val += othor_loss * self.args.w_othor

        crt_vision_lr = self.optimizers().param_groups[0]["lr"]
        self.log("vision_lr", crt_vision_lr)
        self.log("train_loss", loss_val)

        # train batch stats
        with torch.no_grad():
            if self.args.dataset == "food" or self.args.dataset == "snli":
                outputs = self.final_act(outputs)  # for metric calc
                pred_label = torch.argmax(outputs, dim=1)
                correct_cnt = torch.sum(pred_label == gt_label.squeeze()).item()
                all_cnt = pred_label.size(0)
                acc = correct_cnt / all_cnt
                self.log("train_acc", acc)

            elif self.args.dataset == "imdb":  # multilable, calc f1
                pred_label = (outputs > 0.5).int()
                f1_micro = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="micro",
                )
                self.log("train_f1_micro", f1_micro)
                f1_macro = f1_score(
                    gt_label.squeeze().cpu().numpy(),
                    pred_label.cpu().numpy(),
                    average="macro",
                )
                self.log("train_f1_macro", f1_macro)
            return loss_val

    def validation_step(self, batch, batch_idx):
        text_input, img_input, gt_label = batch
        # Compute outputs
        outputs, extra_out = self.classifier(img_input, text_input)
        # outputs = self.text_classifier(text_input, dump_attn=True)
        outputs = self.final_act(outputs)
        # loss_val = self.loss(outputs, label.squeeze())
        if self.args.dataset == "food" or self.args.dataset == "snli":
            pred_label = torch.argmax(outputs, dim=1)
        elif self.args.dataset == "imdb":
            pred_label = (outputs > 0.5).int()

        ret_dict = {
            "pred_label": pred_label,
            "gt_label": gt_label.squeeze(),
            "text_input": text_input,
            "img_input": img_input,
        }

        if extra_out is not None:
            if "moe_scores" in extra_out:
                ret_dict["moe_scores"] = extra_out["moe_scores"]
            if "cls_" in extra_out:
                ret_dict["cls"] = extra_out["cls_"].detach().cpu()

        self.validation_step_outputs.append(ret_dict)
        return ret_dict

    def on_validation_epoch_end(self) -> None:
        if self.args.dataset == "food" or self.args.dataset == "snli":
            all_cnt = 0
            correct_cnt = 0
            expert_img_dir = os.path.join("debug", "route")
            os.makedirs(expert_img_dir, exist_ok=True)
            cls_features = []
            gt_labels = []
            for step_out in self.validation_step_outputs:
                pred_label = step_out["pred_label"]
                gt_label = step_out["gt_label"]
                # loss_val = step_out["loss"]
                all_cnt += pred_label.size(0)
                correct_cnt += torch.sum(pred_label == gt_label).item()

                if "cls" in step_out:
                    cls_features.append(step_out["cls"])
                gt_labels.append(gt_label)

            acc = correct_cnt / all_cnt
            self.log("val_acc", acc)
        elif self.args.dataset == "imdb":
            all_preds = []
            all_labels = []
            
            for step_out in self.validation_step_outputs:
                pred_label = step_out["pred_label"]
                gt_label = step_out["gt_label"]

                all_preds.append(pred_label)
                all_labels.append(gt_label)
            f1_macro = f1_score(
                torch.cat(all_labels).squeeze().cpu().numpy(),
                torch.cat(all_preds).cpu().numpy(),
                average="macro",
            )
            self.log("val_f1_macro", f1_macro)
            f1_micro = f1_score(
                torch.cat(all_labels).squeeze().cpu().numpy(),
                torch.cat(all_preds).cpu().numpy(),
                average="micro",
            )
            self.log("val_f1_micro", f1_micro)

        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> Any:

        vision_lr = self.args.lr_vis
        text_lr = self.args.lr_text

        optimizer_cfg = [
            {
                "params": self.vision_classifier.parameters(),
                "lr": vision_lr,
                "weight_decay": self.args.wd_vis,
            },
            {
                "params": self.text_classifier.parameters(),
                "lr": text_lr,
                "weight_decay": self.args.wd_text,
            },
        ]
        # todo refactor as returned dict of params from init of the classifier
        if self.fuse_method == "late_concat":
            optimizer_cfg.append(
                {
                    "params": self.classifier.fusion_head.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_t2v":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_v2t":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": text_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_moe_t2v":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
            optimizer_cfg.append(
                {
                    "params": self.classifier.moe_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "instruct_moe_v2t":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
            optimizer_cfg.append(
                {
                    "params": self.classifier.moe_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "promptfuse":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": vision_lr,
                    "weight_decay": 1e-4,
                }
            )
        elif self.fuse_method == "p_sequential":
            optimizer_cfg.append(
                {
                    "params": self.classifier.instruct_proj.parameters(),
                    "lr": text_lr,
                    "weight_decay": 1e-4,
                }
            )
        optimizer = torch.optim.AdamW(optimizer_cfg)
        # optimizer = ScheduledOptim(optimizer,)
        # step lr,
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[3, 6], gamma=0.4
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def overwrite_args(self, args):
        """Avoid the exception caused by lighting when loading incompatible args from model ckpt."""
        self.args = args

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # remove the frozen parameter
        filter = ["attention", "mlp", "attn", "downsample", "intermediate"]
        for k in list(checkpoint["state_dict"].keys()):
            for f in filter:
                if f in k:
                    del checkpoint["state_dict"][k]

        return super().on_save_checkpoint(checkpoint)

    def benchmark_memory(self):
        # do forward pass with batch size 1 for 10000 times
        # measure the memory
        # ensure cuda classifer
        self.classifier.to("cuda").eval()
        dummy_input_img = torch.randn(16, 3, 224, 224).cuda()
        dummy_input_text = ["test"] * 16
        peak_mems = []
        self.eval()
        # optim = self.configure_optimizers()["optimizer"]
        device = torch.cuda.current_device()
        # warm up
        for i in range(2):
            out, _ = self.classifier(dummy_input_img, dummy_input_text)
            dummy_loss = out.sum()
            # dummy_loss.backward()
            # optim.step()
        for i in range(50):
            # optim.zero_grad()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before_mem = torch.cuda.memory_allocated(device) / 1024**2

            # Perform inference
            with torch.no_grad():
                out, __ = self.classifier(dummy_input_img, dummy_input_text)
            # dummy_loss = out.sum()
            # dummy_loss.backward()
            # optim.step()
            # Measure memory after inference
            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_mems.append(peak_mem)
            torch.cuda.empty_cache()
            # measure the memory
        # print stat
        print(f"Mean peak memory: {np.mean(peak_mems)} MB")
        print(f"Std peak memory: {np.std(peak_mems)} MB")

    def benchmark_inference_speed(self):
        # do forward pass with batch size 1 for 10000 times
        # measure the time
        dummy_input_img = torch.randn(16, 3, 224, 224).cuda().half()
        dummy_input_text = ["test"] * 16
        self.eval()
        self.classifier.eval()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        self.classifier.to("cuda")
        repetitions = 1000
        with torch.no_grad():  # warm up
            for i in range(50):
                self.classifier(dummy_input_img, dummy_input_text)
        print("Start timing...")
        timings = []
        with torch.no_grad():
            torch.cuda.synchronize()
            for i in range(repetitions):
                starter.record()
                self.classifier(dummy_input_img, dummy_input_text)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
            torch.cuda.synchronize()
        print(f"Mean time: {np.mean(timings)} ms")
        print(f"Std time: {np.std(timings)} ms")


def main():
    args = get_arguments()
    seed_everything(args.random_seed)

    if args.dataset == "food":
        data_path = "data/food101"
    elif args.dataset == "imdb":
        data_path = "data/mmimdb"
    elif args.dataset == "snli":
        data_path = "data/snli-ve/data"
    if args.dataset == "food":
        batch_size = 38
    elif args.dataset == "imdb":
        batch_size = 16
    elif args.dataset == "snli":
        batch_size = 128
    # batch_size = 64 if args.dataset == "snli" else 32
    batch_size_arg = args.batch_size
    batch_size = min(batch_size, batch_size_arg)

    train_loader, val_loader, test_loader = create_loaders(
        data_path,
        batch_size,
        args.num_workers,
        args.n_shot,
    )

    model = Model(args)
    if args.dataset == "food" or args.dataset == "snli":
        save_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_acc",
            filename="{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            mode="max",
        )
    elif args.dataset == "imdb":  # for mm-imdb the metric is f1 score instead of acc.
        save_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_f1_micro",
            filename="{epoch:02d}-{f1_micro:.2f}",
            save_top_k=1,
            mode="max",
        )
    logger = pl.loggers.TensorBoardLogger("logs", name=args.exp_name)

    max_epoch = args.max_epoch

    trainer = Trainer(
        accelerator="gpu",
        # callbacks=[save_callback],
        precision=16,
        logger=logger,
        max_epochs=max_epoch,
        val_check_interval=0.33,
        # check_val_every_n_epoch=check_val_every_n_epoch,
        gradient_clip_val=1.0,
    )



    if args.evaluate:
        if args.ckpt is not None and args.ckpt != "":

            model = Model.load_from_checkpoint(args.ckpt, strict=False)
            model.overwrite_args(args)
            # save all scripts to logger dir
            model.eval()
            trainer.validate(model, test_loader)
        else:
            raise Warning("Trying to evaluate model but with no checkpoint provided") 
        model.benchmark_inference_speed()
        model.benchmark_memory()
    else:
        # assert args.ckpt is not None
        if args.ckpt is not None and args.ckpt != "":
            model = Model.load_from_checkpoint(args.ckpt, strict=False)
        # save all scripts to logger dir
        os.system("cp -r *py models utils %s" % logger.log_dir)
        trainer.fit(model, train_loader, val_loader)
        trainer.validate(model, test_loader)


if __name__ == "__main__":

    main()
