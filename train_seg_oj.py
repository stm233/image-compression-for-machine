# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import requests
# from transformers import CLIPProcessor, CLIPVisionModel
from collections import OrderedDict
from compressai.models.retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from compressai.models.retinanet import losses
import torch.nn.functional as F

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b)
    return -10 * math.log10(mse)

# file_dir = os.path.dirname(__file__)
# sys.path.append(file_dir)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.focalLoss = losses.FocalLoss()

    def forward(self, input, output, target):
        N, _, H, W = input.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        # out["mse_loss"],out["feature_loss"]  = 0,0
        out['obect_loss'] = [0,0]
        out["mse_loss"] = nn.MSELoss(reduction='none')(input, output["decompressedImage"]).mean()
        feature_loss_p2 = nn.MSELoss(reduction='none')(output["Student_output_features"]["p2"],
                                                       output["Teacher_output_features"]["p2"]).mean()
        feature_loss_p3 = nn.MSELoss(reduction='none')(output["Student_output_features"]["p3"],
                                                       output["Teacher_output_features"]["p3"]).mean()
        feature_loss_p4 = nn.MSELoss(reduction='none')(output["Student_output_features"]["p4"],
                                                       output["Teacher_output_features"]["p4"]).mean()
        feature_loss_p5 = nn.MSELoss(reduction='none')(output["Student_output_features"]["p5"],
                                                       output["Teacher_output_features"]["p5"]).mean()
        feature_loss_p6 = nn.MSELoss(reduction='none')(output["Student_output_features"]["p6"],
                                                       output["Teacher_output_features"]["p6"]).mean()

        out["feature_loss"] = feature_loss_p2 + feature_loss_p3 + feature_loss_p4 + feature_loss_p5 + feature_loss_p6

        # out['obect_loss'] = self.focalLoss(output["Student_classification"],
        #                                    output["Student_regression"],
        #                                    output["Student_anchors"],
        #                                    target)
        out["loss"] =  1000  * out["mse_loss"] + 100 * out["feature_loss"] + \
                       0 * (out['obect_loss'][0] + out['obect_loss'][1]) + \
                       self.lmbda * out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    #
    # parameters = {
    #     n
    #     for n, p in net.named_parameters()
    #     if not n.endswith(".quantiles") and p.requires_grad
    # }
    # aux_parameters = {
    #     n
    #     for n, p in net.named_parameters()
    #     if n.endswith(".quantiles") and p.requires_grad and not "teacherNet" and not "studentNet" in n
    # }
    # print(parameters)
    # TrainList = ['mu_Swin','sigma_Swin','LRP_Swin','cc_mean_transforms',
    #              'cc_scale_transforms','lrp_transforms']
                 # 'h_mean_s','h_scale_s']
    # NotTrainList = ['teacher'] # ,'student'
    # parameters = []
    # for name, param in net.named_parameters():
    #     boolTraining = True
    #     for paraName in NotTrainList:
    #         if paraName in name and param.requires_grad and not name.endswith(".quantiles"):
    #             boolTraining = False
    #             continue
    #     if boolTraining:
    #         parameters.append(name)

    TrainList = ['seg']
    parameters = []
    for name, param in net.named_parameters():
        boolTraining = False
        for paraName in TrainList:
            if paraName in name and param.requires_grad and not name.endswith(".quantiles"):
                boolTraining = True
                continue
        if boolTraining:
            parameters.append(name)

    aux_parameters = []
    for name, param in net.named_parameters():
        if param.requires_grad and name.endswith(".quantiles"):
            aux_parameters.append(name)
    # # print(parameters)
    parameters = set(parameters)
    aux_parameters = set(aux_parameters)

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    # inter_params = parameters & aux_parameters
    # union_params = parameters | aux_parameters

    # assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # clipimage = Image.open(requests.get(url, stream=True).raw)
    # clipimage = clipimage.resize((256,256))
    # clipinputs = transforms.ToTensor()(clipimage).unsqueeze(0)
    # print(clipinputs)
    # clipProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # clipinputs = clipProcessor(images=clipimage, return_tensors="pt")
    # clipinputs = clipinputs.to(device)
    start = time.time()

    for i, d in enumerate(train_dataloader):
        inputIMG = d.to(device)
        annotations= []
        # original_img, up_x4_img = d
        # inputIMG = d['img']
        # annotations = d['annot']
        # inputIMG = inputIMG.to(device)
        # annotations = annotations.to(device)
        # original_img = original_img.to(device)
        # up_x4_img = up_x4_img.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # inputData={}
        # inputData['pixel_values']=clipinputs
        out_net = model(inputIMG)

        out_criterion = criterion(inputIMG,out_net, annotations)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 200 == 0:
            enc_time = time.time() - start
            start = time.time()
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(inputIMG)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"] :.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f'\tfeature loss: {out_criterion["feature_loss"].item()*100:.2f} |'
                # f'\tobect loss: {out_criterion["obect_loss"][0].item():.2f} {out_criterion["obect_loss"][1].item():.2f}|'
                f"\tAux loss: {aux_loss.item():.2f} |"
                f"\ttime: {enc_time:.1f}"
            )
        #
        # if i > 100:
        #     break


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    objective_loss1 = AverageMeter()
    objective_loss2 = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            inputIMG = d.to(device)
            annotations = 0
            # out_net = model(d)
            # out_criterion = criterion(out_net, d)

            # inputIMG = d['img']
            # annotations = d['annot']
            inputIMG = inputIMG.to(device)
            # annotations = annotations.to(device)

            out_net = model(inputIMG)
            out_criterion = criterion(inputIMG, out_net, annotations)

            # scale = d['scale']
            # scale = scale.to(device)
            # scores, labels, boxes = out_net["scores"], out_net["labels"], out_net["boxes"]
            # boxes /= scale

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            objective_loss1.update(out_criterion["feature_loss"])
            objective_loss2.update(out_criterion["obect_loss"][1])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg.item():.3f} |"
        f"\tPSNR: {-10 * math.log10(mse_loss.avg):.3f} |"
        f'\tfearure loss: {objective_loss1.avg.item()*100:.2f} |'

        # f'\tobect loss: {objective_loss1.avg.item():.2f} {objective_loss2.avg.item():.2f}|'
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, filename[:-5]+"_best"+filename[-5:])


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="seg_oj_ICM",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, 
        default='/home/exx/Documents/Tianma/', # /data/Dataset/IRdataset # /data/Dataset/coco2017/
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=10,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-4,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda",  default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="/data/checkpoint/masked_RCNN_oj_ICM/10/", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--teachercheckpoint",
                         default="",  # ./train0008/18.ckpt
                         type=str, help="Path to a checkpoint")
    parser.add_argument("--checkpoint",
                        default="/data/checkpoint/masked_RCNN_oj_ICM/10/1044.ckpt",  # ./save_model/czigzag_1/8.ckpt
                        #\ /home/exx/Documents/Tianma/ICM/save_model/CRC_20/895.ckpt
                        # /home/tianma/Documents/ICM/save_model/promot_object_20/16.ckpt
                        # /data/checkpoint/faster_RCNN_ICM/10/985.ckpt
                        type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    # tv.transforms.RandomCrop(args.patchsize, pad_if_needed=True)
    # clipinputs = clipProcessor(images=clipimage, return_tensors="pt")
    # train_transforms = transforms.Compose(
    #     [transforms.RandomCrop(args.patch_size, pad_if_needed=True), transforms.ToTensor()]
    # )
    # transforms.CenterCrop(args.patch_size),
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size, pad_if_needed=True), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size, pad_if_needed=True), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="val2017", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="val2017", transform=test_transforms)

    # dataset_train = CocoDataset(args.dataset, set_name='train2017',
    #                             transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    # dataset_val = CocoDataset(args.dataset, set_name='val2017',
    #                           transform=transforms.Compose([Normalizer(), Resizer()]))

    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=args.batch_size, drop_last=True)
    # train_dataloader = DataLoader(dataset_train, num_workers=args.num_workers, collate_fn=collater, batch_sampler=sampler)
    #
    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=args.test_batch_size, drop_last=False)
    # test_dataloader = DataLoader(dataset_val, num_workers=args.num_workers, collate_fn=collater, batch_sampler=sampler_val)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = models[args.model]()
    net = net.to(device)

    print('GPU:',torch.cuda.device_count())

    # if args.cuda and torch.cuda.device_count() == 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.6, patience=6)
    criterion = RateDistortionLoss(lmbda=args.lmbda)



    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        new_state_dict = checkpoint["state_dict"]
        # new_state_dict = OrderedDict()
        #
        # for k, v in checkpoint["state_dict"].items():
        #     # print(k)
        #     if 'mu_Swin2' in k:
        #         print(k)
        #         continue
        #
        #     if 'sigma_Swin2' in k:
        #         print(k)
        #         continue
        #
        #     if 'LRP_Swin2' in k:
        #         print(k)
        #         continue
        #
        #     if 'mu_layers' in k:
        #         print(k)
        #         continue
        #     # k = k[7:]
        #     new_state_dict[k]=v

        net.load_state_dict(new_state_dict) # ,strict=False

        # net.load_state_dict(checkpoint["state_dict"])
# 
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        if epoch % 1 == 0:
            loss = test_epoch(epoch, test_dataloader, net, criterion)
            lr_scheduler.step(loss)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save and is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    args.save_path +str(epoch)+'.ckpt',
                )


if __name__ == "__main__":

    main(sys.argv[1:])
