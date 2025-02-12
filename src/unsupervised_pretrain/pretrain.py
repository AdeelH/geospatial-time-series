#!/usr/bin/env python3

# BSD 3-Clause License
#
# Copyright (c) 2022-23, Azavea, Element84, James McClain
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import logging
import math
import sys

import numpy as np
import torch
import tqdm
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader

from datasets import DigestDataset, SeriesDataset
from models import (SeriesEfficientNetb0, SeriesMobileNetv3, SeriesResNet18,
                    freeze, unfreeze)

if __name__ == "__main__":
    # yapf: disable
    # Command line arguments
    parser = argparse.ArgumentParser(description="Pretrain a model using a bunch unlabeled Sentinel-2 time series")
    parser.add_argument("cog_dirs", nargs="+", type=str, help="Paths to the data")
    parser.add_argument("--architecture", type=str, default="resnet18", choices=["resnet18", "mobilenetv3", "efficientnetb0"], help="The model architecture to use (default: resnet18)")
    parser.add_argument("--batch-size", type=int, default=6, help="The batch size (default: 6)")
    parser.add_argument("--dataset", type=str, default="series", choices=["series", "digest"], help="The type of data found in the data directories (default: series)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use for training (default: cuda)")
    parser.add_argument("--epochs", type=int, default=8, help="The number of epochs (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate (default: 1e-3)")
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', default=True, help='Whether to start from pretrained weights (default: True)')
    parser.add_argument("--num-workers", type=int, default=2, help="Number of worker processes for the DataLoader (default: 2)")
    parser.add_argument("--output-dir", type=str, default=".", help="The directory where logs and artifacts will be deposited (default: .)")
    parser.add_argument("--pth-in", type=str, help="Optional path to a .pth file to use as a starting point for model training")
    parser.add_argument("--pth-out", type=str, default="model.pth", help="The name of the output .pth file (default: model.pth)")
    parser.add_argument("--series-length", type=int, default=8, help="The number of time steps in each sample (default: 8)")
    parser.add_argument("--size", type=int, default=512, help="The tile size (default: 512)")
    # yapf: enable
    args = parser.parse_args()

    # Dataset and DataLoader for training set
    if args.dataset == "series":
        assert args.size % 64 == 0
        dataset = SeriesDataset(args.cog_dirs,
                                dim=args.size,
                                series_length=args.series_length)
    elif args.dataset == "digest":
        dataset = DigestDataset(args.cog_dirs)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Logging
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format="%(asctime)-15s %(message)s")
    log = logging.getLogger()
    if args.output_dir is not None:
        fh = logging.FileHandler(f"{args.output_dir}/output.log")
        log.addHandler(fh)

    log.info(args.__dict__)

    # PyTorch device
    device = torch.device(args.device)

    # Model
    if args.pth_in is None:
        if args.architecture == "resnet18":
            model = SeriesResNet18(pretrained=args.pretrained).to(device)
        elif args.architecture == "mobilenetv3":
            model = SeriesMobileNetv3(pretrained=args.pretrained).to(device)
        elif args.architecture == "efficientnetb0":
            model = SeriesEfficientNetb0(pretrained=args.pretrained).to(device)
    elif args.pth_in:
        model = torch.load(args.pth_in, map_location=device).to(device)
        log.info(f"Model weights loaded from {args.pth_in}")

    # Loss function, optimizer, scheduler, miner
    base_obj = losses.TripletMarginLoss().to(device)
    obj = losses.SelfSupervisedLoss(base_obj).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        steps_per_epoch=len(dataloader),
        epochs=args.epochs)
    # miner = miners.BatchEasyHardMiner(pos_strategy="semihard",
    #                                   neg_strategy="easy")

    model.train()
    for epoch in range(0, args.epochs):
        training_losses = []
        for data in tqdm.tqdm(dataloader):
            imagery_a, imagery_b = data
            opt.zero_grad()
            embeddings_a = model(imagery_a.to(device))
            embeddings_b = model(imagery_b.to(device))
            # good_pairs = miner(embeddings_a, embeddings_b).to(device)
            # loss = obj(embeddings_a, embeddings_b, good_pairs)
            loss = obj(embeddings_a, embeddings_b)
            loss.backward()
            training_losses.append(loss.item())
            opt.step()
            sched.step()
        training_losses = np.mean(training_losses)
        log.info(f"epoch={epoch} training_loss={training_losses}")
        if args.output_dir is not None:
            model_save_path = f"{args.output_dir}/{args.pth_out}"
            model_save_path = model_save_path.replace(".pth", f"-{epoch}.pth")
            torch.save(model, model_save_path)

    # Save the model after the last epoch if output_dir is provided
    if args.output_dir is not None:
        model_save_path = f"{args.output_dir}/{args.pth_out}"
        torch.save(model, model_save_path)
        log.info(f"Model saved to {model_save_path}")
