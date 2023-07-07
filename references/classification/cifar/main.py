import os
import datetime
import argparse
import copy
import wandb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.utils.data

import utils.common as utils
from data import load_data
from utils.train import train, validate
from models import *


def parse_args():
    parser = argparse.ArgumentParser("Cifar-10 TVDCN")

    parser.add_argument(
        "--data_dir", type=str, default="~/data", help="path to dataset"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg16",
        help="architecture",
    )
    parser.add_argument(
        "--job_dir", type=str, default="result", help="path for saving models"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--epochs", type=int, default=400, help="num of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="init learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--gpu", type=str, default="0", help="Select gpu to use")
    parser.add_argument("--name", type=str, default="", help="wandb project name")

    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.job_dir = os.path.join(args.job_dir, args.arch)
if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logger = utils.get_logger(os.path.join(args.job_dir, now + ".txt"))


def main():
    logger.info("args = %s", args)
    wandb.init(project=f"TVDCN_{args.name}_{args.arch}", config=vars(args))

    # setup
    train_loader, val_loader = load_data(args.data_dir, args.batch_size)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    # load model
    logger.info("Building model")
    model = eval(args.arch)().cuda()
    logger.info(model)

    # train
    logger.info("Training model:")
    model = finetune(model, train_loader, val_loader, args.epochs, criterion)

    # save model
    path = os.path.join(args.job_dir, f"{args.arch}.pt")
    torch.save({"state_dict": model.state_dict()}, path)


def finetune(model, train_loader, val_loader, epochs, criterion):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    _, best_top1_acc, _ = validate(val_loader, model, criterion, logger)
    best_model_state = copy.deepcopy(model.state_dict())
    epoch = 0
    while epoch < epochs:
        train(epoch, train_loader, model, criterion, optimizer, scheduler, logger)
        _, valid_top1_acc, _ = validate(val_loader, model, criterion, logger)

        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_model_state = copy.deepcopy(model.state_dict())

        cur_lr = optimizer.param_groups[0]["lr"]
        wandb.log(
            {
                "best_acc": max(valid_top1_acc, best_top1_acc),
                "top1": valid_top1_acc,
                "lr": cur_lr,
            }
        )

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))

    model.load_state_dict(best_model_state)

    return model


if __name__ == "__main__":
    main()
