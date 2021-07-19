import argparse
import os
import platform
import shutil
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from data.transforms import TorchTransforms
from data.utils import get_dataset, get_loader
from model.utils import get_network, get_optimizer, get_scheduler, resume_state
from utils.setup import (
    get_logger,
    get_saved_dir,
    get_storage_dir,
    load_config,
    set_seed,
)
from utils.trainer.img_clf import test, train
from utils.trainer.log import result2csv


def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/train_image_clf/example.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="checkpoint name (empty string means the latest checkpoint)\
            or False (means training from scratch).",
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-port",
        default="23456",
        type=str,
        help="port used to set up distributed training",
    )
    args = parser.parse_args()

    config, inner_dir, config_name = load_config(args.config)
    args.saved_dir, args.log_dir = get_saved_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.saved_dir)
    args.storage_dir, args.ckpt_dir = get_storage_dir(
        config, inner_dir, config_name, args.resume
    )
    shutil.copy2(args.config, args.storage_dir)
    set_seed(**config["seed"])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = True if ngpus_per_node > 1 else False
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        print("Distributed training on GPUs: {}.".format(args.gpu))
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, config),
        )
    else:
        print("Training on a single GPU: {}.".format(args.gpu))
        main_worker(0, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    logger = get_logger(args.log_dir, resume=args.resume, is_rank0=(gpu == 0))
    start_time = time.asctime(time.localtime(time.time()))
    logger.info("Start at: {} at: {}".format(start_time, platform.node()))
    torch.cuda.set_device(gpu)
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:{}".format(args.dist_port),
            world_size=args.world_size,
            rank=args.rank,
        )
        logger.warning("Only log rank 0 in distributed training!")

    logger.info("===Prepare data===")
    if "torch_transforms" in config:
        train_transform = TorchTransforms(config["torch_transforms"]["train"])
        test_transform = TorchTransforms(config["torch_transforms"]["test"])
    else:
        train_transform, test_transform = None, None
    logger.info("Torch training transformations:\n{}".format(train_transform))
    logger.info("Torch test transformations:\n{}".format(test_transform))
    logger.info("Load dataset from: {}".format(config["dataset_dir"]))
    train_data = get_dataset(config["dataset_dir"], train_transform)
    test_data = get_dataset(config["dataset_dir"], test_transform, train=False)
    prefetch = "prefetch" in config and config["prefetch"]
    logger.info("Prefetch: {}".format(prefetch))
    if args.distributed:
        train_sampler = DistributedSampler(train_data)
        # Divide batch size equally among multiple GPUs,
        # to keep the same learning rate used in a single GPU.
        batch_size = int(config["loader"]["batch_size"] / ngpus_per_node)
        num_workers = config["loader"]["num_workers"]
        train_loader = get_loader(
            train_data,
            prefetch=prefetch,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
        )
    else:
        train_sampler = None
        train_loader = get_loader(
            train_data, prefetch=prefetch, loader_config=config["loader"], shuffle=True
        )
    test_loader = get_loader(
        test_data, prefetch=prefetch, loader_config=config["loader"]
    )

    logger.info("\n===Setup training===")
    model = get_network(config["network"])
    logger.info("Create network: {}".format(config["network"]))
    model = model.cuda(gpu)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(gpu)
    logger.info("Create criterion: {}".format(criterion))
    optimizer = get_optimizer(model, config["optimizer"])
    logger.info("Create optimizer: {}".format(optimizer))
    scheduler = get_scheduler(optimizer, config["lr_scheduler"])
    logger.info("Create scheduler: {}".format(config["lr_scheduler"]))
    resumed_epoch, best_acc, best_epoch = resume_state(
        model,
        args.resume,
        args.ckpt_dir,
        logger,
        optimizer=optimizer,
        scheduler=scheduler,
        is_best=True,
    )
    if args.distributed:
        # Convert BatchNorm*D layer to SyncBatchNorm before wrapping Network with DDP.
        if "sync_bn" in config and config["sync_bn"]:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info("Turn on synchronized batch normalization in ddp.")
        model = DistributedDataParallel(model, device_ids=[gpu])

    for epoch in range(config["num_epochs"] - resumed_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        logger.info(
            "===Epoch: {}/{}===".format(epoch + resumed_epoch + 1, config["num_epochs"])
        )
        logger.info("Training...")
        train_result = train(
            model,
            train_loader,
            criterion,
            optimizer,
            logger,
            amp=args.amp,
        )
        logger.info("Test...")
        test_result = test(model, test_loader, criterion, logger)

        if scheduler is not None:
            scheduler.step()
            logger.info(
                "Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"])
            )

        # Save result and checkpoint.
        if not args.distributed or (args.distributed and gpu == 0):
            result = {"train": train_result, "test": test_result}
            result2csv(result, args.log_dir)

            saved_dict = {
                "epoch": epoch + resumed_epoch + 1,
                "result": result,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "best_epoch": best_epoch,
            }
            if not "parallel" in str(type(model)):
                saved_dict["model_state_dict"] = model.state_dict()
            else:
                # DP or DDP.
                saved_dict["model_state_dict"] = model.module.state_dict()
            if scheduler is not None:
                saved_dict["scheduler_state_dict"] = scheduler.state_dict()

            is_best = False
            if test_result["acc"] > best_acc:
                is_best = True
                best_acc = test_result["acc"]
                best_epoch = epoch + resumed_epoch + 1
            logger.info(
                "Best test accuaracy {} in epoch {}".format(best_acc, best_epoch)
            )
            if is_best:
                ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")
                torch.save(saved_dict, ckpt_path)
                logger.info("Save the best model to {}".format(ckpt_path))
            ckpt_path = os.path.join(args.ckpt_dir, "latest_model.pt")
            torch.save(saved_dict, ckpt_path)
            logger.info("Save the latest model to {}".format(ckpt_path))

    end_time = time.asctime(time.localtime(time.time()))
    logger.info("End at: {} at: {}".format(end_time, platform.node()))


if __name__ == "__main__":
    main()
