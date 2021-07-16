import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from .network import preact_resnet_cifar, resnet_cifar, resnet_imagenet


def get_network(network_config):
    if "resnet18_cifar" in network_config:
        model = resnet_cifar.resnet18(**network_config["resnet18_cifar"])
    elif "preact_resnet18_cifar" in network_config:
        model = preact_resnet_cifar.preact_resnet18(
            **network_config["preact_resnet18_cifar"]
        )
    elif "resnet18_imagenet" in network_config:
        model = resnet_imagenet.resnet18(**network_config["resnet18_imagenet"])
    else:
        raise NotImplementedError("Network {} is not supported.".format(network_config))

    return model


def get_criterion(criterion_config):
    if "cross_entropy" in criterion_config:
        criterion = nn.CrossEntropyLoss(**criterion_config("cross_entropy"))
    else:
        raise NotImplementedError(
            "Criterion {} is not supported.".format(criterion_config)
        )

    return criterion


def get_optimizer(model, optimizer_config):
    if "Adam" in optimizer_config:
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config["Adam"])
    elif "SGD" in optimizer_config:
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_config["SGD"])
    else:
        raise NotImplementedError(
            "Optimizer {} is not supported.".format(optimizer_config)
        )

    return optimizer


def get_scheduler(optimizer, lr_scheduler_config):
    if lr_scheduler_config is None:
        scheduler = None
    elif "multi_step" in lr_scheduler_config:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, **lr_scheduler_config["multi_step"]
        )
    elif "cosine_annealing" in lr_scheduler_config:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, **lr_scheduler_config["cosine_annealing"]
        )
    else:
        raise NotImplementedError(
            "Learning rate scheduler {} is not supported.".format(lr_scheduler_config)
        )

    return scheduler


def load_model(model, ckpt_path, key=None):
    """Load model state dict from the checkpoint.

    Args:
        model (torch.nn.Module): The model to load.
        ckpt_path (str): The checkpoint path.
        key (str, optional): The key to the model state dict.
    """
    # To avoid GPU RAM surge when loading a model checkpoint saved on GPU.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Find the key to the model's state_dict if not specified.
    if key is None:
        for k in ckpt.keys():
            if isinstance(ckpt[k], OrderedDict):
                key = k
                break
    if not "parallel" in str(type(model)):
        # Remove "module." in `model_state_dict` if saved from DP or DDP
        # wrapped model in the single GPU training.
        model_state_dict = OrderedDict()
        for k, v in ckpt[key].items():
            if k.startswith("module."):
                k = k.replace("module.", "")
                model_state_dict[k] = v
            else:
                model_state_dict[k] = v
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(ckpt[key])


def resume_state(
    model, resume, ckpt_dir, logger, optimizer=None, scheduler=None, is_best=False
):
    """Resume training state from checkpoint.

    Args:
        model (torch.nn.Module): Model to resume.
        resume (str): Checkpoint name (empty string means the latest checkpoint)
            or False (means training from scratch).
        ckpt_dir (str): Checkpoint directory.
        optimizer (torch.optim.Optimizer, optional): Optimizer to resume. Default is None.
        scheduler (torch.optim._LRScheduler, optional): Learning rate scheduler to
            resume. Default is None.
        is_best (boolean): Set True to load checkpoint
            with ``best_acc``. Default is False.

    Returns:
        resumed_epoch (int): The epoch to resume (0 means training from scratch.)
        best_acc (float): The best test accuracy in the training.
        best_epoch (int): The epoch getting the ``best_acc``.
    """
    if resume == "False":
        logger.warning("Training from scratch.")
        resumed_epoch = 0
        if is_best:
            best_acc = 0
            best_epoch = 0
            return resumed_epoch, best_acc, best_epoch
        else:
            return resumed_epoch
    else:
        ckpt_name = "latest_model.pt" if resume == "" else resume
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        logger.info("Load training state from the checkpoint {}:".format(ckpt_path))
        logger.info("Epoch: {}, result: {}".format(ckpt["epoch"], ckpt["result"]))
        load_model(model, ckpt_path)
        resumed_epoch = ckpt["epoch"]
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if is_best:
            best_acc = ckpt["best_acc"]
            best_epoch = ckpt["best_epoch"]
            return resumed_epoch, best_acc, best_epoch
        else:
            return resumed_epoch


def set_mode(model, training_mode=True, requires_grad=True):
    """Set ``training`` and ``requires_grad`` status for all modules in the model, and
    return the original status. It's useful to restore model statuses to avoid
    in-place modification.
    """
    # TODO: fine-grained per-module status.
    ori_training_mode = model.training
    model.train() if training_mode else model.eval()
    ori_requires_grad = next(model.parameters()).requires_grad
    for param in model.parameters():
        param.requires_grad = requires_grad

    return ori_training_mode, ori_requires_grad
