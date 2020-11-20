import os
from functools import cmp_to_key

import torch
import torch.nn as nn
from torch.optim import lr_scheduler


def get_network(network_config):
    if "resnet18" in network_config:
        model = resnet.resnet18()
    else:
        raise NotImplementedError("Network {} is not supported.".format(network_config))

    return model


def get_criterion(criterion_config):
    if "CrossEntropy" in criterion_config:
        criterion = nn.CrossEntropyLoss()
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
    elif "MultiStepLR" in lr_scheduler_config:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, **lr_scheduler_config["MultiStepLR"]
        )
    elif "CosineAnnealingLR" in lr_scheduler_config:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, **lr_scheduler_config["CosineAnnealingLR"]
        )
    else:
        raise NotImplementedError(
            "Learning rate scheduler {} is not supported.".format(lr_scheduler_config)
        )

    return scheduler


def ckpt_key():
    def ckpt_cmp(x, y):
        epoch_x = int(x.split("h")[1].split(".")[0])
        epoch_y = int(y.split("h")[1].split(".")[0])
        return epoch_x - epoch_y

    return cmp_to_key(ckpt_cmp)


def resume_state(model, optimizer, resume, ckpt_dir, gpu, logger, scheduler=None):
    """Resume training state from checkpoint.

    Args:
        model (torch.nn.Module): Model to resume.
        optimizer (torch.optim): Optimizer to resume.
        resume (str): Checkpoint name or False (means training from scratch).
        ckpt_dir (str): Checkpoint directory.
        gpu (str or int): The specified single gpu to load checkpoint.
        logger (logging.logger): Program logger.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler to resume (default: None).
    
    Returns:
        resumed_epoch: The epoch to resume (0 means training from scratch.)
    """
    file_list = sorted(os.listdir(ckpt_dir), key=ckpt_key())
    if resume == "False":
        logger.warning("Training from scratch!")
        resumed_epoch = 0
        return resumed_epoch
    elif resume == "" and file_list:
        ckpt_path = os.path.join(ckpt_dir, file_list[-1])
        logger.info(
            "Resume training state from the latest checkpoint: {}".format(ckpt_path)
        )
    else:
        ckpt_path = os.path.join(ckpt_dir, resume)
        logger.info("Resume training state from: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cuda:{}".format(gpu))
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    logger.info("Result of the resumed model: {}".format(ckpt["result"]))

    return ckpt["epoch"]
