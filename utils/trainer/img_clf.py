import time

import torch
from torch.cuda.amp import GradScaler, autocast

from .dist import get_dist_info, reduce_tensor
from .log import AverageMeter, tabulate_epoch_meter, tabulate_step_meter


def train(
    model,
    loader,
    criterion,
    optimizer,
    logger,
    amp=False,
):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    meter_list = [loss_meter, acc_meter]

    model.train()
    gpu = next(model.parameters()).device
    initialized, _, world_size = get_dist_info()
    scaler = GradScaler() if amp else None
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        data = batch["img"].cuda(gpu, non_blocking=True)
        target = batch["target"].cuda(gpu, non_blocking=True)

        optimizer.zero_grad()
        if amp:
            with autocast():
                output = model(data)
        else:
            output = model(data)
        loss = criterion(output, target)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc = torch.sum(truth).float() / len(truth)
        if initialized:
            loss = reduce_tensor(loss, world_size)
            acc = reduce_tensor(acc, world_size)
        loss_meter.update(loss.item())
        acc_meter.update(acc.item())

        tabulate_step_meter(batch_idx, len(loader), 3, meter_list, logger)

    logger.info("Training summary:")
    tabulate_epoch_meter(time.time() - start_time, meter_list, logger)
    result = {m.name: m.total_avg for m in meter_list}

    return result


def test(model, loader, criterion, logger):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    meter_list = [loss_meter, acc_meter]

    model.eval()
    gpu = next(model.parameters()).device
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        data = batch["img"].cuda(gpu, non_blocking=True)
        target = batch["target"].cuda(gpu, non_blocking=True)
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        loss_meter.update(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc_meter.update((torch.sum(truth).float() / len(truth)).item())

        tabulate_step_meter(batch_idx, len(loader), 2, meter_list, logger)

    logger.info("Test summary:")
    tabulate_epoch_meter(time.time() - start_time, meter_list, logger)
    result = {m.name: m.total_avg for m in meter_list}

    return result
