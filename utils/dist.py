import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backward propagation.
    
    Modified from https://github.com/open-mmlab/OpenSelfSup.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]

        return grad_out


def get_dist_info():
    """Get DistributedDataParallel informations (initialized, rank, world_size).

    Modified from https://github.com/open-mmlab/mmcv.
    """
    initialized = dist.is_initialized() if dist.is_available() else False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return initialized, rank, world_size


def reduce_tensor(tensor, world_size):
    """world_size = num_gpus_per_node * num_nodes.

    Modified from https://github.com/fastai/imagenet-fast.
    """
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    reduced_tensor /= world_size

    return reduced_tensor


def gather_tensor(input):
    output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
    dist.all_gather(output, input)

    return tuple(output)
