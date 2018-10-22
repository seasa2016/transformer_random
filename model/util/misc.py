import torch

def aeq(*args):

    arguments = (arg for arg in args)
    first = next(arguments)

    assert all(arg == first for arg in arguments),\
            "all the value should be the same"

def use_gpu(opt):
    return (hasattr(opt,'gpuid') and len(opt.gpuid)>0) or \
            (hasattr(opt,'gpu') and opt.gpu > -1)

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.shape)))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.shape)
    out_size[0] *= count
    batch = x.shape[0]
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
