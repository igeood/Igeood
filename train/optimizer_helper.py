import inspect

import torch.optim as optim

optimizer_registry = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}
OPTIMIZERS = set(optimizer_registry.keys())

scheduler_registry = {
    "step": optim.lr_scheduler.StepLR,
    "multi_step": optim.lr_scheduler.MultiStepLR,
    "cosine_annealing": optim.lr_scheduler.CosineAnnealingLR,
}
SCHEDULERS = set(scheduler_registry.keys())


class AdamWrapper(optim.Adam):
    def __init__(self, params, **kwargs):
        filt = inspect.getfullargspec(super().__init__).args
        arguments = {k: v for k, v in kwargs.items() if k in filt}
        super().__init__(params, **arguments)


class AdamWWrapper(optim.AdamW):
    def __init__(self, params, **kwargs):
        filt = inspect.getfullargspec(super().__init__).args
        arguments = {k: v for k, v in kwargs.items() if k in filt}
        super().__init__(params, **arguments)


class SGDWrapper(optim.SGD):
    def __init__(self, params, **kwargs):
        filt = inspect.getfullargspec(super().__init__).args
        arguments = {k: v for k, v in kwargs.items() if k in filt}
        super().__init__(params, **arguments)


def cls_kwargs_wrapper(cls, *args, **kwargs):
    filt = inspect.getfullargspec(cls.__init__).args
    arguments = {k: v for k, v in kwargs.items() if k in filt}
    return cls(*args, **arguments)


optimizer_cls_registry = {
    "adam": AdamWrapper,
    "adamw": AdamWWrapper,
    "sgd": SGDWrapper,
}


def get_optimizer(optimizer_name: str):  # , params, **kwargs):
    return optimizer_registry.get(optimizer_name.lower())


def get_optimizer_cls(optimizer_name: str):
    return optimizer_cls_registry.get(optimizer_name.lower())


def get_scheduler(scheduler_name, optimizer, **kwargs):
    scheduler_cls = scheduler_registry.get(scheduler_name.lower())
    return cls_kwargs_wrapper(scheduler_cls, optimizer, **kwargs)
