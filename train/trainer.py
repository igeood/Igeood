"""Train a model using Pytorch's DistributedDataParallel"""
import os
import subprocess
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

import train.config_helper as config
from train.optimizer_helper import (
    OPTIMIZERS,
    SCHEDULERS,
    get_optimizer_cls,
    get_scheduler,
)
from train.registry import DATASETS, MODELS, get_dataset, get_model, get_num_classes


def add_dataset_arguments(dataset_parser):
    dataset_parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/",
        help="folder to load and download datasets (default: datasets/)",
    )
    dataset_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    dataset_parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for validation (default: 1000)",
    )
    dataset_parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        metavar="N",
        help="number of workers to load data (default: 8)",
    )
    dataset_parser.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="shuffle the dataset (default: False)",
    )
    dataset_parser.add_argument(
        "--data-augmentation",
        type=bool,
        default=False,
        help="load the dataset with data augmentation (default: False)",
    )
    return dataset_parser


def add_model_arguments(model_parser):
    model_parser.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="load pre-trained model (default: False)",
    )
    model_parser.add_argument(
        "--save-model",
        type=bool,
        default=True,
        help="save the current model (default: True)",
    )
    model_parser.add_argument(
        "--model-dir",
        type=str,
        default="checkpoints/",
        help="root directory to load and save models (default: checkpoints/)",
    )
    return model_parser


def add_trainer_arguments(training_parser):
    training_parser.add_argument(
        "-cfg",
        "--config-file",
        default=None,
        type=str,
        help="a yaml file for overriding some parameters specification in this module",
    )
    training_parser.add_argument(
        "--resume-checkpoint",
        type=int,
        default=None,
        help="resume training epoch from checkpoint file (default: None)",
    )
    training_parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    training_parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="how many epochs to wait before logging training status (default: 1)",
    )
    training_parser.add_argument(
        "--checkpoints",
        type=int,
        default=5,
        metavar="N",
        help="how many training checkpoints to save (default: 5)",
    )
    training_parser.add_argument(
        "--logger",
        type=str,
        default=None,
        choices=["tensorboard", "wandb"],
        help="external logger (default: None)",
    )
    training_parser.add_argument(
        "--gpu",
        type=bool,
        default=True,
        help="enable CUDA training (default: True)",
    )
    training_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    training_parser.add_argument(
        "--val-phase",
        type=bool,
        default=True,
        help="calculate validation performance during training (default: True)",
    )
    return training_parser


def add_optimizer_arguments(optimizer_parser):
    optimizer_parser.add_argument(
        "-optim",
        "--optimizer",
        type=str,
        default="adam",
        help="optimization algorithm (default: adam)",
        choices=OPTIMIZERS,
    )
    optimizer_parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    optimizer_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        metavar="WD",
        help="weight decay coefficient (default: 0)",
    )
    optimizer_parser.add_argument(
        "--dampening",
        type=float,
        default=0,
        metavar="D",
        help="dampening for momentum (default: 0)",
    )
    optimizer_parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    optimizer_parser.add_argument(
        "--nesterov",
        type=bool,
        default=False,
        metavar="N",
        help="enables Nesterov momentum (default: False)",
    )
    # scheduler
    optimizer_parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=None,
        # metavar="",
        help="learning rate scheduler (default: None)",
        choices=SCHEDULERS,
    )
    optimizer_parser.add_argument(
        "--steps-list",
        nargs="+",
        default=None,
        metavar="S",
        help="list of epoch proportion ratios, must be increasing (default: None)",
    )
    optimizer_parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        metavar="G",
        help="multiplicative factor of learning rate decay (default: 0.1)",
    )
    return optimizer_parser


def save_checkpoint_fn(model_name, dataset_name, seed, model_dir):
    def fn(epoch, state):
        PATH = os.path.join(model_dir, model_name, dataset_name, str(seed))
        os.makedirs(PATH, exist_ok=True)
        print("==> Saving checkpoint ...")
        torch.save(state, os.path.join(PATH, f"ckpt_{epoch}.pt"))

    return fn


def make_ckpt_path(model_name, dataset_name, seed, epoch):
    return os.path.join(
        "checkpoints/", model_name, dataset_name, str(seed), f"ckpt_{epoch}.pt"
    )


def load_checkpoint(ckpt_path, model, optimizer, scheduler, device):
    try:
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location=torch.device(device))

        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                ckpt_path, checkpoint["epoch"]
            )
        )
    except AttributeError as error:
        print("=> no checkpoint found at '{}'".format(ckpt_path))
        raise error

    return start_epoch


def get_gpu_memory_map():
    """Get the current gpu usage.
    usage: returns dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    # gpu memory map
    return dict(zip(range(len(gpu_memory)), gpu_memory))


def seed_worker(worker_id):
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loader(
    dataset, batch_size, num_workers, shuffle=False, seed=0, *args, **kwargs
):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        *args,
        **kwargs,
    )


def test_step(model, dataloader, device):
    model.eval()
    acc_sum, total = 0, 0
    with torch.no_grad():
        for batch, labels in dataloader:
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(batch)
            _, pred = torch.max(out, 1)

            acc_sum += torch.sum(pred == labels).item()
            total += batch.shape[0]

    return acc_sum / total


def training_loop(
    model,
    criterion,
    optimizer,
    trainloader,
    testloader,
    device,
    epochs,
    log_interval,
    checkpoint_interval=10,
    checkpoint_fn=None,
    start_epoch=0,
    scheduler=None,
):
    """Train a model with data"""
    start = datetime.now()
    for epoch in tqdm(range(start_epoch, epochs + start_epoch)):
        running_loss = 0.0
        model.train()
        train_accuracy, test_accuracy = 0, 0
        acc_sum, total = 0, 0
        for _, (batch, labels) in enumerate(trainloader):
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # forward + backward + optimize
            outputs = model(batch)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()

            # optimizer step
            optimizer.step()

            # training statistics
            running_loss += loss.item()
            # training accuracy
            _, pred = torch.max(outputs.detach(), 1)
            acc_sum += torch.sum(pred == labels).item()
            total += batch.shape[0]

        # normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)

        # scheduler
        if scheduler is not None:
            scheduler.step()

        # logging
        if (epoch) % log_interval == 0:
            # calculate training and test set accuracy of the existing model
            test_accuracy = test_step(model, testloader, device)
            train_accuracy = acc_sum / total
            print(
                "Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(
                    epoch + 1,
                    epochs,
                    running_loss,
                    train_accuracy * 100,
                    test_accuracy * 100,
                )
            )

        # save checkpoint
        if (epoch + 1) % checkpoint_interval == 0 and checkpoint_fn is not None:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }
            checkpoint_fn(epoch, state)

    print("==> Finished Training ...")
    print("Training completed in: {}s".format(str(datetime.now() - start)))


def train_step():
    return


def make_reproducible(seed: int):
    import random

    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn

    # randomness reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ## may decrease performance
    cudnn.deterministic = True
    cudnn.benchmark = False


def main(args):
    """Create objects and call training loop"""
    print(args)
    # data
    train_dataset = get_dataset(args.dataset_name, args.data_dir, True)
    train_loader = get_data_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        args.shuffle,
    )
    test_dataset = get_dataset(args.dataset_name, args.data_dir, False)
    test_loader = get_data_loader(
        test_dataset,
        args.batch_size,
        args.num_workers,
        args.shuffle,
    )

    # model
    n_classes = get_num_classes(args.dataset_name)
    model = get_model(args.model, n_classes)

    # loss
    criterion = nn.CrossEntropyLoss()

    # GPU support
    if torch.cuda.is_available() and args.gpu:
        device = "cuda"
        print(f"{torch.cuda.device_count()} GPUs detected")
    else:
        device = "cpu"
        print("No GPU. switching to CPU")

    model = model.to(device)
    model = nn.DataParallel(model)

    # optimizer
    optimizer_cls = get_optimizer_cls(args.optimizer)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        momentum=args.momentum,
    )
    # scheduler
    scheduler = None
    if args.lr_scheduler is not None:
        milestones = [args.epochs * s for s in args.steps_list]
        scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer,
            milestones=milestones,
            gamma=args.gamma,
        )
    # checkpoints
    args.checkpoint_interval = round(args.epochs / args.checkpoints)
    checkpoint_fn = save_checkpoint_fn(
        args.model,
        args.dataset_name,
        args.seed,
        args.model_dir,
    )
    # load checkpoint
    start_epoch = 0
    if args.resume_checkpoint is not None:
        ckpt_path = make_ckpt_path(
            args.model, args.dataset_name, args.seed, args.resume_checkpoint
        )
        start_epoch = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)

    # train
    training_loop(
        model,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        device,
        args.epochs,
        args.log_interval,
        args.checkpoint_interval,
        checkpoint_fn,
        scheduler=scheduler,
        start_epoch=start_epoch,
    )

    if args.save_model:
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(
            state_dict,
            os.path.join(
                args.model_dir, f"{args.model}_{args.dataset_name}_{args.seed}.pt"
            ),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a DNN model with GPU if available.",
        epilog="example: python trainer.py densenet100 cifar10 --gpu True",
        allow_abbrev=True,
    )

    # positional args
    parser.add_argument("model", type=str, choices=MODELS, help="architecture name")
    parser.add_argument("dataset_name", type=str, choices=DATASETS, help="dataset name")

    # model args
    model_parser = parser.add_argument_group(title="model args")
    model_parser = add_model_arguments(model_parser)

    # dataset args
    dataset_parser = parser.add_argument_group(title="dataset args")
    dataset_parser = add_dataset_arguments(dataset_parser)

    # training args
    training_parser = parser.add_argument_group(title="training args")
    add_trainer_arguments(training_parser)

    # optimizer args
    optimizer_parser = parser.add_argument_group(title="optimizer args")
    optimizer_parser = add_optimizer_arguments(optimizer_parser)

    # parse args
    args = parser.parse_args()

    make_reproducible(args.seed)

    # ovewrite cli arguments with config file arguments
    if args.config_file is not None:
        config_args_dict = config.read_yaml_file(args.config_file)
        args_dict = vars(args)

        for k, v in config_args_dict.items():
            args_dict[k] = v

        args = argparse.Namespace(**args_dict)

    main(args)
