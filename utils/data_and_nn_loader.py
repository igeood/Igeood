from tqdm import tqdm
import os
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data

torch.manual_seed(0)

# Uncomment these lines if you are having problem downloading PyTorch datasets:
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
from pathlib import Path

ROOT = str(os.path.dirname(os.path.dirname(__file__)))

sys.path.append("{}/datasets/".format(ROOT))
sys.path.append("{}/pre_trained/".format(ROOT))
sys.path.append("{}/models/".format(ROOT))

from utils.logger import logger
from models.densenet import DenseNetBC100
from models.resnet import ResNet34

# Train set statistics
transform_dict = dict()
transform_dict["CIFAR10"] = (
    (0.4913997, 0.4821584, 0.446531),
    (0.2470323, 0.243485, 0.2615876),
)
transform_dict["CIFAR100"] = (
    (0.5070752, 0.486549, 0.4409178),
    (0.2673342, 0.2564384, 0.2761506),
)
transform_dict["SVHN"] = (
    (0.4379793, 0.4439904, 0.4729508),
    (0.1981116, 0.2011045, 0.1970895),
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_statistics(dataset_name):
    dataset_name = dataset_name.upper()
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                transform_dict[dataset_name][0], transform_dict[dataset_name][1]
            ),
        ]
    )


def transform_statistics_tensor(dataset_name):
    dataset_name = dataset_name.upper()
    return transforms.Normalize(
        transform_dict[dataset_name][0], transform_dict[dataset_name][1]
    )


def transform_tensor(*args, **kwargs):
    return transforms.ToTensor()


def transform_none(*args, **kwargs):
    return transforms.Lambda(lambda x: x)


def load_train_dataset(name, transform_name, transform=transform_statistics):
    transform_name = transform_name.upper()
    if transform is None or transform_name is None:
        transform = transform_tensor

    if name.upper() == "CIFAR10":
        return torchvision.datasets.CIFAR10(
            root="{}/datasets".format(ROOT),
            train=True,
            download=True,
            transform=transform(transform_name),
        )
    elif name.upper() == "CIFAR100":
        return torchvision.datasets.CIFAR100(
            root="{}/datasets".format(ROOT),
            train=True,
            download=True,
            transform=transform(transform_name),
        )
    elif name.upper() == "SVHN":
        return torchvision.datasets.SVHN(
            root="{}/datasets/SVHN".format(ROOT),
            split="train",
            download=True,
            transform=transform(transform_name),
        )
    else:
        return torchvision.datasets.ImageFolder(
            "{}/datasets/{}".format(ROOT, name),
            transform=transform(transform_name),
        )


def train_dataloader(
    name, transform_name="CIFAR10", transform=transform_statistics, *args, **kwargs
):
    trainset = load_train_dataset(name, transform_name, transform)
    logger.info("dataset {} found. Preparing DataLoader".format(name))
    batch_size = kwargs.get("batch_size", 1)
    trainloader = torch.utils.data.DataLoader(
        trainset, shuffle=False, num_workers=2, batch_size=batch_size
    )
    logger.info("dataset {} loaded with batch size {}".format(name, batch_size))
    return trainloader


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = int(self.tensors[1][index])
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def load_test_dataset(name, transform_dataset, transform=transform_statistics):
    if transform is None or transform_dataset is None:
        transform = transform_tensor
    elif name.upper() == "PLACES365":
        # original size: 3x256x256
        # original length 36500
        pre_transform = torchvision.transforms.Resize(32)

        return LimitDataset(
            torchvision.datasets.ImageFolder(
                "{}/datasets/Places365/".format(
                    ROOT,
                ),
                transform=torchvision.transforms.Compose(
                    [pre_transform, transform(transform_dataset)]
                ),
            ),
            10000,
        )
    elif name.upper() == "TEXTURES":
        # original size: 3x300x300
        # length 5640
        pre_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(32),
                torchvision.transforms.CenterCrop(32),
            ]
        )
        return torchvision.datasets.ImageFolder(
            "{}/datasets/Textures/dtd/images".format(
                ROOT,
            ),
            transform=torchvision.transforms.Compose(
                [
                    pre_transform,
                    transform(transform_dataset),
                ]
            ),
        )
    elif name.upper() == "CHARS74K":
        # original size: 3x42x34
        # length 7705
        pre_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(32),
                torchvision.transforms.CenterCrop(32),
            ]
        )
        return torchvision.datasets.ImageFolder(
            "{}/datasets/Chars74K/English/Img/GoodImg/Bmp/".format(
                ROOT,
            ),
            transform=torchvision.transforms.Compose(
                [
                    pre_transform,
                    transform(transform_dataset),
                ]
            ),
        )
    elif name.upper() == "CIFAR100":
        return torchvision.datasets.CIFAR100(
            root="{}/datasets".format(ROOT),
            train=False,
            download=True,
            transform=transform(transform_dataset),
        )
    elif name.upper() == "CIFAR10":
        return torchvision.datasets.CIFAR10(
            root="{}/datasets".format(ROOT),
            train=False,
            download=True,
            transform=transform(transform_dataset),
        )
    elif name.upper() == "SVHN":
        return torchvision.datasets.SVHN(
            root="{}/datasets/SVHN".format(ROOT),
            split="test",
            download=True,
            transform=transform(transform_dataset),
        )
    elif name.upper() == "GAUSSIAN_NOISE_DATASET":
        nb_samples = 10000
        return CustomTensorDataset(
            [
                torch.clamp(torch.randn(nb_samples, 3, 32, 32) + 0.5, 0, 1),
                torch.tensor([1.0] * nb_samples),
            ],
            transform=transform_statistics_tensor(transform_dataset),
        )
    elif name.upper() == "DENSENET10_ADV":
        return load_adv_dataset("densenet10")

    elif name.upper() == "RESNET_CIFAR10_ADV":
        return load_adv_dataset("resnet_cifar10")

    elif name.upper() == "DENSENET100_ADV":
        return load_adv_dataset("densenet100")

    elif name.upper() == "RESNET_CIFAR100_ADV":
        return load_adv_dataset("resnet_cifar100")

    elif name.upper() == "DENSENET_SVHN_ADV":
        return load_adv_dataset("densenet_svhn")

    elif name.upper() == "RESNET_SVHN_ADV":
        return load_adv_dataset("resnet_svhn")
    else:
        return torchvision.datasets.ImageFolder(
            "{}/datasets/{}".format(ROOT, name),
            transform=transform(transform_dataset),
        )


def dataset_channel_statistics(dataloader: torch.utils.data.DataLoader, decimal=4):
    """Calculate a dataset's empirical mean and standard deviation and round to `decimal` points"""
    channels_sum, channels_sq_sum, total_batches = 0, 0, 0
    for (data, _) in tqdm(dataloader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sq_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        total_batches += 1

    logger.debug(f"total samples {total_batches}")
    mean = channels_sum / total_batches
    meansq = channels_sq_sum / total_batches
    std = (meansq - mean ** 2) ** 0.5
    return list(np.around(mean.numpy(), decimal)), list(np.around(std.numpy(), decimal))


def test_dataloader(
    name, transform_name="CIFAR10", transform=transform_statistics, *args, **kwargs
):
    shuffle = False
    if name == "densenet10_adv":
        testset = load_adv_dataset("densenet10", *args, **kwargs)
    elif name == "densenet100_adv":
        testset = load_adv_dataset("densenet100", *args, **kwargs)
    elif name == "densenet_svhn_adv":
        testset = load_adv_dataset("densenet_svhn", *args, **kwargs)
    elif name == "resnet_cifar10_adv":
        testset = load_adv_dataset("resnet_cifar10", *args, **kwargs)
    elif name == "resnet_cifar100_adv":
        testset = load_adv_dataset("resnet_cifar100", *args, **kwargs)
    elif name == "resnet_svhn_adv":
        testset = load_adv_dataset("resnet_svhn", *args, **kwargs)
    elif "densenet" in name or "resnet" in name:
        testset = load_adv_dataset(name.split("ADV")[-1], *args, **kwargs)
    else:
        testset = load_test_dataset(name, transform_name, transform)

    batch_size = kwargs.get("batch_size", 1)
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=shuffle, num_workers=0, batch_size=batch_size
    )
    logger.info("dataset {} loaded with batch size {}".format(name, batch_size))
    return testloader


def load_adv_dataset(nn_name, *args, **kwargs):
    in_dataset_name = get_in_dataset_name(nn_name)
    filename = "{}/tensors/{}/{}/adv_data_fgsm.pth".format(
        ROOT, nn_name, in_dataset_name
    )
    adv_data = load_tensor(filename)
    filename = "{}/tensors/{}/{}/label_fgsm.pth".format(ROOT, nn_name, in_dataset_name)
    adv_labels = load_tensor(filename)

    if adv_data is not None and adv_labels is not None:
        return torch.utils.data.TensorDataset(
            adv_data[:10000].cpu(), adv_labels[:10000].cpu()
        )


def gradient_trasform(in_dataset_name):
    """Transforms the standard deviation of the gradient tensor to adapt to the in-distribution data.

    Args:
        in_dataset_name (str): name of the in-distribution dataset.

    Returns:
        transforms.Normalize: callable torch based normalizer.
    """
    n_channels = get_number_channels(in_dataset_name)
    return transforms.Normalize(
        np.zeros(n_channels), transform_dict[in_dataset_name.upper()][1]
    )


def load_pre_trained_nn(nn_name, gpu=None):
    if type(gpu) == int:
        map_location = torch.device("cuda:{}".format(gpu))
    elif gpu is None:
        map_location = torch.device("cpu")
    else:
        map_location = gpu
    num_c = get_num_classes(get_in_dataset_name(nn_name))
    if "densenet" in nn_name:
        model = DenseNetBC100(num_c)
        if "svhn" not in nn_name and "cifar" not in nn_name:
            return load_nn(nn_name, map_location)
        return load_nn_from_state_dict(nn_name, model, map_location)
    elif "resnet" in nn_name:
        model = ResNet34(num_c)
        return load_nn_from_state_dict(nn_name, model, map_location)
    return load_nn(nn_name, map_location)


def load_nn(nn_name, map_location=torch.device("cpu")):
    logger.info("model {}/pre_trained/{}.pth loaded".format(ROOT, nn_name))
    model = torch.load("{}/pre_trained/{}.pth".format(ROOT, nn_name), map_location)
    return model


def load_nn_from_state_dict(nn_name, model, map_location=torch.device("cpu")):
    model.load_state_dict(load_nn(nn_name), strict=False)
    model.to(map_location)
    model.eval()
    return model


def save_model(model, PATH):
    torch.save(model.module.state_dict(), PATH)


def get_in_dataset_name(nn_name):
    if "1" not in nn_name:
        return "SVHN"
    l = nn_name.split("1")
    if len(l[1]) == 1:
        return "CIFAR10"
    else:
        return "CIFAR100"


def get_nn_name(architecture, in_dataset_name):
    if architecture.lower() == "densenet":
        if "CIFAR10" == in_dataset_name.upper():
            nn_name = "densenet10"
        elif "CIFAR100" == in_dataset_name.upper():
            nn_name = "densenet100"
        else:
            nn_name = "densenet_svhn"
    elif architecture.lower() == "resnet":
        nn_name = "{}_{}".format(architecture.lower(), in_dataset_name.lower())
    else:
        return
    return nn_name


def get_number_channels(dataset_name):
    if "MNIST" in dataset_name.upper():
        return 1
    return 3


def get_num_classes(dataset_name):
    dataset_name = dataset_name.upper()
    if "MNIST" in dataset_name or "SVHN" in dataset_name or "CIFAR10" == dataset_name:
        return 10
    else:
        return 100


def save_tensor(x, filepath):
    torch.save(x, "{}/tensors/{}".format(ROOT, filepath))


def load_tensor(filename):
    if os.path.isfile(filename):
        return torch.load(filename, map_location=DEVICE)
    logger.warning("file {} not found, returning None".format(filename))
    return None


def pred_loop(model, dataloader, gpu, *args, **kwargs):
    logits = []
    targets = []
    with torch.no_grad():
        for (data, target) in tqdm(dataloader):
            if gpu is not None:
                data = data.to(gpu)
            pred = model(data, *args, **kwargs)
            logits.append(pred.detach().cpu())
            targets.append(target.detach().cpu().reshape(-1, 1))

    logits = torch.vstack(logits)
    targets = torch.vstack(targets).reshape(-1)
    return logits, targets


def hidden_features_pred_loop(model, dataloader, gpu, *args, **kwargs):
    features = {}
    targets = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(dataloader)):
            if gpu is not None:
                data = data.to(gpu)
            logits, out_features = model.feature_list(data)
            for i, out_feature in enumerate(out_features):
                if idx == 0:
                    features[i] = []
                out_feature = out_feature.reshape(
                    out_feature.shape[0], out_feature.shape[1], -1
                )
                out_feature = torch.mean(out_feature, 2)
                features[i].append(out_feature)
            targets.append(target.detach().cpu().reshape(-1, 1))

    for k in features:
        features[k] = torch.vstack(features[k])
    targets = torch.vstack(targets).reshape(-1)
    return features, targets


def get_and_save_test_logits(nn_name, dataset_name, batch_size, gpu, *args, **kwargs):
    os.makedirs(
        "{}/tensors/{}/{}".format(ROOT, nn_name, dataset_name),
        exist_ok=True,
    )
    model = load_pre_trained_nn(nn_name, gpu)
    in_dataset_name = get_in_dataset_name(nn_name)
    dataloader = test_dataloader(dataset_name, in_dataset_name, batch_size=batch_size)
    logits, targets = pred_loop(model, dataloader, gpu, *args, **kwargs)
    torch.save(
        logits,
        "{}/tensors/{}/{}/logits_test.pt".format(ROOT, nn_name, dataset_name),
    )
    torch.save(
        targets,
        "{}/tensors/{}/{}/targets_test.pt".format(ROOT, nn_name, dataset_name),
    )
    logits = load_test_logits(nn_name, dataset_name)
    return logits


def get_and_save_test_features(nn_name, dataset_name, batch_size, gpu, *args, **kwargs):
    os.makedirs(
        "{}/tensors/{}/{}".format(ROOT, nn_name, dataset_name),
        exist_ok=True,
    )
    model = load_pre_trained_nn(nn_name, gpu)
    in_dataset_name = get_in_dataset_name(nn_name)
    dataloader = test_dataloader(dataset_name, in_dataset_name, batch_size=batch_size)
    features, targets = hidden_features_pred_loop(
        model, dataloader, gpu, *args, **kwargs
    )
    torch.save(
        features,
        "{}/tensors/{}/{}/features_test.pt".format(ROOT, nn_name, dataset_name),
    )
    features = load_test_features(nn_name, dataset_name)
    return features


def load_test_logits(nn_name, dataset_name):
    filename = "{}/tensors/{}/{}/logits_test.pt".format(ROOT, nn_name, dataset_name)
    return load_tensor(filename)


def load_test_features(nn_name, dataset_name):
    filename = "{}/tensors/{}/{}/features_test.pt".format(ROOT, nn_name, dataset_name)
    return load_tensor(filename)


def load_test_targets(nn_name, dataset_name):
    filename = "{}/tensors/{}/{}/targets_test.pt".format(ROOT, nn_name, dataset_name)
    return load_tensor(filename)


def load_test_penultimate_features(nn_name, dataset_name):
    filename = "{}/tensors/{}/{}/penultimate_features_test.pt".format(
        ROOT, nn_name, dataset_name
    )
    return load_tensor(filename)


def get_and_save_train_logits(nn_name, dataset_name, batch_size, gpu, *args, **kwargs):
    os.makedirs(
        "{}/tensors/{}/{}".format(ROOT, nn_name, dataset_name),
        exist_ok=True,
    )
    model = load_pre_trained_nn(nn_name, gpu)
    in_dataset_name = get_in_dataset_name(nn_name)
    dataloader = train_dataloader(
        dataset_name,
        in_dataset_name,
        batch_size=batch_size,
    )
    logits, targets = pred_loop(model, dataloader, gpu, *args, **kwargs)
    torch.save(
        logits,
        "{}/tensors/{}/{}/logits_train.pt".format(ROOT, nn_name, dataset_name),
    )
    torch.save(
        targets,
        "{}/tensors/{}/{}/targets_train.pt".format(ROOT, nn_name, dataset_name),
    )
    logits = load_train_logits(nn_name, dataset_name)
    return logits


def get_and_save_train_features(
    nn_name, dataset_name, batch_size, gpu, *args, **kwargs
):
    os.makedirs(
        "{}/tensors/{}/{}".format(ROOT, nn_name, dataset_name),
        exist_ok=True,
    )
    model = load_pre_trained_nn(nn_name, gpu)
    in_dataset_name = get_in_dataset_name(nn_name)
    dataloader = train_dataloader(
        dataset_name,
        in_dataset_name,
        batch_size=batch_size,
    )
    features, targets = hidden_features_pred_loop(
        model, dataloader, gpu, *args, **kwargs
    )
    torch.save(
        features,
        "{}/tensors/{}/{}/features_train.pt".format(ROOT, nn_name, dataset_name),
    )
    features = load_train_features(nn_name, dataset_name)
    return features


def load_train_logits(nn_name, dataset_name):
    filename = "{}/tensors/{}/{}/logits_train.pt".format(ROOT, nn_name, dataset_name)
    return load_tensor(filename)


def load_train_features(nn_name, dataset_name):
    filename = "{}/tensors/{}/{}/features_train.pt".format(ROOT, nn_name, dataset_name)
    return load_tensor(filename)


def load_train_targets(nn_name, dataset_name):
    filename = "{}/tensors/{}/{}/targets_train.pt".format(ROOT, nn_name, dataset_name)
    return load_tensor(filename)


def load_logits_centroid(nn_name, dataset_name, method_name=None, new=False, cap=1000):
    # check file existence
    filename = "{}/tensors/centroid_logits_{}_{}.pt".format(ROOT, nn_name, dataset_name)
    if new:
        filename = "{}/tensors/{}/{}/{}_centroid_logits.pt".format(
            ROOT, nn_name, dataset_name, method_name
        )
    if os.path.isfile(filename):
        centroid = torch.load(filename, map_location=DEVICE)
    else:
        logger.warning("file not found. Returning None")
        return None
    return centroid.detach()


def load_test_logits_centroid(nn_name, dataset_name, cap=1000):
    # check file existence
    filename = "{}/tensors/{}/{}/ige_centroid_logits_{}.pt".format(
        ROOT, nn_name, dataset_name, cap
    )
    if os.path.isfile(filename):
        centroid = torch.load(filename, map_location=DEVICE)
    else:
        logger.warning("file not found. Returning None")
        return None
    return centroid.detach()


def get_feature_list(model, gpu):
    if gpu is not None:
        temp_x = torch.rand(2, 3, 32, 32).cuda()
    else:
        temp_x = torch.rand(2, 3, 32, 32)
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    for count, out in enumerate(temp_list):
        feature_list[count] = out.size(1)
    return feature_list


def load_hidden_features_cov(
    nn_name, dataset_name, diag=False, cap=None, per_class=False
):
    mat_type = ""
    if diag:
        mat_type = "_diag"
    cap_name = "_{}".format(cap) if cap is not None else ""
    if per_class:
        cap_name += "_per_class"

    filename = "{}/tensors/{}/{}/hidden_features{}_cov_mat{}.pt".format(
        ROOT, nn_name, dataset_name, mat_type, cap_name
    )
    if os.path.isfile(filename):
        logger.info("loading cov tensors from file {}".format(filename.split("/")[-1]))
        return torch.load(filename, map_location=DEVICE)
    else:
        logger.warn("file {} not found, returning".format(filename.split("/")[-1]))
        return


def load_hidden_features_inv(
    nn_name, dataset_name, diag=False, cap=None, per_class=False
):
    cap_name = "_{}".format(cap) if cap is not None else ""
    if per_class:
        cap_name += "_per_class"
    mat_type = ""
    if diag:
        mat_type = "_diag"

    filename = "{}/tensors/{}/{}/hidden_features{}_invs_cov{}.pt".format(
        ROOT, nn_name, dataset_name, mat_type, cap_name
    )
    if os.path.isfile(filename):
        logger.info("loading inv tensors from file {}".format(filename.split("/")[-1]))
        return torch.load(filename, map_location=DEVICE)
    else:
        logger.warn("file {} not found, returning".format(filename.split("/")[-1]))
        return


def load_hidden_features_means(nn_name, dataset_name, cap=None):
    cap_name = "_{}".format(cap) if cap is not None else ""
    filename = "{}/tensors/{}/{}/hidden_features_means{}.pt".format(
        ROOT, nn_name, dataset_name, cap_name
    )
    if os.path.isfile(filename):
        logger.info(
            "loading means tensors from file {}".format(filename.split("/")[-1])
        )
        return torch.load(filename, map_location=DEVICE)
    else:
        logger.warn("file {} not found, returning".format(filename.split("/")[-1]))
        return


class LimitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, nb_samples):
        self.dataset = dataset
        self.nb_samples = min(nb_samples, len(dataset))

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return self.nb_samples
