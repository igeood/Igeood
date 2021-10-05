import torchvision
import torchvision.transforms as transforms
from models.densenet import DenseNetBC100
from models.resnet import ResNet34

transform_cifar10_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # mirroring
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4913997, 0.4821584, 0.446531],
            [0.2470323, 0.243485, 0.2615876],
        ),
    ]
)
transform_cifar10_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4913997, 0.4821584, 0.446531],
            [0.2470323, 0.243485, 0.2615876],
        ),
    ]
)

# registries
transform_registry_train = {
    "cifar10": transform_cifar10_train,
    "cifar10_data_augmentation": transform_cifar10_train,
}
transform_registry_test = {
    "cifar10": transform_cifar10_test,
}
n_classes_registry = {
    "cifar10": 10,
    "cifar100": 100,
}
models_registry = {
    "densenet100": DenseNetBC100,
    "densenetbc100": DenseNetBC100,
    "resnet34": ResNet34,
}

##
DATASETS = (
    set(list(transform_registry_train.keys()))
    & set(list(transform_registry_test.keys()))
    & set(list(n_classes_registry.keys()))
)

MODELS = set(models_registry.keys())


def get_dataset(dataset_name, data_dir, train: bool):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        transform = (
            transform_registry_train.get("cifar10")
            if train
            else transform_registry_test.get("cifar10")
        )
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )

    return dataset


def get_num_classes(dataset_name: str):
    return n_classes_registry.get(dataset_name.lower())


def get_model(model_name, n_classes):
    print(model_name, n_classes)
    return models_registry.get(model_name.lower())(n_classes)
