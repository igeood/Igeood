import argparse
import itertools

import utils.data_and_nn_loader as dl
from src.adv_samples import generate_fgsm_adv_samples
from src.ensemble_method import AdvWeightRegression, WeightRegression
from src.igeood import main as igeood_main
from src.logits_benchmark import main as logits_main
from src.mahalanobis import main as mahalanobis_main
from utils.logger import logger

parser = argparse.ArgumentParser(
    description="Reproduce results from IGEOOD: An Information Geometry Approach to Out-of-Distribution Detection.",
    epilog="example: python eval.py igeoodlogits -nn densenet10 -o Imagenet_resize -e 0.0015 -T 5 -gpu 0",
    allow_abbrev=True,
)
parser.add_argument(
    "method",
    default="igeoodlogits",
    type=str,
    help="OOD detection method",
    choices=[
        "igeood",
        "igeood_plus",
        "igeood_adv",
        "igeood_adv_plus",
        "min_igeoodlogits",
        "igeood_logits",
        "igeoodlogits",
        "msp",
        "odin",
        "energy",
        "mahalanobis",
        "mahalanobis_adv",
        "kl_score_sum",
        "kl_score_min",
    ],
)
parser.add_argument(
    "-nn",
    "--nn",
    default="densenet",
    type=str,
    help="Neural network architecture",
    choices=["densenet", "resnet", "densenet2"],
)
parser.add_argument(
    "-i",
    "--in-dataset",
    default="CIFAR10",
    type=str,
    help="In-distribution dataset name",
    choices=["CIFAR10", "cifar10", "CIFAR100", "cifar100", "SVHN", "svhn"],
)
parser.add_argument(
    "-o",
    "--out-dataset",
    default="Imagenet",
    type=str,
    help="Out-of-distribution dataset name",
)
parser.add_argument(
    "-eps",
    "--epsilon",
    default=0,
    type=float,
    help="Noise magnitude for input pre-processing",
)
parser.add_argument(
    "-EPSS",
    "--epsilons",
    nargs="+",
    default=None,
    type=float,
    help="List of noise magnitudes for input pre-processing",
)
parser.add_argument(
    "-T",
    "--temperature",
    default=1,
    type=float,
    help="Logits scaling temperature",
)
parser.add_argument(
    "-TS",
    "--temperatures",
    nargs="+",
    default=None,
    type=float,
    metavar="TS",
    help="List of temperatures",
)
parser.add_argument(
    "-r",
    "--rewrite",
    default=False,
    type=bool,
    help="Re-calculate scores if true. If false, use available score files",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    help="Calculate scores in batches",
)
parser.add_argument(
    "-gpu",
    "--gpu",
    default=None,
    type=int,
    help="GPU index",
)

if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(args)

    method = args.method
    nn_name = args.nn
    in_dataset_name = args.in_dataset.upper()
    nn_name = dl.get_nn_name(nn_name, in_dataset_name)
    out_dataset_name = args.out_dataset
    eps = args.epsilon
    temperature = args.temperature
    rewrite = args.rewrite
    batch_size = args.batch_size
    gpu = args.gpu

    # multiple temperature and eps
    temperature_list = args.temperatures
    eps_list = args.epsilons
    if temperature_list is None:
        temperature_list = [temperature]
    if eps_list is None:
        eps_list = [eps]

    in_dataset_name = dl.get_in_dataset_name(nn_name)

    for temperature, eps in itertools.product(temperature_list, eps_list):
        if method in [
            "msp",
            "odin",
            "energy",
            "igeood_logits",
            "igeoodlogits",
            "min_igeoodlogits",
            "kl_score_sum",
            "kl_score_min",
        ]:
            logits_main(
                method,
                nn_name,
                in_dataset_name,
                out_dataset_name,
                temperature,
                eps,
                rewrite,
                batch_size,
                gpu,
            )
        elif method == "mahalanobis":
            # Mahalanobis uses only the blocks' outputs
            mahalanobis_main(
                WeightRegression(ignore_dim=1),
                nn_name,
                in_dataset_name,
                out_dataset_name,
                eps,
                batch_size,
                gpu,
                rewrite,
            )
        elif method == "mahalanobis_adv":
            adv_set = dl.load_adv_dataset(nn_name)
            if adv_set is None or rewrite:
                generate_fgsm_adv_samples(nn_name, batch_size, gpu)
            mahalanobis_main(
                AdvWeightRegression(ignore_dim=1),
                nn_name,
                in_dataset_name,
                out_dataset_name,
                eps,
                batch_size,
                gpu,
                rewrite,
            )
        elif method == "igeood_plus":
            igeood_main(
                WeightRegression(),
                nn_name,
                in_dataset_name,
                out_dataset_name,
                out_dataset_name,
                out_dataset_name,
                temperature,
                eps,
                batch_size,
                gpu,
                rewrite,
                True,
            )
        elif method == "igeood_adv":
            adv_set = dl.load_adv_dataset(nn_name)
            if adv_set is None or rewrite:
                generate_fgsm_adv_samples(nn_name, batch_size, gpu)
            igeood_main(
                AdvWeightRegression(ignore_dim=0),
                nn_name,
                in_dataset_name,
                out_dataset_name,
                None,
                None,
                temperature,
                eps,
                batch_size,
                gpu,
                rewrite,
                True,
            )
        elif method == "igeood_adv_plus":
            adv_set = dl.load_adv_dataset(nn_name)
            if adv_set is None or rewrite:
                generate_fgsm_adv_samples(nn_name, batch_size, gpu)
            igeood_main(
                AdvWeightRegression(ignore_dim=0),
                nn_name,
                in_dataset_name,
                out_dataset_name,
                "ADV",
                None,
                temperature,
                eps,
                batch_size,
                gpu,
                rewrite,
                True,
            )
        elif method == "igeood":
            igeood_main(
                WeightRegression(),
                nn_name,
                in_dataset_name,
                out_dataset_name,
                None,
                None,
                temperature,
                eps,
                batch_size,
                gpu,
                rewrite,
                True,
            )
        else:
            print("Method {} not found.".format(method))
