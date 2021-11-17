import argparse

from hyperparameter_tuning.utils import METHOD_REGISTRY, MODELS

parser = argparse.ArgumentParser(
    description="Tune hyperparameters of an OOD detection method",
    epilog="example: python -m hyper_tuning.hyper_tuning densenet10 igeoodlogits -t-T True",
    allow_abbrev=True,
)

parser.add_argument(
    # "--nn-name",
    "--nn-names",
    nargs="+",
    default=None,
    help="architectures name list",
)
parser.add_argument(
    "--method-names",
    nargs="+",
    default=None,
    help="methods name list",
)
parser.add_argument(
    "--dataset-names",
    nargs="+",
    default=None,
    help="tuning datasets name list",
)
parser.add_argument(
    "--eval-datasets",
    nargs="+",
    default=None,
    help="evaluation datasets name list",
)
parser.add_argument(
    "-t-T",
    "--tune-temperature",
    type=bool,
    default=False,
)
parser.add_argument(
    "-t-eps",
    "--tune-eps",
    type=bool,
    default=False,
)
parser.add_argument(
    "-t-all",
    "--tune-all-params",
    type=bool,
    default=False,
)
parser.add_argument(
    "-t-seq",
    "--tune-sequential",
    type=bool,
    default=False,
)
parser.add_argument(
    "-T",
    "--temperature",
    type=float,
    default=1,
    help="temperature start value",
)
parser.add_argument(
    "-eps",
    "--eps",
    type=float,
    default=0,
    help="epsilon start value",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="make experiments tuning reproducible",
)
parser.add_argument(
    "--save",
    type=bool,
    default=True,
    help="save experiments results",
)
