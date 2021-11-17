import glob
import os

import numpy as np
import pandas as pd

from utils.data_and_nn_loader import ROOT
from utils.logger import logger


def make_output_folders(nn, out_dataset):
    os.makedirs(
        "{}/results/scores/{}/{}".format(ROOT, nn, out_dataset),
        exist_ok=True,
    )
    os.makedirs(
        "{}/results/figures/{}/{}".format(ROOT, nn, out_dataset),
        exist_ok=True,
    )
    os.makedirs(
        "{}/results/metrics/{}/{}".format(ROOT, nn, out_dataset),
        exist_ok=True,
    )


def make_image_dataset_folder(dataset_name):
    os.makedirs("{}/datasets/{}".format(ROOT, dataset_name), exist_ok=True)


def make_tensor_folder(nn_name, dataset_name):
    os.makedirs(
        "{}/tensors/{}/{}".format(ROOT, nn_name, dataset_name),
        exist_ok=True,
    )


def make_metric_folder(nn, out_dataset):
    os.makedirs(
        "{}/results/metrics/{}/{}".format(ROOT, nn, out_dataset),
        exist_ok=True,
    )


def make_score_file(nn, out_dataset, filename):
    make_output_folders(nn, out_dataset)
    return open(
        "{}/results/scores/{}/{}/{}".format(ROOT, nn, out_dataset, filename),
        "w",
    )


def write_score_file(f, data):
    np.savetxt(f, data, delimiter=",")
    f.close()


def load_score_file(nn, dataset_name, filename):
    path = "{}/results/scores/{}/{}/{}".format(ROOT, nn, dataset_name, filename)
    logger.info("loading scores from {}".format(path))
    return np.loadtxt(path, delimiter=",")


def find_score_file(nn, dataset_name, query):
    logger.info("searching for file {}/{}/{}".format(nn, dataset_name, query))
    prefix = "{}/results/scores/{}/{}/".format(ROOT, nn, dataset_name)
    path = glob.glob(prefix + query)
    if len(path) > 0:
        return path[0].split("/")[-1]
    logger.warn("file not found")
    return


def check_existence_score_file(nn_name, dataset_name, filename):
    path = glob.glob(
        "{}/results/scores/{}/{}/{}".format(ROOT, nn_name, dataset_name, filename)
    )
    return len(path) > 0


def make_evaluation_metrics_file(nn, out_dataset, filename):
    make_metric_folder(nn, out_dataset)
    return open(
        "{}/results/metrics/{}/{}/{}".format(ROOT, nn, out_dataset, filename),
        "w",
    )


def write_evaluation_metrics_file(f, header, content):
    for item in header:
        f.write("%s\n" % item)
    for item in content:
        f.write("%s\n" % item)
    return f


def clean_title(title):
    return "_".join(title.lower().split(" ")) + ".txt"


def append_results_to_file(
    nn_name,
    out_dataset_name,
    method_name,
    eps,
    temperature,
    fpr_at_tpr_in,
    fpr_at_tpr_out,
    detection,
    auroc,
    aupr_in,
    aupr_out,
    filename="results",
):
    results = pd.DataFrame.from_dict(
        {
            "nn": [nn_name],
            "out_dataset": [out_dataset_name],
            "method": [method_name],
            "eps": [eps],
            "T": [temperature],
            "fpr_at_tpr95_in": [fpr_at_tpr_in],
            "fpr_at_tpr95_out": [fpr_at_tpr_out],
            "detection": [detection],
            "auroc": [auroc],
            "aupr_in": [aupr_in],
            "aupr_out": [aupr_out],
        }
    )

    filename = "{}/results/{}.csv".format(ROOT, filename)
    if not os.path.isfile(filename):
        results.to_csv(filename, header=True, index=False)
    else:  # else it exists so append without writing the header
        results.to_csv(filename, mode="a", header=False, index=False)


def remove_duplicates(filename):
    filename = "{}/results/{}.csv".format(ROOT, filename)
    df = pd.read_csv(filename)
    logger.info("df has length {}".format(len(df)))
    df.drop_duplicates(
        subset=["nn", "out_dataset", "method", "eps", "T"], keep="last", inplace=True
    )
    df.to_csv(filename, index=False, header=True)
    logger.info("length reduced to {}".format(len(df)))
