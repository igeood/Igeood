import os

import numpy as np
import sklearn
import sklearn.covariance
import torch
import utils.data_and_nn_loader as dl
from torch.autograd import Variable
from utils.logger import logger

from src.measures import fisher_rao_logits_distance

ROOT = dl.ROOT


def run_logits_centroid_estimator(
    nn_name, epochs=100, batch_size=128, gpu=None, lr=0.01, *args, **kwargs
):
    model = dl.load_pre_trained_nn(nn_name, gpu)
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    dataloader = dl.train_dataloader(
        in_dataset_name, in_dataset_name, batch_size=batch_size
    )
    logger.info("diagonal matrix initialization")
    init_tensor = torch.eye(dl.get_num_classes(in_dataset_name))
    distance = fisher_rao_logits_distance
    logits, targets = dl.pred_loop(model, dataloader, gpu)
    if gpu is not None:
        init_tensor = init_tensor.cuda(gpu)
        logits = logits.cuda(gpu)
        targets = targets.cuda(gpu)

    centroid, epoch_loss = logits_centroid_estimator(
        logits, targets, init_tensor, distance, epochs, lr, *args, **kwargs
    )
    # save tensor
    os.makedirs("{}/tensors".format(ROOT), exist_ok=True)
    filename = "{}/tensors/centroid_logits_{}_{}.pt".format(
        ROOT, nn_name, in_dataset_name
    )
    torch.save(centroid, filename)
    logger.info("first loss is approx {}".format(epoch_loss[0][0]))
    logger.info("last loss is approx {}".format(epoch_loss[0][-1]))
    return centroid, epoch_loss, logits, targets


def logits_centroid_estimator(
    logits, targets, init_tensor, distance, epochs, lr, *args, **kwargs
):
    n_classes = init_tensor.shape[1]
    centroid = [
        Variable(init_tensor[i].reshape(1, -1), requires_grad=True)
        for i in range(n_classes)
    ]
    logger.info("Initialized centroid: {}".format(centroid))

    epoch_loss = [[] for _ in range(n_classes)]
    for epoch in range(epochs):
        for c in range(n_classes):
            filt = targets == c
            if filt.sum() == 0:
                continue

            d = distance(logits[filt].detach(), centroid[c], *args, **kwargs)
            loss = torch.mean(d)
            epoch_loss[c].append(loss.item())
            loss.backward()
            # optimizer.step()
            with torch.no_grad():
                aux = centroid[c]
                tmp = aux - lr * aux.grad
                centroid[c].copy_(tmp)

    logger.info("converged centroid: {}".format(centroid))
    return torch.vstack(centroid), epoch_loss


def get_hidden_features_sample(model, dataloader, gpu, cap=None):
    model.eval()
    feature_list = dl.get_feature_list(model, gpu)
    num_hidden_features = len(feature_list)

    hidden_feature_sample = {i: {} for i in range(num_hidden_features)}
    logger.info("cap is {}".format(cap))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            batch_size = data.shape[0]
            if gpu is not None:
                data = data.cuda(gpu)
                target = target.cuda(gpu)

            data = Variable(data)
            output, features = model.feature_list(data)

            for i, feature in enumerate(features):
                features[i] = torch.mean(
                    feature.reshape(feature.shape[0], feature.shape[1], -1), 2
                )

            pred = output.max(1)[1]
            # construct the sample matrix
            for b in range(batch_size):
                label = pred[b]
                index = int(label.cpu().numpy())
                for j, feature in enumerate(features):
                    if index not in hidden_feature_sample[j].keys():
                        hidden_feature_sample[j][index] = []
                    hidden_feature_sample[j][index].append(feature[b].reshape(1, -1))

            if cap is not None and batch_size * batch_idx >= cap:
                logger.warning("cap of {} exceeded, breaking...".format(cap))
                break

    for j in range(num_hidden_features):
        for c, sample in hidden_feature_sample[j].items():
            hidden_feature_sample[j][c] = torch.vstack(sample)

    return hidden_feature_sample


def get_hidden_feat_sample_mean(hidden_feature_sample):
    # sample mean per feature
    num_features = len(hidden_feature_sample)
    sample_class_mean = {}
    for i in range(num_features):
        x = hidden_feature_sample[i]
        sample_class_mean[i] = {
            c: torch.mean(sample, 0).reshape(1, -1)
            for c, sample in x.items()
            if len(sample) > 0
        }

    return sample_class_mean


def get_hidden_feat_cov_inv_matrix(
    hidden_feature_sample, sample_class_mean, diag=False, eps=1e-6
):
    num_features = len(hidden_feature_sample)

    inv = {}
    cov = {}
    for i in range(num_features):
        mu = sample_class_mean[i]
        X = [x - mu[c] for c, x in hidden_feature_sample[i].items()]
        X = torch.vstack(X)
        X = X.cpu().numpy()
        if diag:
            # diagonal covariance matrix estimation
            temp_cov_mat = [
                torch.from_numpy(np.cov(X[:, col].T, rowvar=False)).float()
                for col in range(X.shape[1])
            ]
            cov[i] = torch.diag(torch.tensor(temp_cov_mat))
            inv[i] = torch.diag(1 / (torch.tensor(temp_cov_mat) + 1e-12))
        else:
            # Maximum likelihood covariance estimator
            group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            # find pseudo-inverse
            group_lasso.fit(X)
            inv[i] = torch.from_numpy(group_lasso.precision_).float()
            cov[i] = torch.from_numpy(group_lasso.covariance_).float()

    return inv, cov


def hidden_feature_estimator(
    nn_name,
    dataset_name=None,
    batch_size=512,
    gpu=None,
    train=True,
    diag=False,
    cap=None,
    *args,
    **kwargs
):
    # test features
    cap_str = "_{}".format(cap) if cap is not None else ""
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    if dataset_name is None:
        dataset_name = in_dataset_name
    if train:
        dataloader = dl.train_dataloader(
            dataset_name, in_dataset_name, batch_size=batch_size
        )
    else:
        dataloader = dl.test_dataloader(dataset_name, in_dataset_name, batch_size=100)
    model = dl.load_pre_trained_nn(nn_name, gpu)

    sample = get_hidden_features_sample(model, dataloader, gpu, cap)
    means = get_hidden_feat_sample_mean(sample)
    inv, cov = get_hidden_feat_cov_inv_matrix(sample, means, diag, *args, **kwargs)

    # Save hidden features means
    os.makedirs("{}/tensors/{}/{}".format(ROOT, nn_name, dataset_name), exist_ok=True)
    filename = "{}/tensors/{}/{}/hidden_features_means{}.pt".format(
        ROOT, nn_name, dataset_name, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(means, filename)

    mat_type = ""
    if diag:
        mat_type = "_diag"

    # Save hidden features inv cov
    filename = "{}/tensors/{}/{}/hidden_features{}_invs_cov{}.pt".format(
        ROOT, nn_name, dataset_name, mat_type, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(inv, filename)

    # Save hidden feature cov
    filename = "{}/tensors/{}/{}/hidden_features{}_cov_mat{}.pt".format(
        ROOT, nn_name, dataset_name, mat_type, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(cov, filename)

    return (means, inv, cov)


if __name__ == "__main__":
    nn_name = "densenet10"
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    init_tensor = torch.eye(dl.get_num_classes(in_dataset_name))
    distance = fisher_rao_logits_distance
    epochs = 100
    out_dataset_names = [
        "SVHN",
        "Imagenet_resize",
        "iSUN",
        "LSUN_resize",
        "CIFAR100",
        "Textures",
        "Places365",
        "Chars74K",
        "gaussian_noise_dataset",
    ]
    for out_dataset_name in out_dataset_names:
        hidden_feature_estimator(nn_name, out_dataset_name, train=False, cap=1000)
    run_logits_centroid_estimator(nn_name, epochs=100, batch_size=128, gpu=None)
