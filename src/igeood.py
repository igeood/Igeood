import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils.data_and_nn_loader as dl
import utils.evaluation_metrics as em
import utils.file_manager as fm
from torch.autograd import Variable
from utils.logger import logger, timing

from src.ensemble_method import *
from src.estimators import (hidden_feature_estimator,
                            run_logits_centroid_estimator)
from src.measures import *

cudnn.benchmark = True


def get_prefix(
    cov_mat_ood,
    means_ood,
    logits_flag,
    per_class=False,
    distance=fr_distance_multivariate_gaussian,
):
    # File naming
    prefix = "igeoodfeatures"
    if cov_mat_ood is not None:
        prefix += "cov" + cov_mat_ood
    if logits_flag:
        prefix += "combinelogits"
    if per_class:
        prefix += "_per_class"
    if means_ood:
        prefix += "_means_ood"
    return prefix


def get_filename(prefix, temperature, eps):
    return "{}{:.1f}_{:.4f}.txt".format(prefix, temperature, eps)


@timing
def main(
    ensemble_method,
    nn_name,
    in_dataset_name,
    out_dataset_name,
    cov_mat_ood,
    means_ood,
    temperature,
    eps,
    batch_size,
    gpu,
    rewrite=False,
    logits_flag=True,
    per_class=False,
    distance=fr_distance_multivariate_gaussian,
):
    if cov_mat_ood == "same":
        cov_mat_ood = out_dataset_name
    elif cov_mat_ood == "ADV":
        cov_mat_ood += nn_name
    if out_dataset_name == "ADV":
        out_dataset_name += nn_name

    prefix = get_prefix(cov_mat_ood, means_ood, logits_flag, per_class, distance)
    filename = get_filename(prefix, temperature, eps)

    # in_dataset_name = dl.get_in_dataset_name(nn_name)
    fm.make_output_folders(nn_name, in_dataset_name)
    fm.make_output_folders(nn_name, out_dataset_name)

    # In scores
    f = fm.find_score_file(nn_name, in_dataset_name, filename)
    if rewrite is True or f is None:
        in_scores = igeoodwb_score(
            nn_name,
            in_dataset_name,
            batch_size,
            gpu,
            rewrite,
            cov_mat_ood,
            means_ood,
            logits_flag,
            temperature,
            eps,
            per_class=per_class,
            distance=distance,
        )
        fw = fm.make_score_file(nn_name, in_dataset_name, filename)
        fm.write_score_file(fw, in_scores)
        fw.close()
    else:
        in_scores = fm.load_score_file(nn_name, in_dataset_name, filename)

    # Out scores
    f = fm.find_score_file(nn_name, out_dataset_name, filename)
    if rewrite is True or f is None:
        out_scores = igeoodwb_score(
            nn_name,
            out_dataset_name,
            batch_size,
            gpu,
            rewrite,
            cov_mat_ood,
            means_ood,
            logits_flag,
            temperature,
            eps,
            per_class=per_class,
            distance=distance,
        )
        fw = fm.make_score_file(nn_name, out_dataset_name, filename)
        fm.write_score_file(fw, out_scores)
        fw.close()
    else:
        out_scores = fm.load_score_file(nn_name, out_dataset_name, filename)

    # Validation data
    if "val" in ensemble_method.__name__:
        val_dataset_name = ensemble_method.val_dataset_name
    elif "adv" in ensemble_method.__name__:
        val_dataset_name = "ADV" + nn_name
    else:
        val_dataset_name = out_dataset_name
    val_scores = None
    if "val" in ensemble_method.__name__ or "adv" in ensemble_method.__name__:
        val_filename = "{}{:.1f}_{:.4f}.txt".format(prefix, temperature, eps)
        fm.make_output_folders(nn_name, val_dataset_name)
        f = fm.find_score_file(nn_name, val_dataset_name, val_filename)
        if rewrite is True or f is None:
            val_scores = igeoodwb_score(
                nn_name,
                val_dataset_name,
                batch_size,
                gpu,
                rewrite,
                cov_mat_ood,
                means_ood,
                logits_flag,
                temperature,
                eps,
                per_class=per_class,
                distance=distance,
            )
            fw = fm.make_score_file(nn_name, val_dataset_name, val_filename)
            fm.write_score_file(fw, val_scores)
            fw.close()
        else:
            val_scores = fm.load_score_file(nn_name, val_dataset_name, val_filename)

    # Ensemble method
    # length = min(len(in_scores), len(out_scores))
    combine_in_score, combine_out_score = ensemble_method(
        in_scores, out_scores, val_scores
    )

    if np.isnan(combine_in_score.max()) or np.isnan(combine_out_score.max()):
        logger.warning("nan value found in score, returning without evaluating")
        return

    # Evaluation metric
    method_name = "{}_{}".format(filename.split(".txt")[0], ensemble_method.__name__)
    (
        fpr_at_tpr_in,
        fpr_at_tpr_out,
        detection,
        auroc,
        aupr_in,
        aupr_out,
    ) = em.print_metrics_and_info(
        combine_in_score,
        combine_out_score,
        nn_name,
        in_dataset_name,
        out_dataset_name,
        method_name,
        True,
        False,
        True,
    )

    # Save to results file
    method_name = "{}_{}".format(prefix, ensemble_method.__name__)
    fm.append_results_to_file(
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
    )
    return fpr_at_tpr_in, detection, auroc, aupr_in


def igeoodwb_score(
    nn_name,
    dataset_name,
    batch_size,
    gpu,
    rewrite=False,
    cov_mat_ood=None,
    means_ood=None,
    logits_flag=True,
    temperature=1,
    eps=0,
    dataloader=None,
    per_class=False,
    distance=fr_distance_multivariate_gaussian,
):
    in_dataset_name = dl.get_in_dataset_name(nn_name)

    # Model
    num_classes = dl.get_num_classes(in_dataset_name)
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()

    # Matrices
    sample_mean_in = dl.load_hidden_features_means(nn_name, in_dataset_name)
    cov_matrix_in = dl.load_hidden_features_cov(
        nn_name, in_dataset_name, True, None, per_class=per_class
    )
    if cov_matrix_in is None or sample_mean_in is None or rewrite:
        hidden_feature_estimator(
            nn_name, in_dataset_name, batch_size, gpu, True, True, None
        )
        cov_matrix_in = dl.load_hidden_features_cov(
            nn_name, in_dataset_name, True, None, per_class=per_class
        )
        sample_mean_in = dl.load_hidden_features_means(nn_name, in_dataset_name)

    if cov_mat_ood is not None:
        if cov_mat_ood == "ADV":
            cov_mat_ood += nn_name
            cap = None
        else:
            cap = 1000
        cov_val_dataset_name = cov_mat_ood
        logger.info("cov. val. matrix {}".format(cov_val_dataset_name))
        cov_matrix_out = dl.load_hidden_features_cov(
            nn_name, cov_val_dataset_name, True, cap
        )
        if cov_matrix_out is None or rewrite:
            hidden_feature_estimator(
                nn_name,
                cov_val_dataset_name,
                batch_size,
                gpu,
                False,
                True,
                cap,
            )
            cov_matrix_out = dl.load_hidden_features_cov(
                nn_name, cov_val_dataset_name, True, cap
            )
    else:
        cov_matrix_out = None

    if means_ood is not None:
        sample_mean_out = dl.load_hidden_features_means(
            nn_name, cov_val_dataset_name, cap=cap
        )
        if sample_mean_out is None or rewrite:
            hidden_feature_estimator(
                nn_name,
                cov_val_dataset_name,
                batch_size,
                gpu,
                False,
                True,
                cap,
            )
            sample_mean_out = dl.load_hidden_features_means(
                nn_name, cov_val_dataset_name, cap=cap
            )
    else:
        sample_mean_out = sample_mean_in

    # Logits
    logits_centroids = None
    if logits_flag:
        logits_centroids = dl.load_logits_centroid(nn_name, in_dataset_name)
        if logits_centroids is None or rewrite:
            logger.info("calculating logits centroids for IGEOOD score")
            logits_centroids, _, _, _ = run_logits_centroid_estimator(
                nn_name, gpu=gpu, batch_size=batch_size
            )

    logger.info("tensors loaded")

    # Get scores
    if dataloader is None:
        dataloader = dl.test_dataloader(
            dataset_name, in_dataset_name, batch_size=batch_size
        )
    return igeoodwb(
        model,
        dataloader,
        num_classes,
        sample_mean_in,
        cov_matrix_in,
        gpu,
        sample_mean_out,
        cov_matrix_out,
        logits_flag,
        temperature,
        eps,
        in_dataset_name,
        logits_centroids,
        distance=distance,
    )


def igeoodwb(
    model,
    dataloader,
    num_classes,
    sample_mean_in,
    cov_mat_in,
    gpu,
    sample_mean_out=None,
    cov_mat_out=None,
    logits_flag=True,
    temperature=None,
    eps=None,
    in_dataset_name=None,
    centroid_logits=None,
    distance=fr_distance_multivariate_gaussian,
):
    t0 = time.time()
    length = len(dataloader)
    model.eval()
    n_layers = len(sample_mean_in)

    igeoodfeature_scores = {i: [] for i in range(n_layers)}
    igeoodlogits_scores = []
    for batch_idx, data in enumerate(dataloader):
        if type(data) in [tuple, list]:
            data, _ = data
        if gpu is not None:
            data = data.cuda()
        data = Variable(data, requires_grad=True)
        logits, out_features = model.feature_list(data)

        # Logits
        if logits_flag:
            dist = igeoodlogits(logits, temperature, centroid_logits)

            if eps > 0:
                loss = torch.mean(-dist)
                loss.backward()

                # Normalizing the gradient to binary in {0, 1}
                gradient = torch.ge(data.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2

                # Normalizing the gradient to the same space of image
                gradient = dl.gradient_trasform(in_dataset_name)(gradient)
                with torch.no_grad():
                    temp_inputs = torch.add(data, gradient, alpha=-eps)
                    noised_logits, out_features = model.feature_list(temp_inputs)
                dist = igeoodlogits(noised_logits, temperature, centroid_logits)

            igeoodlogits_scores.extend(dist.detach().cpu().numpy().reshape(-1, 1))

        # Hidden features
        with torch.no_grad():
            for layer_idx, out_feature in enumerate(out_features):
                out_feature = out_feature.reshape(
                    out_feature.shape[0], out_feature.shape[1], -1
                )
                out_feature = torch.mean(out_feature, 2)

                # Compute Fisher-Rao score
                score1 = igeoodfeature(
                    out_feature,
                    sample_mean_in,
                    cov_mat_in,
                    cov_mat_in,
                    layer_idx,
                    num_classes,
                    distance=distance,
                )
                score1, _ = torch.min(score1, dim=1)
                score1 = score1.detach().cpu().numpy().reshape(-1, 1)

                if cov_mat_out is not None:
                    score2 = igeoodfeature(
                        out_feature,
                        sample_mean_out,
                        cov_mat_in,
                        cov_mat_out,
                        layer_idx,
                        num_classes,
                        distance=distance,
                    )
                    score2, _ = torch.min(score2, dim=1)
                    score2 = score2.detach().cpu().numpy().reshape(-1, 1)
                    igeoodfeature_scores[layer_idx].extend(np.hstack([score1, score2]))
                else:
                    igeoodfeature_scores[layer_idx].extend(score1)

        # Verbose
        if batch_idx % (int(length / 10) + 1) == 0 and batch_idx > 0:
            logger.info(
                "Batch {}/{}, {:.2f} seconds used.".format(
                    batch_idx + 1, length, time.time() - t0
                )
            )
            t0 = time.time()

    scores = np.hstack(
        [np.asarray(igeoodfeature_scores[i], dtype=np.float32) for i in range(n_layers)]
    )
    if logits_flag:
        scores = np.hstack([scores, np.vstack(igeoodlogits_scores)])

    return scores
