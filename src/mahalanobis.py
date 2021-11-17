import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils.data_and_nn_loader as dl
import utils.evaluation_metrics as em
import utils.file_manager as fm
from torch.autograd import Variable
from utils.logger import logger, timing

from src.ensemble_method import *
from src.ensemble_method import MeanScore, WeightRegression
from src.estimators import hidden_feature_estimator

cudnn.benchmark = True


@timing
def main(
    ensemble_method,
    nn_name,
    in_dataset_name,
    out_dataset_name,
    eps,
    batch_size,
    gpu,
    rewrite=False,
    *args,
    **kwargs
):
    # Ensemble method
    mat_type = ""
    # File naming
    prefix = "mahalanobis"

    # Model
    # in_dataset_name = dl.get_in_dataset_name(nn_name)
    num_classes = dl.get_num_classes(in_dataset_name)
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()
    feature_list = dl.get_feature_list(model, gpu)
    num_features = len(feature_list)

    fm.make_output_folders(nn_name, in_dataset_name)
    fm.make_output_folders(nn_name, out_dataset_name)

    # Matrices
    inverse = dl.load_hidden_features_inv(nn_name, in_dataset_name)
    sample_mean = dl.load_hidden_features_means(nn_name, in_dataset_name)

    if inverse is None or sample_mean is None:
        hidden_feature_estimator(nn_name, in_dataset_name, batch_size, gpu, True)
        inverse = dl.load_hidden_features_inv(nn_name, in_dataset_name)
        sample_mean = dl.load_hidden_features_means(nn_name, in_dataset_name)
    logger.info("tensors loaded")

    filename = "{}{}_{:.4f}.txt".format(prefix, mat_type, eps)

    # Get in scores
    f = fm.find_score_file(nn_name, in_dataset_name, filename)
    if rewrite is True or f is None:
        logger.info(
            "Calculating mahalanobis score for nn {} and dataset {}".format(
                nn_name, in_dataset_name
            )
        )
        in_dataloader = dl.test_dataloader(
            in_dataset_name, in_dataset_name, batch_size=batch_size
        )
        in_score = get_mahalanobis_score(
            model,
            in_dataloader,
            sample_mean,
            inverse,
            num_classes,
            nn_name,
            num_features,
            eps,
            gpu,
        )
        fw = fm.make_score_file(nn_name, in_dataset_name, filename)
        fm.write_score_file(fw, in_score)
        fw.close()
    else:
        in_score = fm.load_score_file(nn_name, in_dataset_name, filename)

    # Get out scores
    f = fm.find_score_file(nn_name, out_dataset_name, filename)
    if rewrite or f is None:
        logger.info(
            "Calculating mahalanobis score for nn {} and dataset {}".format(
                nn_name, out_dataset_name
            )
        )
        out_dataloader = dl.test_dataloader(
            out_dataset_name, in_dataset_name, batch_size=batch_size
        )
        out_score = get_mahalanobis_score(
            model,
            out_dataloader,
            sample_mean,
            inverse,
            num_classes,
            nn_name,
            num_features,
            eps,
            gpu,
        )
        fw = fm.make_score_file(nn_name, out_dataset_name, filename)
        fm.write_score_file(fw, out_score)
        fw.close()
    else:
        out_score = fm.load_score_file(nn_name, out_dataset_name, filename)

    # Validation data
    ensemble_name = ensemble_method.__name__
    if "val" in ensemble_method.__name__:
        val_dataset_name = ensemble_method.val_dataset_name
    elif "adv" in ensemble_method.__name__:
        val_dataset_name = "ADV" + nn_name
    else:
        val_dataset_name = out_dataset_name
    val_score = None
    if "val" in ensemble_method.__name__ or "adv" in ensemble_method.__name__:
        val_filename = "{}_{:.4f}.txt".format(prefix, eps)
        fm.make_output_folders(nn_name, val_dataset_name)
        f = fm.find_score_file(nn_name, val_dataset_name, val_filename)
        if rewrite is True or f is None:
            val_dataloader = dl.test_dataloader(
                val_dataset_name, in_dataset_name, batch_size=batch_size
            )
            val_score = get_mahalanobis_score(
                model,
                val_dataloader,
                sample_mean,
                inverse,
                num_classes,
                nn_name,
                num_features,
                eps,
                gpu,
            )
            fw = fm.make_score_file(nn_name, val_dataset_name, val_filename)
            fm.write_score_file(fw, val_score)
            fw.close()
        else:
            val_score = fm.load_score_file(nn_name, val_dataset_name, val_filename)

    # Ensemble method
    method_name = "{}_{}".format(prefix, ensemble_name)
    # length = min(len(in_score), len(out_score))
    in_s, out_s = ensemble_method(in_score, out_score, val_score)

    if np.isnan(in_s.max()) or np.isnan(out_s.max()):
        logger.warning("nan value found in score, returning without evaluating")
        return

    (
        fpr_at_tpr_in,
        fpr_at_tpr_out,
        detection,
        auroc,
        aupr_in,
        aupr_out,
    ) = em.print_metrics_and_info(
        in_s,
        out_s,
        nn_name,
        in_dataset_name,
        out_dataset_name,
        method_name,
        True,
        False,
        True,
    )

    fm.append_results_to_file(
        nn_name,
        out_dataset_name,
        method_name,
        eps,
        1,
        fpr_at_tpr_in,
        fpr_at_tpr_out,
        detection,
        auroc,
        aupr_in,
        aupr_out,
    )
    return fpr_at_tpr_in, detection, auroc, aupr_in


def get_mahalanobis_score(
    model,
    dataloader,
    sample_mean,
    inverse,
    num_classes,
    nn_name,
    num_features,
    eps=0.0,
    gpu=None,
):

    logger.info("get Mahalanobis scores")
    logger.info("noise magnitude: " + str(eps))
    mahalanobis = []
    for i in range(num_features):
        m = get_mahalanobis_layer_score(
            model,
            dataloader,
            num_classes,
            nn_name,
            sample_mean,
            inverse,
            i,
            eps,
            gpu,
        )
        mahalanobis.append(m)
    mahalanobis = np.hstack(mahalanobis)
    return mahalanobis


def get_mahalanobis_layer_score(
    model,
    test_loader,
    num_classes,
    net_type,
    sample_mean,
    inverse,
    layer_index,
    eps,
    gpu,
) -> np.ndarray:
    """
    Compute the Mahalanobis confidence score
    return: Mahalanobis score from layer_index
    """
    model.eval()
    Mahalanobis = []
    for data in test_loader:
        if type(data) in [tuple, list]:
            data, _ = data
        if gpu is not None:
            data = data.cuda()
        data = Variable(data, requires_grad=True)

        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = compute_mahalanobis_distance(
            out_features, sample_mean, inverse, layer_index, num_classes
        )
        if eps > 0:
            # Input_processing in the direction of the predicted class
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = torch.vstack(
                [sample_mean[layer_index][i] for i in sample_pred.cpu().numpy()]
            )
            zero_f = out_features - Variable(batch_sample_mean)
            pure_gau = (
                -0.5
                * torch.mm(
                    torch.mm(zero_f, Variable(inverse[layer_index])), zero_f.t()
                ).diag()
            )
            loss = torch.mean(-pure_gau)
            loss.backward()

            gradient = torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            gradient = dl.gradient_trasform(dl.get_in_dataset_name(net_type))(gradient)
            tempInputs = torch.add(data.data, -eps, gradient)

            with torch.no_grad():
                noise_out_features = model.intermediate_forward(
                    Variable(tempInputs), layer_index
                )
            noise_out_features = noise_out_features.view(
                noise_out_features.size(0), noise_out_features.size(1), -1
            )
            noise_out_features = torch.mean(noise_out_features, 2)
            gaussian_score = compute_mahalanobis_distance(
                noise_out_features, sample_mean, inverse, layer_index, num_classes
            )
        gaussian_score, _ = torch.max(gaussian_score, dim=1)

        Mahalanobis.extend(gaussian_score.cpu().numpy())

    Mahalanobis = np.asarray(Mahalanobis, dtype=np.float32).reshape(-1, 1)
    return Mahalanobis


def compute_mahalanobis_distance(
    out_features, sample_mean, inverse, layer_index, num_classes
):
    gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = (
            -0.5 * torch.mm(torch.mm(zero_f, inverse[layer_index]), zero_f.t()).diag()
        )
        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
    return gaussian_score


def predict(
    nn_name,
    dataloader,
    batch_size,
    gpu,
    eps=0,
):
    logger.info("get Mahalanobis scores")
    logger.info("noise magnitude: " + str(eps))
    mahalanobis = []

    # model
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    num_classes = dl.get_num_classes(in_dataset_name)
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()

    feature_list = dl.get_feature_list(model, gpu)
    num_features = len(feature_list)

    # Matrices
    inverse = dl.load_hidden_features_inv(nn_name, in_dataset_name)
    sample_mean = dl.load_hidden_features_means(nn_name, in_dataset_name)

    if inverse is None or sample_mean is None:
        hidden_feature_estimator(nn_name, in_dataset_name, batch_size, gpu, True)
        inverse = dl.load_hidden_features_inv(nn_name, in_dataset_name)
        sample_mean = dl.load_hidden_features_means(nn_name, in_dataset_name)

    for i in range(num_features):
        m = get_mahalanobis_layer_score(
            model,
            dataloader,
            num_classes,
            nn_name,
            sample_mean,
            inverse,
            i,
            eps,
            gpu,
        )
        mahalanobis.append(m)
    mahalanobis = np.hstack(mahalanobis)
    return mahalanobis
