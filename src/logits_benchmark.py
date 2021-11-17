import torch
import torch.backends.cudnn as cudnn
import utils.data_and_nn_loader as dl
import utils.evaluation_metrics as em
import utils.file_manager as fm
from torch.autograd import Variable
from tqdm import tqdm

ROOT = dl.ROOT
from utils.logger import logger, timing

from src.estimators import run_logits_centroid_estimator
from src.measures import *

cudnn.benchmark = True

method_dispatcher = {
    "msp": msp,
    "odin": odin,
    "energy": energy,
    "igeood_logits": igeoodlogits,
    "igeoodlogits": igeoodlogits,
}


@timing
def main(
    method,
    nn_name,
    in_dataset_name,
    out_dataset_name,
    temperature,
    eps,
    rewrite,
    batch_size,
    gpu,
    early_stopping=False,
):
    if type(method) == str:
        method = method_dispatcher[method]
    logger.info("temperature {}, eps {}".format(temperature, eps))
    # temperature = int(temperature)
    prefix = "{}".format(method.__name__)
    filename = "{}{:.1f}_{:.4f}".format(prefix, temperature, eps)
    # in_dataset_name = dl.get_in_dataset_name(nn_name)
    fm.make_output_folders(nn_name, in_dataset_name)
    fm.make_output_folders(nn_name, out_dataset_name)

    # early stopping condition
    if early_stopping:
        f_in = fm.find_score_file(nn_name, in_dataset_name, filename + ".txt")
        f_out = fm.find_score_file(nn_name, out_dataset_name, filename + ".txt")
        if f_in is not None and f_out is not None and rewrite is False:
            logger.warning("Scores already calculated, returning.")
            return
    if method in [
        igeoodlogits,
    ]:
        logits_centroid = dl.load_logits_centroid(
            nn_name, dl.get_in_dataset_name(nn_name)
        )
        if logits_centroid is None:
            logger.info("calculating logits centroids for IGEOOD score")
            logits_centroid, _, _, _ = run_logits_centroid_estimator(nn_name, gpu=gpu)
    else:
        logits_centroid = None

    if eps == 0:
        logits_in = dl.load_test_logits(nn_name, in_dataset_name)
        if logits_in is None or rewrite:
            fm.make_tensor_folder(nn_name, in_dataset_name)
            logits_in = dl.get_and_save_test_logits(
                nn_name, in_dataset_name, batch_size, gpu
            )
        logits_out = dl.load_test_logits(nn_name, out_dataset_name)
        if logits_out is None or rewrite:
            fm.make_tensor_folder(nn_name, out_dataset_name)
            logits_out = dl.get_and_save_test_logits(
                nn_name, out_dataset_name, batch_size, gpu
            )
    else:
        # In
        f = "{}/tensors/{}/{}/{}.pt".format(ROOT, nn_name, in_dataset_name, filename)
        logits_in = dl.load_tensor(f)
        if logits_in is None or rewrite:
            fm.make_tensor_folder(nn_name, in_dataset_name)
            model = dl.load_pre_trained_nn(nn_name, gpu)
            dataloader_in = dl.test_dataloader(
                in_dataset_name, in_dataset_name, batch_size=batch_size
            )
            logits_in = input_pre_processing(
                method,
                model,
                dataloader_in,
                nn_name,
                temperature,
                eps,
                gpu,
                logits_centroid,
            )
            torch.save(logits_in, f)

        # Out
        f = "{}/tensors/{}/{}/{}.pt".format(ROOT, nn_name, out_dataset_name, filename)
        logits_out = dl.load_tensor(f)
        if logits_out is None or rewrite:
            fm.make_tensor_folder(nn_name, out_dataset_name)
            model = dl.load_pre_trained_nn(nn_name, gpu)
            dataloader_out = dl.test_dataloader(
                out_dataset_name, in_dataset_name, batch_size=batch_size
            )
            logits_out = input_pre_processing(
                method,
                model,
                dataloader_out,
                nn_name,
                temperature,
                eps,
                gpu,
                logits_centroid,
            )
            # torch.save(logits_out, f)

    in_score = (
        method(logits_in, temperature, logits_centroid)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    out_score = (
        method(logits_out, temperature, logits_centroid)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    fw = fm.make_score_file(nn_name, in_dataset_name, filename + ".txt")
    fm.write_score_file(fw, in_score)
    fw.close()

    fw = fm.make_score_file(nn_name, out_dataset_name, filename + ".txt")
    fm.write_score_file(fw, out_score)
    fw.close()

    prefix = "{}".format(method.__name__)
    filename = "{}{:.1f}_{:.4f}".format(prefix, temperature, eps)
    # in_dataset_name = dl.get_in_dataset_name(nn_name)
    in_score = fm.load_score_file(nn_name, in_dataset_name, filename + ".txt")
    out_score = fm.load_score_file(nn_name, out_dataset_name, filename + ".txt")

    (
        fpr_at_tpr_in,
        fpr_at_tpr_out,
        detection,
        auroc,
        aupr_in,
        aupr_out,
    ) = em.print_metrics_and_info(
        in_score,
        out_score,
        nn_name,
        in_dataset_name,
        out_dataset_name,
        filename,
        True,
        False,
        True,
    )

    fm.append_results_to_file(
        nn_name,
        out_dataset_name,
        prefix,
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


def input_pre_processing(
    method, model, dataloader, nn_name, temperature, eps, gpu, *args, **kwargs
):
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    logits = []
    for (data, target) in tqdm(dataloader):
        if gpu is not None:
            data = data.to(gpu)
        data = Variable(data, requires_grad=True)
        pred = model(data)
        dist = method(pred, temperature, *args, **kwargs)
        loss = torch.mean(-dist)
        loss.backward()

        # Transforming the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Normalizing the gradient to the in distribution statistics
        gradient = dl.gradient_trasform(in_dataset_name)(gradient)
        with torch.no_grad():
            temp_inputs = torch.add(data, gradient, alpha=-eps)
            noised_logits = model(temp_inputs)

        logits.append(noised_logits.detach())

    logits = torch.vstack(logits)
    logger.info("scores calculated for {}.".format(method.__name__))
    return logits


def batch_prediction(method, data, model, nn_name, gpu, temperature, eps):
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    if type(method) == str:
        method = method_dispatcher[method]
    if igeoodlogits == method:
        logits_centroid = dl.load_logits_centroid(nn_name, in_dataset_name)
        if logits_centroid is None:
            logger.info("calculating logits centroids for IGEOOD score")
            logits_centroid, _, _, _ = run_logits_centroid_estimator(nn_name, gpu=gpu)
    else:
        logits_centroid = None

    if gpu is not None:
        data = data.cuda(gpu)
    data = Variable(data, requires_grad=True)
    pred = model(data)
    dist = method(pred, temperature, logits_centroid)

    if eps > 0:
        loss = torch.mean(-dist)
        loss.backward()

        # Transforming the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Normalizing the gradient to the in distribution statistics
        gradient = dl.gradient_trasform(in_dataset_name)(gradient)
        with torch.no_grad():
            temp_inputs = torch.add(data, gradient, alpha=-eps)
            noised_logits = model(temp_inputs)

        dist = method(noised_logits, temperature, logits_centroid)

    return dist.detach().cpu().numpy()
