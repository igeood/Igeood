from src.logits_benchmark import input_pre_processing
import optuna
from optuna import trial as trial_module
import torch
import torch.utils.data
import utils.data_and_nn_loader as dl
import utils.evaluation_metrics as em
from utils.logger import timing
import numpy as np
import src.logits_benchmark
import itertools
from hyperparameter_tuning.utils import (
    get_method,
    report_results,
    write_results,
    DATASETS,
)
from hyperparameter_tuning.parser import parser

# EPS = [0, 0.0005, 0.001, 0.0014, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]
# TEMPERATURTE = [1, 2, 5, 10, 100, 500, 1000]


def temperature_objective(trial: trial_module.Trial):
    # sourcery skip: inline-immediately-returned-variable
    temperature = trial.suggest_float(name="temperature", low=1, high=1000, step=0.1)

    in_scores = metric(in_logits, temperature, args.in_centroids).detach().cpu().numpy()
    out_scores = (
        metric(out_logits, temperature, args.in_centroids).detach().cpu().numpy()
    )

    fpr_at_tpr = em.compute_metrics(in_scores, out_scores, fpr_only=True)

    return fpr_at_tpr


def tune_temperature(
    metric_fn, nn_name, in_dataset_name, out_dataset_name, gpu, **kwargs
):
    global metric, in_logits, out_logits, args
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name="temperature tuning", direction="minimize", sampler=sampler
    )
    metric = metric_fn
    in_logits = dl.load_test_logits(nn_name, in_dataset_name)
    if in_logits is None:
        in_logits = dl.get_and_save_test_logits(
            nn_name, in_dataset_name, args.batch_size, gpu
        )

    out_logits = dl.load_test_logits(nn_name, out_dataset_name)
    if out_logits is None:
        out_logits = dl.get_and_save_test_logits(
            nn_name, out_dataset_name, args.batch_size, gpu
        )

    # in_centroids = dl.load_logits_centroid(nn_name, in_dataset_name).to(gpu)

    study.optimize(
        temperature_objective,
        n_trials=500,
        timeout=600,
        n_jobs=1,
        show_progress_bar=True,
        **kwargs,
    )
    return study


def eps_objective(trial: trial_module.Trial, gpu, temperature=1):
    # sourcery skip: inline-immediately-returned-variable
    eps = trial.suggest_float(name="eps", low=0, high=0.004, step=1e-4)

    in_logits = input_pre_processing(
        metric, model, dataloader_in, nn, temperature, eps, gpu, args.in_centroids
    )
    out_logits = input_pre_processing(
        metric, model, dataloader_out, nn, temperature, eps, gpu, args.in_centroids
    )

    in_scores = metric(in_logits, temperature, args.in_centroids).detach().cpu().numpy()
    out_scores = (
        metric(out_logits, temperature, args.in_centroids).detach().cpu().numpy()
    )
    fpr_at_tpr = em.compute_metrics(in_scores, out_scores, fpr_only=True)

    return fpr_at_tpr


def tune_eps(metric_fn, nn_name, in_dataset_name, out_dataset_name, gpu, **kwargs):
    global metric, nn, temperature, model, dataloader_in, dataloader_out
    # sampler = optuna.samplers.TPESampler(seed=args.seed)
    search_space = {"eps": list(np.linspace(0, 0.004, 21).round(4))}
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        study_name="eps tuning", direction="minimize", sampler=sampler
    )

    nn = nn_name
    metric = metric_fn

    temperature = kwargs.get("temperature", 1)
    nb_samples = 10000

    model = dl.load_pre_trained_nn(nn_name, gpu)
    dataset_in = dl.LimitDataset(
        dl.load_test_dataset(in_dataset_name, in_dataset_name), nb_samples
    )
    dataset_out = dl.LimitDataset(
        dl.load_test_dataset(out_dataset_name, in_dataset_name), nb_samples
    )
    dataloader_in = torch.utils.data.DataLoader(dataset_in, batch_size=args.batch_size)
    dataloader_out = torch.utils.data.DataLoader(
        dataset_out, batch_size=args.batch_size
    )
    # in_centroids = dl.load_logits_centroid(nn_name, in_dataset_name).to(gpu)

    study.optimize(
        lambda trial: eps_objective(trial, gpu, temperature), n_trials=21, n_jobs=1
    )
    return study


def all_objective(trial: trial_module.Trial, gpu):
    # sourcery skip: inline-immediately-returned-variable
    eps = trial.suggest_float(name="eps", low=0, high=0.01, step=1e-4)
    temperature = trial.suggest_float(name="temperature", low=1, high=1000, step=0.1)

    in_logits = input_pre_processing(
        metric, model, dataloader_in, nn, temperature, eps, gpu, args.in_centroids
    )
    out_logits = input_pre_processing(
        metric, model, dataloader_out, nn, temperature, eps, gpu, args.in_centroids
    )

    in_scores = metric(in_logits, temperature, args.in_centroids).detach().cpu().numpy()
    out_scores = (
        metric(out_logits, temperature, args.in_centroids).detach().cpu().numpy()
    )
    fpr_at_tpr = em.compute_metrics(in_scores, out_scores, fpr_only=True)

    return fpr_at_tpr


@timing
def tune_all_params(
    metric_fn, nn_name, in_dataset_name, out_dataset_name, gpu, **kwargs
):
    global metric, nn, temperature, model, dataloader_in, dataloader_out
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name="eps and temperature tuning", direction="minimize", sampler=sampler
    )
    nn = nn_name
    metric = metric_fn
    nb_samples = 10000

    model = dl.load_pre_trained_nn(nn_name, gpu)
    dataset_in = dl.LimitDataset(
        dl.load_test_dataset(in_dataset_name, in_dataset_name), nb_samples
    )
    dataset_out = dl.LimitDataset(
        dl.load_test_dataset(out_dataset_name, in_dataset_name), nb_samples
    )
    dataloader_in = torch.utils.data.DataLoader(dataset_in, batch_size=args.batch_size)
    dataloader_out = torch.utils.data.DataLoader(
        dataset_out, batch_size=args.batch_size
    )

    study.optimize(lambda trial: all_objective(trial, gpu), n_trials=100, n_jobs=1)
    return study


@timing
def tune_sequential(
    metric_fn, nn_name, in_dataset_name, out_dataset_name, gpu, **kwargs
):
    global metric, nn, temperature, metric, in_logits, out_logits, in_centroids, model, dataloader_in, dataloader_out
    # temperature tuning
    temp_study = tune_temperature(
        metric_fn,
        nn_name,
        in_dataset_name,
        out_dataset_name,
        gpu,
        **kwargs,
    )
    temperature = temp_study.best_trial.params.get("temperature")
    args.temperature = temperature
    report_results(temp_study)
    if args.save:
        write_results(temp_study, args, name="tune_temperature")

    # eps tuning
    eps_study = tune_eps(
        metric_fn,
        nn_name,
        in_dataset_name,
        out_dataset_name,
        gpu,
        temperature=temperature,
    )
    eps = eps_study.best_trial.params.get("eps")
    args.eps = eps
    report_results(eps_study)
    if args.save:
        write_results(eps_study, args, name=f"tune_eps_with_T_{temperature}")

    temp_trial = temp_study.best_trial
    eps_trial = eps_study.best_trial

    study = optuna.create_study()
    study_trial = optuna.create_trial(
        params={
            "temperature": temp_trial.params.get("temperature"),
            "eps": eps_trial.params.get("eps"),
        },
        distributions={
            "temperature": temp_trial.distributions.get("temperature"),
            "eps": eps_trial.distributions.get("eps"),
        },
        value=eps_trial.values[0],
    )
    study.add_trial(study_trial)
    return study


@timing
def main_hyper_logits(args):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.gpu = 0 if args.device == "cuda" else None
    args.batch_size = 128

    kwargs = {}
    if args.tune_all_params:
        args.tune_fn = tune_all_params
    elif args.tune_temperature:
        args.tune_fn = tune_temperature
    elif args.tune_eps:
        args.tune_fn = tune_eps
        kwargs["temperature"] = args.temperature
    elif args.tune_sequential:
        args.tune_fn = tune_sequential
    else:
        print("Invalid tuning mode. Returning.")
        return

    if args.eval_datasets is None:
        args.eval_datasets = args.dataset_names
    elif "all" in args.eval_datasets:
        args.eval_datasets = DATASETS

    print(args)
    for args.method, args.nn_name, args.dataset_name in itertools.product(
        args.method_names, args.nn_names, args.dataset_names
    ):
        print("Tune on", args.nn_name, args.dataset_name, args.method)
        args.in_dataset_name = dl.get_in_dataset_name(args.nn_name)
        if "ige" in args.method:
            distance_name = "fisher_rao_logits_distance"
        elif "kl" in args.method:
            distance_name = "kl_divergence_logits"
        args.in_centroids = dl.load_logits_centroid(
            args.nn_name, args.in_dataset_name, distance_name, True
        ).to(args.gpu)
        args.metric = get_method(args.method)
        study = args.tune_fn(
            args.metric,
            args.nn_name,
            args.in_dataset_name,
            args.dataset_name,
            args.gpu,
            **kwargs,
        )
        args.eps = round(study.best_trial.params.get("eps", args.eps), 4)
        args.temperature = round(
            study.best_trial.params.get("temperature", args.temperature), 1
        )

        report_results(study)
        if args.save:
            write_results(study, args)

        # eval
        for ds in args.eval_datasets:
            print("Eval. on", ds)
            src.logits_benchmark.main(
                args.method,
                args.nn_name,
                args.in_dataset_name,
                ds,
                args.temperature,
                args.eps,
                False,
                args.batch_size,
                args.gpu,
                early_stopping=False,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    main_hyper_logits(args)
