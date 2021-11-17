import os

import optuna
import src.measures as m
from optuna.trial import TrialState

import utils.file_manager as fm

MODELS = [
    "densenet10",
    "resnet_cifar10",
    "densenet100",
    "resnet_cifar100",
    "densenet_svhn",
    "resnet_svhn",
]
DATASETS = [
    "iSUN",
    "SVHN",
    "Imagenet_resize",
    "LSUN_resize",
    "CIFAR10",
    "CIFAR100",
    "Textures",
    "Places365",
    "Chars74K",
    "gaussian_noise_dataset",
]
METHOD_REGISTRY = {
    "igeoodlogits": m.igeoodlogits,
    "min_igeoodlogits": m.min_igeoodlogits,
    "odin": m.odin,
    "energy": m.energy,
    "kl_score_min": m.kl_score_min,
    "kl_score_sum": m.kl_score_sum,
}
WB_METHOD_REGISTRY = {
    "mahalanobis": "mahalanobis",
}


def get_method(method_name: str):
    try:
        return METHOD_REGISTRY.get(method_name.lower())
    except KeyError:
        raise KeyError(
            "Method not registered in `method_registry` or not implemented yet."
        )


def write_results(study: optuna.Study, args, name=None):
    # study results
    if name is None:
        name = args.tune_fn.__name__
    path = os.path.join("results/hyper", args.nn_name, args.dataset_name, args.method)
    filename = f"{name}_{args.seed}"
    os.makedirs(path, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(path, filename + ".csv"))

    # overall results
    data = {
        "nn": args.nn_name,
        "out_dataset": args.dataset_name,
        "method": args.method,
        "eps": args.eps,
        "T": args.temperature,
        "fpr_at_tpr95_in": study.best_trial.value,
        "seed": args.seed,
    }
    fm.append_to_file(data, "ood_hyper_tuning")


def report_results(study: optuna.Study):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    trial = study.best_trial

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
