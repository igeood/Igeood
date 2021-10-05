import torch
import torch.nn.functional as F

import utils.data_and_nn_loader as dl


def msp(logits, *args, **kwargs):
    return torch.max(F.softmax(logits, dim=1), dim=1)[0]


def odin(logits, temperature, *args, **kwargs):
    return torch.max(F.softmax(logits / temperature, dim=1), dim=1)[0]


def energy(logits, temperature, *args, **kwargs):
    return temperature * torch.log(torch.sum(torch.exp(logits / temperature), dim=1))


def fisher_rao_logits_distance(
    output: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6
):
    inner = torch.sum(
        torch.sqrt(F.softmax(output, dim=1) * F.softmax(target, dim=1)), dim=1
    )
    return 2 * torch.acos(torch.clamp(inner, -1 + epsilon, 1 - epsilon))


def igeoodlogits(logits, temperature, centroids, epsilon=1e-12, *args, **kwargs):
    d = [
        fisher_rao_logits_distance(
            logits / temperature, mu.reshape(1, -1) / temperature, epsilon
        ).reshape(-1, 1)
        for mu in centroids
    ]
    stack = torch.hstack(d)
    return torch.sum(stack, 1)


def fr_distance_univariate_gaussian(
    mu_1: torch.Tensor, sig_1: torch.Tensor, mu_2: torch.Tensor, sig_2: torch.Tensor
) -> torch.Tensor:
    """Calculates the Fisher-Rao distance between univariate gaussian distributions in prallel.

    Args:
        mu_1 (torch.Tensor): Tensor of dimension (N,*) containing the means of N different univariate gaussians
        sig_1 (torch.Tensor): Standard deviations of univariate gaussian distributions
        mu_2 (torch.Tensor): Means of the second univariate gaussian distributions
        sig_2 (torch.Tensor): Standard deviation of the second univariate gaussian distributions
    Returns:
        torch.Tensor: Distance tensor of size (N,*)
    """
    dim = len(mu_1.shape)
    mu_1, mu_2 = mu_1.reshape(*mu_1.shape, 1), mu_2.reshape(*mu_2.shape, 1)
    sig_1, sig_2 = sig_1.reshape(*sig_1.shape, 1), sig_2.reshape(*sig_2.shape, 1)

    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=dl.DEVICE))
    a = torch.norm(
        torch.cat((mu_1 / sqrt_2, sig_1), dim=dim)
        - torch.cat((mu_2 / sqrt_2, -1 * sig_2), dim=dim),
        p=2,
        dim=dim,
    )
    b = torch.norm(
        torch.cat((mu_1 / sqrt_2, sig_1), dim=dim)
        - torch.cat((mu_2 / sqrt_2, sig_2), dim=dim),
        p=2,
        dim=dim,
    )

    num = a + b + 1e-12
    den = a - b + 1e-12
    return sqrt_2 * torch.log(num / den)


def fr_distance_multivariate_gaussian(
    x: torch.Tensor, y: torch.Tensor, cov_x: torch.Tensor, cov_y: torch.Tensor
) -> torch.Tensor:
    num_examples = x.shape[0]
    # Replicate std dev. matrix to match the batch size
    sig_x = torch.vstack([torch.sqrt(torch.diag(cov_x)).reshape(1, -1)] * num_examples)
    sig_y = torch.vstack([torch.sqrt(torch.diag(cov_y)).reshape(1, -1)] * num_examples)
    return torch.sqrt(
        torch.sum(
            fr_distance_univariate_gaussian(x, sig_x, y, sig_y) ** 2,
            dim=1,
        )
    ).reshape(-1, 1)


def igeoodfeature(out_feature, sample_mean, cov_in, cov_out, layer_index, num_classes):
    score = []
    for i in range(num_classes):
        if i not in sample_mean[layer_index].keys():
            continue
        batch_sample_mean = torch.vstack(
            [sample_mean[layer_index][i]] * out_feature.data.shape[0]
        )
        score.append(
            fr_distance_multivariate_gaussian(
                out_feature,
                batch_sample_mean,
                cov_out[layer_index],
                cov_in[layer_index],
            )
        )

    return torch.hstack(score)


def euclidian_distance(x, y):
    return torch.cdist(x, y, p=2)


def mahalanobis_distance(x, y, cov):
    delta = x - y
    delta_t = delta.transpose(1, 0)
    inverse = torch.inverse(cov)
    aux = torch.matmul(inverse, delta_t)
    m = (delta_t * aux).sum(0)  # equivalent of dot product
    return torch.sqrt(m)
