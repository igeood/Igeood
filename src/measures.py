import torch
import torch.linalg
import torch.nn.functional as F


def msp(logits, *args, **kwargs):
    return torch.max(F.softmax(logits, dim=1), dim=1)[0]


def odin(logits, temperature, *args, **kwargs):
    return torch.max(F.softmax(logits / temperature, dim=1), dim=1)[0]


def energy(logits: torch.Tensor, temperature: int, *args, **kwargs) -> torch.Tensor:
    return temperature * torch.log(torch.sum(torch.exp(logits / temperature), dim=1))


def norml2(logits, temperature, *args, **kwargs):
    return torch.sum(F.softmax(logits / temperature, dim=1) ** 2, dim=1)


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


def min_igeoodlogits(logits, temperature, centroids, epsilon=1e-12, *args, **kwargs):
    d = [
        fisher_rao_logits_distance(
            logits / temperature, mu.reshape(1, -1) / temperature, epsilon
        ).reshape(-1, 1)
        for mu in centroids
    ]
    stack = torch.hstack(d)
    return -torch.min(stack, 1)[0]


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

    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=mu_1.device))
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
    num_examples_x = x.shape[0]
    num_examples_y = y.shape[0]
    # Replicate std dev. matrix to match the batch size
    sig_x = torch.vstack(
        [torch.sqrt(torch.diag(cov_x)).reshape(1, -1)] * num_examples_x
    )
    sig_y = torch.vstack(
        [torch.sqrt(torch.diag(cov_y)).reshape(1, -1)] * num_examples_y
    )
    return torch.sqrt(
        torch.sum(
            fr_distance_univariate_gaussian(x, sig_x, y, sig_y) ** 2,
            dim=1,
        )
    ).reshape(-1, 1)


def igeoodfeature(
    out_feature,
    sample_mean,
    cov1,
    cov2,
    layer_idx,
    n_classes,
    distance=fr_distance_multivariate_gaussian,
):
    score = []
    for i in range(n_classes):
        if i not in sample_mean[layer_idx].keys():
            continue
        # per class sample mean
        if type(sample_mean[layer_idx]) == dict:
            batch_sample_mean = torch.vstack(
                [sample_mean[layer_idx][i]] * out_feature.data.shape[0]
            )
        else:
            batch_sample_mean = torch.vstack(
                [sample_mean[layer_idx]] * out_feature.data.shape[0]
            )
        # per class cov matrix
        if type(cov1[layer_idx]) == dict:
            c1 = cov1[layer_idx][i]
        else:
            c1 = cov1[layer_idx]
        if type(cov2[layer_idx]) == dict:
            c2 = cov2[layer_idx][i]
        else:
            c2 = cov2[layer_idx]

        score.append(
            distance(
                out_feature,
                batch_sample_mean,
                c1,
                c2,
            )
        )

    return torch.hstack(score)


def _igeood_layer_score(x, mus, cov_x, cov_mus):
    if type(mus) == dict:
        mus = torch.vstack([mu.reshape(1, -1) for k, mu in mus.items()])
    else:
        mus = [mus.reshape(1, -1)]

    stack = torch.hstack(
        [
            fr_distance_multivariate_gaussian(
                x, mu.reshape(1, -1), cov_x, cov_mus
            ).reshape(-1, 1)
            for mu in mus
        ]
    )
    return stack


def igeood_layer_score_min(x, mus, cov_x, cov_mus):
    stack = _igeood_layer_score(x, mus, cov_x, cov_mus)
    return torch.min(stack, dim=1)[0]


def igeood_layer_score_sum(x, mus, cov_x, cov_mus):
    stack = _igeood_layer_score(x, mus, cov_x, cov_mus)
    return torch.sum(stack, dim=1)


def euclidian_distance(x, y):
    return torch.cdist(x, y, p=2)


def mahalanobis_distance(
    x: torch.Tensor, y: torch.Tensor, cov: torch.Tensor, *args, **kwargs
):
    delta = x - y
    delta_t = delta.T
    inverse = torch.linalg.pinv(cov, hermitian=True)
    inverse_prod = inverse @ delta_t
    m = (delta_t * inverse_prod).sum(0)  # equivalent of dot product
    return torch.nan_to_num(torch.sqrt(m), 0)


def mahalanobis_distance_inv(
    x: torch.Tensor, y: torch.Tensor, cov: torch.Tensor, *args, **kwargs
):
    delta = x - y
    delta_t = delta.T
    # inverse = torch.linalg.pinv(cov, hermitian=True)
    inverse_prod = torch.linalg.lstsq(cov, delta_t, driver="gelsd")[0]
    m = (delta_t * inverse_prod).sum(0)  # equivalent of dot product
    return torch.sqrt(m)


def mahalanobis_distance_inverse(
    x: torch.Tensor, y: torch.Tensor, cov_inv: torch.Tensor, *args, **kwargs
):
    # be careful with memory overflow
    delta = x - y
    delta_t = delta.transpose(1, 0)
    m = torch.diag(delta @ cov_inv @ delta_t)
    return torch.sqrt(m)


def kl_divergence(p, q, eps=1e-6):
    return (p * torch.log(p / (q + eps))).sum(1)


def kl_divergence_logits(logits1, logits2):
    return kl_divergence(
        F.softmax(logits1, dim=1),
        F.softmax(logits2, dim=1),
    )


def _kl_score(logits, temperature, centroids):
    d = [
        kl_divergence(
            F.softmax(logits / temperature, dim=1),
            F.softmax(mu.reshape(1, -1) / temperature, dim=1),
        ).reshape(-1, 1)
        for mu in centroids
    ]
    stack = torch.hstack(d)
    return stack


def kl_score_sum(logits, temperature, centroids, *args, **kwargs):
    stack = _kl_score(logits, temperature, centroids)
    return torch.sum(stack, 1)


def kl_score_min(logits, temperature, centroids, *args, **kwargs):
    stack = _kl_score(logits, temperature, centroids)
    return -torch.min(stack, 1)[0]
