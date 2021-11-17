import torch
import torch.nn as nn
import utils.data_and_nn_loader as dl
import utils.file_manager as fm
from torch.autograd import Variable
from tqdm import tqdm

ROOT = dl.ROOT


def generate_fgsm_adv_samples(nn_name, batch_size=128, save=False):
    fm.make_output_folders(nn_name, "ADV{}".format(nn_name))
    gpu = None
    if torch.cuda.is_available():
        gpu = 0
    # Fast Gradient Sign Method
    dataset_name = dl.get_in_dataset_name(nn_name)
    # Load network
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model = nn.DataParallel(model)
    model.to(gpu)
    adv_noise = 0.05

    if "densenet" in nn_name:
        # Noise magnitude compatible with mahalobis score paper
        min_pixel = -1.98888885975
        max_pixel = 2.12560367584
        if dataset_name == "CIFAR10":
            noise_magnitude = 0.21 / 4
        elif dataset_name == "CIFAR100":
            noise_magnitude = 0.21 / 8
        else:
            noise_magnitude = 0.21 / 4

    if "resnet" in nn_name:
        min_pixel = -2.42906570435
        max_pixel = 2.75373125076
        if dataset_name == "CIFAR10":
            noise_magnitude = 0.25 / 4
        elif dataset_name == "CIFAR100":
            noise_magnitude = 0.25 / 8
        else:
            noise_magnitude = 0.25 / 4

    dataloader = dl.train_dataloader(dataset_name, dataset_name, batch_size=batch_size)

    print("FGSM, Dist: " + dataset_name)
    model.eval()
    adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
    label_tot = 0

    correct, adv_correct, noise_correct = 0, 0, 0
    total, generated_noise = 0, 0

    criterion = nn.CrossEntropyLoss().cuda()

    selected_list = []
    selected_index = 0

    for (data, target) in tqdm(dataloader):
        if gpu is not None:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()

        noisy_data = torch.add(
            data.data, torch.randn(data.size()).to(gpu), alpha=noise_magnitude
        )
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

        if total == 0:
            clean_data_tot = data.clone().data.cpu()
            label_tot = target.clone().data.cpu()
            noisy_data_tot = noisy_data.clone().cpu()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()), 0)
            label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()), 0)

        # generate adversarial
        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient = dl.gradient_trasform(dataset_name)(gradient)

        adv_data = torch.add(inputs.data, gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, min_pixel, max_pixel)

        # measure the noise
        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)

        if total == 0:
            adv_data_tot = adv_data.clone().cpu()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()), 0)

        # compute the accuracy
        with torch.no_grad():
            output = model(Variable(adv_data))
        pred = output.data.max(1)[1]
        equal_flag_adv = pred.eq(target.data).cpu()
        adv_correct += equal_flag_adv.sum()

        # compute the accuracy
        with torch.no_grad():
            output = model(Variable(noisy_data))
        pred = output.data.max(1)[1]
        equal_flag_noise = pred.eq(target.data).cpu()
        noise_correct += equal_flag_noise.sum()

        for i in range(data.size(0)):
            if (
                equal_flag[i] == 1
                and equal_flag_noise[i] == 1
                and equal_flag_adv[i] == 0
            ):
                selected_list.append(selected_index)
            selected_index += 1

        total += data.size(0)
        break

    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)

    print("Adversarial Noise:({:.2f})\n".format(generated_noise / total))
    print(
        "Final Accuracy: {}/{} ({:.2f}%)\n".format(
            correct, total, 100.0 * correct / total
        )
    )
    print(
        "Adversarial Accuracy: {}/{} ({:.2f}%)\n".format(
            adv_correct, total, 100.0 * adv_correct / total
        )
    )
    print(
        "Noisy Accuracy: {}/{} ({:.2f}%)\n".format(
            noise_correct, total, 100.0 * noise_correct / total
        )
    )

    if save:
        fm.make_tensor_folder(nn_name, dataset_name)
        torch.save(
            adv_data_tot,
            "{}/tensors/{}/{}/adv_data_fgsm.pth".format(ROOT, nn_name, dataset_name),
        )
        torch.save(
            label_tot,
            "{}/tensors/{}/{}/label_fgsm.pth".format(ROOT, nn_name, dataset_name),
        )


if __name__ == "__main__":
    nn_name = "densenet10"
    generate_fgsm_adv_samples(nn_name, batch_size=200)
    nn_name = "densenet100"
    generate_fgsm_adv_samples(nn_name, batch_size=200)
    nn_name = "densenet_svhn"
    generate_fgsm_adv_samples(nn_name, batch_size=64)
    nn_name = "resnet_cifar10"
    generate_fgsm_adv_samples(nn_name, batch_size=200)
    nn_name = "resnet_cifar100"
    generate_fgsm_adv_samples(nn_name, batch_size=200)
    nn_name = "resnet_svhn"
    generate_fgsm_adv_samples(nn_name, batch_size=64)
