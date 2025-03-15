import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip
from models.model import ResNet18, ResNet34, ResNet50, vgg19_bn, vgg11_bn, vgg13_bn, vgg16_bn
from models.model_i import ResNet18 as ResNet18_i
from models.model_i import ResNet34 as ResNet34_i
from models.model_i import ResNet50 as ResNet50_i
from models.model_i import vgg11_bn as vgg11_bn_i
from models.model_i import vgg13_bn as vgg13_bn_i
from models.model_i import vgg16_bn as vgg16_bn_i
from models.model_i import vgg19_bn as vgg19_bn_i
from models.baseline_MNIST_network import BaselineMNISTNetwork
from SSBW import StegaStampEncoder, StegaStampDecoder, Discriminator, MNISTStegaStampEncoder, MNISTStegaStampDecoder, MNISTDiscriminator
from torchvision.datasets import CIFAR10, DatasetFolder, MNIST
import torchvision
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import cv2


def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Normalize:
    """Normalization of images.

    Args:
        dataset_name (str): the name of the dataset to be normalized.
        expected_values (float): the normalization expected values.
        variance (float): the normalization variance.
    """

    def __init__(self, dataset_name, expected_values, variance):
        if dataset_name == "cifar" or dataset_name == 'imagenet' or dataset_name == 'tinyimagenet':
            self.n_channels = 3
        elif dataset_name == "mnist":
            self.n_channels = 1
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class ProbTransform(torch.nn.Module):
    """The data augmentation transform by the probability.

    Args:
        f (nn.Module): the data augmentation transform operation.
        p (float): the probability of the data augmentation transform.
    """

    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    """The data augmentation transform.

    Args:
        dataset_name (str): the name of the dataset.
    """

    def __init__(self, dataset_name):
        super(PostTensorTransform, self).__init__()
        if dataset_name == 'mnist':
            input_height, input_width = 28, 28
        elif dataset_name == 'cifar':
            input_height, input_width = 32, 32
        self.random_crop = ProbTransform(transforms.RandomCrop((input_height, input_width), padding=5),
                                         p=0.8)  # ProbTransform(A.RandomCrop((input_height, input_width), padding=5), p=0.8)
        self.random_rotation = ProbTransform(transforms.RandomRotation(10),
                                             p=0.5)  # ProbTransform(A.RandomRotation(10), p=0.5)
        if dataset_name == "cifar":
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)  # A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path, 'a') as f:
            f.write(msg)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def model_test(model, dataset, device, batch_size=16, num_workers=4, normalizer=None):
    with torch.no_grad():
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True
        )

        model = model.to(device)
        model.eval()

        predict_digits = []
        labels = []
        for batch in test_loader:
            batch_img, batch_label = batch
            if normalizer:
                batch_img = normalizer(batch_img)
            batch_img = batch_img.to(device)
            batch_img = model(batch_img)
            batch_img = batch_img.cpu()
            predict_digits.append(batch_img)
            labels.append(batch_label)

        predict_digits = torch.cat(predict_digits, dim=0)
        labels = torch.cat(labels, dim=0)
        return predict_digits, labels


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_transform(dataset_name):
    if dataset_name == 'cifar':
        transform_train = Compose([
            transforms.Resize((32, 32)),
            RandomHorizontalFlip(),
            ToTensor(),
        ])

        transform_test = Compose([
            transforms.Resize((32, 32)),
            ToTensor(),
        ])
    elif dataset_name == 'imagenet':
        transform_train = Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            torchvision.transforms.RandomRotation(20),
            RandomHorizontalFlip(0.5),
            ToTensor(),
        ])
        transform_test = Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            ToTensor(),
        ])
    elif dataset_name == 'mnist':
        transform_train = Compose([
            transforms.Resize((28, 28)),
            RandomHorizontalFlip(),
            ToTensor(),
        ])
        transform_test = Compose([
            transforms.Resize((28, 28)),
            ToTensor(),
        ])
    elif dataset_name == 'tinyimagenet':
        transform_train = Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            torchvision.transforms.RandomRotation(20),
            RandomHorizontalFlip(0.5),
            ToTensor(),
        ])
        transform_test = Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            ToTensor(),
        ])
    else:
        print("We do not use this dataset in transformation now.")
        return
    return transform_train, transform_test


def define_dataset(dataset_name, transform_train, transform_test):
    if dataset_name == 'cifar':
        cls_num = 10
        trainset = CIFAR10(
            root='/home/sun/sunming/Untargeted_Backdoor_Watermark/data/cifar',
            # please replace this with path to your dataset
            transform=transform_train,
            target_transform=None,
            train=True,
            download=False)
        testset = CIFAR10(
            root='/home/sun/sunming/Untargeted_Backdoor_Watermark/data/cifar',
            # please replace this with path to your dataset
            transform=transform_test,
            target_transform=None,
            train=False,
            download=False)
        print("CIFAR10 is adopted")

    elif dataset_name == 'tinyimagenet':
        cls_num = 50
        trainset = DatasetFolder(
            root='/home/sun/sunming/dataset/sub-imagenet-50/train',
            # please replace this with path to your training set
            loader=cv2.imread,
            extensions=('jpeg',),
            transform=transform_train,
            target_transform=None,
            is_valid_file=None)

        testset = DatasetFolder(
            root='/home/sun/sunming/dataset/sub-imagenet-50/val',  # please replace this with path to your test set
            loader=cv2.imread,
            extensions=('jpeg',),
            transform=transform_test,
            target_transform=None,
            is_valid_file=None)
        print("Tiny-ImageNet is adopted")

    elif dataset_name == 'imagenet':
        cls_num = 12
        trainset = DatasetFolder(
            root='/home/sun/sunming/Untargeted_Backdoor_Watermark/data/imagenet12/train',
            # please replace this with path to your training set
            loader=cv2.imread,
            extensions=('jpeg',),
            transform=transform_train,
            target_transform=None,
            is_valid_file=None)

        testset = DatasetFolder(
            root='/home/sun/sunming/Untargeted_Backdoor_Watermark/data/imagenet12/val',
            # please replace this with path to your test set
            loader=cv2.imread,
            extensions=('jpeg',),
            transform=transform_test,
            target_transform=None,
            is_valid_file=None)
        print("ImageNet is adopted")

    elif dataset_name == 'mnist':
        cls_num = 10
        trainset = MNIST(
            root='/home/sun/sunming/Untargeted_Backdoor_Watermark/data/mnist',
            # please replace this with path to your dataset
            transform=transform_train,
            target_transform=None,
            train=True,
            download=False)
        testset = MNIST(
            root='/home/sun/sunming/Untargeted_Backdoor_Watermark/data/mnist',
            # please replace this with path to your dataset
            transform=transform_test,
            target_transform=None,
            train=False,
            download=False)
        print("MNIST is adopted")
    else:
        print("We do not use this dataset.")
        return

    return trainset, testset, cls_num


def define_model(dataset_name, model_name, surrogate_model_name, cls_num, pretrain=False):
    if dataset_name == 'cifar':
        if model_name == 'resnet18':
            main_model = ResNet18(num_classes=cls_num)
            clean_model = ResNet18(num_classes=cls_num)
            print("ResNet-18 is adopted")
        elif model_name == 'resnet34':
            main_model = ResNet34(num_classes=cls_num)
            clean_model = ResNet34(num_classes=cls_num)
            print("ResNet-34 is adopted")
        elif model_name == 'resnet50':
            main_model = ResNet50(num_classes=cls_num)
            clean_model = ResNet50(num_classes=cls_num)
            print("ResNet-50 is adopted")
        elif model_name == 'vgg11':
            main_model = vgg11_bn(num_classes=cls_num)
            clean_model = vgg11_bn(num_classes=cls_num)
            print("VGG-11 is adopted")
        elif model_name == 'vgg13':
            main_model = vgg13_bn(num_classes=cls_num)
            clean_model = vgg13_bn(num_classes=cls_num)
            print("VGG-13 is adopted")
        elif model_name == 'vgg16':
            main_model = vgg16_bn(num_classes=cls_num)
            clean_model = vgg16_bn(num_classes=cls_num)
            print("VGG-16 is adopted")
        elif model_name == 'vgg19':
            main_model = vgg19_bn(num_classes=cls_num)
            clean_model = vgg19_bn(num_classes=cls_num)
            print("VGG-19 is adopted")
        else:
            print("We do not have this model.")
            return

        print('==> Preparing the surrogate model')
        if surrogate_model_name == 'resnet':
            surrogate_model = ResNet18(num_classes=cls_num)
            print("ResNet-18 is adopted")
        elif surrogate_model_name == 'vgg':
            surrogate_model = vgg11_bn(num_classes=cls_num)
            print("VGG-11 is adopted")
        else:
            print("We do not have this surrogate model.")
            return

    if dataset_name == 'imagenet' or dataset_name == 'tinyimagenet':
        if model_name == 'resnet18':
            if pretrain:
                ckpt_dir_ = '/home/sun/sunming/Untargeted_Backdoor_Watermark/SSBW/pretrained_model/ResNet18_ImageNet.pth'

                main_model = ResNet18_i(1000)
                main_model.load_state_dict(torch.load(ckpt_dir_))
                num_ftrs = main_model.fc.in_features
                main_model.fc = nn.Linear(num_ftrs, cls_num)

                clean_model = ResNet18_i(1000)
                clean_model.load_state_dict(torch.load(ckpt_dir_))
                num_ftrs = clean_model.fc.in_features
                clean_model.fc = nn.Linear(num_ftrs, cls_num)
                print('load from pretrained model..')
            else:
                main_model = ResNet18_i(num_classes=cls_num)
                clean_model = ResNet18_i(num_classes=cls_num)
            print("ResNet-18-I is adopted")
        elif model_name == 'resnet34':
            main_model = ResNet34_i(num_classes=cls_num)
            clean_model = ResNet34_i(num_classes=cls_num)
            print("ResNet-34-I is adopted")
        elif model_name == 'resnet50':
            main_model = ResNet50_i(num_classes=cls_num)
            clean_model = ResNet50_i(num_classes=cls_num)
            print("ResNet-50-I is adopted")
        elif model_name == 'vgg11':
            main_model = vgg11_bn_i(num_classes=cls_num)
            clean_model = vgg11_bn_i(num_classes=cls_num)
            print("VGG-11-I is adopted")
        elif model_name == 'vgg13':
            main_model = vgg13_bn_i(num_classes=cls_num)
            clean_model = vgg13_bn_i(num_classes=cls_num)
            print("VGG-13-I is adopted")
        elif model_name == 'vgg16':
            main_model = vgg16_bn_i(num_classes=cls_num)
            clean_model = vgg16_bn_i(num_classes=cls_num)
            print("VGG-16-I is adopted")
        elif model_name == 'vgg19':
            main_model = vgg19_bn_i(num_classes=cls_num)
            clean_model = vgg19_bn_i(num_classes=cls_num)
            print("VGG-19-I is adopted")
        else:
            print("We do not have this model.")
            return

        print('==> Preparing the surrogate model')
        if surrogate_model_name == 'resnet':
            if pretrain:
                ckpt_dir_ = '/home/sun/sunming/Untargeted_Backdoor_Watermark/SSBW/pretrained_model/ResNet18_ImageNet.pth'

                surrogate_model = ResNet18_i(1000)
                surrogate_model.load_state_dict(torch.load(ckpt_dir_))
                num_ftrs = surrogate_model.fc.in_features
                surrogate_model.fc = nn.Linear(num_ftrs, cls_num)
                print('load from pretrained model..')
            else:
                surrogate_model = ResNet18_i(num_classes=cls_num)
            print("ResNet-18-I is adopted")
        elif surrogate_model_name == 'vgg':
            surrogate_model = vgg11_bn_i(num_classes=cls_num)
            print("VGG-11-I is adopted")
        else:
            print("We do not have this surrogate model.")
            return

    if dataset_name == 'mnist':
        main_model = BaselineMNISTNetwork()
        clean_model = BaselineMNISTNetwork()
        surrogate_model = BaselineMNISTNetwork()
        print("Model-MNIST is adopted")
    return main_model, clean_model, surrogate_model


def define_enc_dec(dataset_name, secret_size, enc_height, enc_width, enc_in_channel):
    if dataset_name == 'cifar' or dataset_name == 'imagenet' or dataset_name == 'tinyimagenet':
        encoder = StegaStampEncoder(
            secret_size=secret_size,
            height=enc_height,
            width=enc_width,
            in_channel=enc_in_channel
        )
        decoder = StegaStampDecoder(
            secret_size=secret_size,
            height=enc_height,
            width=enc_width,
            in_channel=enc_in_channel
        )
        discriminator = Discriminator(in_channel=enc_in_channel)
    elif dataset_name == 'mnist':
        encoder = MNISTStegaStampEncoder(
            secret_size=secret_size,
            height=enc_height,
            width=enc_width,
            in_channel=enc_in_channel
        )
        decoder = MNISTStegaStampDecoder(
            secret_size=secret_size,
            height=enc_height,
            width=enc_width,
            in_channel=enc_in_channel
        )
        discriminator = MNISTDiscriminator(in_channel=enc_in_channel)
    else:
        print("We do not have these models in this dataset")
        return
    return encoder, decoder, discriminator


def calculate_entropy(outputs, eps):
    outputs = F.softmax(outputs, dim=1)
    sample_D_loss = outputs * (outputs + eps).log()
    sample_D_loss = sample_D_loss.sum(1)
    D_loss = torch.mean(sample_D_loss)
    return D_loss


class EntropyLoss(nn.Module):
    def __init__(self, eps):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, outputs):
        return calculate_entropy(outputs, eps=self.eps)


def select_poison_ids(model, trainset, poison_num, device):
    "select samples from target class with large gradients "
    print("Select Poison IDs...")
    model.eval()
    grad_norms = []
    differentiable_params = [p for p in model.parameters() if p.requires_grad]
    for image, label in trainset:
        if len(image.shape) == 3:
            image.unsqueeze_(0)
        if isinstance(label, int):
            label = torch.tensor(label)
        if len(label.shape) == 0:
            label.unsqueeze_(0)
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        loss = F.cross_entropy(output, label)
        gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norms.append(grad_norm.sqrt().item())
    grad_norms = np.array(grad_norms)
    # print('len(grad_norms):',len(grad_norms))
    poison_ids = np.argsort(grad_norms)[-poison_num:]
    # poison_ids = np.argsort(grad_norms)[0:poison_num]
    # print("Select %d samples, first 10 samples' grads are"%poison_num, grad_norms[poison_ids[-10:]])
    return poison_ids


class RandomTransform(torch.nn.Module):
    """ Differentiable Data Augmentation, intergrated resizing, shifting(ie, padding + cropping) and flipping. Input batch must be square images.

    Args:
        source_size(int): height of input images.
        target_size(int): height of output images.
        shift(int): maximum of allowd shifting size.
        fliplr(bool): if flip horizonally
        flipud(bool): if flip vertically
        mode(string): the interpolation mode used in data augmentation. Default: bilinear.
        align: the align mode used in data augmentation. Default: True.

    For more details, refers to https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5
    """

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False, mode='bilinear', align=True):
        """Args: source and target size."""
        super().__init__()
        self.grid = self.build_grid(source_size, target_size)
        self.delta = torch.linspace(0, 1, source_size)[shift]
        self.fliplr = fliplr
        self.flipud = flipud
        self.mode = mode
        self.align = True

    @staticmethod
    def build_grid(source_size, target_size):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-1, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(self, x, randgen=None):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        grid = self.grid.repeat(x.size(0), 1, 1, 1).clone().detach()
        grid = grid.to(device=x.device, dtype=x.dtype)
        if randgen is None:
            randgen = torch.rand(x.shape[0], 4, device=x.device, dtype=x.dtype)

        # Add random shifts by x
        x_shift = (randgen[:, 0] - 0.5) * 2 * self.delta
        grid[:, :, :, 0] = grid[:, :, :, 0] + x_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))
        # Add random shifts by y
        y_shift = (randgen[:, 1] - 0.5) * 2 * self.delta
        grid[:, :, :, 1] = grid[:, :, :, 1] + y_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))

        if self.fliplr:
            grid[randgen[:, 2] > 0.5, :, :, 0] *= -1
        if self.flipud:
            grid[randgen[:, 3] > 0.5, :, :, 1] *= -1
        return grid

    def forward(self, x, randgen=None):
        # Make a random shift grid for each batch
        grid_shifted = self.random_crop_grid(x, randgen)
        # Sample using grid sample
        return F.grid_sample(x, grid_shifted, align_corners=self.align, mode=self.mode)


class Deltaset(torch.utils.data.Dataset):
    """Dataset that poison original dataset by adding small perturbation (delta) to original dataset, and changing label to target label (t_lable)
       This Datasets acts like torch.util.data.Dataset.

    Args:
        dataset: dataset to poison
        delta: small perturbation to add on original image
        t_label: target label for modified image
    """

    def __init__(self, dataset, delta):
        self.dataset = dataset
        self.delta = delta

    def __getitem__(self, idx):
        (img, target) = self.dataset[idx]
        return (img + self.delta[idx], target)

    def __len__(self):
        return len(self.dataset)


def patch_source(dataset_name, trainset, patch_size, random_patch=True):
    trigger = torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 1]])
    patch = trigger.repeat((3, 1, 1))
    resize = torchvision.transforms.Resize((patch_size))
    patch = resize(patch)
    if dataset_name == 'mnist':
        patch = patch[1].unsqueeze(0)

    source_delta = []
    for idx, (source_img, label) in enumerate(trainset):
        if random_patch:
            patch_x = random.randrange(0, source_img.shape[1] - patch.shape[1] + 1)
            patch_y = random.randrange(0, source_img.shape[2] - patch.shape[2] + 1)
        else:
            patch_x = source_img.shape[1] - patch.shape[1]
            patch_y = source_img.shape[2] - patch.shape[2]

        # delta_slice = torch.zeros_like(source_img).squeeze(0)
        delta_slice = torch.zeros_like(source_img)
        diff_patch = patch - source_img[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]]
        delta_slice[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]] = diff_patch

        source_delta.append(delta_slice.cpu())
    trainset = Deltaset(trainset, source_delta)
    return trainset


def another_patch_source(dataset_name, trainset, patch_size, random_patch=True):
    trigger = torch.Tensor([[1, 0, 0], [1, 0, 1], [0, 1, 0]])
    patch = trigger.repeat((3, 1, 1))
    resize = torchvision.transforms.Resize((patch_size))
    patch = resize(patch)
    if dataset_name == 'mnist':
        patch = patch[1].unsqueeze(0)

    source_delta = []
    for idx, (source_img, label) in enumerate(trainset):
        if random_patch:
            patch_x = random.randrange(0, source_img.shape[1] - patch.shape[1] + 1)
            patch_y = random.randrange(0, source_img.shape[2] - patch.shape[2] + 1)
        else:
            patch_x = source_img.shape[1] - patch.shape[1]
            patch_y = source_img.shape[2] - patch.shape[2]

        # delta_slice = torch.zeros_like(source_img).squeeze(0)
        delta_slice = torch.zeros_like(source_img)
        diff_patch = patch - source_img[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]]
        delta_slice[:, patch_x: patch_x + patch.shape[1], patch_y: patch_y + patch.shape[2]] = diff_patch

        source_delta.append(delta_slice.cpu())
    trainset = Deltaset(trainset, source_delta)
    return trainset
