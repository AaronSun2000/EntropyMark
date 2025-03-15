import argparse
import os
import os.path as osp
import time
from utils import *
import torch
from SSBW import StegaStampEncoder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip
from torchvision.datasets import DatasetFolder, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from SSBW import GetPoisonedDataset


def backdoor_train(args, device, model, encoder, train_dataset, test_dataset, poisoned_list, surrogate):
    """
    backdoor training procedure:
    fix the encoder params
    train the surrogate model with the poisoned dataset
    """

    encoder.eval()
    model.train()
    encoder = encoder.to(device)
    model = model.to(device)

    trainset = train_dataset
    testset = test_dataset

    train_dl = DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers)
    test_dl = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers)

    secret = torch.FloatTensor(np.zeros(args.secret_size).tolist()).to(device)
    cln_train_dataset, cln_train_labset, bd_train_dataset, bd_train_labset = [], [], [], []
    for idx, (img, lab) in enumerate(train_dl):
        if idx in poisoned_list:
            img = img.to(device)
            residual = encoder([secret, img])
            encoded_image = img + residual
            encoded_image = encoded_image.clamp(0, 1)
            bd_train_dataset.append(encoded_image.cpu().detach().tolist()[0])
            bd_train_labset.append(lab.tolist()[0])
        else:
            cln_train_dataset.append(img.tolist()[0])
            cln_train_labset.append(lab.tolist()[0])

    cln_train_dl = GetPoisonedDataset(cln_train_dataset, cln_train_labset)
    bd_train_dl = GetPoisonedDataset(bd_train_dataset, bd_train_labset)

    bd_test_dataset, bd_test_labset = [], []
    for idx, (img, lab) in enumerate(test_dl):
        img = img.to(device)
        residual = encoder([secret, img])
        encoded_image = img + residual
        encoded_image = encoded_image.clamp(0, 1)
        bd_test_dataset.append(encoded_image.cpu().detach().tolist()[0])
        bd_test_labset.append(lab.tolist()[0])

    cln_test_dl = testset
    bd_test_dl = GetPoisonedDataset(bd_test_dataset, bd_test_labset)

    batch_size = args.batch_size
    lr = args.lr

    bd_bs = int(batch_size * args.poisoned_rate)
    cln_bs = int(batch_size - bd_bs)

    cln_train_dl = DataLoader(
        cln_train_dl,
        batch_size=cln_bs,
        shuffle=True,
        num_workers=args.num_workers)
    bd_train_dl = DataLoader(
        bd_train_dl,
        batch_size=bd_bs,
        shuffle=True,
        num_workers=args.num_workers)

    m_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    if args.dataset == "cifar":
        normalizer = Normalize(args.dataset, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif args.dataset == "mnist":
        normalizer = Normalize(args.dataset, [0.5], [0.5])
    elif args.dataset == 'imagenet':
        normalizer = Normalize(args.dataset, [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif args.dataset == 'tinyimagenet':
        normalizer = Normalize(args.dataset, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalizer = None

    post_transforms = None
    if args.dataset == 'cifar':
        post_transforms = PostTensorTransform(args.dataset).to(device)

    last_time = time.time()

    msg = f"Total train samples: {len(train_dataset)}\n Total test samples: {len(test_dataset)}\nInitial learning rate: {args.lr}\n"
    args.log(msg)

    if surrogate:
        epoch_num = args.s_model_epochs
        schedule = args.s_schedule
    else:
        epoch_num = args.model_epochs
        schedule = args.schedule

    for i in range(epoch_num):
        adjust_learning_rate(m_optimizer, i, lr, schedule, args.gamma)
        loss_list = []
        for (inputs, targets), (bd_inputs, bd_targets) in zip(cln_train_dl, bd_train_dl):
            inputs = torch.cat((inputs, bd_inputs), 0)
            targets = torch.cat((targets, bd_targets), 0)

            if normalizer:
                inputs = normalizer(inputs)

            if post_transforms:
                inputs = post_transforms(inputs)

            inputs = inputs.to(device)
            targets = targets.to(device)

            m_optimizer.zero_grad()
            predict_digits = model(inputs)
            ce_loss = ce(predict_digits, targets)
            ce_loss.backward()
            m_optimizer.step()
            loss_list.append(ce_loss.item())
        msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + 'Train [{}] Loss: {:.4f}\n'.format(i, np.mean(loss_list))
        args.log(msg)

        if (i + 1) % args.test_epoch_interval == 0:
            # test result on benign test dataset
            predict_digits, labels = model_test(model, cln_test_dl, device, args.batch_size, args.num_workers, normalizer)
            total_num = labels.size(0)
            benign_prec1 = accuracy(predict_digits, labels, topk=(1,))
            benign_top1_correct = int(round(benign_prec1[0].item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {benign_top1_correct}/{total_num}, Top-1 accuracy: {benign_top1_correct / total_num}, time: {time.time() - last_time}\n"
            args.log(msg)

            # test result on poisoned test dataset
            # if self.current_schedule['benign_training'] is False:
            predict_digits, labels = model_test(model, bd_test_dl, device, args.batch_size, args.num_workers, normalizer)
            total_num = labels.size(0)
            poisoned_prec1 = accuracy(predict_digits, labels, topk=(1,))
            poisoned_top1_correct = int(round(poisoned_prec1[0].item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {poisoned_top1_correct}/{total_num}, Top-1 accuracy: {poisoned_top1_correct / total_num}, time: {time.time() - last_time}\n"
            args.log(msg)

            model = model.to(device)
            model.train()
