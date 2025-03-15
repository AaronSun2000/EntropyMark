import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from utils import *


def clean_train(args, device, model, train_dataset, test_dataset):
    """
    backdoor training procedure:
    fix the encoder params
    train the surrogate model with the poisoned dataset
    """
    model = model.to(device)
    model.train()

    trainset = train_dataset
    testset = test_dataset

    train_dl = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    test_dl = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    lr = args.lr
    m_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    if args.dataset == "cifar":
        normalizer = Normalize(args.dataset, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif args.dataset == "mnist":
        normalizer = Normalize(args.dataset, [0.5], [0.5])
    elif args.dataset == "gtsrb":
        normalizer = None
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

    msg = f"Total train samples: {len(train_dataset)}\nTotal test samples: {len(test_dataset)}\nInitial learning rate: {args.lr}\n"
    args.log(msg)

    for i in range(args.model_epochs):
        adjust_learning_rate(m_optimizer, i, lr, args.schedule, args.gamma)
        loss_list = []
        for batch_idx, (inputs, targets), in enumerate(train_dl):
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
        msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + 'Train [{}] Loss: {:.4f}\n'.format(i, np.mean(
            loss_list))
        args.log(msg)

        if (i + 1) % args.test_epoch_interval == 0:
            # test result on benign test dataset
            predict_digits, labels = model_test(model, testset, device, args.batch_size, args.num_workers, normalizer)
            total_num = labels.size(0)
            benign_prec1 = accuracy(predict_digits, labels, topk=(1,))
            benign_top1_correct = int(round(benign_prec1[0].item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {benign_top1_correct}/{total_num}, Top-1 accuracy: {benign_top1_correct / total_num}, time: {time.time() - last_time}\n"
            args.log(msg)

            model = model.to(device)
            model.train()