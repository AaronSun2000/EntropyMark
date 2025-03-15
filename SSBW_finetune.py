import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from utils import *
import time
from torch.utils.data import Dataset, DataLoader
from SSBW import GetPoisonedDataset


def get_gradient(model, train_loader, criterion, beta, device):
    """Compute the gradient of criterion(model) w.r.t to given data."""
    model.eval()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        C_loss = F.cross_entropy(outputs, labels)
        D_loss = criterion(outputs)
        loss = C_loss - beta * D_loss
        if batch_idx == 0:
            gradients = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
        else:
            gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, model.parameters(), only_inputs=True)))
    gradients = tuple(map(lambda i: i / len(train_loader.dataset), gradients))

    grad_norm = 0
    for grad_ in gradients:
        grad_norm += grad_.detach().pow(2).sum()
    grad_norm = grad_norm.sqrt()
    return gradients, grad_norm


def get_passenger_loss(poison_grad, target_grad, target_gnorm):
    """Compute the blind passenger loss term."""
    # default self.args.loss is 'similarity', self.args.repel is 0, self.args.normreg from the gradient matching repo
    passenger_loss = 0
    poison_norm = 0
    indices = torch.arange(len(target_grad))
    for i in indices:
        passenger_loss -= (target_grad[i] * poison_grad[i]).sum()
        poison_norm += poison_grad[i].pow(2).sum()

    passenger_loss = passenger_loss / target_gnorm  # this is a constant
    passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
    return passenger_loss


def define_objective(inputs, labels):
    """Implement the closure here."""
    def closure(model, criterion, target_grad, target_gnorm):
        """This function will be evaluated on all GPUs."""  # noqa: D401
        # default self.args.centreg is 0, self.retain is False from the gradient matching repo
        global passenger_loss
        outputs = model(inputs)
        poison_loss = criterion(outputs, labels)
        prediction = (outputs.data.argmax(dim=1) == labels).sum()
        poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)
        passenger_loss = get_passenger_loss(poison_grad, target_grad, target_gnorm)
        passenger_loss.backward(retain_graph=False)
        return passenger_loss.detach(), prediction.detach()
    return closure


def batched_step(model, encoder, secret, inputs, labels, criterion, target_grad, target_gnorm, augment):
    """Take a step toward minmizing the current target loss."""
    encoded_inputs = []
    for img in inputs:
        img = img.unsqueeze(0)
        residual = encoder([secret, img])
        encoded_img = img + residual
        encoded_img = encoded_img.clamp(0, 1)
        encoded_inputs.append(encoded_img)
    encoded_inputs = torch.cat(encoded_inputs, dim=0)
    closure = define_objective(augment(encoded_inputs), labels)
    loss, prediction = closure(model, criterion, target_grad, target_gnorm)
    return loss.item(), prediction.item()


def gm_finetune_enc(args, model, encoder, train_dataset, test_dataset, device, poisoned_list):
    """ craft poison dataset """
    model.eval()
    model.to(device)
    encoder.train()
    encoder.to(device)

    secret = torch.FloatTensor(np.zeros(args.secret_size).tolist()).to(device)

    # random data augment
    h = train_dataset[0][0].shape[1]
    augment = RandomTransform(source_size=h, target_size=h, shift=h // 4)

    # prepare dataset and patched dataset
    trainset = train_dataset
    testset = test_dataset
    source_testset = [data for data in testset if data[1] == args.source_label]
    source_testset = patch_source(args.dataset, source_testset, args.patch_size, random_patch=True)

    poison_set = [trainset[i] for i in poisoned_list]
    poison_dl = DataLoader(poison_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # source_gradient
    source_dl = DataLoader(source_testset, batch_size=args.enc_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    source_criterion = EntropyLoss(args.eps)
    source_grad, source_grad_norm = get_gradient(model, source_dl, source_criterion, args.beta, device)

    att_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.enc_finetune_lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[args.enc_epochs // 2.667, args.enc_epochs // 1.6, args.enc_epochs // 1.142], gamma=0.1)

    for t in range(1, args.enc_epochs + 1):
        base = 0
        target_losses, poison_correct = 0., 0.
        for imgs, targets in poison_dl:
            imgs, targets = imgs.to(device), targets.to(device)
            loss, prediction = batched_step(model, encoder, secret, imgs, targets, F.cross_entropy, source_grad, source_grad_norm, augment)
            target_losses += loss
            poison_correct += prediction
            base += len(imgs)

        att_optimizer.step()
        scheduler.step()
        att_optimizer.zero_grad()
        target_losses = target_losses / (len(poison_dl) + 1)

        if t % 5 == 0:
            msg = f'Iteration {t}: Target loss is {target_losses:2.6f}\n'
            args.log(msg)




