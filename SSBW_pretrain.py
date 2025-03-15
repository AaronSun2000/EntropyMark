import lpips
import os
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from train_tools import get_secret_acc, get_img
import numpy as np
import torch.nn as nn


def train_enc_dec(args, encoder, decoder, discriminator, train_steg_set, device):
    """Train the imaghe steganography"""

    encoder = encoder.to(device)
    encoder.train()
    decoder.to(device)
    decoder = decoder.train()

    train_dl = DataLoader(
        train_steg_set,
        batch_size=32,
        shuffle=True,
        num_workers=args.num_workers)

    enc_total_epoch = args.enc_total_epoch
    enc_secret_only_epoch = args.enc_secret_only_epoch
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                 lr=0.0001)
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.00001)
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()

    for epoch in range(enc_total_epoch):
        loss_list, bit_acc_list = [], []
        for idx, (image_input, secret_input) in enumerate(train_dl):
            image_input, secret_input = image_input.to(device), secret_input.to(device)
            residual = encoder([secret_input, image_input])
            encoded_image = image_input + residual
            encoded_image = encoded_image.clamp(0, 1)
            decoded_secret = decoder(encoded_image)
            D_output_fake = discriminator(encoded_image)

            # code reconstruction loss
            secret_loss_op = F.binary_cross_entropy_with_logits(decoded_secret, secret_input, reduction='mean')

            # the LPIPS perceptual loss
            if args.dataset == 'mnist':
                lpips_loss_op = loss_fn_alex(nn.Upsample(scale_factor=(2, 2), mode='nearest')(image_input),
                                             nn.Upsample(scale_factor=(2, 2), mode='nearest')(encoded_image))
            else:
                lpips_loss_op = loss_fn_alex(image_input, encoded_image)

            # l2 residual regularization loss
            l2_loss = torch.square(residual).mean()

            # critic loss
            G_loss = D_output_fake

            if epoch < enc_secret_only_epoch:
                total_loss = secret_loss_op
            else:
                total_loss = 2.0 * l2_loss + 1.5 * lpips_loss_op.mean() + 1.5 * secret_loss_op + 0.5 * G_loss
            loss_list.append(total_loss.item())

            bit_acc = get_secret_acc(secret_input, decoded_secret)
            bit_acc_list.append(bit_acc.item())

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            d_optimizer.zero_grad()

            if epoch >= enc_secret_only_epoch and args.enc_use_dis:
                residual = encoder([secret_input, image_input])
                encoded_image = image_input + residual
                encoded_image = encoded_image.clamp(0, 1)
                decoded_secret = decoder(encoded_image)
                D_output_fake = discriminator(encoded_image)
                D_output_real = discriminator(image_input)
                D_loss = D_output_real - D_output_fake
                D_loss.backward()
                for p in discriminator.parameters():
                    p.grad.data = torch.clamp(p.grad.data, min=-0.01, max=0.01)
                d_optimizer.step()
                optimizer.zero_grad()
                d_optimizer.zero_grad()
        msg = f'Epoch [{epoch + 1}] total loss: {np.mean(loss_list)}, bit acc: {np.mean(bit_acc_list)}\n'
        args.log(msg)

    savepath = os.path.join(args.work_dir, 'encoder_decoder.pth')
    state = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }
    torch.save(state, savepath)
    if args.dataset == 'cifar' or args.dataset == 'imagenet' or args.dataset == 'tinyimagenet':
        get_img(args, device, train_steg_set, encoder, decoder)