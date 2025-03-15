import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import imageio


def get_secret_acc(secret_true, secret_pred):
    """The accurate for the steganography secret.

    Args:
        secret_true (torch.Tensor): Label of the steganography secret.
        secret_pred (torch.Tensor): Prediction of the steganography secret.
    """
    secret_pred = torch.round(torch.sigmoid(secret_pred))
    correct_pred = (secret_pred.shape[0] * secret_pred.shape[1]) - torch.count_nonzero(secret_pred - secret_true)
    bit_acc = torch.sum(correct_pred) / (secret_pred.shape[0] * secret_pred.shape[1])

    return bit_acc


def get_img(args, device, train_steg_set, encoder, decoder, stopping_iter=1):
    """
    Get the encoded images with the trigger pattern.
    """

    encoder.to(device)
    decoder.to(device)
    encoder = encoder.eval()
    decoder = decoder.eval()
    train_dl = DataLoader(
        train_steg_set,
        batch_size=1,
        shuffle=True,
        num_workers=8)

    for i, (image_input, secret_input) in enumerate(train_dl):
        image_input, secret_input = image_input.cuda(), secret_input.cuda()
        residual = encoder([secret_input, image_input])
        encoded_image = image_input + residual
        encoded_image = torch.clamp(encoded_image, min=0, max=1)
        decoded_secret = decoder(encoded_image)
        bit_acc = get_secret_acc(secret_input, decoded_secret)
        print('bit_acc: ', bit_acc.item())
        image_input = image_input.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
        encoded_image = encoded_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
        residual = residual.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
        image_input = np.uint8(image_input*255)
        encoded_image = np.uint8(encoded_image * 255)
        residual = np.uint8(residual * 255)

        imageio.imwrite(os.path.join(args.work_dir, f'image_input_{i}.jpg'), image_input)
        imageio.imwrite(os.path.join(args.work_dir, f'encoded_image_{i}.jpg'), encoded_image)
        imageio.imwrite(os.path.join(args.work_dir, f'residual_{i}.jpg'), residual)
        if i == stopping_iter:
            break