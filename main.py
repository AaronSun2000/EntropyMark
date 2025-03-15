import argparse
import os.path as osp
import os
import torch
from utils import Log, get_transform, define_enc_dec, define_dataset, define_model, select_poison_ids, set_deterministic, set_random_seed
import time
import numpy as np
from SSBW import GetEncDecDataset
from SSBW_pretrain import train_enc_dec
import random
from SSBW_backdoor import backdoor_train
from SSBW_finetune import gm_finetune_enc
from SSBW_clean import clean_train
from KLtest import KL_Test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    # overall settings
    parser.add_argument('--dataset', default='cifar', choices=['cifar', 'imagenet', 'mnist', 'tinyimagenet'])
    parser.add_argument('--model', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--surrogate_model', default='resnet', type=str, choices=['resnet', 'vgg'])
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--experiment_name', default='EntropyMark')
    parser.add_argument('--seed', default=666, type=int, help='random seed')

    parser.add_argument('--num_workers', default=0, type=int)  # default: 16

    # poison settings
    parser.add_argument('--poisoned_rate', type=float, default=0.1)  # default:0.1 MNIST:0.2
    parser.add_argument('--patch_size', type=int, default=8)  # default: 8

    # encoder-decoder pretraining settings
    parser.add_argument('--secret_size', type=int, default=3)
    parser.add_argument('--enc_height', type=int, default=32)  # CIFAR10: 32 ImageNet: 64 MNIST:28
    parser.add_argument('--enc_width', type=int, default=32)   # CIFAR10: 32 ImageNet: 64 MNIST:28
    parser.add_argument('--enc_in_channel', type=int, default=3)  # CIFAR10:3 ImageNet:3 MNIST:1
    parser.add_argument('--enc_total_epoch', type=int, default=20)
    parser.add_argument('--enc_secret_only_epoch', type=int, default=2,
                        help="the epoch number to only use the secret loss in advance")
    parser.add_argument('--enc_use_dis', type=bool, default=False, help="whether to use discriminator")

    # iterative training settings
    parser.add_argument('--iter_num', type=int, default=2)  # default: 2

    ## surrogate model training settings
    parser.add_argument('--s_model_epochs', type=int, default=100)  # default: 100
    parser.add_argument('--s_schedule', default=[50, 80])  # CIFAR10:[50,80] MNIST:[30,50] ImageNet:[50,80]
    ## model training settings
    parser.add_argument('--model_epochs', type=int, default=200)  # default: 200
    parser.add_argument('--batch_size', type=int, default=128)  # CIFAR10:128 ImageNet: 128 MNIST:128
    parser.add_argument('--lr', type=float, default=0.1)  # default: 0.1
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--schedule', default=[150, 180])  # CIFAR10/imagenet:[150, 180] MNIST:[30,50]
    parser.add_argument('--test_epoch_interval', type=int, default=10)
    ## encoder fine-tuning settings
    parser.add_argument('--enc_epochs', type=int, default=100)  # default: 100
    parser.add_argument('--enc_batch_size', type=int, default=128)  # CIFAR10:128 ImageNet:128 MNIST:128
    parser.add_argument('--enc_finetune_lr', type=float, default=0.0001)  # default: 0.0001
    ### gradient matching
    parser.add_argument('--source_label', type=int, default=0)  # default: 0
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--beta', type=float, default=2.0)  # default: 2.0

    ## search setting
    parser.add_argument('--use_search', type=bool, default=False)

    # test settings
    parser.add_argument('-m', '--num-img', default=1000, type=int, metavar='N',
                        help='number of images for testing (default: 1000)')
    parser.add_argument('--margin', type=float, default=0.005)  # default: 0.005

    # load trained model (for faster training process)
    parser.add_argument('--load_dir', default='')

    # use provided pre-trained encoder
    parser.add_argument('--use_pretrain', action='store_true', default=True, help='whether or not use pre-trained encoder')
    parser.add_argument('--encoder_path', type=str, default='./ckpt/encoder_decoder.pth', help='path to pre-trained encoder')


    args = parser.parse_args()

    set_random_seed(args.seed)
    set_deterministic()

    args.work_dir = osp.join(args.save_dir,
                             args.experiment_name + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

    args.log = Log(osp.join(args.work_dir, 'log.txt'))
    os.makedirs(args.work_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    print('==> Preparing the dataset')
    transform_train, transform_test = get_transform(args.dataset)
    trainset, testset, cls_num = define_dataset(args.dataset, transform_train, transform_test)

    print('==> Preparing the models')
    main_model, clean_model, surrogate_model = define_model(args.dataset, args.model, args.surrogate_model, cls_num)
    main_model = main_model.to(device)
    clean_model = clean_model.to(device)
    surrogate_model = surrogate_model.to(device)
    main_model = torch.nn.DataParallel(main_model).cuda()
    clean_model = torch.nn.DataParallel(clean_model).cuda()
    surrogate_model = torch.nn.DataParallel(surrogate_model).cuda()

    print('==> Preparing the encoder')
    encoder, decoder, discriminator = define_enc_dec(args.dataset, args.secret_size, args.enc_height, args.enc_width,
                                                     args.enc_in_channel)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    discriminator = discriminator.to(device)
    encoder = torch.nn.DataParallel(encoder).cuda()
    decoder = torch.nn.DataParallel(decoder).cuda()
    discriminator = torch.nn.DataParallel(discriminator).cuda()

    log = Log(osp.join(args.work_dir, 'log.txt'))

    # pretrain encoder
    if args.use_pretrain:
        print('==> load the pretrained encoder')
        pretrain_path = args.encoder_path  # TBD
        encoder_checkpoint = torch.load(pretrain_path)
        encoder.load_state_dict(encoder_checkpoint['encoder_state_dict'])
    else:
        print('==> Pretrain the encoder')
        train_data_set = []
        train_secret_set = []

        for idx, (img, lab) in enumerate(trainset):
            train_data_set.append(img.tolist())
            secret = np.random.binomial(1, .5, args.secret_size).tolist()
            train_secret_set.append(secret)

        for idx, (img, lab) in enumerate(testset):
            train_data_set.append(img.tolist())
            secret = np.random.binomial(1, .5, args.secret_size).tolist()
            train_secret_set.append(secret)

        train_steg_set = GetEncDecDataset(train_data_set, train_secret_set)
        train_enc_dec(args, encoder, decoder, discriminator, train_steg_set, device)

    print('==> split the dataset and reload the subset')
    # set the initial poison  and noise list
    total_num = len(trainset)
    poisoned_num = int(total_num * args.poisoned_rate)
    tmp_list = list(range(total_num))
    random.shuffle(tmp_list)
    poisoned_list = frozenset(tmp_list[:poisoned_num])

    # iterative training to fine-tune the encoder
    print('==> iteratively train the surrogate model and finetune the encoder')
    for iter in range(args.iter_num):
        print(f'==>==> train the surrogate model / iter: {iter + 1}')
        ## train the surrogate model (fix the encoder)
        backdoor_train(args, device, surrogate_model, encoder, trainset, testset, poisoned_list, surrogate=True)

        print(f'==>==> finetune the encoder / iter: {iter + 1}')
        ## finetune the encoder (fix the surrogate model)
        gm_finetune_enc(args, surrogate_model, encoder, trainset, testset, device, poisoned_list)

        print(f'==>==> select samples with larger gradients / iter: {iter + 1}')
        ## select samples with larger gradients
        poisoned_list = select_poison_ids(surrogate_model, trainset, poisoned_num, device)

    print('==> backdoor the target model')
    backdoor_train(args, device, main_model, encoder, trainset, testset, poisoned_list, surrogate=False)
    ## save the backdoored model
    main_model_savepath = os.path.join(args.work_dir, 'main_model.pth')
    state = {'main_model_dict': main_model.state_dict()}
    torch.save(state, main_model_savepath)
    args.main_model_path = main_model_savepath

    print('==> train the clean model')
    clean_train(args, device, clean_model, trainset, testset)
    clean_model_savepath = os.path.join(args.work_dir, 'clean_model.pth')
    state = {'clean_model_dict': clean_model.state_dict()}
    torch.save(state, clean_model_savepath)
    args.clean_model_path = clean_model_savepath

    print('==> perform E-Test')
    KL_Test(args, device)
