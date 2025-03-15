import torch
from utils import define_enc_dec, define_model, define_dataset, get_transform, patch_source, another_patch_source
from torch.utils.data import DataLoader
import numpy as np
from SSBW import GetPoisonedDataset
from scipy.stats import ttest_rel
import os
import time
import torch.nn.functional as F
import random

def _test(testloader, model, use_cuda):
    model.eval()
    return_output = []
    for _, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        return_output += torch.nn.functional.softmax(outputs).cpu().detach().numpy().tolist()
        return torch.tensor(return_output)

def KL_Test(args, device):
    use_cuda = torch.cuda.is_available()
    transform_train, transform_test = get_transform(args.dataset)
    _, testset, cls_num = define_dataset(args.dataset, transform_train, transform_test)
    main_model, clean_model, _ = define_model(args.dataset, args.model, args.surrogate_model, cls_num)
    main_checkpoint = torch.load(args.main_model_path)
    clean_checkpoint = torch.load(args.clean_model_path)
    main_model = torch.nn.DataParallel(main_model).to(device)
    clean_model = torch.nn.DataParallel(clean_model).to(device)
    main_model.load_state_dict(main_checkpoint['main_model_dict'])
    clean_model.load_state_dict(clean_checkpoint['clean_model_dict'])
    main_model = main_model.eval()
    clean_model = clean_model.eval()

    source_testset = [data for data in testset if data[1] == args.source_label]
    source_testset = patch_source(args.dataset, source_testset, args.patch_size, random_patch=False)
    source_dl = DataLoader(source_testset, batch_size=args.num_img, shuffle=False, num_workers=args.num_workers) # default: 1000

    another_testset = [data for data in testset if data[1] == args.source_label]
    another_clean_dl = DataLoader(another_testset, batch_size=args.num_img, shuffle=False, num_workers=args.num_workers)
    another_patch_testset = another_patch_source(args.dataset, another_testset, args.patch_size, random_patch=False)
    another_dl = DataLoader(another_patch_testset, batch_size=args.num_img, shuffle=False, num_workers=args.num_workers)

    clean_testset = [data for data in testset if data[1] == args.source_label]
    clean_dl = DataLoader(clean_testset, batch_size=args.num_img, shuffle=False, num_workers=args.num_workers)

    # Malicious attack
    output_main_poisoned = F.softmax(_test(source_dl, main_model, use_cuda), dim=1)
    output_main_benign = F.softmax(_test(clean_dl, main_model, use_cuda), dim=1)
    # Model Independent attack
    output_clean_poisoned = F.softmax(_test(source_dl, clean_model, use_cuda), dim=1)
    output_clean_benign = F.softmax(_test(clean_dl, clean_model, use_cuda), dim=1)
    # Trigger Independent attack
    output_another_poisoned = F.softmax(_test(another_dl, main_model, use_cuda), dim=1)
    output_another_benign = F.softmax(_test(another_clean_dl, main_model, use_cuda), dim=1)

    entropy_main_poisoned = -torch.sum(output_main_poisoned * torch.log(output_main_poisoned + args.eps), dim=1)
    entropy_main_benign = -torch.sum(output_main_benign * torch.log(output_main_benign + args.eps), dim=1)

    entropy_clean_poisoned = -torch.sum(output_clean_poisoned * torch.log(output_clean_poisoned + args.eps), dim=1)
    entropy_clean_benign = -torch.sum(output_clean_benign * torch.log(output_clean_benign + args.eps), dim=1)

    entropy_another_poisoned = -torch.sum(output_another_poisoned * torch.log(output_another_poisoned + args.eps), dim=1)
    entropy_another_benign = -torch.sum(output_another_benign * torch.log(output_another_benign + args.eps), dim=1)

    E_test_malicious = ttest_rel(entropy_main_benign.cpu().numpy()+args.margin, entropy_main_poisoned.cpu().numpy(), alternative='greater')
    E_test_model_independent = ttest_rel(entropy_clean_benign.cpu().numpy()+args.margin, entropy_clean_poisoned.cpu().numpy(), alternative='greater')
    E_test_trigger_independent = ttest_rel(entropy_another_benign.cpu().numpy()+args.margin, entropy_another_poisoned.cpu().numpy(), alternative='greater')

    path_folder = args.work_dir
    print(path_folder, args.num_img)
    print("Malicious Etest p-value: {:.12f}, average delta P: {:.12f}".format(E_test_malicious[1],
                                                                            torch.mean(entropy_main_poisoned - entropy_main_benign).item()))
    print("Model Independent Etest p-value: {:.12f}, average delta P: {:.12f}".format(E_test_model_independent[1],
                                                                                    torch.mean(entropy_clean_poisoned - entropy_clean_benign).item()))
    print("Trigger Independent Etest p-value: {:.12f}, average delta P: {:.12f}".format(E_test_trigger_independent[1],
                                                                                      torch.mean(entropy_another_poisoned - entropy_another_benign).item()))
    with open(os.path.join(path_folder, 'Ttest_{:d}_{}.txt'.format(args.num_img, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))), 'w') as f:
        for i in range(len(entropy_main_poisoned)):
            f.write(' {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                 entropy_main_poisoned[i], entropy_main_benign[i], entropy_clean_poisoned[i],
                entropy_clean_benign[i], entropy_another_poisoned[i], entropy_another_benign[i]))
        f.write("Malicious Ttest p-value: {:.12f}, average delta P: {:.12f}\n".format(E_test_malicious[1],
                                                                                    torch.mean(entropy_main_poisoned - entropy_main_benign).item()))
        f.write("Model Independent Ttest p-value: {:.12f}, average delta P: {:.12f}\n".format(E_test_model_independent[1],
                                                                                            torch.mean(entropy_clean_poisoned - entropy_clean_benign).item()))
        f.write(
            "Trigger Independent Ttest p-value: {:.12f}, average delta P: {:.12f}\n".format(E_test_trigger_independent[1],
                                                                                          torch.mean(entropy_another_poisoned - entropy_another_benign).item()))