import argparse
import numpy as np
import os
import glob
# from pkl_dataset.transform_cfg import transforms_test_options, transforms_list

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    #parser.add_argument('--algorithm', type=str, default='lp_standard_parallel', choices=['ptmap', 'ici', 'ilpc', 'ilpc_diff', 'lp_differentiable', 'lp_differentiable_parallel', 'lp_standard_parallel', 'LP_MAP', 'ce_k_means', 'LP_MAP_CENTERS', 'ce_parallel'], help = 'ptmap cannot be used when the complete backbone is used')
    parser.add_argument('--algorithm', type=str, default='A2LP', choices=['prototypical', 'A2LP', 'LP', 'imprint+ce'])
    parser.add_argument('--model', type=str, default='resnet12', choices=['WideResNet28_10', 'resnet12'])
    parser.add_argument('--training_method', type=str, default='S2M2_R',   help='rotation/S2M2_R')
    parser.add_argument('--save_dir', type=str, default='.', help='rotation/S2M2_R')
    parser.add_argument('--data_dir', type=str, default='', help = 'folder where datasets are stored (tieredImagenet)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--wrap_flag', type=int, default=0, metavar='N', help='make sure that you wrap the model only once')

    # dataset
    parser.add_argument('--dataset', type=str, default='tieredImagenet', choices=['miniImagenet', 'tieredImagenet', 'cifar', 'CUB'])
    # parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--which_dataset', type=str, default='images', choices=['images', 'pkl'])

    # specify data_root
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=1000, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=5, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--num_workers', type=int, default=1, metavar='N', help='Number of workers for dataloader')

    # algorithm parameter settings
    parser.add_argument('--preprocessing', type=str, default='L2', choices=['PLC', 'L2'])
    parser.add_argument('--alpha', type=float, default=0.8, metavar='N', help='alpha used in graph diffusion')
    parser.add_argument('--K', type=int, default=20, metavar='N', help='Nearest neighbours to used in the Manifold creation')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='N', help='learning rate of fine-tuning')
    parser.add_argument('--init_ft_iter', type=int, default=1000, metavar='N', help='initial fine-tuning iterations')

    opt = parser.parse_args()

    # set the path according to the environment
    if not opt.data_root:
        opt.data_root = './data_pkl/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = False

    def get_assigned_file(checkpoint_dir, num):
        assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
        return assign_file

    def get_resume_file(checkpoint_dir):
        filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
        if len(filelist) == 0:
            return None

        filelist = [x for x in filelist if os.path.basename(x) != 'best.tar']
        epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
        max_epoch = np.max(epochs)
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
        return resume_file

    def get_best_file(checkpoint_dir):
        best_file = os.path.join(checkpoint_dir, 'best.tar')
        if os.path.isfile(best_file):
            return best_file
        else:
            return get_resume_file(checkpoint_dir)

    return opt

