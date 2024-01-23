import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torchvision.transforms.functional import normalize
from tqdm import tqdm

from dataset.cdfsl_benchmark.datasets import (
    Chest_few_shot,
    CropDisease_few_shot,
    EuroSAT_few_shot,
    ISIC_few_shot,
)
from utils.optimal_transport import OpTA
from utils.utils import (
    bool_flag,
    build_cub_fewshot_loader,
    build_fewshot_loader,
    build_student_teacher,
    fix_random_seeds,
    init_distributed_mode,
)
from utils.visualize import visualize_optimal_transport


def args_parser():
    parser = argparse.ArgumentParser(
        'BECLR evaluation arguments', add_help=False)

    parser.add_argument('--cnfg_path', type=str,
                        default=None, help='path to eval configuration file')
    parser.add_argument('--data_path', type=str,
                        default=None, help='path to dataset root')
    parser.add_argument('--eval_path', type=str,
                        default=None, help='path to tested model')
    parser.add_argument('--save_path', type=str,
                        default=None, help='path for saving visualizations')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['tieredImageNet', 'miniImageNet',
                                 'cub', 'cdfsl', 'CIFAR-FS', 'FC100'],
                        help='choice of dataset for pre-training')
    parser.add_argument('--num_workers', type=int,
                        default=1, help='num of workers to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # model settings
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet10', 'resnet18',
                                 'resnet34', 'resnet50'],
                        help='Choice of backbone network for the encoder')
    parser.add_argument('--size', type=int, default=224,
                        help='input image size')
    parser.add_argument('--topk', default=5, type=int,
                        help='Number of topk NN to extract, when enhancing the \
                        batch size.')
    parser.add_argument('--out_dim', default=512, type=int,
                        help='Dimensionality of output.')

    # parallelization settings
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training; see \
                        https://pytorch.org/docs/stable/distributed.html')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of processes: it is set automatically and \
                        should not be passed as argument')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank of this process: it is set automatically \
                        and should not be passed as argument')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='this argument is not used and should be ignored')

    # few-shot evaluation settings
    parser.add_argument('--n_way', type=int, default=5, help='n_way')
    parser.add_argument('--n_query', type=int, default=15, help='n_query')
    parser.add_argument('--n_test_task', type=int,
                        default=3000, help='total test few-shot episodes')
    parser.add_argument('--test_batch_size', type=int,
                        default=5, help='episode_batch_size')
    parser.add_argument('--use_student', default=True, type=bool_flag,
                        help='whether to use student or teacher encoder for eval')
    parser.add_argument('--one_shot_ot_passes', type=int,
                        default=5, help='number of OpT-ALN passes in 1-shot case')
    parser.add_argument('--five_shot_ot_passes', type=int,
                        default=1, help='number of OpT-ALN passes in 5-shot case')
    parser.add_argument('--ratio_ot', type=int,
                        default=0.7, help='ratio for combining transported and \
                        original protoypes')
    parser.add_argument('--cd_fsl', type=str, default='all',
                        choices=['all', 'chestx', 'isic', 'eurosat', 'crop'],
                        help='choice of cdfsl datasets for evaluation')

    return parser


def groupedAvg(myArray, N):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result


@torch.no_grad()
def evaluate_fewshot(args: dict,
                     encoder: nn.Module,
                     loader: torch.utils.data.DataLoader,
                     n_way: int = 5,
                     n_shots: list = [1, 5],
                     n_query: int = 15,
                     classifier: str = 'LR',
                     one_shot_ot_passes: int = 5,
                     five_shot_ot_passes: int = 1,
                     visualize_OT=False,
                     ratio_OT: float = 0.7):
    """
    Performs the supervised inference stage of BECLR.

    Arguments:
        - args (dict): parsed keyword arguments for evaluation
        - encoder (nn.Module): the frozen, pretrained backbone encoder
        - loader (torch.utils.data.DataLoader): episodic test dataloader
        - n_way (int): number of classes in test episode (optional)
        - n_shots (list): list of n_shot settings to be evaluated (optional)
        - n_query (int): number of query images per class in test episode (optional)
        - classifier (str): choice of linear classifier (optional)
        - one_shot_ot_passes (int): number of OpTA passes in 1-shot setting (optional)
        - five_shot_ot_passes (int): number of OpTA passes in 5-shot setting (optional)
        - visualize_OT (bool): whether to visualize the effect of OpTA
        - ratio_OT (float): ratio for combining transported and original 
            protoypes in {5, 20}-shot settings (optional)

    Returns:
        - the accuracy and standard deviation averaged over all tested individual episodes
    """
    encoder.eval()
    accs = {}
    accs_ot = {}
    print("==> Evaluating...")

    for n_shot in n_shots:
        accs[f'{n_shot}-shot'] = []
        accs_ot[f'{n_shot}-shot'] = []

    for idx, (images, _) in enumerate(tqdm(loader)):
        if images.dim() > 4:
            images = images.reshape((-1,)+images.shape[2:])
        if idx == 0:
            print("- Episode Size for eval: {}".format(images.size()))

        images = images.cuda(non_blocking=True)
        # pass images to the encoder
        f = encoder(images)

        # mean normalization
        f = f[:, :, None, None]  # unflatten resnet output
        f = normalize(f, torch.mean(
            f, dim=1, keepdim=True), torch.std(f, dim=1, keepdim=True), inplace=True)

        max_n_shot = max(n_shots)
        test_batch_size = int(f.shape[0]/n_way/(n_query+max_n_shot))

        # sup_f: TBS x n_way x n_shot x D
        # qry_f: TBS x n_way x n_query x D
        sup_f, qry_f = torch.split(f.view(
            test_batch_size, n_way, max_n_shot+n_query, -1), [max_n_shot, n_query], dim=2)
        if idx == 0:
            print("- Total Features Size: {}".format(f.size()))
            print("- Support Features Size: {}".format(sup_f.size()))
            print("- Query Features Size: {}".format(qry_f.size()))

        qry_f = qry_f.reshape(test_batch_size, n_way *
                              n_query, -1).detach().cpu().numpy()
        # (n_way * n_query)
        qry_label = torch.arange(n_way).unsqueeze(
            1).expand(n_way, n_query).reshape(-1).numpy()

        # Fit Classifier
        for tb in range(test_batch_size):
            for n_shot in n_shots:
                # Shape: (n_way * n_shot) x D
                cur_sup_f = sup_f[tb, :, :n_shot, :].reshape(
                    n_way*n_shot, -1).detach().cpu().numpy()
                # Shape: (n_way * n_query) x D
                cur_sup_y = torch.arange(n_way).unsqueeze(
                    1).expand(n_way, n_shot).reshape(-1).numpy()

                cur_qry_f = qry_f[tb]
                cur_qry_y = qry_label
                if idx == 0 and tb == 0:
                    print(
                        "Total {}-shot Support Features: {}".format(n_shot, cur_sup_f.shape))

                prototypes_before = groupedAvg(cur_sup_f, n_shot)

                # initilize OpTA module
                transportation_module = OpTA(regularization=0.05,
                                             max_iter=1000,
                                             stopping_criterion=1e-4)

                ot_passes = one_shot_ot_passes if n_shot == 1 else five_shot_ot_passes
                prototypes = prototypes_before
                for i in range(ot_passes):
                    prototypes, cur_qry_f = transportation_module(
                        torch.from_numpy(prototypes), torch.from_numpy(cur_qry_f))

                    if i == 0:
                        first_pass_prototypes = prototypes

                    prototypes = prototypes.detach().cpu().numpy()
                    cur_qry_f = cur_qry_f.detach().cpu().numpy()

                if n_shot != 1:
                    prototypes = prototypes * ratio_OT + \
                        prototypes_before * (1 - ratio_OT)

                # initialize classifier
                if classifier == 'LR':
                    clf = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1.0,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                elif classifier == 'SVM':
                    clf = LinearSVC(C=1.0)

                # fit classifier, using support set prototypes
                clf.fit(prototypes, cur_sup_y[::n_shot])
                # predict for quey set
                cur_qry_pred = clf.predict(cur_qry_f)
                acc = metrics.accuracy_score(cur_qry_y, cur_qry_pred)

                # visualization
                if visualize_OT:
                    visualize_optimal_transport(prototypes_before,
                                                first_pass_prototypes,
                                                prototypes, cur_qry_f,
                                                cur_sup_y[::n_shot],
                                                cur_qry_y, "tsne", idx+1,
                                                n_shot, save_path=args.save_path,
                                                n_way=n_way, n_query=n_query)

                accs[f'{n_shot}-shot'].append(acc)

    results = []
    for n_shot in n_shots:
        results_shot = []
        acc = np.array(accs[f'{n_shot}-shot'])
        mean = acc.mean()
        std = acc.std()
        c95 = 1.96*std/math.sqrt(acc.shape[0])

        print('Accuracy, {}-way {}-shot: {:.2f}+{:.2f}'.format(
            n_way, n_shot, mean*100, c95*100))
        print("-------------------------------------\n")
        results_shot.append(mean*100)
        results_shot.append(c95*100)
        results.append(results_shot)

    return results


def evaluate_imagenet(args: dict, n_shots: list = [1, 5]):
    """
    Creates episodic FSL test dataloader for miniImageNet, tieredImageNet,
    CIFAR-FS and FC100, and evaluates the performance.

    Arguments:
        - args (dict): parsed keyword evaluation arguments
        - n_shots (list): list of n-shot settings to be evalauated (optional)
    """
    # build episodic dataloader
    test_loader = build_fewshot_loader(args, 'test', max_n_shot=max(n_shots))
    # build BECLR model
    student, teacher = build_student_teacher(args)

    # load checkpoit fo model to be evaluated
    if args.eval_path is not None:
        student.load_state_dict(torch.load(args.eval_path)
                                ['student'], strict=True)
        teacher.load_state_dict(torch.load(args.eval_path)
                                ['teacher'], strict=True)

        model = student if args.use_student else teacher

        evaluate_fewshot(args, model.module.encoder, test_loader,
                         n_way=args.n_way,
                         n_shots=n_shots, n_query=args.n_query,
                         one_shot_ot_passes=args.one_shot_ot_passes,
                         five_shot_ot_passes=args.five_shot_ot_passes,
                         classifier='LR', ratio_OT=args.ratio_ot)
        return


def evaluate_cub(args: dict, n_shots: list = [1, 5]):
    """
    Creates episodic FSL test dataloader for CUB, and evaluates the performance.

    Arguments:
        - args (dict): parsed keyword evaluation arguments
        - n_shots (list): list of n-shot settings to be evalauated (optional)
    """
    # build episodic dataloader
    test_loader = build_cub_fewshot_loader(
        args, n_shot=max(n_shots), download=False, mode='test')
    # build BECLR model
    student, teacher = build_student_teacher(args)

    # load checkpoit fo model to be evaluated
    if args.eval_path is not None:
        student.load_state_dict(torch.load(args.eval_path)
                                ['student'], strict=True)
        teacher.load_state_dict(torch.load(args.eval_path)
                                ['teacher'], strict=True)

        model = student if args.use_student else teacher

        evaluate_fewshot(args, model.module.encoder, test_loader,
                         n_way=args.n_way,
                         n_shots=n_shots, n_query=args.n_query,
                         one_shot_ot_passes=args.one_shot_ot_passes,
                         five_shot_ot_passes=args.five_shot_ot_passes,
                         classifier='LR', ratio_OT=args.ratio_ot)
        return


def evaluate_cdfsl(args: dict, n_shots: list = [5, 20]):
    """
    Creates episodic FSL test dataloaders for CDFSL, and evaluates the performance.

    Arguments:
        - args (dict): parsed keyword evaluation arguments
        - n_shots (list): list of n-shot settings to be evalauated (optional)
    """
    test_loaders = []

    # build episodic dataloader for each CDFSL dataset
    if args.cd_fsl == "all" or args.cd_fsl == "chestx":
        loader_name = "ChestX"
        print("Loading {}".format(loader_name))
        datamgr = Chest_few_shot.SetDataManager(Path(args.data_path) / Path(
            "chestX"), args.size, n_eposide=args.n_test_task, n_support=max(n_shots), n_query=args.n_query)
        chest_loader = datamgr.get_data_loader(aug=False)

        test_loaders.append((loader_name, chest_loader))

    if args.cd_fsl == "all" or args.cd_fsl == "isic":
        loader_name = "ISIC"
        print("Loading {}".format(loader_name))
        datamgr = ISIC_few_shot.SetDataManager(Path(args.data_path) / Path(
            "ISIC"), args.size, n_eposide=args.n_test_task, n_support=max(n_shots), n_query=args.n_query)
        isic_loader = datamgr.get_data_loader(aug=False)

        test_loaders.append((loader_name, isic_loader))

    if args.cd_fsl == "all" or args.cd_fsl == "eurosat":
        loader_name = "EuroSAT"
        print("Loading {}".format(loader_name))
        datamgr = EuroSAT_few_shot.SetDataManager(Path(args.data_path) / Path(
            "EuroSAT/2750"), args.size, n_eposide=args.n_test_task, n_support=max(n_shots), n_query=args.n_query)
        eurosat_loader = datamgr.get_data_loader(aug=False)

        test_loaders.append((loader_name, eurosat_loader))

    if args.cd_fsl == "all" or args.cd_fsl == "crop":
        loader_name = "CropDisease"
        print("Loading {}".format(loader_name))
        datamgr = CropDisease_few_shot.SetDataManager(Path(args.data_path) / Path(
            "plant-disease"), args.size, n_eposide=args.n_test_task, n_support=max(n_shots), n_query=args.n_query)
        cropdis_loader = datamgr.get_data_loader(aug=False)

        test_loaders.append((loader_name, cropdis_loader))

    # build BECLR model
    student, teacher = build_student_teacher(args)

    # load checkpoit fo model to be evaluated
    if args.eval_path is not None:
        student.load_state_dict(torch.load(args.eval_path)
                                ['student'], strict=True)
        teacher.load_state_dict(torch.load(args.eval_path)
                                ['teacher'], strict=True)

        model = student if args.use_student else teacher

        for idx, (loader_name, test_loader) in enumerate(test_loaders):
            print("---------- {} ------------".format(loader_name))
            evaluate_fewshot(args, model.module.encoder, test_loader,
                             n_way=args.n_way,
                             n_shots=n_shots, n_query=args.n_query,
                             one_shot_ot_passes=args.one_shot_ot_passes,
                             five_shot_ot_passes=args.five_shot_ot_passes,
                             classifier='LR', ratio_OT=args.ratio_ot)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'BECLR evaluation arguments', parents=[args_parser()])

    args = parser.parse_args()
    with open(args.cnfg_path) as f:
        parser = argparse.ArgumentParser()
        argparse_dict = vars(args)
        argparse_dict.update(json.load(f))

        args = argparse.Namespace()
        args.__dict__.update(argparse_dict)

    args.split_path = Path(Path(__file__).parent)/Path('dataset')/Path('split')

    # initialize distributed parallel training & fix random seed
    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    print("\n".join("%s: %s" % (k, str(v))
                    for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.dataset == "cub":
        evaluate_cub(args)
    elif args.dataset == "cdfsl":
        evaluate_cdfsl(args)
    else:
        evaluate_imagenet(args)
