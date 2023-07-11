import argparse
import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torchvision.transforms.functional import normalize
from tqdm import tqdm

from cdfsl_benchmark.datasets import (Chest_few_shot, CropDisease_few_shot,
                                      EuroSAT_few_shot, ISIC_few_shot)
from memory import NNmemoryBankModule
from optimal_transport import OptimalTransport
from supervised_finetuning import (supervised_finetuning,
                                   supervised_finetuning_test)
from utils import (bool_flag, build_cub_fewshot_loader, build_fewshot_loader,
                   build_student_teacher, fix_random_seeds, get_params_groups,
                   init_distributed_mode, load_student_teacher)
from visualize import visualize_memory, visualize_optimal_transport


def args_parser():
    parser = argparse.ArgumentParser(
        'BECLR evaluation arguments', add_help=False)

    parser.add_argument('--data_path', type=str,
                        default=None, help='path to dataset root')
    parser.add_argument('--eval_path', type=str,
                        default=None, help='path to tested model')
    parser.add_argument('--save_path', type=str,
                        default=None, help='path for saving visualizations')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['tieredImageNet',
                                 'miniImageNet', 'cub', 'cdfsl'],
                        help='choice of dataset for pre-training')
    parser.add_argument('--num_workers', type=int,
                        default=1, help='num of workers to use')
    parser.add_argument("--seed", type=int, default=42, help="random seed")

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
                        help="""Dimensionality of output.""")

    # parallelization settings
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="""url used to set up distributed training; see \
                        https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int,
                        help="""number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int,
                        help="""rank of this process: it is set automatically \
                        and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    # few-shot evaluation settings
    parser.add_argument('--n_way', type=int, default=5, help='n_way')
    parser.add_argument('--n_query', type=int, default=15, help='n_query')
    parser.add_argument('--n_test_task', type=int,
                        default=3000, help='total test few-shot episodes')
    parser.add_argument('--test_batch_size', type=int,
                        default=5, help='episode_batch_size')
    parser.add_argument('--use_student', default=True, type=bool_flag,
                        help='whether to use student or teacher encoder for eval')

    # supervised finetuning settings
    parser.add_argument('--fine_tune', default=False, type=bool_flag,
                        help='whether to perform supervised fine-tuning')
    parser.add_argument('--sup_finetune_lr', type=float, default=0.001,
                        help='Learning rate for finetuning.')
    parser.add_argument('--sup_finetune_epochs', type=int, default=15,
                        help='How many epochs to finetune.')
    parser.add_argument('--ft_freeze_backbone', default=True, type=bool_flag,
                        help='Whether to freeze the backbone during finetuning.')
    parser.add_argument('--finetune_batch_norm', default=False, type=bool_flag,
                        help='Whether to update the batch norm parameters during finetuning.')
    parser.add_argument('--ft_use_ot', default=True, type=bool_flag,
                        help='Whether to use OT during finetuning.')
    parser.add_argument('--use_prototypes_in_tuning', default=False, type=bool_flag,
                        help='Whether to use prototypes during finetuning or just for initialization.')

    parser.add_argument('--cd_fsl', type=str, default='all',
                        choices=['all', 'chestx', 'isic', 'eurosat', 'crop'], help='dataset')

    return parser


def groupedAvg(myArray, N):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result


def finetune_fewshot(
        args, encoder, loader, n_way=5, n_shots=[1, 5], n_query=15, classifier='LR'):

    encoder.eval()
    accs = {}
    accs_ot = {}
    print("==> Fine-tuning...")

    for n_shot in n_shots:
        accs[f'{n_shot}-shot'] = []
        accs_ot[f'{n_shot}-shot'] = []

    original_encoder_state = copy.deepcopy(encoder.state_dict())

    for idx, (episode, _) in enumerate(tqdm(loader)):
        if episode.dim() > 4:
            if idx == 0:
                print("Episode Size CD-FSL: {}".format(episode.size()))
            episode = episode.reshape((-1,)+episode.shape[2:])
        if idx == 0:
            print("Episode Size for eval: {}".format(episode.size()))

        for n_shot in n_shots:
            acc = supervised_finetuning(encoder,
                                        episode=episode,
                                        inner_lr=args.sup_finetune_lr,
                                        total_epochs=args.sup_finetune_epochs,
                                        freeze_backbone=args.ft_freeze_backbone,
                                        finetune_batch_norm=args.finetune_batch_norm,
                                        device=torch.device('cuda'),
                                        n_way=args.n_way,
                                        n_query=n_query,
                                        n_shot=n_shot,
                                        max_n_shot=max(n_shots),
                                        use_ot=args.ft_use_ot,
                                        use_prototypes_in_tuning=args.use_prototypes_in_tuning)

            encoder.load_state_dict(original_encoder_state)
            accs[f'{n_shot}-shot'].append(acc)

    results = []
    for n_shot in n_shots:
        results_shot = []
        acc = np.array(accs[f'{n_shot}-shot'])
        mean = acc.mean()
        std = acc.std()
        c95 = 1.96*std/math.sqrt(acc.shape[0])

        print('classifier: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
            classifier, n_way, n_shot, mean*100, c95*100))
        print("------------------------------------------------------\n")
        results_shot.append(mean*100)
        results_shot.append(c95*100)
        results.append(results_shot)
    return results


@torch.no_grad()
def evaluate_fewshot(
        args, encoder, loader, n_way=5, n_shots=[1, 5], n_query=15, classifier='LR', visualize_OT=False):

    encoder.eval()
    accs = {}
    accs_ot = {}
    print("==> Evaluating...")

    for n_shot in n_shots:
        accs[f'{n_shot}-shot'] = []
        accs_ot[f'{n_shot}-shot'] = []

    for idx, (images, _) in enumerate(tqdm(loader)):
        if images.dim() > 4:
            if idx == 0:
                print("Episode Size CD-FSL: {}".format(images.size()))
            images = images.reshape((-1,)+images.shape[2:])
        if idx == 0:
            print("Episode Size for eval: {}".format(images.size()))

        images = images.cuda(non_blocking=True)

        f = encoder(images)

        # mean normalization
        f = f[:, :, None, None]  # unflatten resnet output
        f = normalize(f, torch.mean(
            f, dim=1, keepdim=True), torch.std(f, dim=1, keepdim=True), inplace=True)

        # print("After Normalization: {}".format(torch.isnan(f).any()))
        # print("After Normalization: {}".format(not torch.isfinite(f).any()))

        max_n_shot = max(n_shots)
        test_batch_size = int(f.shape[0]/n_way/(n_query+max_n_shot))

        # sup_f: TBS x n_way x n_shot x D
        # qry_f: TBS x n_way x n_query x D
        sup_f, qry_f = torch.split(f.view(
            test_batch_size, n_way, max_n_shot+n_query, -1), [max_n_shot, n_query], dim=2)
        if idx == 0:
            print("Total Features Size: {}".format(f.size()))
            print("Support Features Size: {}".format(sup_f.size()))
            print("Query Features Size: {}".format(qry_f.size()))

        qry_f = qry_f.reshape(test_batch_size, n_way *
                              n_query, -1).detach().cpu().numpy()
        # (n_way * n_query)
        qry_label = torch.arange(n_way).unsqueeze(
            1).expand(n_way, n_query).reshape(-1).numpy()

        # -- Fit Logistic Regression Classifier
        for tb in range(test_batch_size):
            for n_shot in n_shots:

                # (n_way * n_shot) x D
                cur_sup_f = sup_f[tb, :, :n_shot, :].reshape(
                    n_way*n_shot, -1).detach().cpu().numpy()
                cur_sup_y = torch.arange(n_way).unsqueeze(
                    1).expand(n_way, n_shot).reshape(-1).numpy()
                # (n_way * n_query) x D
                cur_qry_f = qry_f[tb]
                cur_qry_y = qry_label
                if idx == 0 and tb == 0:
                    print(
                        "Total {}-shot Support Features: {}".format(n_shot, cur_sup_f.shape))

                prototypes_before = groupedAvg(cur_sup_f, n_shot)

                ##################
                transportation_module = OptimalTransport(regularization=0.05, learn_regularization=False, max_iter=1000,
                                                         stopping_criterion=1e-4)

                prototypes, cur_qry_f = transportation_module(
                    torch.from_numpy(prototypes_before), torch.from_numpy(cur_qry_f))

                prototypes = prototypes.detach().cpu().numpy()
                cur_qry_f = cur_qry_f.detach().cpu().numpy()
                ##################

                if classifier == 'LR':
                    clf = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1.0,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                    clf_ot = LogisticRegression(penalty='l2',
                                                random_state=0,
                                                C=1.0,
                                                solver='lbfgs',
                                                max_iter=1000,
                                                multi_class='multinomial')
                elif classifier == 'SVM':
                    clf = LinearSVC(C=1.0)
                    clf_ot = LinearSVC(C=1.0)

                clf.fit(cur_sup_f, cur_sup_y)
                clf_ot.fit(prototypes, cur_sup_y[::n_shot])

                cur_qry_pred = clf.predict(cur_qry_f)
                cur_qry_pred_ot = clf_ot.predict(cur_qry_f)

                acc = metrics.accuracy_score(cur_qry_y, cur_qry_pred)
                acc_ot = metrics.accuracy_score(cur_qry_y, cur_qry_pred_ot)

                if visualize_OT:
                    visualize_optimal_transport(prototypes_before, prototypes, cur_qry_f,
                                                cur_sup_y[::n_shot], cur_qry_y, "umap", idx+1, n_shot, save_path=args.save_path, n_way=n_way, n_query=n_query)

                # if acc_ot - acc > 0.2:
                #     print(
                #         "------------------- Episode {} ---------------\n".format(idx+1))
                #     print("Acc before: {}".format(acc))
                #     print("Acc before: {}".format(acc_ot))
                #     visualize_optimal_transport(prototypes_before, prototypes, cur_qry_f,
                #                                 cur_sup_y[::n_shot], cur_qry_y, "tsne", idx+1, n_shot, save_path=args.save_path, n_way=n_way, n_query=n_query)

                accs[f'{n_shot}-shot'].append(acc)
                accs_ot[f'{n_shot}-shot'].append(acc_ot)

    results = []
    for n_shot in n_shots:
        results_shot = []
        acc = np.array(accs[f'{n_shot}-shot'])
        mean = acc.mean()
        std = acc.std()
        c95 = 1.96*std/math.sqrt(acc.shape[0])

        acc_ot = np.array(accs_ot[f'{n_shot}-shot'])
        mean_ot = acc_ot.mean()
        std_ot = acc_ot.std()
        c95_ot = 1.96*std_ot/math.sqrt(acc_ot.shape[0])

        print('classifier: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
            classifier, n_way, n_shot, mean*100, c95*100))

        print('classifier: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
            classifier, n_way, n_shot, mean_ot*100, c95_ot*100))
        print("------------------------------------------------------\n")
        results_shot.append(mean_ot*100)
        results_shot.append(c95_ot*100)
        results.append(results_shot)
    return results


def evaluate_imagenet(args):
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    n_shots = [1, 5]

    if args.fine_tune:
        args.test_batch_size = 1

    test_loader = build_fewshot_loader(args, 'test', max_n_shot=max(n_shots))

    student, teacher = build_student_teacher(args)
    ##########################################################################################################
    # memory_size = (40 * 200 //
    #                (256 * 2) + 1) * 256 * 2 + 1

    # params_groups = get_params_groups(student)
    # optimizer = torch.optim.SGD(
    #     params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    # fp16_scaler = None
    # teacher_nn_replacer = NNmemoryBankModule(
    #     size=memory_size, origin="teacher")
    # student_nn_replacer = NNmemoryBankModule(
    #     size=memory_size, origin="student")
    # student_f_nn_replacer = NNmemoryBankModule(
    #     size=memory_size, origin="student_f")

    # print("Memory Size: {}".format(memory_size))

    # student, teacher, optimizer, fp16_scaler, start_epoch, loss, batch_size = load_student_teacher(
    #     student, teacher, args.eval_path, teacher_nn_replacer,
    #     student_nn_replacer, student_f_nn_replacer, optimizer=optimizer,
    #     fp16_scaler=fp16_scaler)

    # visualize_memory(teacher_nn_replacer, args.save_path, "teacher")
    # exit()
    ###############################

    if args.eval_path is not None:

        student.load_state_dict(torch.load(args.eval_path)
                                ['student'], strict=True)
        teacher.load_state_dict(torch.load(args.eval_path)
                                ['teacher'], strict=True)

        if "deit" in args.backbone:
            student.module.encoder.masked_im_modeling = False

        if args.use_student:
            model = student
        else:
            model = teacher

        if args.fine_tune:
            finetune_fewshot(args, model.module.encoder, test_loader, n_way=args.n_way, n_shots=n_shots,
                             n_query=args.n_query, classifier='LR')
        else:
            evaluate_fewshot(args, model.module.encoder, test_loader, n_way=args.n_way,
                             n_shots=n_shots, n_query=args.n_query, classifier='LR')

            if "deit" in args.backbone:
                student.module.encoder.masked_im_modeling = True
        return


def evaluate_cub(args):
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    n_shots = [5, 20]

    if args.fine_tune:
        args.test_batch_size = 1

    test_loader = build_cub_fewshot_loader(
        args, n_shot=max(n_shots), download=False, mode='test')

    student, teacher = build_student_teacher(args)

    if args.eval_path is not None:

        student.load_state_dict(torch.load(args.eval_path)
                                ['student'], strict=True)
        teacher.load_state_dict(torch.load(args.eval_path)
                                ['teacher'], strict=True)

        if "deit" in args.backbone:
            student.module.encoder.masked_im_modeling = False

        if args.use_student:
            model = student
        else:
            model = teacher

        if args.fine_tune:
            finetune_fewshot(args, model.module.encoder, test_loader, n_way=args.n_way, n_shots=n_shots,
                             n_query=args.n_query, classifier='LR')
        else:
            evaluate_fewshot(args, model.module.encoder, test_loader, n_way=args.n_way,
                             n_shots=n_shots, n_query=args.n_query, classifier='LR')

            if "deit" in args.backbone:
                student.module.encoder.masked_im_modeling = True
        return


def evaluate_cdfsl(args):
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    n_shots = [5, 20]

    #####################################################
    # few_shot_params = dict(n_way = args.n_way , n_support = params.n_test_shot)
    dataset_names = ["ISIC", "EuroSAT", "CropDisease", "ChestX"]
    test_loaders = []

    if args.cd_fsl == "all" or args.cd_fsl == "chestx":
        loader_name = "ChestX"
        print("Loading {}".format(loader_name))
        datamgr = Chest_few_shot.SetDataManager(Path(args.data_path) / Path("chestX"),
                                                args.size, n_eposide=args.n_test_task, n_support=max(n_shots), n_query=args.n_query)
        chest_loader = datamgr.get_data_loader(aug=False)

        test_loaders.append((loader_name, chest_loader))

    if args.cd_fsl == "all" or args.cd_fsl == "isic":
        loader_name = "ISIC"
        print("Loading {}".format(loader_name))
        datamgr = ISIC_few_shot.SetDataManager(Path(args.data_path) / Path("ISIC"),
                                               args.size, n_eposide=args.n_test_task, n_support=max(n_shots), n_query=args.n_query)
        isic_loader = datamgr.get_data_loader(aug=False)

        test_loaders.append((loader_name, isic_loader))

    if args.cd_fsl == "all" or args.cd_fsl == "eurosat":
        loader_name = "EuroSAT"
        print("Loading {}".format(loader_name))
        datamgr = EuroSAT_few_shot.SetDataManager(Path(args.data_path) / Path("EuroSAT/2750"),
                                                  args.size, n_eposide=args.n_test_task, n_support=max(n_shots), n_query=args.n_query)
        eurosat_loader = datamgr.get_data_loader(aug=False)

        test_loaders.append((loader_name, eurosat_loader))

    if args.cd_fsl == "all" or args.cd_fsl == "crop":
        loader_name = "CropDisease"
        print("Loading {}".format(loader_name))
        datamgr = CropDisease_few_shot.SetDataManager(Path(args.data_path) / Path("plant-disease"),
                                                      args.size, n_eposide=args.n_test_task, n_support=max(n_shots), n_query=args.n_query)
        cropdis_loader = datamgr.get_data_loader(aug=False)

        test_loaders.append((loader_name, cropdis_loader))

    ####################################################

    student, teacher = build_student_teacher(args)

    if args.eval_path is not None:

        student.load_state_dict(torch.load(args.eval_path)
                                ['student'], strict=True)
        teacher.load_state_dict(torch.load(args.eval_path)
                                ['teacher'], strict=True)

        if "deit" in args.backbone:
            student.module.encoder.masked_im_modeling = False

        if args.use_student:
            model = student
        else:
            model = teacher

        # evaluate all datasets of the cd-fsl benchmark
        for idx, (loader_name, test_loader) in enumerate(test_loaders):
            print("---------- {} ------------".format(loader_name))
            if args.fine_tune:
                finetune_fewshot(args, model.module.encoder, test_loader, n_way=args.n_way, n_shots=n_shots,
                                 n_query=args.n_query, classifier='LR')
            else:
                evaluate_fewshot(args, model.module.encoder, test_loader, n_way=args.n_way,
                                 n_shots=n_shots, n_query=args.n_query, classifier='LR')

            if "deit" in args.backbone:
                student.module.encoder.masked_im_modeling = True
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'BECLR evaluation arguments', parents=[args_parser()])
    args = parser.parse_args()
    # need to change that
    args.dist = False

    args.split_path = (Path(__file__).parent).joinpath('split')

    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    if args.dataset == "tieredImageNet" or args.dataset == "miniImageNet":
        evaluate_imagenet(args)
    elif args.dataset == "cub":
        evaluate_cub(args)
    else:
        evaluate_cdfsl(args)
