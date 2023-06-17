import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision.transforms.functional import normalize

import numpy as np
import math
import copy
from supervised_finetuning import supervised_finetuning, supervised_finetuning_test

from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tqdm import tqdm
from pathlib import Path
import torch.backends.cudnn as cudnn
from utils import fix_random_seeds, init_distributed_mode

from optimal_transport import OptimalTransport
from cdfsl_benchmark.datasets import (ISIC_few_shot, CropDisease_few_shot,
                                      EuroSAT_few_shot, Chest_few_shot)


from utils import build_fewshot_loader, bool_flag, build_student_teacher


def args_parser():
    parser = argparse.ArgumentParser(
        'MSiam training arguments', add_help=False)

    parser.add_argument('--data_path', type=str,
                        default=None, help='path to dataset')
    parser.add_argument('--eval_path', type=str,
                        default=None, help='path to tested model')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['tieredImageNet', 'miniImageNet', 'cdfsl'], help='dataset')
    parser.add_argument('--num_workers', type=int,
                        default=1, help='num of workers to use')
    parser.add_argument('--use_feature_align', default=False, type=bool_flag,
                        help="Whether to use feature alignment")
    parser.add_argument('--use_feature_align_teacher', default=False, type=bool_flag,
                        help="Whether to use feature alignment on the teacher features")
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch_size')
    parser.add_argument('--topk', default=5, type=int,
                        help='Number of topk NN to extract, when enhancing the batch size (Default 5).')
    parser.add_argument('--use_clustering', default=False, type=bool_flag,
                        help="Whether to use online clustering")

    ######
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=bool_flag,
                        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
                        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
                        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=False, type=bool_flag,
                        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help="""Drop path rate for student network.""")
    ########

    # Model settings
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet10', 'resnet18', 'resnet34', 'resnet50', 'deit_tiny'], help='Choice of backbone network for the encoder')
    parser.add_argument('--size', type=int, default=224, help='input size')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='size of input square patches for masking in pixels, default 16 (for 16x16 patches)')

    parser.add_argument('--mask_ratio', default=0.0, type=float, nargs='+', help="""Ratio of masked-out patches.
        If a list of ratio is specified, one of them will be randomly choosed for each image.""")
    parser.add_argument('--mask_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--mask_shape', default='block',
                        type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--mask_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")

    # self-supervision settings
    parser.add_argument('--temp', type=float, default=2.0,
                        help='temperature for loss function')
    parser.add_argument('--lamb', type=float, default=0.1,
                        help='lambda for uniform loss')

    # few-shot evaluation settings
    parser.add_argument('--n_way', type=int, default=5, help='n_way')
    parser.add_argument('--n_query', type=int, default=15, help='n_query')
    parser.add_argument('--n_test_task', type=int,
                        default=3000, help='total test few-shot episodes')
    parser.add_argument('--test_batch_size', type=int,
                        default=5, help='episode_batch_size')
    parser.add_argument('--use_student', default=True, type=bool_flag,
                        help='whether to use student or teacher encoder for eval')

    parser.add_argument("--seed", type=int, default=42, help="seed")

    ###############################################################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    # Supervised finetuning parameters
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

    return parser


def groupedAvg(myArray, N):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result


def finetune_fewshot(
        args, encoder, loader, n_way=5, n_shots=[1, 5], n_query=15, classifier='LR', power_norm=False, feature_extractor=None):

    encoder.eval()
    accs = {}
    accs_ot = {}
    print("==> Fine-tuning...")

    for n_shot in n_shots:
        accs[f'{n_shot}-shot'] = []
        accs_ot[f'{n_shot}-shot'] = []

    original_encoder_state = copy.deepcopy(encoder.state_dict())

    # AAAAAAAAAAAAAAAAAA
    # n_shot = 5
    for idx, (episode, _) in enumerate(tqdm(loader)):
        if episode.dim() > 4:
            if idx == 0:
                print("Episode Size CD-FSL: {}".format(episode.size()))
            episode = episode.reshape((-1,)+episode.shape[2:])
        if idx == 0:
            print("Episode Size for eval: {}".format(episode.size()))

        # acc = supervised_finetuning_test(encoder,
        #                                  episode=episode,
        #                                  inner_lr=args.sup_finetune_lr,
        #                                  total_epochs=args.sup_finetune_epochs,
        #                                  freeze_backbone=args.ft_freeze_backbone,
        #                                  finetune_batch_norm=args.finetune_batch_norm,
        #                                  device=torch.device('cuda'),
        #                                  n_way=args.n_way,
        #                                  n_query=n_query,
        #                                  n_shot=n_shot,
        #                                  max_n_shot=max(n_shots),
        #                                  use_ot=args.ft_use_ot,
        #                                  use_prototypes_in_tuning=args.use_prototypes_in_tuning)

        # encoder.load_state_dict(original_encoder_state)
        # accs[f'{n_shot}-shot'].append(acc)

    # acc = np.array(accs[f'{n_shot}-shot'])
    # mean = acc.mean()
    # print('Mean Accuracy {}'.format(mean*100))

        for n_shot in n_shots:
            acc = supervised_finetuning_test(encoder,
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

        print('classifier: {}, power_norm: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
            classifier, power_norm, n_way, n_shot, mean*100, c95*100))
        print("------------------------------------------------------\n")
        results_shot.append(mean*100)
        results_shot.append(c95*100)
        results.append(results_shot)
    return results


@torch.no_grad()
def evaluate_fewshot(
        encoder, use_transformers, loader, n_way=5, n_shots=[1, 5], n_query=15, classifier='LR', power_norm=False, feature_extractor=None):

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

        if feature_extractor != None:
            x = f[:, :, None, None]
            G = f.permute(1, 0)

            w1 = feature_extractor.fe[0].weight.flatten(start_dim=1)
            # only keep first TBS weights from first layer
            # # pooling = nn.AdaptiveAvgPool2d((G.shape[1], 1))
            pooling = nn.AdaptiveMaxPool2d((G.shape[1], 1))
            w1 = pooling(w1[:, :, None]).flatten(start_dim=1)
            # w1 = w1[:, :G.shape[1]]

            w2 = feature_extractor.fe[2].weight.flatten(start_dim=1)

            G = torch.matmul(G, w1.t())
            G = feature_extractor.fe[1](G)
            G = torch.matmul(G, w2.t())[:, :, None, None]
            G = feature_extractor.fe[3](G)

            xq = feature_extractor.f_q(G)
            xk = feature_extractor.f_k(G)
            xv = feature_extractor.f_v(G)

            xq = xq.squeeze(dim=1).squeeze(dim=1).transpose(
                0, 1).reshape(-1, x.shape[2], x.shape[3])
            xk = xk.squeeze(dim=1).squeeze(dim=1).transpose(
                0, 1).reshape(-1, x.shape[2], x.shape[3])
            xv = xv.squeeze(dim=1).squeeze(dim=1).transpose(
                0, 1).reshape(-1, x.shape[2], x.shape[3])

            # Attention Block
            xq = xq.reshape(xq.shape[0], xq.shape[1]*xq.shape[2])
            xk = xk.reshape(xk.shape[0], xk.shape[1]*xk.shape[2])
            xv = xv.reshape(xv.shape[0], xv.shape[1]*xv.shape[2])

            G = torch.mm(xq, xk.transpose(0, 1)/xk.shape[1]**0.5)
            softmax = nn.Softmax(dim=-1)
            G = softmax(G)
            G = torch.mm(G, xv)

            # Transductive Mask transformed input
            G = G.reshape(-1, x.shape[2], x.shape[3])
            x = x * G
            f = nn.Flatten()(x)

        # print(torch.mean(f, dim=1, keepdim=False).size())
        # print("Befor Normalization: {}".format(torch.isnan(f).any()))
        # print("Befor Normalization: {}".format(not torch.isfinite(f).any()))

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

                prototypes = groupedAvg(cur_sup_f, n_shot)

                ##################
                transportation_module = OptimalTransport(regularization=0.05, learn_regularization=False, max_iter=1000,
                                                         stopping_criterion=1e-4)

                # cur_sup_f, cur_qry_f = transportation_module(
                #     torch.from_numpy(cur_sup_f), torch.from_numpy(cur_qry_f))
                prototypes, cur_qry_f = transportation_module(
                    torch.from_numpy(prototypes), torch.from_numpy(cur_qry_f))

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

        print('classifier: {}, power_norm: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
            classifier, power_norm, n_way, n_shot, mean*100, c95*100))

        print('classifier: {}, power_norm: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
            classifier, power_norm, n_way, n_shot, mean_ot*100, c95_ot*100))
        print("------------------------------------------------------\n")
        results_shot.append(mean_ot*100)
        results_shot.append(c95_ot*100)
        results.append(results_shot)
    return results


def evaluate_imagenet(args):
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.fine_tune:
        args.test_batch_size = 1

    test_loader = build_fewshot_loader(args, 'test')

    student, teacher = build_student_teacher(args)

    if args.eval_path is not None:
        # student, teacher = load_student_teacher(
        #     student, teacher, args.eval_path, eval=True)

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

        feature_extractor = model.module.feature_extractor if "feature_extractor" in model.named_buffers() else None

        if args.fine_tune:
            finetune_fewshot(args, model.module.encoder, test_loader, n_way=args.n_way, n_shots=[1, 5],
                             n_query=args.n_query, classifier='LR', power_norm=True, feature_extractor=feature_extractor)
        else:
            evaluate_fewshot(model.module.encoder, model.module.use_transformers, test_loader, n_way=args.n_way, n_shots=[
                1, 5], n_query=args.n_query, classifier='LR', power_norm=True, feature_extractor=feature_extractor)

            if "deit" in args.backbone:
                student.module.encoder.masked_im_modeling = True
        return


def evaluate_cdfsl(args):
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    #####################################################
    # few_shot_params = dict(n_way = args.n_way , n_support = params.n_test_shot)
    dataset_names = ["ISIC", "EuroSAT", "CropDisease", "ChestX"]
    test_loaders = []

    loader_name = "ChestX"
    print("Loading {}".format(loader_name))
    datamgr = Chest_few_shot.SetDataManager(Path(args.data_path) / Path("chestX"),
                                            args.size, n_eposide=args.n_test_task, n_support=20, n_query=args.n_query)
    test_loader = datamgr.get_data_loader(aug=False)

    test_loaders.append((loader_name, test_loader))

    print(len(test_loader))
    exit()

    loader_name = "ISIC"
    print("Loading {}".format(loader_name))
    datamgr = ISIC_few_shot.SetDataManager(
        args.size, n_eposide=args.n_test_task, n_support=20, n_query=args.n_query)
    test_loader = datamgr.get_data_loader(aug=False)

    test_loaders.append((loader_name, test_loader))

    loader_name = "EuroSAT"
    print("Loading {}".format(loader_name))
    datamgr = EuroSAT_few_shot.SetDataManager(
        args.size, n_eposide=args.n_test_task, n_support=20, n_query=args.n_query)
    test_loader = datamgr.get_data_loader(aug=False)

    test_loaders.append((loader_name, test_loader))

    loader_name = "CropDisease"
    print("Loading {}".format(loader_name))
    datamgr = CropDisease_few_shot.SetDataManager(
        args.size, n_eposide=args.n_test_task, n_support=20, n_query=args.n_query)
    test_loader = datamgr.get_data_loader(aug=False)

    test_loaders.append((loader_name, test_loader))

    # for idx, (loader_name, test_loader) in enumerate(test_loaders):
    #     print(len(test_loader))

    # exit()
    ####################################################

    student, teacher = build_student_teacher(args)

    if args.eval_path is not None:
        # student, teacher = load_student_teacher(
        #     student, teacher, args.eval_path, eval=True)

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

        feature_extractor = model.module.feature_extractor if "feature_extractor" in model.named_buffers() else None

        # evaluate all datasets of the cd-fsl benchmark
        for idx, (loader_name, test_loader) in enumerate(test_loaders):
            print("---------- {} ------------".format(loader_name))
            if args.fine_tune:
                finetune_fewshot(args, model.module.encoder, test_loader, n_way=args.n_way, n_shots=[5, 20],
                                 n_query=args.n_query, classifier='LR', power_norm=True, feature_extractor=feature_extractor)
            else:
                evaluate_fewshot(model.module.encoder, model.module.use_transformers, test_loader, n_way=args.n_way, n_shots=[
                    5, 20], n_query=args.n_query, classifier='LR', power_norm=True, feature_extractor=feature_extractor)

            if "deit" in args.backbone:
                student.module.encoder.masked_im_modeling = True
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MSiam evaluation arguments', parents=[args_parser()])
    args = parser.parse_args()
    # need to change that
    args.dist = False

    args.split_path = (Path(__file__).parent).joinpath('split')

    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    if args.dataset == "tieredImageNet" or args.dataset == "miniImageNet":
        evaluate_imagenet(args)
    else:
        evaluate_cdfsl(args)
