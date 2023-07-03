"""Adapted from https://github.com/IBM/cdfsl-benchmark/blob/master/finetune.py"""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
from sklearn import metrics
import os
import glob
from itertools import combinations
from torchvision.transforms.functional import normalize
from optimal_transport import OptimalTransport


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _set_params(self, weight, bias):
        state_dict = dict(weight=weight, bias=bias)
        self.fc.load_state_dict(state_dict)

    def init_params_from_prototypes(self, z_support, n_way, n_support, use_ot=False):
        z_support = z_support.contiguous()
        if use_ot:
            z_proto = z_support.view(n_way, 1, -1).mean(1)
        else:
            z_proto = z_support.view(n_way, n_support, -1).mean(1)
        # Interpretation of ProtoNet as linear layer (see Snell et al. (2017))
        self._set_params(weight=2*z_proto, bias=-
                         torch.norm(z_proto, dim=-1)**2)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def groupedAvg(myArray, N):
    result = np.cumsum(myArray, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result


def torch_groupedAvg(myTensor, N):
    result = torch.cumsum(myTensor, 0)[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    return result


def supervised_finetuning_test(encoder, episode, device='cuda',
                               proto_init=True, freeze_backbone=False,
                               finetune_batch_norm=False,
                               inner_lr=0.001, total_epochs=15, n_way=5, n_query=15, n_shot=5, max_n_shot=5, use_ot=False, use_prototypes_in_tuning=False):

    episode = episode.cuda(non_blocking=True)
    torch.autograd.set_detect_anomaly(True)

    f = encoder(episode)
    # mean normalization
    f = f[:, :, None, None]  # unflatten resnet output
    f = normalize(f, torch.mean(
        f, dim=1, keepdim=True), torch.std(f, dim=1, keepdim=True), inplace=False)

    z_support, z_query = torch.split(f.view(
        1, n_way, max_n_shot+n_query, -1), [max_n_shot, n_query], dim=2)

    z_support = z_support.view(
        1, n_way, max_n_shot, -1)[0, :, :n_shot, :].reshape(n_way*n_shot, -1)
    z_query = z_query.reshape(n_way*n_query, -1)

    n_support = z_support.shape[0] // n_way
    support_size = z_support.shape[0]
    batch_size = n_way

    y_support = Variable(torch.from_numpy(
        np.repeat(range(n_way), n_shot))).type(torch.LongTensor).to(device)  # (25,)

    # print("Support Features Size: {}".format(z_support.size()))
    # print("Query Features Size: {}".format(z_query.size()))

    if use_ot:
        # -------------- Optimal Transport -------------------
        transportation_module = OptimalTransport(regularization=0.05, learn_regularization=False, max_iter=1000,
                                                 stopping_criterion=1e-4, device=device)
        prototypes = torch_groupedAvg(z_support, n_shot)

        z_support_tran, _ = transportation_module(prototypes, z_query)

        if use_prototypes_in_tuning:
            z_support = z_support_tran
            y_support = Variable(torch.from_numpy(np.repeat(range(n_way), 1))).type(
                torch.LongTensor).to(device)
            support_size = n_way
    # -------------------------------------------------

    input_dim = z_support.shape[1]
    classifier = Classifier(input_dim, n_way=n_way)
    classifier.to(device)
    classifier.train()

    loss_fn = nn.CrossEntropyLoss().to(device)
    if proto_init:  # Initialise as distance classifer (distance to prototypes)
        if use_ot and not use_prototypes_in_tuning:
            classifier.init_params_from_prototypes(z_support_tran,
                                                   n_way, n_support, use_ot=use_ot)
        else:
            classifier.init_params_from_prototypes(z_support,
                                                   n_way, n_support, use_ot=use_ot)
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=inner_lr)

    if freeze_backbone is False:
        delta_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder.parameters()), lr=inner_lr)

    # Finetuning
    if freeze_backbone is False:
        encoder.train()
    else:
        encoder.eval()

    classifier.train()

    if not finetune_batch_norm:
        for module in encoder.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()

    # print("Support Feature Size: {}".format(z_support.size()))
    # print("Support Size: {}".format(support_size))
    # print("Support Labels Size: {}".format(y_support.size()))

    for epoch in tqdm(range(total_epochs), total=total_epochs, leave=False):

        # output = classifier(z_support)
        # loss = loss_fn(output, y_support)

        # loss.backward(retain_graph=True)
        # classifier_opt.step()

        rand_id = np.random.permutation(support_size)

        for j in range(0, support_size, batch_size):  # why not entire support set ????
            classifier_opt.zero_grad()
            if freeze_backbone is False:
                delta_opt.zero_grad()

            #####################################
            selected_id = torch.from_numpy(
                rand_id[j: min(j+batch_size, support_size)]).type(
                torch.LongTensor).to(device)

            output = z_support[selected_id, :]
            y_batch = y_support[selected_id]
            #####################################

            output = classifier(output)
            loss = loss_fn(output, y_batch)

            # print(loss)
            # if loss > 0:
            #     print(loss)

            #####################################
            loss.backward(retain_graph=True)

            classifier_opt.step()

            if freeze_backbone is False:
                delta_opt.step()

    classifier.eval()
    encoder.eval()

    y_query = torch.tensor(np.repeat(range(n_way), n_query)).to(device)

    scores = classifier(z_query)
    _, predictions = torch.max(scores, dim=1)

    acc = metrics.accuracy_score(y_query.cpu(), predictions.cpu())

    # print(predictions)
    # print(y_query)
    # print("Accuracy: {}".format(acc))
    # exit()

    return acc


def supervised_finetuning(encoder, episode, device='cuda',
                          proto_init=True, freeze_backbone=False,
                          finetune_batch_norm=False,
                          inner_lr=0.001, total_epochs=15, n_way=5, n_query=15, n_shot=5, max_n_shot=5, use_ot=False, use_prototypes_in_tuning=False):

    episode = episode.cuda(non_blocking=True)
    torch.autograd.set_detect_anomaly(True)

    f = encoder(episode)
    # mean normalization
    f = f[:, :, None, None]  # unflatten resnet output
    f = normalize(f, torch.mean(
        f, dim=1, keepdim=True), torch.std(f, dim=1, keepdim=True), inplace=False)

    z_support, z_query = torch.split(f.view(
        1, n_way, max_n_shot+n_query, -1), [max_n_shot, n_query], dim=2)

    z_support = z_support.view(
        1, n_way, max_n_shot, -1)[0, :, :n_shot, :].reshape(n_way*n_shot, -1)
    z_query = z_query.reshape(n_way*n_query, -1)

    n_support = z_support.shape[0] // n_way
    support_size = z_support.shape[0]
    batch_size = n_way

    y_support = Variable(torch.from_numpy(
        np.repeat(range(n_way), n_shot))).type(torch.LongTensor).to(device)  # (25,)

    # print("Support Features Size: {}".format(z_support.size()))
    # print("Query Features Size: {}".format(z_query.size()))

    if use_ot:
        # -------------- Optimal Transport -------------------
        transportation_module = OptimalTransport(regularization=0.05, learn_regularization=False, max_iter=1000,
                                                 stopping_criterion=1e-4, device=device)
        prototypes = torch_groupedAvg(z_support, n_shot)

        z_support_tran, _ = transportation_module(prototypes, z_query)

        if use_prototypes_in_tuning:
            z_support = z_support_tran
            y_support = Variable(torch.from_numpy(np.repeat(range(n_way), 1))).type(
                torch.LongTensor).to(device)
            support_size = n_way
    # -------------------------------------------------

    input_dim = z_support.shape[1]
    classifier = Classifier(input_dim, n_way=n_way)
    classifier.to(device)
    classifier.train()

    loss_fn = nn.CrossEntropyLoss().to(device)
    if proto_init:  # Initialise as distance classifer (distance to prototypes)
        if use_ot and not use_prototypes_in_tuning:
            classifier.init_params_from_prototypes(z_support_tran,
                                                   n_way, n_support, use_ot=use_ot)
        else:
            classifier.init_params_from_prototypes(z_support,
                                                   n_way, n_support, use_ot=use_ot)
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=inner_lr)

    if freeze_backbone is False:
        delta_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder.parameters()), lr=inner_lr)

    # Finetuning
    if freeze_backbone is False:
        encoder.train()
    else:
        encoder.eval()

    classifier.train()

    if not finetune_batch_norm:
        for module in encoder.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()

    x_support, x_query = torch.split(episode.view(
        1, n_way, max_n_shot+n_query, -1), [max_n_shot, n_query], dim=2)

    x_support = x_support.view(
        1, n_way, max_n_shot, -1)[0, :, :n_shot, :].reshape(n_way*n_shot, 3, 224, 224)
    x_query = x_query.reshape(n_way*n_query, 3, 224, 224)

    # print("Support Images Size: {}".format(x_support.size()))
    # print("Support Size: {}".format(support_size))
    # print("Support Labels Size: {}".format(y_support.size()))

    for epoch in tqdm(range(total_epochs), total=total_epochs, leave=False):
        rand_id = np.random.permutation(support_size)

        for j in range(0, support_size, batch_size):  # why not entire support set ????
            classifier_opt.zero_grad()
            if freeze_backbone is False:
                delta_opt.zero_grad()

            #####################################
            selected_id = torch.from_numpy(
                rand_id[j: min(j+batch_size, support_size)]).type(
                torch.LongTensor).to(device)

            output = encoder(x_support[selected_id, :])[:, :, None, None]
            output = normalize(output, torch.mean(
                output, dim=1, keepdim=True), torch.std(output, dim=1, keepdim=True), inplace=False).squeeze()

            y_batch = y_support[selected_id]
            #####################################

            output = classifier(output.reshape(n_way, input_dim))
            loss = loss_fn(output, y_batch)

            #####################################
            loss.backward()

            classifier_opt.step()

            if freeze_backbone is False:
                delta_opt.step()

    classifier.eval()
    encoder.eval()

    z_query = encoder(x_query)[:, :, None, None]
    z_query = normalize(z_query, torch.mean(
        z_query, dim=1, keepdim=True), torch.std(z_query, dim=1, keepdim=True), inplace=False).reshape(n_query * n_way, input_dim)

    y_query = torch.tensor(np.repeat(range(n_way), n_query)).to(device)

    scores = classifier(z_query)
    _, predictions = torch.max(scores, dim=1)

    acc = metrics.accuracy_score(y_query.cpu(), predictions.cpu())

    return acc