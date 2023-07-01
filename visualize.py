
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
from umap import UMAP  # noqa


def visualize_memory(memory_bank):
    print("--Unique Labels Counts--: {}-------\n".format(
        memory_bank.labels.unique(return_counts=True)[-1]))
    print(int(memory_bank.bank_ptr))
    bank = np.array(memory_bank.bank.T.detach().cpu())
    labels = np.array(memory_bank.labels.detach().cpu())

    print(bank.shape)
    print(labels.shape)
    print(labels)
    exit()

    # random classes
    # indices of these classes
    # random 5 images per class
    # get umap projections
    # plot with or without distributions

    # alternaive first get projections and then do the indexing

    umap = UMAP(n_components=2, init='random', random_state=0)
    umap_proj = umap.fit_transform(bank)

    cmap = cm.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # only plot the first 25 clusters
    num_clusters = 25
    ax.set_prop_cycle(color=[cmap(1.*i/num_clusters)
                      for i in range(num_clusters)])

    for lab in range(num_clusters):
        indices = labels == lab
        indices = np.random.permutation(indices)
        # only plot 5 samples/class
        if len(indices < 5):
            ax.scatter(umap_proj[indices, 0],
                       umap_proj[indices, 1],
                       label=lab,
                       alpha=0.75)
        else:
            ax.scatter(umap_proj[indices[0:5], 0],
                       umap_proj[indices[0:5], 1],
                       label=lab,
                       alpha=0.75)
    # save 2d graph
    plt.savefig(Path("C:\GitHub\msiam") / Path("memory_vis.png"))


@torch.no_grad()
def visualize_optimal_transport(orginal_prototypes, transported_prototypes, z_query,
                                y_support, y_query, proj, episode, n_shot,
                                n_way=5, n_query=15):
    df_sup_before = pd.DataFrame(orginal_prototypes)
    df_sup_before['class'] = pd.Series(y_support)
    df_sup_before['set'] = pd.Series(np.ones(len(df_sup_before.index)))

    df_sup_after = pd.DataFrame(transported_prototypes)
    df_sup_after['class'] = pd.Series(y_support)
    df_sup_after['set'] = pd.Series(np.ones(len(df_sup_after.index)))

    df_quer = pd.DataFrame(z_query)
    df_quer['class'] = pd.Series(y_query)
    df_quer['set'] = pd.Series(np.zeros(len(df_quer.index)))

    df = pd.concat([df_sup_before, df_quer, df_sup_after])
    df = df.reset_index(drop=True)
    features = df.iloc[:, :-2]

    if proj == "tsne":
        tsne = TSNE(n_components=2, verbose=0)
        proj_2d = tsne.fit_transform(features)
    else:
        umap = UMAP(n_components=2, init='random', random_state=0)
        proj_2d = umap.fit_transform(features)

    proj_2d_before = proj_2d[:-n_way, :]
    proj_2d_after = proj_2d[n_way:, :]

    print("Set DBS score before: {}".format(
        davies_bouldin_score(proj_2d_before, df['set'][:-n_way])))
    print("Set DBS score after: {}".format(
        davies_bouldin_score(proj_2d_after, df['set'][n_way:])))

    # sns.relplot(x=proj_2d_before[:, 0], y=proj_2d_before[:, 1], hue=df['class'][:-n_way].astype(
    #     int), palette="Dark2", style=df['set'][:-n_way].astype(int), s=df['set'][:-n_way]*150+50, legend=False)
    # a = sns.kdeplot(x=proj_2d_before[:, 0], y=proj_2d_before[:, 1],
    #                 hue=df['class'][:-n_way].astype(int), palette="Pastel2", legend=False)
    # sfig = a.get_figure()
    # sfig.savefig('C:/GitHub/msiam/visualizations/{}_ep{}_{}-shot_before.jpeg'.format(
    #     proj, episode, n_shot), dpi=1000)

    # sns.relplot(x=proj_2d_after[:, 0], y=proj_2d_after[:, 1], hue=df['class'][n_way:].astype(
    #     int), palette="Dark2", style=df['set'][n_way:].astype(int), s=df['set'][n_way:]*150+50, legend=False)
    # b = sns.kdeplot(x=proj_2d_after[:, 0], y=proj_2d_after[:, 1],
    #                 hue=df['class'][n_way:].astype(int), palette="Pastel2", legend=False)
    # sfig = b.get_figure()
    # sfig.savefig('C:/GitHub/msiam/visualizations/{}_ep{}_{}-shot_after.jpeg'.format(
    #     proj, episode, n_shot), dpi=1000)

    sns.relplot(x=proj_2d_before[:, 0], y=proj_2d_before[:, 1], hue=df['class'][:-n_way].astype(
        int), palette="Dark2", style=df['set'][:-n_way].astype(int), s=df['set'][:-n_way]*150+50, legend=False)
    sns.despine(right=True)
    plt.savefig('C:/GitHub/msiam/visualizations/{}_ep{}_{}-shot_before.jpeg'.format(
        proj, episode, n_shot), dpi=1000)

    sns.relplot(x=proj_2d_after[:, 0], y=proj_2d_after[:, 1], hue=df['class'][n_way:].astype(
        int), palette="Dark2", style=df['set'][n_way:].astype(int), s=df['set'][n_way:]*150+50, legend=False)
    sns.despine(right=True)
    plt.savefig('C:/GitHub/msiam/visualizations/{}_ep{}_{}-shot_after.jpeg'.format(
        proj, episode, n_shot), dpi=1000)

    return


@torch.no_grad()
def visualize_memory_embeddings(memory: torch.Tensor, labels: torch.Tensor,
                                num_clusters: int, save_path: str,
                                epoch: int, origin: str):
    # ----------------- 3D tSNE------------------------------
    tsne = TSNE(n_components=3, verbose=1)
    tsne_proj = tsne.fit_transform(memory)

    cmap = cm.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # only plot the first 25 clusters
    num_clusters = 25
    ax.set_prop_cycle(color=[cmap(1.*i/num_clusters)
                      for i in range(num_clusters)])

    for lab in range(num_clusters):
        indices = labels == lab
        indices = np.random.permutation(indices)
        # only plot 5 samples/class
        if len(indices < 5):
            ax.scatter(tsne_proj[indices, 0],
                       tsne_proj[indices, 1],
                       tsne_proj[indices, 2],
                       label=lab,
                       alpha=0.75)
        else:
            ax.scatter(tsne_proj[indices[0:5], 0],
                       tsne_proj[indices[0:5], 1],
                       tsne_proj[indices[0:5], 2],
                       label=lab,
                       alpha=0.75)
    # plt.show()
    save_path = Path(save_path) / Path(origin)
    save_path.mkdir(parents=True, exist_ok=True)
    # save 3d interactive graph
    pickle.dump(fig, open(save_path / Path("E_"+str(epoch)+".pickle"), 'wb'))

   # ----------------- 2D tSNE------------------------------
    tsne = TSNE(n_components=2, verbose=1)
    tsne_proj = tsne.fit_transform(memory)

    cmap = cm.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # only plot the first 25 clusters
    num_clusters = 25
    ax.set_prop_cycle(color=[cmap(1.*i/num_clusters)
                      for i in range(num_clusters)])

    for lab in range(num_clusters):
        indices = labels == lab
        indices = np.random.permutation(indices)
        # only plot 5 samples/class
        if len(indices < 5):
            ax.scatter(tsne_proj[indices, 0],
                       tsne_proj[indices, 1],
                       label=lab,
                       alpha=0.75)
        else:
            ax.scatter(tsne_proj[indices[0:5], 0],
                       tsne_proj[indices[0:5], 1],
                       label=lab,
                       alpha=0.75)
    # save 2d graph
    plt.savefig(save_path / Path("E_"+str(epoch)+".png"))
