
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


def visualize_memory(memory_bank, save_path, origin, n_class=50, n_samples=30, proj="umap", epoch=0):
    print("==> Visualizing Memory Embeddings...")
    unique_label_counts = np.array(
        memory_bank.labels.unique(return_counts=True)[-1].cpu())
    if len(unique_label_counts) <= 1:
        print("Error with memory configuration")
        return
    print("--Unique Labels Counts--: {}-------\n".format(unique_label_counts))

    bank = np.array(memory_bank.bank.T.detach().cpu())
    labels = np.array(memory_bank.labels.detach().cpu())

    df_memory_bank = pd.DataFrame(bank)
    df_memory_bank['class'] = pd.Series(labels)

    # all classes
    unique_labels = np.arange(unique_label_counts.shape[0])

    # discard classes with fewer than n_samples assgined to them
    discarded_labels = [idx for idx, val in enumerate(
        unique_label_counts) if val <= n_samples]
    discarded_indices = np.argwhere(np.isin(unique_labels, discarded_labels))
    unique_labels = np.delete(unique_labels, discarded_indices)

    # randomly select n_class classes from the memory
    unique_labels = np.random.choice(unique_labels, n_class, replace=False)

    # keep only samples from selected claasses
    df_memory_bank = df_memory_bank[df_memory_bank['class'].isin(
        unique_labels)]

    # keep only n_samples per class for visualization purposes
    df_memory_bank = df_memory_bank.groupby(
        'class', group_keys=False).apply(lambda df: df.sample(n_samples))
    print(df_memory_bank.shape)

    if proj == "tsne":
        tsne = TSNE(n_components=2, verbose=0)
        proj_2d = tsne.fit_transform(df_memory_bank)
    else:
        umap = UMAP(n_components=2, init='random', random_state=0)
        proj_2d = umap.fit_transform(df_memory_bank)

    print("DBI score: {}".format(
        davies_bouldin_score(proj_2d, df_memory_bank['class'])))

    ax = sns.relplot(x=proj_2d[:, 0], y=proj_2d[:, 1], hue=df_memory_bank['class'].astype(
        int), palette="Dark2", style=df_memory_bank['class'].astype(int), s=35, legend=False, facet_kws=dict(despine=False))
    ax.set(yticklabels=[])
    ax.tick_params(left=False)
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)

    save_path = Path(save_path) / Path("memory_visualizations") / Path(origin)
    save_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path / Path("memory_state_" +
                str(epoch)+".jpeg"), dpi=1000)
    return


@torch.no_grad()
def visualize_optimal_transport(orginal_prototypes, transported_prototypes, z_query,
                                y_support, y_query, proj, episode, n_shot, save_path,
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

    print("Set DBI score before: {}".format(
        davies_bouldin_score(proj_2d_before, df['set'][:-n_way])))
    print("Set DBI score after: {}".format(
        davies_bouldin_score(proj_2d_after, df['set'][n_way:])))

    ax1 = sns.relplot(x=proj_2d_before[:, 0], y=proj_2d_before[:, 1], hue=df['class'][:-n_way].astype(
        int), palette="Dark2", style=df['set'][:-n_way].astype(int), s=df['set'][:-n_way]*150+50, legend=False, facet_kws=dict(despine=False))
    ax1.set(yticklabels=[])
    ax1.tick_params(left=False)
    ax1.set(xticklabels=[])
    ax1.tick_params(bottom=False)
    plt.savefig(Path(save_path) / Path(proj+"_ep"+str(episode) +
                "_"+str(n_shot)+"-shot_before.jpeg"), dpi=1000)

    ax2 = sns.relplot(x=proj_2d_after[:, 0], y=proj_2d_after[:, 1], hue=df['class'][n_way:].astype(
        int), palette="Dark2", style=df['set'][n_way:].astype(int), s=df['set'][n_way:]*150+50, legend=False, facet_kws=dict(despine=False))
    ax2.set(yticklabels=[])
    ax2.tick_params(left=False)
    ax2.set(xticklabels=[])
    ax2.tick_params(bottom=False)
    plt.savefig(Path(save_path) / Path(proj+"_ep"+str(episode) +
                "_"+str(n_shot)+"-shot_after.jpeg"), dpi=1000)
    return
