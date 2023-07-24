
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


def visualize_memory_batch(batch_features, batch_labels, centers, save_path, proj="umap", origin="student", idx=0):

    # print(batch_features.size())
    # print(len(batch_labels))
    # print(centers.size())
    #################################################
    unique_label_counts = batch_labels.unique(return_counts=True)

    # keep classes with only 3 elements (3-shot)
    selected_indices = [idx for idx, val in enumerate(
        np.array(unique_label_counts[-1].cpu())) if val == 3]
    selected_labels = np.array(unique_label_counts[0].cpu())[selected_indices]

    # from these only keep at random 3 (3-way)
    selected_labels = np.random.choice(selected_labels, 3, replace=False)

    # keep only batch_featurees corresponding to selected_labels
    indices = sum(batch_labels == i for i in selected_labels).bool().squeeze(0)
    bf = batch_features[indices, :]

    # similarity matrix and top_k clusters between batch features & memory prototypes
    z_center_similarity_matrix = torch.einsum(
        "nd,md->nm", bf, centers)
    # find top cluster centers for each batch embedding
    _, topk_clusters = torch.topk(z_center_similarity_matrix, 1, dim=1)

    if topk_clusters.unique().shape[0] - len(selected_labels) >= 2:
        return
    print("-------------------{}--{}----------------------\n".format(origin, idx))
    print(len(selected_labels))
    print(topk_clusters.unique().shape[0])

    ############################################################
    df_batch = pd.DataFrame(batch_features.detach().cpu().numpy())
    df_batch['Class'] = pd.Series(batch_labels)
    df_batch['Embedding Origin'] = pd.Series(np.zeros(len(df_batch.index)))

    df_centers = pd.DataFrame(centers.cpu().numpy())
    df_centers['Class'] = pd.Series(300 * np.ones(len(df_centers.index)))
    df_centers['Embedding Origin'] = pd.Series(np.ones(len(df_centers.index)))

    ########################################
    # keep in dataframe only embeddings from selected classes
    df_batch = df_batch[df_batch['Class'].isin(selected_labels)]

    top_clusters = topk_clusters.unique().cpu().numpy()
    random_clusters = np.random.choice(
        np.arange(centers.shape[0]), 30, replace=False)
    kept_clusters = np.concatenate((top_clusters, random_clusters), axis=0)

    df_centers = df_centers[df_centers.index.isin(
        kept_clusters)]
    for i in range(len(top_clusters)):
        if i < 3:
            df_centers.loc[df_centers.index ==
                           top_clusters[i], "Class"] = selected_labels[i]
        else:
            df_centers.loc[df_centers.index ==
                           top_clusters[i], "Class"] = selected_labels[0]
    # keep only topk clsuters from memory prototypes

    df_centers = pd.concat([df_centers, df_centers, df_centers, df_centers,
                           df_centers, df_centers, df_centers, df_centers,
                           df_centers, df_centers, df_centers, df_centers,
                           df_centers, df_centers, df_centers, df_centers,
                           df_centers, df_centers, df_centers, df_centers,
                           df_centers, df_centers, df_centers, df_centers,
                           df_centers, df_centers, df_centers, df_centers,
                           df_centers, df_centers, df_centers, df_centers])
    df = pd.concat([df_batch, df_centers])
    df = df.reset_index(drop=True)
    features = df.iloc[:, :-2]

    print(df_batch.shape)
    print(df_centers.shape)
    print(df["Class"].unique())
    df['Class'] = df['Class'].astype(str)
    df['Class'] = df['Class'].replace(
        str(float(selected_labels[0])), "Class I")
    df['Class'] = df['Class'].replace(
        str(float(selected_labels[1])), "Class II")
    df['Class'] = df['Class'].replace(
        str(float(selected_labels[2])), "Class III")
    df['Class'] = df['Class'].replace(str(300.0), "Other Memory Prototypes")
    df['Size'] = df['Embedding Origin']
    df['Embedding Origin'] = df['Embedding Origin'].astype(str)
    df['Embedding Origin'] = df['Embedding Origin'].replace(
        str(0.0), "Episode")
    df['Embedding Origin'] = df['Embedding Origin'].replace(
        str(1.0), "Memory Prototype")

    if proj == "tsne":
        tsne = TSNE(n_components=2, verbose=0, perplexity=5)
        proj_2d = tsne.fit_transform(features)
    else:
        umap = UMAP(n_components=2, init='random', random_state=0)
        proj_2d = umap.fit_transform(features)

    sns.set_context("paper")
    sns.set_style("darkgrid")
    ax = sns.relplot(x=proj_2d[:, 0], y=proj_2d[:, 1], hue=df['Class'], palette=[
                     "C0", "C1", "C2", "C7"], style=df['Embedding Origin'], s=140+50*(1-df['Size']), legend=True, facet_kws=dict(despine=False), markers=[".", "*"])
    ax.set(yticklabels=[])
    ax.tick_params(left=False)
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)
    plt.savefig(Path(save_path) /
                Path("{}_{}_{}.jpeg".format(proj, origin, idx)), dpi=300)
    exit()
    return


def visualize_memory(memory_bank, save_path, origin, n_class=25, n_samples=30, proj="umap", epoch=0):
    print("==> Visualizing Memory Embeddings...")
    print(memory_bank.labels.unique(return_counts=True)[-1])
    unique_label_counts = np.array(
        memory_bank.labels.unique(return_counts=True)[-1].cpu())
    unique_labels = np.array(
        memory_bank.labels.unique(return_counts=True)[0].cpu())
    if len(unique_label_counts) <= 1:
        print("Error with memory configuration")
        return

    bank = np.array(memory_bank.bank.T.detach().cpu())
    labels = np.array(memory_bank.labels.detach().cpu())

    df_memory_bank = pd.DataFrame(bank)
    df_memory_bank['class'] = pd.Series(labels)

    # discard classes with fewer than n_samples assgined to them
    discarded_labels = unique_labels[[idx for idx, val in enumerate(
        unique_label_counts) if val <= n_samples]]
    print("Discarded: {}".format(np.sort(discarded_labels)))
    discarded_indices = np.argwhere(np.isin(unique_labels, discarded_labels))
    unique_labels = np.delete(unique_labels, discarded_indices)
    print("Unique: {}".format(np.sort(unique_labels)))

    if len(unique_labels) < n_class:
        print("Error with memory configuration")
        return

    # randomly select n_class classes from the memory
    unique_labels = np.random.choice(unique_labels, n_class, replace=False)
    print("Unique after Sampling: {}".format(np.sort(unique_labels)))

    # keep only samples from selected claasses
    df_memory_bank = df_memory_bank[~df_memory_bank['class'].isin(
        discarded_labels)]
    # keep only n_samples per class for visualization purposes
    df_memory_bank = df_memory_bank[df_memory_bank['class'].isin(
        unique_labels)]

    # keep only n_samples per class for visualization purposes
    print("Df Unique classes: {}".format(
        np.sort(df_memory_bank['class'].unique())))

    try:
        df_memory_bank = df_memory_bank.groupby(
            'class', group_keys=False).apply(lambda df: df.sample(n_samples))
    except:
        print("-----------------MEMORY ERROR----------------")
        return
    # reset index
    df_memory_bank = df_memory_bank.reset_index(drop=True)

    if proj == "tsne":
        tsne = TSNE(n_components=2, verbose=0)
        proj_2d = tsne.fit_transform(df_memory_bank)
    else:
        umap = UMAP(n_components=2, init='random', random_state=0)
        proj_2d = umap.fit_transform(df_memory_bank)

    print("DBI score: {}".format(
        davies_bouldin_score(proj_2d, df_memory_bank['class'])))

    if n_class == 25:
        # adjust marker sizes in default 25-class visualization setting
        sizes = np.repeat(
            np.array([200, 200, 200, 200, 200, 270, 200, 270, 200, 230, 200, 230, 200, 245, 200, 200, 200, 200, 200, 270, 200, 270, 200, 230, 200]), n_samples)
        df_memory_bank['size'] = pd.Series(sizes)
    else:
        sizes = 240

    sns.set_context("paper")
    sns.set_style("ticks")
    ax = sns.relplot(x=proj_2d[:, 0], y=proj_2d[:, 1], hue=df_memory_bank['class'].astype(
        int), palette="Dark2", style=df_memory_bank['class'].astype(int), s=sizes, legend=False, facet_kws=dict(despine=False))
    # ax = sns.kdeplot(x=proj_2d[:, 0], y=proj_2d[:, 1],
    #                  hue=df_memory_bank['class'].astype(int), palette="Dark2", legend=False, fill=True)
    ax.set(yticklabels=[])
    # ax.tick_params(left=False)
    ax.set(xticklabels=[])
    # ax.tick_params(bottom=False)

    save_path = Path(save_path) / Path("memory_visualizations") / Path(origin)
    save_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path / Path("memory_state_" +
                str(epoch)+".jpeg"), dpi=300)
    return


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
        tsne = TSNE(n_components=2, verbose=0, perplexity=5)
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
    ##############
    proj_2d_query = proj_2d[n_way:-n_way, :]
    proj_2d_sup_before = proj_2d[:n_way]
    proj_2d_sup_after = proj_2d[-n_way:]

    sns.set_context("paper")
    # white, dark, ticks
    sns.set_style("white")
    ax = sns.kdeplot(x=proj_2d_query[:, 0], y=proj_2d_query[:, 1],
                     hue=df['class'][n_way:-n_way].astype(int), palette="Pastel2", legend=False)
    ax = sns.scatterplot(data=proj_2d_query, x=proj_2d_query[:, 0], y=proj_2d_query[:, 1],
                         hue=df['class'][n_way:-n_way].astype(int), s=130, style=df['set'][n_way:-n_way].astype(int), palette="Dark2", markers=["."], legend=False)
    ax = sns.scatterplot(data=proj_2d_sup_after, x=proj_2d_sup_after[:, 0], y=proj_2d_sup_after[:, 1],
                         hue=df['class'][-n_way:].astype(int), s=800, style=df['set'][-n_way:].astype(int), palette="Dark2", markers=["*"], legend=False)

    ax.set(yticklabels=[])
    ax.tick_params(left=False)
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)
    plt.savefig(Path(save_path) / Path(proj+"_ep"+str(episode) +
                "_"+str(n_shot)+"-shot_after.jpeg"), dpi=300)
#######################
    plt.clf()
    ax2 = sns.kdeplot(x=proj_2d_query[:, 0], y=proj_2d_query[:, 1],
                      hue=df['class'][n_way:-n_way].astype(int), palette="Pastel2", legend=False)
    ax2 = sns.scatterplot(data=proj_2d_query, x=proj_2d_query[:, 0], y=proj_2d_query[:, 1],
                          hue=df['class'][n_way:-n_way].astype(int), s=130, style=df['set'][n_way:-n_way].astype(int), palette="Dark2", markers=["."], legend=False)
    ax2 = sns.scatterplot(data=proj_2d_sup_before, x=proj_2d_sup_before[:, 0], y=proj_2d_sup_before[:, 1],
                          hue=df['class'][:n_way].astype(int), s=800, style=df['set'][:n_way].astype(int), palette="Dark2", markers=["*"], legend=False)
    ax2.set(yticklabels=[])
    ax2.tick_params(left=False)
    ax2.set(xticklabels=[])
    ax2.tick_params(bottom=False)
    plt.savefig(Path(save_path) / Path(proj+"_ep"+str(episode) +
                "_"+str(n_shot)+"-shot_before.jpeg"), dpi=300)
    plt.clf()

    return
