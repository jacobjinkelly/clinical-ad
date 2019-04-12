
import os
import pickle
import os, sys
from collections import Counter
import matplotlib.pyplot as plt
from random import shuffle
import random
random.seed(16)
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns; sns.set()


def plot_ce(abbr):
    root = "w5_ns1000_g_20190402"
    dest_dir = "/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/concept_embedding_dataset_" + root
    file = abbr + "_embedding_dataset_rs_gpars_" + root + ".pickle"
    p_in = open(os.path.join(dest_dir, file), "rb")
    data = pickle.load(p_in)
    p_in.close()
    X = []
    labels = []
    for key in data:
        temp = []
        for i in data[key]:
            temp.append(i.embedding[0])
        shuffle(temp)
        num_samples = min(len(temp), 10)
        for i in range(num_samples):
            X.append(temp[i])
            labels.append(key)

    tsne = TSNE(n_components=2, verbose=0, perplexity=10, n_iter=1000, random_state=42)
    results = tsne.fit_transform(X)
    print(results.shape)
    # pca = PCA(n_components=2)
    # results = pca.fit_transform(X)
    vis_x = results[:, 0]
    vis_y = results[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(vis_x, vis_y)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (vis_x[i], vis_y[i]))
    plt.show()


def plot_se(abbr):
    root = "w5_ns1000_g_20190408"
    dest_dir = "/Users/Marta/80k_abbreviations/abbr_dataset_mimic_casi" + root
    file = abbr + "_mimic_casi_" + root + ".pickle"
    p_in = open(os.path.join(dest_dir, file), "rb")
    data = pickle.load(p_in)
    p_in.close()
    X = []
    labels = []

    dir = "/Users/Marta/80k_abbreviations/allacronyms"
    o = open(os.path.join(dir, "allacronyms_meta2name_20190402_NEW.pickle"), 'rb')
    meta2name = pickle.load(o)
    o.close()

    colours =[]
    for key in data["mimic_rs"]:
        temp = []
        if list(meta2name[abbr][key])[0] not in ['lower extremity', 'lymphedema', "life expectancy",
                                                 "left ear", "left eye", "leukocyte esterase"]:
            continue
        for i in data["mimic_rs"][key]:
            temp.append(i.embedding[0])
        shuffle(temp)
        num_samples = min(len(temp), 100)
        for i in range(num_samples):
            X.append(temp[i])
            labels.append(list(meta2name[abbr][key])[0])




    tsne = TSNE(n_components=2, verbose=0, perplexity=10, n_iter=1000, random_state=42)
    results = tsne.fit_transform(X)
    print(results.shape)
    # pca = PCA(n_components=2)
    # results = pca.fit_transform(X)
    vis_x = results[:, 0]
    vis_y = results[:, 1]

    # fig, ax = plt.subplots()
    # ax.scatter(vis_x, vis_y)
    # ax.plot(colours=colours)
    # for i, txt in enumerate(labels):
    #     ax.annotate(txt, (vis_x[i], vis_y[i]),color='xkcd:lavender')

    d = {'Sense': labels, 'Dataset': labels}
    df = pd.DataFrame(data=d)
    sns.scatterplot(x=vis_x, y=vis_y, hue=df["Sense"], palette="husl",legend="full")
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., fontsize=16)
    plt.show()
plot_se("le")