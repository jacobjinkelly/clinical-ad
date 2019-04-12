import build_sparseMatrix
import os
import pickle
from tqdm import tqdm
from get_card_exp import get_card_expansions
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns; sns.set()
import torch
import pandas as pd

import matplotlib.pyplot as plt
#src_dir = "/Volumes/terminator/hpf/concept_embed_model_w5/"


def get_ancestoryHierarchy(word2idx, child2par, input):
    ancestors = set()
    ancestors.add(word2idx[input])
    parents_to_find = set(child2par[input])
    seen_parents = set()
    while len(parents_to_find) > 0:
        curr_parent = parents_to_find.pop()
        seen_parents.add(curr_parent)
        ancestors.add(word2idx[curr_parent])
        if len(child2par[curr_parent]) > 0:
            for new_par in child2par[curr_parent]:
                if new_par not in seen_parents:
                    parents_to_find.add(new_par)
    return list(ancestors)


def get_sparseMatrix(abbr):
    #p_in = open("umls_mimic_sparseMatrix_word2idx_20190315.pickle", 'rb')
    #p_in = open("umls_mimic_sparseMatrix_word2idx_20190315.pickle", 'rb')
    src_dir = "/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/sparse_matrices_20190402_old"
    #abbr = "all"
    #concept_matrix_file = abbr + "_umls_ancestor_sparseMatrix_20190322.pickle"
    concept_matrix_file = "dm_umls_ancestor_sparseMatrix_20190402_full.pickle"
    #p_in = open("sparse_matrix_test.pickle", 'rb')
    p_in = open(os.path.join(src_dir, concept_matrix_file), 'rb')
    umls_rel = pickle.load(p_in)
    p_in.close()

    matrix_hierarchy = umls_rel["matrix_hierarchy"]
    word2idx = umls_rel["word2idx"]
    idx2word = umls_rel["idx2word"]
    all_ancetors = umls_rel["all_concepts"]

    return matrix_hierarchy, word2idx, idx2word, all_ancetors


def sum_ancestors(abbr):
    src_dir = "/Volumes/terminator/hpf/20190322_modelcheckpoints/models_to_label/"
    #concept_matrix_file = abbr + "_20190323_CONCEPTMODEL.pickle"
    #concept_matrix_file ="all_checkpoint150.pickle"
    #concept_matrix_file = "checkpoint50_20190323_0-53_CONCEPTMODEL.pickle"
    #p_in = open(os.path.join(src_dir, concept_matrix_file), "rb")

    #p = pickle.load(p_in)
    #p_in.close()
    #concept_matrix_file = "all_checkpoint150.pickle"
    ancestor_hierarchy, word2idx, idx2word, all_ancestors = get_sparseMatrix(abbr)
    concept_matrix_file = abbr+'_20190402_epoch50_w5_g.pth.tar'
    p = torch.load(os.path.join(src_dir, concept_matrix_file), map_location='cpu')
    concept_embedding_weights = p['embed1.weight']


    concept_embed_dict = {}
    mimic = build_sparseMatrix.get_concepts_mimic()
    child2par, _ = build_sparseMatrix.get_concepts_umls(mimic)
    #concept_embedding_weights = p["weights"]

    #word2idx = p["word2idx"]

    ancestors_dict = {}
    for concept in tqdm(all_ancestors):
        try:
            weight = word2idx[concept]
        except:
            continue
        ancestors = ancestor_hierarchy.rows[weight]
        if len(ancestors) == 0:
            print("EMPTY CONCEPT!")
            print(concept)
        concept_embed_dict[concept] = concept_embedding_weights[ancestors].sum(0)


        if concept not in ancestors_dict:
            ancestors_dict[concept] = []
        for ancestor in ancestors:
            cui = idx2word[ancestor]
            ancestors_dict[concept].append(cui)

    '''
    target_dir = "/Users/Marta/80k_abbreviations/create_samples"
    f = "20190327_MASTER150e.pickle"
    p_out = open(os.path.join(target_dir,f), 'wb')
    pickle.dump(ancestors_dict, p_out)
    p_out.close()
    '''

    dir = "/Users/Marta/80k_abbreviations/allacronyms"
    #n = open(os.path.join(dir, "allacronyms_cui2meta_20190318.pickle"), 'rb')
    n = open(os.path.join(dir, "allacronyms_cui2meta_20190402_NEW.pickle"), 'rb')

    cui2meta = pickle.load(n)
    n.close()

    dir = "/Users/Marta/80k_abbreviations/allacronyms"
    #m = open(os.path.join(dir, "allacronyms_meta2name_20190318.pickle"), 'rb')

    m = open(os.path.join(dir, "allacronyms_meta2name_20190402_NEW.pickle"), 'rb')
    meta2name = pickle.load(m)
    m.close()

    dir = "/Users/Marta/80k_abbreviations/preprocess_pipeline"
    #i = open(os.path.join(dir, "umls_id2name_20190310.pickle"), 'rb')
    i = open(os.path.join(dir, "umls_id2name_20190402.pickle"), 'rb')
    cui2name = pickle.load(i)
    i.close()


    all_concepts = list(cui2meta[abbr].keys()) # Get all CUI codes

    matrix_hierarchy, word2idx, idx2word, all_ancetors = get_sparseMatrix(abbr)

    print("concepts to plot")
    concepts_to_plot = []
    labels = []
    card_exp = get_card_expansions()
    for concept in all_concepts:
        key = cui2meta[abbr][concept]
        #if key not in card_exp[abbr]:
         #   continue
        print(concept)
        if concept in concept_embed_dict:
            concepts_to_plot.append(concept)
            labels.append(list(meta2name[abbr][cui2meta[abbr][concept]])[0])
            ancestors = matrix_hierarchy.rows[word2idx[concept]]
            print("ANCESTORS")
            if concept == "C0011603" or concept == "C0232262":
                print(concept)
                print(ancestors)
            for i in ancestors:
                if idx2word[i] in concept_embed_dict:
                    if idx2word[i] not in concepts_to_plot:
                        concepts_to_plot.append(idx2word[i])
                        labels.append(idx2word[i])
            print()
        else:
            print("NOT IN CONCEPT EMBED DICT")
            print(concept)


    #X = np.zeros((len(concepts_to_plot), 200))
    X = np.zeros((len(concepts_to_plot), 200))
    #X = np.zeros((len(concepts_to_plot), 1024))
    i = 0
    for concept in concepts_to_plot:
        if concept == "C0011603":
            key = i
        X[i, :] = concept_embed_dict[concept]
        i += 1

    #print(np.sum((concept_embed_dict["C0018808"] - concept_embed_dict["C0018799"]) ** 2))
    #print(np.sum((concept_embed_dict["C0018808"] - concept_embed_dict["C0011603"]) ** 2))

    tsne = TSNE(n_components=2, verbose=0, perplexity=10, n_iter=1000, random_state=42)
    results = tsne.fit_transform(X)
    print(X.shape)
    print(results.shape)
    #pca = PCA(n_components=2)
    #results = pca.fit_transform(X)
    vis_x = results[:, 0]
    vis_y = results[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(vis_x, vis_y)


    for i, txt in enumerate(labels):
        ax.annotate(txt, (vis_x[i], vis_y[i]))
    plt.show()



    return concept_embed_dict

sum_ancestors("dm")