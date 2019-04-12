import pickle
import os, sys
import time
import numpy as np
from abbrrep_class import AbbrRep
import argparse
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import Counter

device = 'cpu'
import fastText
wordEmbed_model = os.path.join(os.path.abspath(os.path.join('./', os.pardir)), "word_embeddings.bin")
FASTTEXT_MODEL = fastText.load_model(wordEmbed_model)

results = {}
results["mimic"] = {}
results["casi"] = {}
childparent_dir = '/Users/Marta/80k_abbreviations/preprocess_pipeline/umls_child2parent_20190304.pickle'

p_in = open(childparent_dir, 'rb')
child2parent_full = pickle.load(p_in)
p_in.close()
parent2child = {}

child2parent_full = child2parent_full['child2parent']
child2parent = {}
#for key in mimic:
 #   child2parent[key] = set()
labels = set()

keys_to_aviod = ["C1140093", "C0872087", "C2720507", "C0587755", "C0449234", "C0243095"]
pars_to_avoid = ["MTHMST", "CSP"]
for key in child2parent_full:
    if key in keys_to_aviod:
        continue
    for par in child2parent_full[key]:
        if par[0] in keys_to_aviod:
            continue
        if par[2] in pars_to_avoid:
            continue
        if par[2] == "SNOMEDCT_US" or par[2] == "MTH":
            try:
                child2parent[key].add(par[0])
            except:
                child2parent[key] = set()
                child2parent[key].add(par[0])
            if par[0] not in child2parent:
                child2parent[par[0]] = set()

            # add child as value of parent key
            try:
                parent2child[par[0]].add(key)
            except:
                parent2child[par[0]] = set()

def get_sparseMatrix(abbr):
    src_dir = "/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/sparse_matrices_20190406"
    concept_matrix_file = abbr + "_umls_ancestor_sparseMatrix_20190406_full.pickle"
    p_in = open(os.path.join(src_dir, concept_matrix_file), 'rb')
    umls_rel = pickle.load(p_in)
    p_in.close()

    matrix_hierarchy = umls_rel["matrix_hierarchy"]
    word2idx = umls_rel["word2idx"]
    idx2word = umls_rel["idx2word"]
    all_ancetors = umls_rel["all_concepts"]

    return matrix_hierarchy, word2idx, idx2word, all_ancetors

def load_data(opt, abbr, cui2meta, meta2cui, ns, augment=False, par2child=None, child2par=None):

    from sklearn.utils import shuffle
    import math
    root = "w5_ns1000_g_20190408"
    src_dir = "/Users/Marta/80k_abbreviations/abbr_dataset_mimic_casi" + root + "/"
    fname = abbr + "_mimic_casi_" + root + ".pickle"
    p_in = open(os.path.join(src_dir, fname), "rb")
    data = pickle.load(p_in)
    p_in.close()

    if augment:
        src_dir = "/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/concept_embedding_dataset_w5_ns1000_g_20190408/"
        fname = abbr + "_embedding_dataset_rs_gpars_20190408.pickle"
        p_in = open(os.path.join(src_dir, fname), "rb")
        ancestor_data = pickle.load(p_in)
        p_in.close()

    # unlabelled_data = data['mimic_abbr'][abbr]
    casi_data = data['casi_abbr']
    casi_test = []
    for subkey in casi_data:
        casi_test.extend(casi_data[subkey])
    casi_test = shuffle(casi_test, random_state=42)
    full_data = []
    # for subkey in data["mimic_rs"]:
    num_samples = 0
    for i in data["mimic_rs"]:
        num_samples = len(data["mimic_rs"][i][0].onehot)-1
        break
    for subkey in range(num_samples):
        curr_data = []
        try:
            curr_data = data["mimic_rs"][subkey]
        except:

            meta_id = subkey
            if augment:
                cui = meta2cui[abbr][meta_id]
                if cui not in child2par:
                    continue
                par_data = []
                for par in child2parent[cui]:
                    if par not in child2par and len(par2child[par]) == 1:
                        try:
                            for item in ancestor_data[par]:
                                item.label = subkey
                                par_data.append(item)
                        except KeyError:
                            continue
                par_data = shuffle(par_data, random_state=42)
                curr_data.extend(par_data[:min(int(int(ns)/5), len(par_data))])
        curr_data = shuffle(curr_data, random_state=42)
        full_data.extend(curr_data[:min(int(ns), len(curr_data))])
    full_data = shuffle(full_data, random_state=42)
    split = math.floor(len(full_data) * 0.7)
    mimic_train = full_data[:split]
    mimic_test = full_data[split:]
    for i in mimic_test:
        print(i.label)

    return abbr, data, casi_test, mimic_train, mimic_test

def get_abbr_concepts(abbr, cui2meta, meta2name, limit=False):
    ancestor_hierarchy, word2idx, idx2word, all_ancestors = get_sparseMatrix(abbr)

    return ancestor_hierarchy, word2idx, idx2word, all_ancestors

def knn(abbr, data):
    X = []
    labels = []
    for item in data:
        X.append(item.embedding[0])
        labels.append(item.label)
    return X, labels

def get_NN(abbr, X, labels, query, name2meta, meta2name, num_neighbours=5):
    #dist = np.linalg.norm((X - query), axis=1)
    dist = np.linalg.norm((X - query), axis=1)
    x = dist.argsort()[:num_neighbours]
    print(x)
    poss_labels = []
    for i in x:
        print(labels[i])
        poss_labels.append(labels[i])
    #print(np.argmin(dist))
    # print(labels[np.argmin(dist)])
    # dist = np.dot(X, query.T).reshape((1, -1))
    # x = dist.argsort()[0][-num_neighbours:][::-1]
    # print(x)
    # poss_labels = []
    # for i in x:
    #     print(labels[i])
    #     if augment:
    #         pass
    #     poss_labels.append(labels[i])
    # print(np.argmax(dist))
    # print(labels[np.argmax(dist)])
    # print(dist)

    #dist = np.sum((np.array(X) - np.array(query)) ** 2, axis=1)
    #ind = np.argpartition(dist, -100)[-100:]
    # poss_labels = []
    # for i in ind:
    #     poss_labels.append(labels[i])

    possible_ids = []
    print(poss_labels)
    for i in poss_labels:
        if i in meta2name[abbr]:
            possible_ids.append(i)
        else:
            possible_ids.append(name2meta[abbr][i])
    print(possible_ids)
    x = Counter(possible_ids)
    print(x.most_common())
    maj_vote = x.most_common()[0][0]
    print(maj_vote)
    return maj_vote

def make_map(abbr, ancestor_hierarchy, word2idx, idx2word, all_ancestors, cui2meta):

    par2child = {}
    child2par = {}
    possible_exp = list(cui2meta[abbr].keys())
    for row in range(ancestor_hierarchy.shape[0]):
        for col in ancestor_hierarchy.rows[row]:
            child = idx2word[row]
            if child in possible_exp:
                par = idx2word[col]
                if par == child:
                    continue
                if par not in par2child:
                    par2child[par] = set()
                par2child[par].add(child)
                if child not in child2par:
                    child2par[child] = set()
                child2par[child].add(par)
    return par2child, child2par


def label_data(opt, abbr, augment=True):
    expansion_list = []

    dir = "/Users/Marta/80k_abbreviations/allacronyms"
    n = open(os.path.join(dir, "allacronyms_cui2meta_20190402_NEW.pickle"), 'rb')
    cui2meta = pickle.load(n)
    n.close()

    m = open(os.path.join(dir, "allacronyms_meta2cui_20190402_NEW.pickle"), 'rb')
    meta2cui = pickle.load(m)
    m.close()

    o = open(os.path.join(dir, "allacronyms_meta2name_20190402_NEW.pickle"), 'rb')
    meta2name = pickle.load(o)
    o.close()

    q = open(os.path.join(dir, "allacronyms_name2meta_20190402_NEW.pickle"), 'rb')
    name2meta = pickle.load(q)
    q.close()

    ancestor_hierarchy, word2idx, idx2word, all_ancestors = get_abbr_concepts(abbr, cui2meta, meta2name, limit=opt.limit)
    if augment:
        par2child, child2par = make_map(abbr, ancestor_hierarchy, word2idx, idx2word, all_ancestors, cui2meta)
        abbr, data, unlabelled_data, mimic_train, mimic_test = load_data(opt, abbr, cui2meta, meta2cui, 100, augment, par2child, child2par)
    else:
        abbr, data, unlabelled_data, mimic_train, mimic_test = load_data(opt, abbr, cui2meta, meta2cui, 100)

    concepts = word2idx
    score = 0
    total = 0

    all_training_set, conept_labels = knn(abbr, mimic_train)

    for query in unlabelled_data:
        print("***************NEW QUERY******************")
        print(query.features_left, query.features_right)
        assigned_concept = get_NN(abbr, all_training_set, conept_labels, query.embedding, name2meta, meta2name)
        print(assigned_concept)
        try:
            assigned_meta_id = name2meta[abbr][assigned_concept]
        except KeyError:
            assigned_meta_id = assigned_concept
        ground_truth = name2meta[abbr][query.label]
        ground_truth_name = query.label
        print(meta2name[abbr][assigned_meta_id], ground_truth_name)
        labels.add(ground_truth_name)
        print(assigned_meta_id, ground_truth)
        if assigned_meta_id == ground_truth:
            score += 1
        print(score)
        total += 1
    print(abbr + " from " + "casi" + ": CORRECT=" + str(score) + " TOTAL=" + str(total))
    results["casi"][abbr] = [score, total]
    print(labels)
    print(meta2name[abbr])



    score = 0
    total = 0
    for query in mimic_test:
        print("***************NEW QUERY******************")
        print(query.features_left, query.features_right)
        assigned_concept = get_NN(abbr, all_training_set, conept_labels, query.embedding, name2meta, meta2name)

        assigned_meta_id = assigned_concept

        try:
            ground_truth = name2meta[abbr][query.label]
            ground_truth_name = query.label
        except KeyError:
            ground_truth = query.label
            ground_truth_name = meta2name[abbr][query.label]
        print(assigned_concept)
        print(assigned_meta_id)
        print(meta2name[abbr][assigned_meta_id], ground_truth_name)

        print(assigned_meta_id, ground_truth)
        if assigned_meta_id == ground_truth:
            score += 1
        total += 1
    print(abbr + " from " + "mimic" + ": CORRECT=" + str(score) + " TOTAL=" + str(total))
    results["mimic"][abbr] = [score, total]


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-abbr')
    parser.add_argument('-limit', action="store_true")

    opt = parser.parse_args()

    abbr_dict = ["ac", "asa", "av", "avr", "bal", "ca", "cea", "cvp", "dc", "dm", "dt", "er", "im", "ir", "ivf", "le",
                "op",  "otc", "pa", "pac", "pcp", "pda", "pe", "pr", "ra", "rt", "sbp", "sma"]
    # abbr_dict = ["bmp", "cvp", "fsh", "mom", "ac", "ald", "ama", "asa", "av", "avr", "bal", "bm", "ca", "cea", "cr",
    #             "cva", "cvs", "dc", "dip", "dm", "dt", "er", "et", "gt", "im", "ir", "it", "ivf", "le", "mp", "mr",
    #             "ms", "na", "np", "op", "or", "otc", "pa", "pac", "pcp", "pd", "pda", "pe", "pr", "pt", "ra", "rt", "sbp", "sma", "vad"]
    for abbr in abbr_dict:
        label_data(opt, abbr)

    micro_acc_casi = 0
    total_casi = 0
    score_casi = 0

    micro_acc_mimic = 0
    total_mimic = 0
    score_mimic = 0

    for abbr in results['casi']:
        correct = results['casi'][abbr][0]
        total = results['casi'][abbr][1]
        micro_acc_casi += correct/total
        total_casi += total
        score_casi += correct
    micro_acc_casi /= len(results['casi'])
    macro_acc_casi = score_casi/total_casi

    for abbr in results['mimic']:
        correct = results['mimic'][abbr][0]
        total = results['mimic'][abbr][1]
        micro_acc_mimic += correct/total
        total_mimic += total
        score_mimic += correct
    micro_acc_mimic /= len(results['mimic'])
    macro_acc_mimic = score_mimic/total_mimic
    print("MICRO ACC CASI: " + str(micro_acc_casi))
    print("MACRO ACC CASI: " + str(macro_acc_casi))

    print("MICRO ACC MIMIC: " + str(micro_acc_mimic))
    print("MACRO ACC MIMIC: " + str(macro_acc_mimic))

    print("Done labelling unlabelled data!!! \U0001F388 \U0001F33B")

if __name__ == "__main__":
   main()