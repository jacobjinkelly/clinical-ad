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

class ConceptEmbedModel(nn.Module):
    def __init__(self, a, embedding_dim, pretrained_weight, ancestor_hierarchy, concepts):
        super().__init__()
        self.embed1 = nn.Embedding(a, embedding_dim)
        self.concepts = concepts
        self.embed1.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.ancestor_hierarchy = ancestor_hierarchy
        self.H = Variable(torch.randn(len(self.concepts), 200, dtype=torch.double) * 1e-9, requires_grad=False)

    def forward(self, x):
        local_H = torch.zeros(len(self.concepts), 200, dtype=torch.double, requires_grad=False)
        for i in range(len(self.concepts)):
            running_ancestor_embeds = self.embed1(torch.tensor(self.ancestor_hierarchy.rows[i]).to(device))
            running_ancestor_embeds_sum = running_ancestor_embeds.sum(0)
            local_H[i, :] = running_ancestor_embeds_sum

        concept_vector = torch.mm(local_H.to(device), torch.t(torch.tensor(x)).to(device))  # C * 200 x 200 * len(x)
        m = nn.Softmax(dim=1)
        x = torch.t(concept_vector)
        output = m(x)
        return output

def get_sparseMatrix(abbr):
    src_dir = "/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/sparse_matrices_20190402"
    concept_matrix_file = abbr + "_umls_ancestor_sparseMatrix_20190402_full.pickle"
    p_in = open(os.path.join(src_dir, concept_matrix_file), 'rb')
    umls_rel = pickle.load(p_in)
    p_in.close()

    matrix_hierarchy = umls_rel["matrix_hierarchy"]
    word2idx = umls_rel["word2idx"]
    idx2word = umls_rel["idx2word"]
    all_ancetors = umls_rel["all_concepts"]

    return matrix_hierarchy, word2idx, idx2word, all_ancetors

def load_data(opt):
    from sklearn.utils import shuffle
    import math

    root = "w5_ns1000_g_20190408"
    src_dir = "/Users/Marta/80k_abbreviations/abbr_dataset_mimic_casi" + root + "/"
    fname = opt.abbr + "_mimic_casi_" + root + ".pickle"
    p_in = open(os.path.join(src_dir, fname), "rb")
    data = pickle.load(p_in)
    p_in.close()

    abbr = opt.abbr
    # unlabelled_data = data['mimic_abbr'][abbr]
    casi_data = data['casi_abbr']
    casi_test = []
    for subkey in casi_data:
        casi_test.extend(casi_data[subkey])
    casi_test = shuffle(casi_test, random_state=42)

    full_data = []
    for subkey in data["mimic_rs"]:
        curr_data = data["mimic_rs"][subkey]
        curr_data = shuffle(curr_data, random_state=42)
        full_data.extend(curr_data[:min(int(100), len(data["mimic_rs"][subkey]))])
    full_data = shuffle(full_data, random_state=42)
    split = math.floor(len(full_data) * 0.7)
    mimic_test = full_data[split:]

    dir = "/Users/Marta/80k_abbreviations/allacronyms"
    #n = open(os.path.join(dir,"allacronyms_cui2meta_20190318.pickle"), 'rb')
    n = open(os.path.join(dir, "allacronyms_cui2meta_20190402_NEW.pickle"), 'rb')
    cui2meta = pickle.load(n)
    n.close()

    #m = open(os.path.join(dir, "allacronyms_meta2cui_20190318.pickle"), 'rb')
    m = open(os.path.join(dir, "allacronyms_meta2cui_20190402_NEW.pickle"), 'rb')
    meta2cui = pickle.load(m)
    m.close()

    #o = open(os.path.join(dir, "allacronyms_meta2name_20190318.pickle"), 'rb')
    o = open(os.path.join(dir, "allacronyms_meta2name_20190402_NEW.pickle"), 'rb')
    meta2name = pickle.load(o)
    o.close()

    q = open(os.path.join(dir, "allacronyms_name2meta_20190402_NEW.pickle"), 'rb')
    name2meta = pickle.load(q)
    q.close()


    return abbr, data, casi_test, mimic_test, cui2meta, meta2cui, meta2name, name2meta

def get_abbr_concepts(abbr, cui2meta, meta2name, limit=False):
    ancestor_hierarchy, word2idx, idx2word, all_ancestors = get_sparseMatrix(abbr)

    return ancestor_hierarchy, word2idx, idx2word, all_ancestors


def label_data(opt):
    expansion_list = []
    abbr, data, unlabelled_data, mimic_test, cui2meta, meta2cui, meta2name, name2meta = load_data(opt)

    ancestor_hierarchy, word2idx, idx2word, all_ancestors = get_abbr_concepts(abbr, cui2meta, meta2name, limit=opt.limit)
    concepts = word2idx
    pretrained_weight = np.zeros((len(concepts), 200))

    model = ConceptEmbedModel(len(concepts), 200, pretrained_weight, ancestor_hierarchy, concepts)
    src_dir = "/Volumes/terminator/hpf/20190322_modelcheckpoints/20190403_models_w5_g/"
    try:
        path = opt.abbr + '_20190404_epoch50.pth.tar'
        model.load_state_dict(torch.load(src_dir+path, map_location='cpu'))
    except FileNotFoundError:
        path = opt.abbr + '_20190405_epoch50.pth.tar'
        model.load_state_dict(torch.load(src_dir + path, map_location='cpu'))
    model.eval()

    score = 0
    total = 0

    all_training_set, conept_labels = knn(opt.abbr)
    for query in unlabelled_data:
        print("***************NEW QUERY******************")
        print(query.features_left, query.features_right)
        output = model(query.embedding)
        choices = list(cui2meta[abbr])
        choices_idx = []
        for i in choices:
            choices_idx.append(word2idx[i])
        loss = nn.Softmax()
        choices_idx = torch.tensor(choices_idx)

        values, indices = loss(output.squeeze()[choices_idx]).max(0)
        print(values, indices)
        assigned_concept = idx2word[choices_idx[indices].item()]
        assigned_meta_id = cui2meta[abbr][assigned_concept]
        ground_truth = name2meta[abbr][query.label]
        print(assigned_concept)
        print(meta2name[abbr][assigned_meta_id])
        if assigned_meta_id == ground_truth:
            score += 1

        total += 1
    print(abbr + " from " + "casi" + ": CORRECT=" + str(score) + " TOTAL=" + str(total))
    results["casi"][abbr] = [score, total]

    score = 0
    total = 0
    for query in mimic_test:
        print("***************NEW QUERY******************")
        print(query.features_left, query.features_right)
        output = model(query.embedding)
        choices = list(cui2meta[abbr])
        choices_idx = []
        for i in choices:
            choices_idx.append(word2idx[i])
        loss = nn.Softmax()
        choices_idx = torch.tensor(choices_idx)

        values, indices = loss(output.squeeze()[choices_idx]).max(0)
        print(values, indices)
        assigned_concept = idx2word[choices_idx[indices].item()]
        assigned_meta_id = cui2meta[abbr][assigned_concept]
        ground_truth = name2meta[abbr][query.label]
        print(assigned_concept)
        print(meta2name[abbr][assigned_meta_id])
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

    abbr_dict = ["ac", "asa", "av", "avr", "bal", "ca", "cea", "cvp",
                 "dc", "dm", "dt", "er", "im", "ir", "ivf", "le",
                 "op",  "otc", "pa", "pac", "pcp", "pda", "pe", "pr", "ra", "rt", "sbp", "sma"]

    for abbr in abbr_dict:
        opt.abbr = abbr
        label_data(opt)

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