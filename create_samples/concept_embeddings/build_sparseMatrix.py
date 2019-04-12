concept_dir = '/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/all_umls_cuis.txt'
childparent_dir = '/Users/Marta/80k_abbreviations/preprocess_pipeline/umls_child2parent_20190304.pickle'
import pickle
import argparse
import os
import numpy
import scipy.sparse
import random
from tqdm import tqdm
from sys import getsizeof
import datetime

src_dir = "/Users/Marta/80k_abbreviations/allacronyms"
#f = open(os.path.join(src_dir, 'cleaned_allacronyms_dict_20190318.pickle'), 'rb')
f = open(os.path.join(src_dir, "cleaned_allacronyms_dict_20190402_NEW.pickle"), 'rb')

abbr_expansion_dict = pickle.load(f)
f.close()

#g = open(os.path.join(src_dir, 'allacronyms_cui2meta_20190318.pickle'), 'rb')
g = open(os.path.join(src_dir, 'allacronyms_cui2meta_20190402_NEW.pickle'), 'rb')
cui2meta = pickle.load(g)
g.close()

def get_concepts_mimic():
    g = open("cuis_in_mimic.txt", 'r')
    mimic = []
    for line in g:
        c = line[:-5]
        mimic.append(c)
    return mimic

def get_abbr_expansions(abbrs_to_get_expansions_for):

    expansion_concepts = []
    try:
        for abbr in abbrs_to_get_expansions_for:

            for key in abbr_expansion_dict[abbr]:
                expansion_concepts.extend(list(abbr_expansion_dict[abbr][key].keys()))
    except:
        print("Couldn't get expasnions for abbr: " + abbr)
    print(expansion_concepts)
    return expansion_concepts

def get_all():
    expansion_concepts = set()
    for abbr in cui2meta:
        for key in cui2meta[abbr]:
            expansion_concepts.add(key)
    print("EXPANSION CONCEPTS")
    print(len(expansion_concepts))
    return list(expansion_concepts)

parent2child = {}

# def get_concepts_umls(mimic):
def get_concepts_umls():

    p_in = open(childparent_dir, 'rb')
    child2parent_full = pickle.load(p_in)
    p_in.close()

    child2parent_full = child2parent_full['child2parent']
    child2parent = {}
    #for key in mimic:
     #   child2parent[key] = set()

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

    concepts = list(child2parent.keys())
    print(len(concepts))
    f = open("child2par.pickle", 'wb')
    pickle.dump(child2parent, f)
    f.close()
    return child2parent, concepts


def build_matrixHierarchy(chil2par, concepts_to_get, opt):
    src_dir = "/Users/Marta/80k_abbreviations/preprocess_pipeline"
    f = open(os.path.join(src_dir, 'umls_id2name_20190402.pickle'), 'rb')
    cui2name = pickle.load(f)
    f.close()

    num_concepts = len(concepts_to_get)
    print(len(concepts_to_get))

    global_ancestors = set()
    child2allancestors = {}
    for i in tqdm(range(num_concepts)):
        concept = concepts_to_get[i]
        if concept not in child2allancestors:
            child2allancestors[concept] = set()
            child2allancestors[concept].add(concept)
        try:
            parents_to_find = chil2par[concept]
        except:
            continue

        # GET TWO PARENTS ONLY
        # parents
        for par in parents_to_find:
            child2allancestors[concept].add(par)
            if par not in child2allancestors:
                child2allancestors[par] = set()
                child2allancestors[par].add(par)

        # grandparents
        for parent in parents_to_find:
            grandparents = chil2par[parent]
            for grandparent in grandparents:
                child2allancestors[concept].add(grandparent)
                child2allancestors[parent].add(grandparent)
                if grandparent not in child2allancestors:
                    child2allancestors[grandparent] = set()
                    child2allancestors[grandparent].add(grandparent)

        # siblings
        if opt.sib:
            random.seed(16)
            for par in parents_to_find:
                sibs = list(parent2child[par])
                random.shuffle(sibs)
                num_sibs = min(2, len(sibs))
                for j in range(num_sibs):
                    ancestors.add(sibs[j])

    word2idx, idx2word = vectorize_concepts(list(child2allancestors.keys()))

    num_cuis = len(word2idx)
    mtx = scipy.sparse.lil_matrix((num_cuis, num_cuis), dtype=numpy.int8)
    for item in child2allancestors:
        idx = word2idx[item]
        mtx[idx, idx] = 1
        for ancestor in child2allancestors[item]:
            ancestor_ = word2idx[ancestor]
            mtx[idx, ancestor_] = 1
    global_ancestors = list(child2allancestors.keys())

    return mtx, global_ancestors, word2idx, idx2word


def vectorize_concepts(concepts):
    word2idx = {}
    idx2word = {}
    counter = 0
    for concept in concepts:
        word2idx[concept] = counter
        idx2word[counter] = concept
        counter += 1
    return word2idx, idx2word


def get_umls_concepts(opt):

    #mimic = get_concepts_mimic()
    chil2par, concepts = get_concepts_umls()

    if opt.expansions:
        expansions = opt.expansions.split(",")
        expansion_concepts = get_abbr_expansions(expansions)
    elif opt.all:
        expansion_concepts = get_all()

    matrix_hierarchy, global_ancestors, word2idx, idx2word = build_matrixHierarchy(chil2par, expansion_concepts, opt)


    save = True
    if len(global_ancestors) == 0:
        save = False
        print(opt.expansions + " has no SM!!!")
    # if opt.all:
    #     all_ancestors = global_ancestors
    #     global_ancestors = expansion_concepts
    #     print("FINISH")
    #     print(list(global_ancestors)[:100])
    if save:
        umls_rel = {
            "matrix_hierarchy": matrix_hierarchy,
            "word2idx": word2idx,
            "idx2word": idx2word,
            "all_concepts": global_ancestors
        }

        target_dir = '/Users/Marta/80k_abbreviations/create_samples/concept_embeddings/sparse_matrices_20190406'
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        if opt.outputfile:
            file_name = opt.outputfile
        else:
            date = "20190406"

            file_name = "umls_ancestor_sparseMatrix_" + str(date) + "_full.pickle"
            if opt.expansions:
                file_name = opt.expansions + "_" + file_name
            if opt.all:
                file_name = "all_" + file_name
        pickle_out = open(os.path.join(target_dir, file_name), 'wb')
        pickle.dump(umls_rel, pickle_out)
        pickle_out.close()

    print("Done creating CUI hierarchy matrix! \U0001F335 \U0001F33A")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-expansions', default=None, help="comma separated list of abbreviations to get "
                                                          "e.g. ivf,ca,ab --> create ONE sm")
    parser.add_argument('-all', action="store_true", help="get all abbr expansions")
    parser.add_argument('-outputfile', help="name of outputfile for sparse matrix")
    parser.add_argument('-sib', action="store_true", help="get concept siblings too")
    parser.add_argument('-abbrlist', help="generate sparse matrices from a list, one per line --> create ONE sm PER abbr")

    opt = parser.parse_args()

    if opt.abbrlist:
        abbrs = open(opt.abbrlist, 'r').readlines()
        for abbr in abbrs:
            opt.expansions = abbr[:-1]
            print(opt.expansions)
            get_umls_concepts(opt)
    else:
        get_umls_concepts(opt)


if __name__ == "__main__":
    main()
