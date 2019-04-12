import pickle
from clean_terms import clean_word
import nltk
import inflect
import sys

parents = {}
id2name = {}
name2id = {}
inflect = inflect.engine()


def get_plural(term):
    return inflect.plural_noun(term)


def get_singular(term):
    return inflect.singular_noun(term)


def get_plural_or_sing_noun(term):
    term_list = [term]
    pos_tagged = nltk.pos_tag(term_list)

    tag = pos_tagged[0][1]
    if tag == "NN":
        return get_plural(term)
    elif tag == "NNS":
        return get_singular(term)
    else:
        return ""


def load_snomed_relationships():
    snomed_taxonomy = open("sct2_StatedRelationship_Full_INT_20170731.txt").readlines()
    for x in snomed_taxonomy:
        tokens = x.strip().split("\t")
        if tokens[7] != '116680003':
            continue
        child_v = tokens[4]
        parent_v = tokens[5]
        if child_v not in parents:
            parents[child_v] = set()
        if tokens[2] != '1':
            if parent_v in parents[child_v]:
                parents[child_v].remove(parent_v)
            continue
        parents[child_v].add(parent_v)

def load_snomed_descriptions():
    snomed_def = open("sct2_Description_Full-en_INT_20170731.txt").readlines()
    for x in snomed_def:
        tokens = x.strip().split("\t")
        if tokens[2] != '1':
            continue
        sid = tokens[4]
        term = tokens[7]
        x = term
        term = ' '.join(clean_word(term).split())
        if term == "":
            continue
        try:
            plural_or_sing_noun = get_plural_or_sing_noun(term)
        except:
            continue
        if sid not in parents:
            continue
        if sid not in id2name:
            id2name[sid] = set()

        else:
            try:
                name2id[term].add(sid)
            except KeyError:
                name2id[term] = set()
                name2id[term].add(sid)
            if plural_or_sing_noun != "":
                try:
                    name2id[plural_or_sing_noun].add(sid)
                except KeyError:
                    name2id[plural_or_sing_noun] = set()
                    name2id[plural_or_sing_noun].add(sid)
        id2name[sid].add(term)
        if plural_or_sing_noun != "":
            id2name[sid].add(plural_or_sing_noun)

    print("Finished loading SNOMED files \U00002744 \U000026C4")

if __name__ == "__main__":
    load_snomed_relationships()
    load_snomed_descriptions()

    snomed_dict = {
        "child2parent": parents,
        "id2name": id2name,
        "name2id": name2id
    }

    pickle_out = open("snomed_dict.pickle", 'wb')
    pickle.dump(snomed_dict, pickle_out)
    pickle_out.close()
