
from clean_terms import clean_word
import nltk
import inflect
import pickle

inflect = inflect.engine()
id2name = {}
name2id = {}
name2src = {}

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

def extract_to_dict():
    umls_def = open("umls_description_full_20190304.txt").readlines()
    #umls_def = open("umls_test.txt").readlines()
    for x in umls_def:
        tokens = x[:-1].split("|")
        cui = tokens[0]
        src = tokens[11]
        descr = tokens[14]
        term = ' '.join(clean_word(descr).split())
        if term == "":
            continue
        try:
            plural_or_sing_noun = get_plural_or_sing_noun(term)
        except:
            continue

        try:
            id2name[cui].add(term)
        except KeyError:
            id2name[cui] = set()
            id2name[cui].add(term)

        try:
            name2id[term].add(cui)
        except KeyError:
            name2id[term] = set()
            name2id[term].add(cui)

        try:
            name2src[term].add(src)
        except KeyError:
            name2src[term] = set()
            name2src[term].add(src)

        if plural_or_sing_noun != "":

            id2name[cui].add(plural_or_sing_noun)

            try:
                name2id[plural_or_sing_noun].add(cui)
            except KeyError:
                name2id[plural_or_sing_noun] = set()
                name2id[plural_or_sing_noun].add(cui)

            try:
                name2src[plural_or_sing_noun].add(src)
            except KeyError:
                name2src[plural_or_sing_noun] = set()
                name2src[plural_or_sing_noun].add(src)





def main():
    extract_to_dict()
    umls_dict = {
        "id2name": id2name,
        "name2id": name2id,
        "name2src": name2src
    }

    pickle_out = open("umls_cuiDescriptions_20190505.pickle", 'wb')
    pickle.dump(umls_dict, pickle_out)
    pickle_out.close()

    print("There are " + str(len(id2name)) + " concepts!")
    print("Finished creating dictionary of UMLS descriptions \U0001F422 \U000026F1")

if __name__ == "__main__":
    main()