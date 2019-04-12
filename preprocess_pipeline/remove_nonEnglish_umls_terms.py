from clean_terms import clean_word
import nltk
import inflect
import pickle

inflect = inflect.engine()
nonenglish_id2name = {}

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


def find_nonenglish_terms():
    umls_def = open("umls_description_full_20190304.txt").readlines()
    for x in umls_def:
        tokens = x[:-1].split("|")
        lang = tokens[1]
        if lang == "ENG":
            continue
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
            nonenglish_id2name[cui].add(term)
        except KeyError:
            nonenglish_id2name[cui] = set()
            nonenglish_id2name[cui].add(term)

        if plural_or_sing_noun != "":

            nonenglish_id2name[cui].add(plural_or_sing_noun)

    umls_dict = {
        "nonenglish_id2name": nonenglish_id2name
    }

    pickle_out = open("umls_nonenglishterms_20190310.pickle", 'wb')
    pickle.dump(umls_dict, pickle_out)
    pickle_out.close()

    print("There are " + str(len(nonenglish_id2name)) + " nonenglish concepts!")

def remove_nonenglish_terms_from_umls_dicts():
    p_in = open("umls_nonenglishterms_20190310.pickle", 'rb')
    p = pickle.load(p_in)
    p_in.close()

    p_in = open("umls_cuiDescriptions_20190505.pickle", 'rb')
    o = pickle.load(p_in)
    p_in.close()

    nonenglish_term_dict = p["nonenglish_id2name"]

    id2name = o["id2name"]
    name2id = o["name2id"]
    name2src = o["name2src"]

    for cui in nonenglish_term_dict:
        for term in nonenglish_term_dict[cui]:
            if term in id2name[cui]:
                print(term)
                id2name[cui].remove(term)
            if term in name2id:
                del name2id[term]
            if term in name2src:
                del name2src[term]

    umls_dict = {
        "id2name": id2name,
        "name2id": name2id,
        "name2src": name2src
    }

    pickle_out = open("umls_cuiDescriptions_20190310.pickle", 'wb')
    pickle.dump(umls_dict, pickle_out)
    pickle_out.close()

    print("Finished removing nonEnglish terms from UMLS dicts!!! \U0001F422 \U000026F1")

def main():
    #find_nonenglish_terms()
    remove_nonenglish_terms_from_umls_dicts()

if __name__ == "__main__":
    main()