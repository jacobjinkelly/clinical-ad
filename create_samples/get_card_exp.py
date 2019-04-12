import pickle
import os

def get_card_expansions():
    dir = "/Users/Marta/80k_abbreviations/allacronyms"
    n = open(os.path.join(dir,"allacronyms_cui2meta_20190318.pickle"), 'rb')
    cui2meta = pickle.load(n)
    n.close()

    m = open(os.path.join(dir, "allacronyms_meta2cui_20190318.pickle"), 'rb')
    meta2cui = pickle.load(m)
    m.close()

    o = open(os.path.join(dir, "allacronyms_meta2name_20190318.pickle"), 'rb')
    meta2name = pickle.load(o)
    o.close()

    p = open(os.path.join(dir, "allacronyms_name2meta_20190318.pickle"), 'rb')
    name2meta = pickle.load(p)
    p.close()

    card_dir = "/Users/Marta/Desktop/CARD_javafiles/"
    card = open(os.path.join(card_dir, "12000_pathology_abbreviations.txt"), 'r').readlines()
    card_dict = {}
    for item in card:
        item = item[:-1].split(" = ")
        try:
            abbr = item[0]
            exp = ' '.join(item[1].split())
        except:
            continue
        try:
            card_dict[abbr].add(exp)
        except KeyError:
            card_dict[abbr] = set()
            card_dict[abbr].add(exp)
    metas2keep = {}
    for abbr in card_dict:
        metas2keep[abbr] = set()
        for exp in card_dict[abbr]:
            if abbr not in name2meta or exp not in name2meta[abbr]:
                continue
            metas2keep[abbr].add(name2meta[abbr][exp])

    return metas2keep

