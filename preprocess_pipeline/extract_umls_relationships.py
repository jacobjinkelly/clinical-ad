
"""
values[0] = CUI1
values[3] = relationship of CUI2 --> CUI1
values[4] = CUI2
values[11] = ontology relationship was extracted from =
"""

import argparse
import pickle

def extract_to_text():
    f = open("umls_RelExtracted_full_20190304.txt", 'w')  # file to store parent-child rel
    g = open("umls_rel_full_20190304.txt", 'r')  # umls relationship file to get parent-child rel from
    columns = "CUI1|rel|CUI2|src"
    f.write(columns + '\n')
    for line in g:
        values = line.split("|")
        rel = values[3]
        if rel == "RB" or rel == "PAR":
            cui1 = values[0]
            cui2 = values[4]
            src = values[11]
            extraced_rel = cui1 + "|" + rel + "|" + cui2 + "|" + src
            f.write(extraced_rel + '\n')
    g.close()
    f.close()


def extract_to_dict():
    child2parent = {}

    fname = open("umls_id2name_20190310.pickle", 'rb')
    f = pickle.load(fname)
    fname.close()

    for key in f:
        child2parent[key] = set()

    g = open("umls_rel_full_20190304.txt", 'r')
    for line in g:
        values = line.split("|")
        rel = values[3]
        child_v = values[0]
        if child_v not in child2parent:
            child2parent[child_v] = set()
        if rel == "RB" or rel == "PAR":
            parent_v = values[4]
            src = values[11]
            entry = (parent_v, rel, src)
            child2parent[child_v].add(entry)
    g.close()

    umls_dict = {
        "child2parent": child2parent
    }

    pickle_out = open("umls_child2parent_20190304.pickle", 'wb')
    pickle.dump(umls_dict, pickle_out)
    pickle_out.close()

    print("There are " + str(len(child2parent)) + " parent \U000027A1 child relationships")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-txt', action="store_true", help="extract par-child rel to text file in format"
                                                    "CUI1|rel|CUI2|src")
    parser.add_argument('-dict', action="store_true", help="extract par-child rel to dict in pickle format"
                                                    "CUI_ch = set((CUI_par, rel, src))")


    opt = parser.parse_args()
    if opt.txt:
        extract_to_text()
    else:
        extract_to_dict()

    print("Finished extracting UMLS parent \U000027A1 child relationships \U00002755")

if __name__ == "__main__":
    main()