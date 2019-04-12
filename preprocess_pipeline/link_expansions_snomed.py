import pickle
import argparse


SNOMED_DICT = None
path_to_snomed_dict = "./snomed_dict.pickle"

abbr2snomed = {}

def load_snomed_dict():
    global SNOMED_DICT
    pickle_in = open(path_to_snomed_dict, 'rb')
    SNOMED_DICT = pickle.load(pickle_in)
    pickle_in.close()

def link_expansion_to_snomed(opt):
    compiled_abbrs = opt.compiled_abbrs
    for key in compiled_abbrs:
        abbr2snomed[key] = set()
        for expansion in compiled_abbrs[key]:
            if expansion in SNOMED_DICT["name2id"]:
                abbr2snomed[key].add(SNOMED_DICT["name2id"][expansion])




def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    # parser.add_argument('-snomed_dict', required=True, help="SNOMED dict that contains following dicts: child2parent,"
                                                            # "id2name, name2id")
    parser.add_argument('-compiled_abbrs',  required=True, help="dict where key is abbr, value is list of expansions")
    parser.add_argument('-outputfile', help=".pickle file to store dictionary with key as abbr or length")

    opt = parser.parse_args()
    link_expansion_to_snomed(opt)

if __name__ == "__main__":
   main()
