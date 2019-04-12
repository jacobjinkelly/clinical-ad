import argparse
import pickle
from localglobalembed import AbbrRep

def get_abbr_distr(data):
    print("CASI:")

    key = "casi_abbr"
    new_dict = {}
    for subkey in data[key]:
        new_dict[subkey] =len(data[key][subkey])
    print(new_dict)
    print("MIMIC_RS:")
    key = "mimic_rs"
    new_dict = {}
    for subkey in data[key]:
        new_dict[subkey] = len(data[key][subkey])
    print(new_dict)
    print("MIMIC_RS_3N:")
    key = "mimic_rs_sim"
    new_dict = {}
    for subkey in data[key]:
        new_dict[subkey] = len(data[key][subkey])
    print(new_dict)

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True)

    opt = parser.parse_args()

    pickle_in = open(opt.dataset, 'rb')
    data = pickle.load(pickle_in)
    get_abbr_distr(data)

if __name__ == "__main__":
   main()